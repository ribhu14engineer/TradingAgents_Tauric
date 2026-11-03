"""Google News RSS search: dated headlines with wide non-US coverage.

Yahoo tags no article with most non-US listings (RELIANCE.NS, TCS.NS), so its
ticker feed is empty for them. Google News search covers those markets' press
by company name, and every item carries a publish time, so a historical window
filters on the same look-ahead rule as the other news vendors. The search's own
after:/before: operators only narrow what is fetched; ``in_window`` decides
what the run may read.
"""

import functools
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import requests
from dateutil.relativedelta import relativedelta

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import coverage_gap, in_window
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.fundamentals import get_company_profile

_RSS_URL = "https://news.google.com/rss/search"
# English results from the Indian edition: it carries the Indian business press
# (Economic Times, Mint, Moneycontrol) alongside the wires (Reuters, Bloomberg).
_EDITION = {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
_TIMEOUT = 20

# Legal-form suffixes that make a name search miss headlines: the press writes
# "Reliance Industries", not "Reliance Industries Limited".
_LEGAL_SUFFIX = re.compile(
    r"[\s,]+(limited|ltd\.?|inc\.?|incorporated|corp\.?|corporation|plc|co\.?|company|"
    r"holdings?|group|s\.?a\.?|ag|n\.?v\.?|se)$",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=256)
def _company_name(canonical: str) -> str:
    """The name the press uses for ``canonical``, or its bare symbol when unknown."""
    try:
        info = get_company_profile(canonical)
    except Exception:  # noqa: BLE001 — a name lookup must not cost the news call
        info = {}
    name = (info.get("longName") or info.get("shortName") or "").strip()
    while name and _LEGAL_SUFFIX.search(name):
        name = _LEGAL_SUFFIX.sub("", name).strip()
    return name or canonical.split(".")[0]


def _search(query: str, start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Articles Google News returns for ``query`` around ``[start, end]``."""
    dated = (f"{query} after:{(start_dt - timedelta(days=1)):%Y-%m-%d} "
             f"before:{(end_dt + timedelta(days=1)):%Y-%m-%d}")
    try:
        response = requests.get(_RSS_URL, params={"q": dated, **_EDITION}, timeout=_TIMEOUT,
                                headers={"User-Agent": "Mozilla/5.0 (TradingAgents news)"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as exc:
        raise VendorUnavailableError(f"Google News search failed: {exc}") from exc

    articles = []
    for item in root.iter("item"):
        publisher = (item.findtext("source") or "").strip() or "Unknown"
        title = (item.findtext("title") or "").strip()
        # Titles end in " - <publisher>", which the source line already says.
        if title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3].rstrip()
        pub_date = None
        if item.findtext("pubDate"):
            try:
                pub_date = parsedate_to_datetime(item.findtext("pubDate"))
            except (TypeError, ValueError):
                pub_date = None
        articles.append({
            "title": title or "No title",
            "publisher": publisher,
            "link": (item.findtext("link") or "").strip(),
            "pub_date": pub_date,
        })
    return articles


def _format_articles(articles: list[dict]) -> str:
    out = ""
    for data in articles:
        when = f"{data['pub_date']:%Y-%m-%d}, " if data["pub_date"] else ""
        out += f"### {data['title']} ({when}source: {data['publisher']})\n"
        if data["link"]:
            out += f"Link: {data['link']}\n"
        out += "\n"
    return out


def _newest_first(articles: list[dict], start_dt: datetime, end_dt: datetime, limit: int) -> list[dict]:
    """The in-window articles, deduplicated by title, newest first, at most ``limit``."""
    seen, kept = set(), []
    for a in articles:
        if not in_window(a["pub_date"], start_dt, end_dt) or a["title"] in seen:
            continue
        seen.add(a["title"])
        kept.append(a)
    kept.sort(key=lambda a: a["pub_date"].timestamp() if a["pub_date"] else 0, reverse=True)
    return kept[:limit]


def get_news_google(ticker: str, start_date: str, end_date: str) -> str:
    """News about ``ticker``'s company published in the window, from Google News."""
    canonical = normalize_symbol(ticker)
    resolved = "" if canonical == ticker else f" (resolved to {canonical})"
    name = _company_name(canonical)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    found = _search(f'"{name}"', start_dt, end_dt)
    in_range = _newest_first(found, start_dt, end_dt, get_config()["news_article_limit"])
    if in_range:
        return (f"## {ticker}{resolved} News ({name}), from {start_date} to {end_date}:\n\n"
                f"{_format_articles(in_range)}")
    # The search asks for this window itself, so coverage reaches its first day:
    # an empty past window is a real absence, and only one past today is unobserved.
    gap = coverage_gap((start_dt,), start_date, end_date, "Google News", f"news for {ticker}{resolved}")
    return gap or f"No news found for {ticker}{resolved} ({name}) between {start_date} and {end_date}"


def get_global_news_google(
    as_of_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Macro headlines for the configured ``global_news_queries``, from Google News."""
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    # Each query gets an equal share of the budget, so the first topic cannot
    # crowd out the rest (central banks, flows, currency, energy...).
    queries = config["global_news_queries"]
    share = max(1, -(-limit // max(1, len(queries))))
    picked, seen = [], set()
    for query in queries:
        for a in _newest_first(_search(query, start_dt, curr_dt), start_dt, curr_dt, share):
            if a["title"] not in seen:
                seen.add(a["title"])
                picked.append(a)

    if not picked:
        gap = coverage_gap((start_dt,), start_date, as_of_date, "Google News global news", "market news")
        return gap or f"No global news found between {start_date} and {as_of_date}"
    return (f"## Global Market News, from {start_date} to {as_of_date}:\n\n"
            f"{_format_articles(picked[:limit])}")
