"""Google News RSS vendor: name search, look-ahead filtering, and failure typing."""

from types import SimpleNamespace

import pytest
import requests

import tradingagents.dataflows.vendors.google_news as gnews
from tradingagents.dataflows.errors import VendorUnavailableError


def _rss(*items):
    """An RSS body holding ``items`` as (title, publisher, RFC 2822 date) triples."""
    body = "".join(
        f"<item><title>{t} - {p}</title><link>https://news.example/{i}</link>"
        f"<pubDate>{d}</pubDate><source url='https://{p}.example'>{p}</source></item>"
        for i, (t, p, d) in enumerate(items)
    )
    return f"<?xml version='1.0'?><rss><channel>{body}</channel></rss>".encode()


@pytest.fixture
def served(monkeypatch):
    """Serve a fixed RSS body for every search and record the queries asked."""
    asked = []

    def serve(body):
        def get(url, params=None, timeout=None, headers=None):
            asked.append(params["q"])
            return SimpleNamespace(content=body, raise_for_status=lambda: None)
        monkeypatch.setattr(gnews.requests, "get", get)
        return asked

    monkeypatch.setattr(gnews, "get_company_profile",
                        lambda t: {"longName": "Reliance Industries Limited"})
    gnews._company_name.cache_clear()
    yield serve
    gnews._company_name.cache_clear()


@pytest.mark.unit
def test_searches_the_company_name_without_its_legal_suffix(served):
    asked = served(_rss(("Jio IPO approved", "Reuters", "Mon, 31 Aug 2026 10:00:00 GMT")))
    out = gnews.get_news_google("RELIANCE.NS", "2026-08-25", "2026-09-01")
    assert asked[0].startswith('"Reliance Industries" after:')
    assert "Jio IPO approved (2026-08-31, source: Reuters)" in out
    assert " - Reuters" not in out


@pytest.mark.unit
def test_articles_after_the_window_never_reach_the_run(served):
    served(_rss(
        ("Before", "Mint", "Fri, 25 Sep 2026 09:00:00 GMT"),
        ("Inside", "Mint", "Mon, 28 Sep 2026 09:00:00 GMT"),
        ("Leak", "Mint", "Tue, 29 Sep 2026 09:00:00 GMT"),
    ))
    out = gnews.get_news_google("RELIANCE.NS", "2026-09-26", "2026-09-28")
    assert "Inside" in out
    assert "Leak" not in out and "Before" not in out


@pytest.mark.unit
def test_an_empty_past_window_reads_as_no_news(served):
    served(_rss())
    out = gnews.get_news_google("RELIANCE.NS", "2026-09-26", "2026-09-28")
    assert out.startswith("No news found for RELIANCE.NS")


@pytest.mark.unit
def test_a_failed_request_lets_the_next_vendor_try(monkeypatch, served):
    served(_rss())

    def fail(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(gnews.requests, "get", fail)
    with pytest.raises(VendorUnavailableError):
        gnews.get_news_google("RELIANCE.NS", "2026-09-26", "2026-09-28")


@pytest.mark.unit
def test_global_news_runs_every_configured_query(monkeypatch, served):
    asked = served(_rss(("RBI hikes repo rate", "Economic Times", "Wed, 07 Oct 2026 08:00:00 GMT")))
    config = {"global_news_lookback_days": 7, "global_news_article_limit": 10,
              "global_news_queries": ["RBI repo rate", "Nifty FII flows"]}
    monkeypatch.setattr(gnews, "get_config", lambda: config)
    out = gnews.get_global_news_google("2026-10-08")
    assert [q.split(" after:")[0] for q in asked] == ["RBI repo rate", "Nifty FII flows"]
    assert out.count("RBI hikes repo rate") == 1  # the same headline from two queries is kept once
