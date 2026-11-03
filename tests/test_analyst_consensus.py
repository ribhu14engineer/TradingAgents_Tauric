"""Analyst consensus: live-only, framed as opinion, and honest about empty coverage."""

import pandas as pd
import pytest

import tradingagents.dataflows.vendors.yahoo.fundamentals as yfund
from tradingagents.agents.analysts import fundamentals_analyst
from tradingagents.dataflows.date_window import get_current_date


def _ticker(targets=None, recs=None):
    return type("T", (), {"analyst_price_targets": targets, "recommendations": recs})()


def _trend():
    return pd.DataFrame([
        {"period": "0m", "strongBuy": 6, "buy": 20, "hold": 0, "sell": 0, "strongSell": 0},
        {"period": "-1m", "strongBuy": 5, "buy": 19, "hold": 0, "sell": 0, "strongSell": 1},
    ])


@pytest.mark.unit
def test_a_past_run_is_not_served_todays_consensus(monkeypatch):
    def leaked(symbol):
        raise AssertionError("a historical run must not reach the vendor")
    monkeypatch.setattr(yfund.yf, "Ticker", leaked)
    out = yfund.get_analyst_consensus("RELIANCE.NS", "2026-08-03")
    assert "withheld" in out


@pytest.mark.unit
def test_a_live_run_reads_counts_and_targets_as_opinion(monkeypatch):
    targets = {"current": 1300.0, "mean": 1666.0, "median": 1690.0, "low": 1350.0, "high": 1890.0}
    monkeypatch.setattr(yfund.yf, "Ticker", lambda s: _ticker(targets, _trend()))
    out = yfund.get_analyst_consensus("RELIANCE.NS", get_current_date())
    assert "opinions, not data" in out
    assert "| This month | 6 | 20 | 0 | 0 | 0 | 26 |" in out
    assert "| 1m ago | 5 | 19 | 0 | 0 | 1 | 25 |" in out
    assert "mean 1,666.00" in out and "+28.2%" in out


@pytest.mark.unit
def test_no_coverage_is_said_plainly(monkeypatch):
    monkeypatch.setattr(yfund.yf, "Ticker", lambda s: _ticker({}, pd.DataFrame()))
    monkeypatch.setattr(yfund, "vendor_reachable", lambda url: True)
    assert yfund.get_analyst_consensus("TINY.NS", None).startswith("No analyst coverage")


@pytest.mark.unit
def test_the_fundamentals_analyst_is_offered_the_tool():
    assert "get_analyst_consensus" in {t.name for t in fundamentals_analyst.TOOLS}
