"""
Tests for pipeline/sentiment.py.

test_no_cross_length_bleeding is a regression test for a real bug found
while building this: fuzzy-matching a 2-word phrase against a 1-word
watchlist key let difflib merge two adjacent companies ("Cyient" and
"GlaxoSmithKline") into one bogus match. Restricting comparisons to
same-word-count keys fixed it -- this test makes sure it stays fixed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.sentiment import extract_mentions, score_text, WATCHLIST


def test_extracts_all_watchlist_companies_from_sample_note():
    text = (
        "Weekly Ideas: Quess corp, Cyient, Glaxosmithcline, Shyam metallics, "
        "Q Power. Morepan & Inox India should also be in radar. NHC Foods too."
    )
    mentions = extract_mentions(text)
    found_tickers = {m["ticker"] for m in mentions}
    expected_tickers = set(WATCHLIST.values())
    assert found_tickers == expected_tickers, f"Missing: {expected_tickers - found_tickers}"


def test_no_cross_length_bleeding():
    """Adjacent single-word companies must not fuse into one match."""
    text = "Cyient Glaxosmithcline"
    mentions = extract_mentions(text)
    assert len(mentions) == 2, f"Expected 2 separate matches, got {mentions}"
    matched_texts = {m["matched_text"] for m in mentions}
    assert matched_texts == {"cyient", "glaxosmithcline"}


def test_typo_tolerance_within_reason():
    """Small typos should still match; unrelated words should not."""
    assert len(extract_mentions("Glaxosmithcline reported earnings")) == 1
    assert len(extract_mentions("completely unrelated sentence here")) == 0


def test_score_text_sign_matches_word_balance():
    positive_text = "strong growth bullish rally beat raise"
    negative_text = "weak deficit debt bearish decline concern"
    neutral_text = "the market opened today and closed later"

    assert score_text(positive_text)["score"] > 0
    assert score_text(negative_text)["score"] < 0
    assert score_text(neutral_text)["score"] == 0
