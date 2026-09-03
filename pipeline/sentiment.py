"""
Turns free-text commentary into structured features: which companies are
mentioned, and the sentiment around each mention.

No internet access in this environment means no FinBERT / transformers model
download. So this module is a deliberately simple, fully self-contained
baseline: a hand-built financial lexicon + stdlib fuzzy matching. This is a
legitimate v1 -- plenty of production systems start with lexicon scoring --
and the upgrade path (swap score_text's internals for a FinBERT call) doesn't
require touching anything else in the pipeline, because every other function
here only depends on score_text's *output shape*, not its internals.
"""
import difflib
import re

# --- Financial sentiment lexicon -----------------------------------------
# Deliberately domain-specific: generic English sentiment ("good"/"bad") misses
# finance-specific tone ("dilution", "restrictive", "beat") that generic
# lexicons don't weight correctly. Extend this list as you see more notes.
POSITIVE_WORDS = {
    "growth", "bullish", "buy", "rally", "strong", "beat", "raise", "raised",
    "opportunity", "upgrade", "outperform", "expansion", "recovery", "easy",
    "momentum", "breakout",
}
NEGATIVE_WORDS = {
    "deficit", "debt", "weak", "restrictive", "risk", "risks", "sacrifice",
    "dilution", "concern", "concerns", "bearish", "sell", "decline", "blow",
    "tough", "inflation", "sacrifice", "financed", "refinancing",
}

# --- Watchlist -------------------------------------------------------------
# Maps a lowercase company name (as it's likely to appear in prose) to a
# ticker symbol. THESE TICKERS ARE PLACEHOLDERS for the pipeline demo --
# verify actual exchange symbols before using this for anything real.
WATCHLIST = {
    "quess corp": "QUESSCORP",
    "cyient": "CYIENT",
    "glaxosmithkline": "GLAXO",
    "shyam metallics": "SHYAMMETL",
    "q power": "QPOWER",
    "morepen": "MOREPENLAB",
    "inox india": "INOXINDIA",
    "nhc foods": "NHCFOODS",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def score_text(text: str) -> dict:
    """Lexicon-based sentiment score in [-1, 1]. Swap this function's internals
    for a model call later without changing any caller."""
    words = _tokenize(text)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total_hits = pos + neg
    score = 0.0 if total_hits == 0 else (pos - neg) / total_hits
    return {"score": round(score, 3), "positive_hits": pos, "negative_hits": neg, "word_count": len(words)}


def extract_mentions(text: str, watchlist: dict = WATCHLIST, cutoff: float = 0.75) -> list[dict]:
    """
    Finds watchlist companies in text, tolerating small typos (the sample note
    has "Glaxosmithcline" for "GlaxoSmithKline" -- real commentary is full of
    this).

    IMPORTANT: a candidate phrase of length L is only ever fuzzy-matched
    against watchlist keys that ALSO have L words. Comparing across word
    counts (e.g. testing "cyient glaxosmithkline" against the single word
    "glaxosmithkline") lets difflib's ratio() find high overlap on a
    shared substring and merge two separate companies into one bogus
    match -- this bit us during development (see tests/test_sentiment.py)
    and is worth knowing about if you extend this matcher.
    """
    words = _tokenize(text)
    keys_by_length: dict[int, list[str]] = {}
    for key in watchlist:
        keys_by_length.setdefault(len(key.split()), []).append(key)
    lengths_desc = sorted(keys_by_length, reverse=True)

    matches = []
    i = 0
    n = len(words)
    while i < n:
        found = None
        for length in lengths_desc:
            if i + length <= n:
                phrase = " ".join(words[i:i + length])
                close = difflib.get_close_matches(phrase, keys_by_length[length], n=1, cutoff=cutoff)
                if close:
                    found = (close[0], length)
                    break
        if found:
            key, length = found
            matches.append({
                "matched_text": " ".join(words[i:i + length]),
                "company": key,
                "ticker": watchlist[key],
                "word_index": i,
            })
            i += length
        else:
            i += 1
    return matches


def score_mentions(text: str, mentions: list[dict], window: int = 12) -> list[dict]:
    """Scores sentiment in a word-window AROUND each mention, not the whole
    document -- a note can be bearish on the market and bullish on one stock."""
    words = _tokenize(text)
    scored = []
    for m in mentions:
        start = max(0, m["word_index"] - window)
        end = min(len(words), m["word_index"] + window)
        local_text = " ".join(words[start:end])
        local_score = score_text(local_text)
        scored.append({**m, "local_sentiment": local_score["score"]})
    return scored


def process_note(note: dict) -> dict:
    """Full pipeline for one note: macro sentiment + per-company mentions."""
    macro = score_text(note["text"])
    mentions = extract_mentions(note["text"])
    mentions = score_mentions(note["text"], mentions)
    return {
        "date": note["date"],
        "source_file": note["source_file"],
        "macro_sentiment": macro["score"],
        "mentions": mentions,
    }


if __name__ == "__main__":
    from pipeline.notes_loader import load_notes_folder

    for note in load_notes_folder():
        result = process_note(note)
        print(f"\n{result['date']}  macro_sentiment={result['macro_sentiment']}")
        for m in result["mentions"]:
            print(f"  - {m['company']:20s} ({m['ticker']:10s})  matched '{m['matched_text']}'  local_sentiment={m['local_sentiment']}")
