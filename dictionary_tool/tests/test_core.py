from __future__ import annotations

from pathlib import Path

from core import DictionaryIndex


def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_lexicon"


def test_list_pairs() -> None:
    idx = DictionaryIndex(fixture_dir())
    pairs = idx.list_pairs()
    assert {"src_lang": "eng_Latn", "tgt_lang": "zho_Hans", "size": 3} in pairs
    assert {"src_lang": "zho_Hans", "tgt_lang": "eng_Latn", "size": 3} in pairs


def test_exact_lookup() -> None:
    idx = DictionaryIndex(fixture_dir())
    out = idx.lookup("eng_Latn", "zho_Hans", "dictionary", top_k=3)
    assert out["match_type"] == "exact"
    assert out["results"][0]["target_text"] == "词典"


def test_contains_lookup() -> None:
    idx = DictionaryIndex(fixture_dir())
    out = idx.lookup("eng_Latn", "zho_Hans", "lang", top_k=3)
    assert out["match_type"] == "contains"
    assert out["results"][0]["source_text"] == "language"


def test_traditional_chinese_normalization() -> None:
    idx = DictionaryIndex(fixture_dir())
    out = idx.lookup("zho_Hans", "eng_Latn", "后", top_k=3)
    assert out["results"][0]["target_text"] == "after"


def test_search_pairs() -> None:
    idx = DictionaryIndex(fixture_dir())
    out = idx.search_pairs("dictionary", top_k=10)
    assert len(out["results"]) >= 2
    assert any(row["src_lang"] == "eng_Latn" for row in out["results"])
