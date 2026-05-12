from __future__ import annotations

from pathlib import Path

from core import DictionaryIndex


def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_lexicon"


def test_list_pairs() -> None:
    idx = DictionaryIndex(fixture_dir())
    pairs = idx.list_pairs()
    assert {"src_lang": "vie_Latn", "tgt_lang": "eng_Latn", "size": 3} in pairs


def test_exact_lookup() -> None:
    idx = DictionaryIndex(fixture_dir())
    out = idx.lookup("vie_Latn", "eng_Latn", "đi", top_k=3, offset=0)
    assert out["match_type"] == "contains"
    assert out["total_matches"] >= 1
    assert out["results"][0]["target_text"] == "to go"


def test_contains_lookup() -> None:
    idx = DictionaryIndex(fixture_dir())
    out = idx.lookup("vie_Latn", "eng_Latn", "nhà", top_k=3, offset=0)
    assert out["match_type"] == "contains"
    assert out["results"][0]["source_text"] == "nhà"


def test_examples_in_results() -> None:
    idx = DictionaryIndex(fixture_dir())
    out = idx.lookup("vie_Latn", "eng_Latn", "đi", top_k=3, offset=0)
    assert out["results"][0].get("examples")


def test_lookup_offset_pagination() -> None:
    idx = DictionaryIndex(fixture_dir())
    out1 = idx.lookup("vie_Latn", "eng_Latn", "n", top_k=1, offset=0)
    out2 = idx.lookup("vie_Latn", "eng_Latn", "n", top_k=1, offset=1)
    assert out1["total_matches"] >= 2
    assert out1["results"]
    assert out2["results"]
    assert out1["results"][0]["source_text"] != out2["results"][0]["source_text"]


def test_search_pairs() -> None:
    idx = DictionaryIndex(fixture_dir())
    out = idx.search_pairs("nước", top_k=10)
    assert len(out["results"]) >= 1
    assert all(row["src_lang"] == "vie_Latn" for row in out["results"])
