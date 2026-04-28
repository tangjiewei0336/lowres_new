from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


TRAD_TO_SIMP = str.maketrans(
    {
        "歲": "岁",
        "隊": "队",
        "長": "长",
        "國": "国",
        "學": "学",
        "會": "会",
        "說": "说",
        "這": "这",
        "個": "个",
        "們": "们",
        "來": "来",
        "時": "时",
        "對": "对",
        "發": "发",
        "過": "过",
        "還": "还",
        "後": "后",
        "裡": "里",
        "為": "为",
        "與": "与",
        "開": "开",
        "關": "关",
        "點": "点",
        "種": "种",
        "體": "体",
        "頭": "头",
        "無": "无",
        "業": "业",
        "東": "东",
        "車": "车",
        "門": "门",
        "問": "问",
        "間": "间",
        "現": "现",
        "電": "电",
        "話": "话",
        "書": "书",
        "買": "买",
        "賣": "卖",
        "貓": "猫",
        "馬": "马",
        "鳥": "鸟",
        "魚": "鱼",
        "風": "风",
        "愛": "爱",
        "聽": "听",
        "讀": "读",
        "寫": "写",
        "線": "线",
        "網": "网",
        "氣": "气",
        "漢": "汉",
        "語": "语",
        "萬": "万",
    }
)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", clean_text(text)).casefold()
    text = text.translate(TRAD_TO_SIMP)
    return re.sub(r"\s+", " ", text)


@dataclass(frozen=True)
class DictionaryEntry:
    source: str
    src_lang: str
    tgt_lang: str
    source_text: str
    target_text: str
    target_candidates: list[str]
    confidence: float
    relation: str | None
    source_url: str | None
    license_note: str | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "DictionaryEntry":
        candidates = row.get("target_candidates")
        if not isinstance(candidates, list):
            candidates = [row.get("target_text")] if isinstance(row.get("target_text"), str) else []
        return cls(
            source=str(row.get("source", "")),
            src_lang=str(row.get("src_lang", "")),
            tgt_lang=str(row.get("tgt_lang", "")),
            source_text=str(row.get("source_text", "")),
            target_text=str(row.get("target_text", "")),
            target_candidates=[str(x) for x in candidates if isinstance(x, str)],
            confidence=float(row.get("confidence", 0.0) or 0.0),
            relation=str(row.get("relation")) if row.get("relation") is not None else None,
            source_url=str(row.get("source_url")) if row.get("source_url") is not None else None,
            license_note=str(row.get("license_note")) if row.get("license_note") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "target_candidates": self.target_candidates,
            "confidence": self.confidence,
            "relation": self.relation,
            "source_url": self.source_url,
            "license_note": self.license_note,
        }


class DictionaryIndex:
    def __init__(self, lexicon_dir: Path) -> None:
        self.lexicon_dir = lexicon_dir
        self._pair_entries: dict[tuple[str, str], list[DictionaryEntry]] = {}
        self._pair_exact: dict[tuple[str, str], dict[str, list[DictionaryEntry]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.lexicon_dir.is_dir():
            raise FileNotFoundError(f"Lexicon directory not found: {self.lexicon_dir}")
        for path in sorted(self.lexicon_dir.glob("dict_terms_*__*.jsonl")):
            entries: list[DictionaryEntry] = []
            exact: dict[str, list[DictionaryEntry]] = {}
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    entry = DictionaryEntry.from_row(row)
                    if not entry.src_lang or not entry.tgt_lang or not entry.source_text:
                        continue
                    entries.append(entry)
                    key = normalize_key(entry.source_text)
                    exact.setdefault(key, []).append(entry)
            if not entries:
                continue
            pair = (entries[0].src_lang, entries[0].tgt_lang)
            self._pair_entries[pair] = entries
            self._pair_exact[pair] = exact

    def list_pairs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for (src_lang, tgt_lang), entries in sorted(self._pair_entries.items()):
            out.append({"src_lang": src_lang, "tgt_lang": tgt_lang, "size": len(entries)})
        return out

    def lookup(
        self,
        src_lang: str,
        tgt_lang: str,
        term: str,
        top_k: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        pair = (src_lang, tgt_lang)
        if pair not in self._pair_entries:
            raise ValueError(f"Dictionary pair not found: {src_lang}->{tgt_lang}")

        key = normalize_key(term)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if offset < 0:
            raise ValueError("offset must be >= 0")

        # 统一为包含关系检索（包含 exact），便于分页遍历全部命中。
        contains_matches = [
            entry
            for entry in self._pair_entries[pair]
            if key in normalize_key(entry.source_text) or normalize_key(entry.source_text) in key
        ]
        ranked = sorted(
            contains_matches,
            key=lambda x: (
                normalize_key(x.source_text) != key,
                -x.confidence,
                len(x.source_text),
                x.target_text,
            ),
        )
        paged = ranked[offset : offset + top_k]
        return {
            "query": term,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "match_type": "contains",
            "total_matches": len(ranked),
            "offset": offset,
            "top_k": top_k,
            "has_more": offset + top_k < len(ranked),
            "results": [x.to_dict() for x in paged],
        }

    def search_pairs(
        self,
        term: str,
        *,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        key = normalize_key(term)
        rows: list[dict[str, Any]] = []
        for (pair_src, pair_tgt), entries in sorted(self._pair_entries.items()):
            if src_lang and pair_src != src_lang:
                continue
            if tgt_lang and pair_tgt != tgt_lang:
                continue
            for entry in entries:
                source_key = normalize_key(entry.source_text)
                if key == source_key or key in source_key or source_key in key:
                    row = entry.to_dict()
                    row["match_type"] = "exact" if key == source_key else "contains"
                    rows.append(row)
        rows.sort(
            key=lambda x: (
                x["match_type"] != "exact",
                -float(x.get("confidence", 0.0) or 0.0),
                x.get("src_lang", ""),
                x.get("tgt_lang", ""),
                x.get("source_text", ""),
            )
        )
        return {"query": term, "results": rows[:top_k]}


def default_lexicon_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "training" / "data" / "dictionaries" / "moe_lexicon"


@lru_cache(maxsize=8)
def get_index(lexicon_dir: str | None = None) -> DictionaryIndex:
    base = Path(lexicon_dir) if lexicon_dir else default_lexicon_dir()
    return DictionaryIndex(base.resolve())
