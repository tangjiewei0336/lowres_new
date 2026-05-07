from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from core import normalize_key


class FineWebRareTermIndex:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._by_lang: dict[str, list[dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.data_dir.is_dir():
            return
        for path in sorted(self.data_dir.glob("rare_terms_*.jsonl")):
            lang = path.stem.removeprefix("rare_terms_")
            rows: list[dict[str, Any]] = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not row.get("term"):
                        continue
                    row.setdefault("lang", lang)
                    rows.append(row)
            if rows:
                self._by_lang[lang] = rows

    def list_languages(self) -> list[dict[str, Any]]:
        return [
            {"lang": lang, "size": len(rows)}
            for lang, rows in sorted(self._by_lang.items())
        ]

    def lookup(self, *, lang: str, term: str, top_k: int = 10, offset: int = 0) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if offset < 0:
            raise ValueError("offset must be >= 0")

        rows = self._by_lang.get(lang, [])
        query_key = normalize_key(term)
        matches: list[dict[str, Any]] = []
        for row in rows:
            term_key = normalize_key(str(row.get("term", "")))
            if query_key == term_key or query_key in term_key or term_key in query_key:
                out = dict(row)
                out["match_type"] = "exact" if query_key == term_key else "contains"
                matches.append(out)

        matches.sort(
            key=lambda r: (
                r.get("match_type") != "exact",
                int(r.get("frequency", 10**12)),
                int(r.get("document_frequency", 10**12)),
                str(r.get("term", "")),
            )
        )
        paged = matches[offset : offset + top_k]
        return {
            "query": term,
            "lang": lang,
            "total_matches": len(matches),
            "offset": offset,
            "top_k": top_k,
            "has_more": offset + top_k < len(matches),
            "results": paged,
        }


def default_rare_terms_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "training" / "data" / "dictionaries" / "fineweb_rare_terms"


@lru_cache(maxsize=8)
def get_rare_term_index(data_dir: str | None = None) -> FineWebRareTermIndex:
    base = Path(data_dir) if data_dir else default_rare_terms_dir()
    return FineWebRareTermIndex(base.resolve())
