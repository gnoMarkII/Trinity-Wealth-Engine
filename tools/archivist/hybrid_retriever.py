"""Read-only lexical/vector hybrid retrieval for the Vault V2 corpus.

The previous offline hash embedding is useful as a deterministic fallback but
does not understand exact tickers or Thai phrases reliably.  This reranker
adds a cached lexical evidence layer: exact phrase/ticker matches are joined
to the current catalog before a vector result can be returned.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from langchain_core.documents import Document

from tools.archivist.metadata import parse_note


_TOKEN_RE = re.compile(r"[\w\u0E00-\u0E7F]+", re.UNICODE)
_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9.-]{1,9}(?![A-Za-z0-9])")


@dataclass(frozen=True)
class _Record:
    relative_path: str
    note_id: str
    title: str
    ticker: str
    text: str
    body: str
    metadata: dict[str, Any]


def _terms(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(value.lower()) if token.strip()]


def _retired_stock_tickers(vault_root: Path) -> set[str]:
    path = vault_root / ".system" / "retired_notes.jsonl"
    result: set[str] = set()
    if not path.is_file():
        return result
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            rel = str(value.get("relative_path") or "").replace("\\", "/")
            parts = rel.split("/")
            if len(parts) >= 4 and parts[:3] == ["30_Knowledge_Base", "Stocks", parts[2]]:
                result.add(parts[2].upper())
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        return set()
    return result


def _cache_key(vault_root: Path, catalog_path: Path) -> tuple[str, int, int]:
    stat = catalog_path.stat()
    return (str(vault_root.resolve()), stat.st_mtime_ns, stat.st_size)


class HybridRetriever:
    def __init__(self, vault_root: Path, catalog: Any, catalog_path: Path) -> None:
        self.vault_root = vault_root.resolve()
        self.catalog = catalog
        self.catalog_path = catalog_path.resolve()
        self.key = _cache_key(self.vault_root, self.catalog_path)
        self.records: list[_Record] = []
        self.by_path: dict[str, _Record] = {}
        self.retired_tickers = _retired_stock_tickers(self.vault_root)
        self._build()

    def _build(self) -> None:
        entries = self.catalog.iter_notes(page_size=500)
        for entry in entries:
            rel = str(entry.relative_path).replace("\\", "/")
            parts = rel.split("/")
            if len(parts) >= 4 and parts[:3] == ["30_Knowledge_Base", "Stocks", parts[2]] and parts[2].upper() in self.retired_tickers:
                continue
            path = self.vault_root / rel
            if not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8")
                metadata, body, issues = parse_note(raw)
                if issues:
                    body = raw
            except (OSError, UnicodeDecodeError):
                continue
            title = str(metadata.get("title") or entry.title or path.stem)
            ticker = str(metadata.get("ticker") or entry.ticker or "").upper()
            tickers = metadata.get("tickers") or []
            ticker_text = " ".join(str(item).upper() for item in tickers) if isinstance(tickers, list) else str(tickers)
            searchable = " ".join((title, ticker, ticker_text, body))
            record = _Record(
                relative_path=rel,
                note_id=str(entry.note_id),
                title=title,
                ticker=ticker,
                text=searchable.lower(),
                body=body,
                metadata=metadata,
            )
            self.records.append(record)
            self.by_path[rel] = record

    def _score(self, query: str, record: _Record) -> float:
        q = query.strip().lower()
        if not q:
            return 0.0
        score = 0.0
        title = record.title.lower()
        if q in record.text:
            score += 8.0
        if q in title:
            score += 14.0
        for term in _terms(query):
            occurrences = record.text.count(term)
            if occurrences:
                score += min(5.0, float(occurrences))
                if term in title:
                    score += 4.0
        query_tickers = {item for item in _TICKER_RE.findall(query.upper()) if item not in {"AND", "THE", "FOR"}}
        if query_tickers:
            fields = {record.ticker}
            raw_tickers = record.metadata.get("tickers")
            if isinstance(raw_tickers, list):
                fields.update(str(item).upper() for item in raw_tickers)
            if query_tickers.intersection(fields):
                score += 30.0
            elif any(f"/{ticker}/" in record.relative_path.upper() for ticker in query_tickers):
                score += 25.0
        terms = _terms(query)
        if terms and all(term in record.text for term in terms):
            score += 6.0
        return score

    def _document(
        self,
        record: _Record,
        score: float,
        method: str,
        matched_document: Optional[Document] = None,
        query: str = "",
    ) -> Document:
        matched_chunk = str(getattr(matched_document, "page_content", "") or "").strip()
        if matched_chunk:
            content = matched_chunk
            content_kind = "matched_chunk"
            content_truncated = False
            chunk_index = (getattr(matched_document, "metadata", {}) or {}).get("chunk_index")
        else:
            # Lexical-only results still return a useful local excerpt.  The
            # result explicitly advertises that it is a preview; callers can
            # use read_note_chunk to obtain the rest without mistaking it for
            # a complete note.
            content = record.body.strip()
            if query:
                lowered_body = record.body.lower()
                positions = [lowered_body.find(term) for term in _terms(query)]
                positions = [position for position in positions if position >= 0]
                if positions:
                    center = min(positions)
                    half = 1200
                    start = max(0, center - half)
                    end = min(len(record.body), center + half)
                    content = record.body[start:end].strip()
            content_truncated = len(content) < len(record.body)
            if content_truncated:
                content += "\n...[preview; use read_note_chunk for the complete note]"
            content_kind = "note_preview"
            chunk_index = None
        metadata = dict(record.metadata)
        metadata.update(
            {
                "relative_path": record.relative_path,
                "note_id": record.note_id,
                "title": record.title,
                "retrieval_score": round(score, 6),
                "retrieval_method": method,
                "trust_tier": metadata.get("trust_tier", "T3"),
                "production_eligible": bool(metadata.get("production_eligible", False)),
                "content_kind": content_kind,
                "content_truncated": content_truncated,
                "content_total_chars": len(record.body),
                "matched_chunk_index": chunk_index,
            }
        )
        return Document(page_content=content, metadata=metadata)

    def search(self, query: str, *, vector_results: Optional[Iterable[Document]] = None, k: int = 5) -> list[Document]:
        lexical = [(self._score(query, record), record) for record in self.records]
        lexical = [(score, record) for score, record in lexical if score > 0]
        lexical.sort(key=lambda item: (-item[0], item[1].relative_path))
        lexical_map = {record.relative_path: score for score, record in lexical}
        combined: dict[str, tuple[float, _Record, str, Optional[Document]]] = {
            record.relative_path: (score, record, "lexical", None)
            for score, record in lexical[: max(k * 8, 40)]
        }

        vector_items = list(vector_results or [])
        for rank, document in enumerate(vector_items):
            metadata = getattr(document, "metadata", {}) or {}
            rel = str(metadata.get("relative_path") or metadata.get("source") or "").replace("\\", "/").strip("/")
            record = self.by_path.get(rel)
            if record is None:
                continue
            base = lexical_map.get(rel, 0.0)
            # Exact lexical evidence remains stronger, while a published
            # multilingual vector hit may introduce a cross-language result
            # that has no shared surface token with the query.
            score = base + (1.5 / (rank + 1) if base else 18.0 / (rank + 1))
            prior = combined.get(rel)
            if prior is None or score > prior[0]:
                combined[rel] = (score, record, "hybrid" if base else "semantic", document)
            elif prior[3] is None:
                # Keep the stronger lexical score but return the actual vector
                # chunk that matched the query instead of the beginning of a
                # potentially very long note.
                combined[rel] = (prior[0], prior[1], "hybrid", document)

        if not combined and vector_items:
            for rank, document in enumerate(vector_items):
                metadata = getattr(document, "metadata", {}) or {}
                rel = str(metadata.get("relative_path") or metadata.get("source") or "").replace("\\", "/").strip("/")
                record = self.by_path.get(rel)
                if record is not None:
                    combined[rel] = (1.0 / (rank + 1), record, "vector-fallback", document)

        ranked = sorted(combined.values(), key=lambda item: (-item[0], item[1].relative_path))[:k]
        return [
            self._document(record, score, method, matched_document, query=query)
            for score, record, method, matched_document in ranked
        ]


_CACHE: dict[tuple[str, int, int], HybridRetriever] = {}


def get_retriever(vault_root: Path, catalog: Any, catalog_path: Path) -> HybridRetriever:
    key = _cache_key(vault_root.resolve(), catalog_path.resolve())
    retriever = _CACHE.get(key)
    if retriever is None:
        retriever = HybridRetriever(vault_root.resolve(), catalog, catalog_path.resolve())
        _CACHE.clear()
        _CACHE[key] = retriever
    return retriever


def hybrid_search(vault_root: Path, catalog: Any, catalog_path: Path, query: str, *, vector_results: Optional[Iterable[Document]] = None, k: int = 5) -> list[Document]:
    return get_retriever(vault_root, catalog, catalog_path).search(query, vector_results=vector_results, k=k)
