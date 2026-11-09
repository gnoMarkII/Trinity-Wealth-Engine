"""Read-only lexical/vector hybrid retrieval for the Vault V2 corpus.

The previous offline hash embedding is useful as a deterministic fallback but
does not understand exact tickers or Thai phrases reliably.  This reranker
adds a cached lexical evidence layer: exact phrase/ticker matches are joined
to the current catalog before a vector result can be returned.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from langchain_core.documents import Document

from tools.archivist.metadata import parse_note


_TOKEN_RE = re.compile(r"[\w\u0E00-\u0E7F]+", re.UNICODE)
_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9.-]{1,9}(?![A-Za-z0-9])")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _Record:
    relative_path: str
    note_id: str
    title: str
    ticker: str
    text: str
    body: str
    metadata: dict[str, Any]
    file_mtime_ns: int = 0
    file_size: int = 0
    actual_sha256: str = ""
    frontmatter_title: str = ""
    frontmatter_ticker: str = ""
    stem: str = ""
    base_metadata: Optional[dict[str, Any]] = None


def _terms(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(value.lower()) if token.strip()]


@dataclass(frozen=True)
class _RetiredTombstones:
    stock_tickers: frozenset[str]
    relative_paths: frozenset[str]
    note_ids: frozenset[str]
    document_keys: frozenset[str]
    unreadable: bool = False


def _load_retired_tombstones(vault_root: Path) -> _RetiredTombstones:
    path = vault_root / ".system" / "retired_notes.jsonl"
    if not path.is_file():
        return _RetiredTombstones(frozenset(), frozenset(), frozenset(), frozenset())
    ticker_retired: dict[str, bool] = {}
    retired_paths: set[str] = set()
    retired_note_ids: set[str] = set()
    retired_doc_keys: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                continue
            status = str(value.get("status") or "retired").strip().lower()
            is_retired = status != "restored"
            rel = str(value.get("relative_path") or "").replace("\\", "/").lstrip("/")
            if rel:
                if is_retired:
                    retired_paths.add(rel)
                else:
                    retired_paths.discard(rel)
                parts = rel.split("/")
                if len(parts) >= 4 and parts[:2] == ["30_Knowledge_Base", "Stocks"]:
                    ticker_retired[parts[2].upper()] = is_retired
            raw_ticker = str(value.get("ticker") or "").strip().upper()
            if raw_ticker:
                ticker_retired[raw_ticker] = is_retired
            raw_note_id = str(value.get("note_id") or "").strip()
            if raw_note_id:
                if is_retired:
                    retired_note_ids.add(raw_note_id)
                else:
                    retired_note_ids.discard(raw_note_id)
            raw_doc_key = str(value.get("document_key") or "").strip()
            if raw_doc_key:
                if is_retired:
                    retired_doc_keys.add(raw_doc_key)
                else:
                    retired_doc_keys.discard(raw_doc_key)
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        return _RetiredTombstones(frozenset(), frozenset(), frozenset(), frozenset(), unreadable=True)
    return _RetiredTombstones(
        stock_tickers=frozenset(ticker for ticker, flag in ticker_retired.items() if flag),
        relative_paths=frozenset(retired_paths),
        note_ids=frozenset(retired_note_ids),
        document_keys=frozenset(retired_doc_keys),
        unreadable=False,
    )


def _retired_stock_tickers(vault_root: Path) -> set[str]:
    return set(_load_retired_tombstones(vault_root).stock_tickers)


def _tombstone_signature(vault_root: Path) -> tuple[int, int, int]:
    path = vault_root / ".system" / "retired_notes.jsonl"
    try:
        if path.is_file():
            st = path.stat()
            return (1, st.st_mtime_ns, st.st_size)
    except OSError:
        return (2, 0, 0)
    return (0, 0, 0)


def _cache_key(vault_root: Path, catalog_path: Path) -> tuple[str, int, int, int, int, int]:
    resolved_vault = vault_root.resolve()
    stat = catalog_path.stat()
    tomb_sig = _tombstone_signature(resolved_vault)
    return (str(resolved_vault), stat.st_mtime_ns, stat.st_size, *tomb_sig)


_RECORD_CACHE_MAX = 4096
_RECORD_CACHE: dict[tuple[str, str, str], _Record] = {}


def _evict_record_cache_for_path(vault_key: str, rel: str, specific_key: Optional[tuple[str, str, str]] = None) -> None:
    if specific_key is not None:
        _RECORD_CACHE.pop(specific_key, None)
    for k in [key for key in _RECORD_CACHE if key[0] == vault_key and key[1] == rel]:
        _RECORD_CACHE.pop(k, None)


def _parse_catalog_metadata_json(entry: Any) -> dict[str, Any]:
    raw_json = getattr(entry, "metadata_json", None)
    if isinstance(raw_json, dict):
        return dict(raw_json)
    if isinstance(raw_json, str) and raw_json.strip() and raw_json.strip() != "{}":
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            return {}
    return {}


def _apply_catalog_metadata(
    *,
    rel: str,
    stem: str,
    body: str,
    base_metadata: dict[str, Any],
    frontmatter_title: str,
    frontmatter_ticker: str,
    file_mtime_ns: int,
    file_size: int,
    actual_sha256: str,
    entry: Any,
) -> _Record:
    metadata = dict(base_metadata)
    catalog_meta = _parse_catalog_metadata_json(entry)
    if catalog_meta:
        metadata.update(catalog_meta)

    entry_note_id = str(getattr(entry, "note_id", "") or "")
    entry_title = str(getattr(entry, "title", "") or "").strip()
    entry_ticker = str(getattr(entry, "ticker", "") or "").strip()
    title = entry_title or frontmatter_title or stem
    ticker = (entry_ticker or frontmatter_ticker or "").upper()
    metadata["note_id"] = entry_note_id
    metadata["title"] = title
    if ticker:
        metadata["ticker"] = ticker

    for attr, meta_key in (
        ("document_key", "document_key"),
        ("current_revision_id", "current_revision_id"),
        ("current_revision", "current_revision"),
        ("date", "date"),
        ("entity_type", "entity_type"),
        ("source_key", "source_key"),
    ):
        val = getattr(entry, attr, None)
        if val is not None and val != "":
            metadata[meta_key] = val

    tickers = metadata.get("tickers") or []
    ticker_text = " ".join(str(item).upper() for item in tickers) if isinstance(tickers, list) else str(tickers)
    searchable = " ".join((title, ticker, ticker_text, body))
    return _Record(
        relative_path=rel,
        note_id=entry_note_id,
        title=title,
        ticker=ticker,
        text=searchable.lower(),
        body=body,
        metadata=metadata,
        file_mtime_ns=file_mtime_ns,
        file_size=file_size,
        actual_sha256=actual_sha256,
        frontmatter_title=frontmatter_title,
        frontmatter_ticker=frontmatter_ticker,
        stem=stem,
        base_metadata=dict(base_metadata),
    )


class HybridRetriever:
    def __init__(self, vault_root: Path, catalog: Any, catalog_path: Path) -> None:
        self.vault_root = vault_root.resolve()
        self.catalog = catalog
        self.catalog_path = catalog_path.resolve()
        self.key = _cache_key(self.vault_root, self.catalog_path)
        self.records: list[_Record] = []
        self.by_path: dict[str, _Record] = {}
        self.tombstones = _load_retired_tombstones(self.vault_root)
        self.retired_tickers = set(self.tombstones.stock_tickers)
        self._file_sigs: dict[str, tuple[Path, int, int]] = {}
        self._build()

    def _is_entry_retired(self, rel: str, entry: Any) -> bool:
        if self.tombstones.unreadable:
            return True
        if rel in self.tombstones.relative_paths:
            return True
        entry_note_id = str(getattr(entry, "note_id", "") or "").strip()
        if entry_note_id and entry_note_id in self.tombstones.note_ids:
            return True
        entry_doc_key = str(getattr(entry, "document_key", "") or "").strip()
        if entry_doc_key and entry_doc_key in self.tombstones.document_keys:
            return True
        parts = rel.split("/")
        if len(parts) >= 4 and parts[:2] == ["30_Knowledge_Base", "Stocks"] and parts[2].upper() in self.retired_tickers:
            return True
        entry_ticker = str(getattr(entry, "ticker", "") or "").strip().upper()
        if entry_ticker and entry_ticker in self.retired_tickers:
            return True
        return False

    def _records_still_current(self) -> bool:
        for path, expected_mtime_ns, expected_size in self._file_sigs.values():
            try:
                if not path.is_file() or not os.access(path, os.R_OK):
                    return False
                stat = path.stat()
                if stat.st_mtime_ns != expected_mtime_ns or stat.st_size != expected_size:
                    return False
                with path.open("rb"):
                    pass
            except OSError:
                return False
        return True

    def _build(self) -> None:
        vault_key = str(self.vault_root)
        entries = self.catalog.iter_notes(page_size=500)
        for entry in entries:
            rel = str(getattr(entry, "relative_path", "") or "").replace("\\", "/").lstrip("/")
            content_sha256 = str(getattr(entry, "content_sha256", "") or "").strip()
            cache_key = (vault_key, rel, content_sha256) if (rel and content_sha256) else None

            if not rel:
                continue

            record_state = str(getattr(entry, "record_state", "active") or "active").strip().lower()
            storage_scope = str(getattr(entry, "storage_scope", "active") or "active").strip().lower()
            if record_state != "active" or storage_scope != "active":
                _evict_record_cache_for_path(vault_key, rel, cache_key)
                continue

            if self._is_entry_retired(rel, entry):
                _evict_record_cache_for_path(vault_key, rel, cache_key)
                continue

            try:
                path = (self.vault_root / rel).resolve()
                if path == self.vault_root or not path.is_relative_to(self.vault_root):
                    _evict_record_cache_for_path(vault_key, rel, cache_key)
                    continue
                if not path.is_file() or not os.access(path, os.R_OK):
                    _evict_record_cache_for_path(vault_key, rel, cache_key)
                    continue
                stat = path.stat()
                with path.open("rb"):
                    pass
            except (OSError, ValueError):
                _evict_record_cache_for_path(vault_key, rel, cache_key)
                continue

            entry_file_size = getattr(entry, "file_size", 0) or 0
            if entry_file_size and int(entry_file_size) != stat.st_size:
                _evict_record_cache_for_path(vault_key, rel, cache_key)
                # Catalog file size disagrees with disk; fall through to read disk below.

            if cache_key is not None:
                cached_record = _RECORD_CACHE.get(cache_key)
                if cached_record is not None:
                    size_matches_entry = (not entry_file_size) or (int(entry_file_size) == stat.st_size)
                    if (
                        size_matches_entry
                        and cached_record.file_mtime_ns == stat.st_mtime_ns
                        and cached_record.file_size == stat.st_size
                    ):
                        refreshed = _apply_catalog_metadata(
                            rel=rel,
                            stem=cached_record.stem or path.stem,
                            body=cached_record.body,
                            base_metadata=cached_record.base_metadata or cached_record.metadata,
                            frontmatter_title=cached_record.frontmatter_title,
                            frontmatter_ticker=cached_record.frontmatter_ticker,
                            file_mtime_ns=stat.st_mtime_ns,
                            file_size=stat.st_size,
                            actual_sha256=cached_record.actual_sha256,
                            entry=entry,
                        )
                        self._file_sigs[rel] = (path, stat.st_mtime_ns, stat.st_size)
                        self.records.append(refreshed)
                        self.by_path[rel] = refreshed
                        continue
                    _RECORD_CACHE.pop(cache_key, None)

            try:
                raw = path.read_text(encoding="utf-8")
                metadata, body, issues = parse_note(raw)
                if issues:
                    body = raw
            except (OSError, UnicodeDecodeError):
                _evict_record_cache_for_path(vault_key, rel, cache_key)
                continue

            actual_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            if (
                content_sha256
                and _SHA256_HEX_RE.match(content_sha256.lower())
                and actual_sha256 != content_sha256.lower()
            ):
                # Stale catalog row: evict any entry cached under the stale digest
                # and refuse to cache the modified disk body under the stale key.
                _evict_record_cache_for_path(vault_key, rel, cache_key)

            frontmatter_title = str(metadata.get("title") or "")
            frontmatter_ticker = str(metadata.get("ticker") or "").upper()
            record = _apply_catalog_metadata(
                rel=rel,
                stem=path.stem,
                body=body,
                base_metadata=metadata,
                frontmatter_title=frontmatter_title,
                frontmatter_ticker=frontmatter_ticker,
                file_mtime_ns=stat.st_mtime_ns,
                file_size=stat.st_size,
                actual_sha256=actual_sha256,
                entry=entry,
            )

            cache_sha = (
                content_sha256
                if (
                    content_sha256
                    and (
                        not _SHA256_HEX_RE.match(content_sha256.lower())
                        or actual_sha256 == content_sha256.lower()
                    )
                )
                else actual_sha256
            )
            effective_cache_key = (vault_key, rel, cache_sha)
            if len(_RECORD_CACHE) >= _RECORD_CACHE_MAX:
                for old_k in list(_RECORD_CACHE.keys())[: _RECORD_CACHE_MAX // 2]:
                    _RECORD_CACHE.pop(old_k, None)
            _RECORD_CACHE[effective_cache_key] = record
            self._file_sigs[rel] = (path, stat.st_mtime_ns, stat.st_size)
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


_CACHE: dict[tuple[str, int, int, int, int, int], HybridRetriever] = {}


def get_retriever(vault_root: Path, catalog: Any, catalog_path: Path) -> HybridRetriever:
    resolved_vault = vault_root.resolve()
    resolved_catalog = catalog_path.resolve()
    key = _cache_key(resolved_vault, resolved_catalog)
    retriever = _CACHE.get(key)
    if (
        retriever is None
        or retriever.catalog is not catalog
        or not retriever._records_still_current()
    ):
        retriever = HybridRetriever(resolved_vault, catalog, resolved_catalog)
        _CACHE.clear()
        _CACHE[key] = retriever
    return retriever


def hybrid_search(vault_root: Path, catalog: Any, catalog_path: Path, query: str, *, vector_results: Optional[Iterable[Document]] = None, k: int = 5) -> list[Document]:
    return get_retriever(vault_root, catalog, catalog_path).search(query, vector_results=vector_results, k=k)
