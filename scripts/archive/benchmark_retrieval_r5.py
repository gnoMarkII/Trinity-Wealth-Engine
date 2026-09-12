"""Run the R5 hybrid retrieval regression set and record auditable metrics."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter  # noqa: E402
from tools.archivist.catalog_runtime import resolve_catalog_path  # noqa: E402
from tools.archivist.hybrid_retriever import hybrid_search  # noqa: E402
from tools.archivist.search import get_embeddings, get_query_vector_runtime  # noqa: E402
from tools.archivist.vector_generation import load_active_manifest, vector_runtime_path  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402


def _load_cases(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"invalid retrieval dataset: {path}")
    return payload


def _relevant(document: Any, case: dict[str, Any]) -> bool:
    metadata = getattr(document, "metadata", {}) or {}
    rel = str(metadata.get("relative_path") or "").replace("\\", "/")
    content = f"{rel}\n{getattr(document, 'page_content', '')}".lower()
    path_targets = [str(item).lower() for item in case.get("target_path_any", [])]
    term_targets = [str(item).lower() for item in case.get("target_terms", [])]
    term_targets_any = [str(item).lower() for item in case.get("target_terms_any", [])]
    path_ok = bool(path_targets) and any(target in rel.lower() for target in path_targets)
    terms_ok = bool(term_targets) and all(term in content for term in term_targets)
    terms_any_ok = bool(term_targets_any) and any(term in content for term in term_targets_any)
    if path_targets and term_targets:
        return path_ok or terms_ok or terms_any_ok
    if path_targets:
        return path_ok
    if term_targets:
        return terms_ok or terms_any_ok
    if term_targets_any:
        return terms_any_ok
    return True


def _forbidden(document: Any, case: dict[str, Any]) -> bool:
    rel = str((getattr(document, "metadata", {}) or {}).get("relative_path") or "").replace("\\", "/")
    return any(str(item).lower() in rel.lower() for item in case.get("forbidden_path_any", []))


def _exact_ticker_top1(document: Any, ticker: str) -> bool:
    metadata = getattr(document, "metadata", {}) or {}
    rel = str(metadata.get("relative_path") or "").replace("\\", "/").upper()
    title = str(metadata.get("title") or "").upper()
    values = metadata.get("tickers") or []
    tickers = {str(metadata.get("ticker") or "").upper()}
    if isinstance(values, list):
        tickers.update(str(item).upper() for item in values)
    return ticker.upper() in tickers or ticker.upper() in rel or ticker.upper() in title


def run(vault: Path, dataset: Path, output: Path) -> dict[str, Any]:
    vault = vault.resolve()
    dataset_payload = _load_cases(dataset.resolve())
    catalog_path = resolve_catalog_path(vault, require_exists=True)
    catalog = SqliteNoteCatalogAdapter(vault_root=vault, read_only=True)
    cases = list(dataset_payload["cases"])
    vectorstore = None
    try:
        manifest = load_active_manifest(vault)
        if manifest:
            query_runtime = get_query_vector_runtime(vault, vector_runtime_path(vault), manifest)
            vectorstore = Chroma(
                collection_name=str(manifest["collection_name"]),
                persist_directory=str(query_runtime),
                embedding_function=get_embeddings(),
            )
    except Exception:
        vectorstore = None

    # Prime the process-local text index outside the measured warm-query loop.
    if cases:
        vector_seed = vectorstore.similarity_search(str(cases[0]["query"]), k=20) if vectorstore else []
        hybrid_search(vault, catalog, catalog_path, str(cases[0]["query"]), vector_results=vector_seed, k=5)

    results: list[dict[str, Any]] = []
    warm_latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        vector_results = vectorstore.similarity_search(str(case["query"]), k=20) if vectorstore else []
        documents = hybrid_search(vault, catalog, catalog_path, str(case["query"]), vector_results=vector_results, k=5)
        warm_latencies.append((time.perf_counter() - started) * 1000.0)
        relevant_ranks = [index + 1 for index, document in enumerate(documents) if _relevant(document, case)]
        forbidden = [str((getattr(document, "metadata", {}) or {}).get("relative_path") or "") for document in documents if _forbidden(document, case)]
        citations = [
            str((getattr(document, "metadata", {}) or {}).get("relative_path") or "").replace("\\", "/")
            for document in documents
        ]
        citation_missing = [rel for rel in citations if not (vault / rel).is_file()]
        exact_ticker = case.get("exact_ticker")
        exact_top1 = bool(documents and _exact_ticker_top1(documents[0], str(exact_ticker))) if exact_ticker else None
        results.append(
            {
                "id": case.get("id"),
                "query": case.get("query"),
                "result_count": len(documents),
                "relevant_ranks": relevant_ranks,
                "precision_at_5": len(relevant_ranks) / 5.0,
                "reciprocal_rank_at_5": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
                "forbidden_paths": forbidden,
                "citation_missing": citation_missing,
                "exact_ticker_top1": exact_top1,
                "paths": citations,
                "language_pair": case.get("language_pair"),
                "language": case.get("language"),
            }
        )

    precision = statistics.fmean(item["precision_at_5"] for item in results) if results else 0.0
    mrr = statistics.fmean(item["reciprocal_rank_at_5"] for item in results) if results else 0.0
    exact_cases = [item for item in results if item["exact_ticker_top1"] is not None]
    exact_top1 = statistics.fmean(bool(item["exact_ticker_top1"]) for item in exact_cases) if exact_cases else 1.0
    pair_values: dict[str, dict[str, list[float]]] = {}
    for item in results:
        if item.get("language_pair"):
            group = str(item["language_pair"])
            language = str(item.get("language") or "unknown")
            pair_values.setdefault(group, {}).setdefault(language, []).append(float(item["reciprocal_rank_at_5"]))
    language_gap = 0.0
    for languages in pair_values.values():
        values = [statistics.fmean(value) for value in languages.values()]
        if len(values) > 1:
            language_gap = max(language_gap, max(values) - min(values))
    p95_index = max(0, math.ceil(len(warm_latencies) * 0.95) - 1)
    warm_sorted = sorted(warm_latencies)
    p95 = warm_sorted[p95_index] if warm_sorted else 0.0
    report = {
        "status": "PASS" if precision >= 0.80 and mrr >= 0.80 and exact_top1 >= 1.0 and not any(item["forbidden_paths"] for item in results) and not any(item["citation_missing"] for item in results) and language_gap <= 0.10 and p95 <= 1000.0 else "FAIL",
        "readiness_status": "PROVISIONAL_LABELS" if dataset_payload.get("label_status") != "reviewed" else "REVIEWED",
        "dataset": str(dataset.resolve()),
        "dataset_label_status": dataset_payload.get("label_status"),
        "catalog_path": str(catalog_path),
        "case_count": len(results),
        "precision_at_5": precision,
        "mrr_at_5": mrr,
        "exact_ticker_top1": exact_top1,
        "forbidden_retired_results": sum(len(item["forbidden_paths"]) for item in results),
        "citation_join_missing": sum(len(item["citation_missing"]) for item in results),
        "language_gap": language_gap,
        "warm_p95_ms": p95,
        "warm_latency_ms": warm_latencies,
        "thresholds": {
            "precision_at_5": 0.80,
            "mrr_at_5": 0.80,
            "exact_ticker_top1": 1.0,
            "forbidden_retired_results": 0,
            "citation_join_missing": 0,
            "language_gap": 0.10,
            "warm_p95_ms": 1000.0,
        },
        "results": results,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"results", "warm_latency_ms"}}, ensure_ascii=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--dataset", type=Path, default=Path("data/retrieval_r5_golden.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.vault, args.dataset, args.output)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
