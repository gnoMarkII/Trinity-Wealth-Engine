"""Vault Benchmark and Synthetic Corpus Generator for Obsidian Vault V2.

Provides streaming synthetic corpus generation with strict O(1) memory overhead.
Ensures peak RAM overhead stays well below 64 MiB above baseline during generation,
even for 50,000 notes.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional


def get_process_rss_bytes() -> int:
    """Returns current process Resident Set Size (RSS) / Working Set in bytes."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            psapi = ctypes.WinDLL("psapi.dll")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                return int(c.WorkingSetSize)
        except Exception:
            pass
    else:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            pass
    return 0


# Standard entity types and templates for synthetic generation
_ENTITY_TYPES = [
    "stock_hub",
    "equity_analysis",
    "company_news",
    "macro_snapshot",
    "concept",
    "youtube_summary",
]

_SECTORS = ["Technology", "Healthcare", "Financials", "Energy", "Consumer", "Industrials"]


def _generate_body_text(target_bytes: int, rng: random.Random) -> str:
    thai_phrases = [
        "แนวโน้มการเติบโตของรายได้และผลตอบแทนส่วนของผู้ถือหุ้นอยู่ในระดับแข็งแกร่ง",
        "สภาวะเศรษฐกิจมหภาคยังคงมีความผันผวนจากการปรับอัตราดอกเบี้ยนโยบายของธนาคารกลาง",
        "อัตรากำไรขั้นต้นและกระแสเงินสดจากการดำเนินงานสะท้อนถึงประสิทธิภาพการบริหารจัดการต้นทุน",
        "การลงทุนในโครงสร้างพื้นฐานดิจิทัลและเทคโนโลยีปัญญาประดิษฐ์เป็นปัจจัยขับเคลื่อนหลัก",
        "การประเมินมูลค่าหุ้นด้วยวิธีคิดลดกระแสเงินสดให้อัตราผลตอบแทนที่น่าสนใจเมื่อเทียบกับความเสี่ยง",
    ]
    english_paragraphs = [
        "In this quarterly assessment, operating margins expanded primarily due to disciplined expense controls and strategic price realizations across core product categories.",
        "Management emphasized long-term capital allocation strategies, focusing on accretive acquisitions, sustained research and development, and opportunistic share repurchases.",
        "Risk factors encompass macroeconomic tightening, foreign currency exchange headwinds, and supply chain adjustments across key manufacturing corridors.",
        "Comparative sector analysis indicates superior return on invested capital relative to median peer benchmarks within the industry.",
    ]
    chunks = []
    current_len = 0
    while current_len < target_bytes:
        p = f"{rng.choice(thai_phrases)} — {rng.choice(english_paragraphs)}\n\n"
        chunks.append(p)
        current_len += len(p.encode("utf-8"))
    return "".join(chunks)


def _format_note_content(note_idx: int, entity_type: str, ticker: str, rng: random.Random) -> str:
    """Generates synthetic markdown content with realistic size distribution (80% 4K, 15% 16K, 5% 64K)."""
    year = 2026
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    date_str = f"{year}-{month:02d}-{day:02d}"
    
    # Linked notes targeting other potential synthetic notes
    link_idx1 = rng.randint(0, max(0, note_idx - 1)) if note_idx > 0 else 0
    link_idx2 = rng.randint(0, max(0, note_idx - 1)) if note_idx > 0 else 0
    
    frontmatter_lines = [
        "---",
        "schema_version: 2",
        f"note_id: synth_note_{note_idx:06d}",
        f"entity_type: {entity_type}",
        f"title: บทวิเคราะห์ Synthetic Note {note_idx:06d} — {ticker}",
        f"date: {date_str}",
        f"ticker: {ticker}",
        f"tags: [synthetic, benchmark, {entity_type.lower()}, {ticker.lower()}]",
        "---",
        "",
    ]

    # Target body size according to R08 spec: 80% ~4 KiB, 15% ~16 KiB, 5% ~64 KiB
    rand_roll = rng.random()
    if rand_roll < 0.80:
        target_body_size = 4096
    elif rand_roll < 0.95:
        target_body_size = 16384
    else:
        target_body_size = 65536

    content_body = _generate_body_text(target_body_size, rng)
    
    body_lines = [
        f"# บทวิเคราะห์ Synthetic Note {note_idx:06d} — {ticker}",
        "",
        f"> Generated synthetic note for benchmark and scaling evaluation.",
        f"> Entity type: `{entity_type}` | Sector: `{rng.choice(_SECTORS)}`",
        "",
        "## Cross References & Links",
        f"- Reference A: [[Synthetic Note {link_idx1:06d}]]",
        f"- Reference B: [[Synthetic Note {link_idx2:06d}]]",
        f"- Anchor Hub: [[{ticker}]]",
        "",
        "## Detailed Analysis Section",
        content_body,
    ]
    
    return "\n".join(frontmatter_lines + body_lines)


def generate_corpus(
    output_dir: Path | str,
    count: int,
    seed: int = 42,
    batch_flush_size: int = 500,
) -> dict[str, Any]:
    """Generates synthetic markdown notes in a streaming fashion.
    
    Ensures zero accumulating lists in RAM to satisfy O(1) memory budget
    (< 64 MiB overhead above baseline).
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    rng = random.Random(seed)
    
    # Sample pool of tickers
    tickers = [f"SYNTH{i:03d}" for i in range(1, 101)]
    
    # Start memory tracing
    base_rss = get_process_rss_bytes()
    peak_rss = base_rss
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()
    peak_overhead = 0
    start_time = time.perf_counter()
    
    total_bytes_written = 0
    
    for i in range(count):
        etype = rng.choice(_ENTITY_TYPES)
        ticker = rng.choice(tickers)
        
        # Determine folder structure based on entity type
        if etype in ("stock_hub", "equity_analysis"):
            sub_dir = out_path / "30_Knowledge_Base" / "Stocks" / ticker
        elif etype == "company_news":
            sub_dir = out_path / "30_Knowledge_Base" / "News" / "2026" / f"{rng.randint(1, 12):02d}"
        elif etype == "macro_snapshot":
            sub_dir = out_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots"
        else:
            sub_dir = out_path / "30_Knowledge_Base" / "Concepts"
            
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"Synthetic Note {i:06d}.md"
        file_path = sub_dir / filename
        
        content = _format_note_content(i, etype, ticker, rng)
        encoded = content.encode("utf-8")
        
        with file_path.open("wb") as f:
            f.write(encoded)
            
        total_bytes_written += len(encoded)
        
        # Sample memory overhead periodically
        if (i + 1) % batch_flush_size == 0 or i == count - 1:
            curr_rss = get_process_rss_bytes()
            if curr_rss > peak_rss:
                peak_rss = curr_rss
            curr_mem, peak_mem = tracemalloc.get_traced_memory()
            overhead = peak_mem - baseline_current
            if overhead > peak_overhead:
                peak_overhead = overhead
                
    elapsed = time.perf_counter() - start_time
    _, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_overhead = max(peak_overhead, final_peak - baseline_current)
    peak_rss_overhead = max(0, peak_rss - base_rss)
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    
    metadata = {
        "count": count,
        "seed": seed,
        "output_dir": str(out_path),
        "elapsed_seconds": round(elapsed, 4),
        "total_bytes_written": total_bytes_written,
        "peak_memory_overhead_bytes": peak_overhead,
        "peak_memory_overhead_mib": round(peak_overhead / (1024 * 1024), 2),
        "peak_rss_overhead_bytes": peak_rss_overhead,
        "peak_rss_overhead_mib": round(peak_rss_overhead / (1024 * 1024), 2),
        "system_info": system_info,
    }
    
    # Save manifest into output_dir
    manifest_path = out_path / "corpus_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    return metadata


def benchmark_catalog(
    vault_dir: Path | str,
    db_path: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Measures synchronization and lookup performance of SqliteNoteCatalogAdapter."""
    from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter

    v_path = Path(vault_dir).resolve()
    db_file = Path(db_path).resolve() if db_path else v_path / ".system" / "benchmark_catalog.db"
    if db_file.exists():
        db_file.unlink()

    adapter = SqliteNoteCatalogAdapter(db_path=db_file, vault_root=v_path)

    # Measure sync
    t0 = time.perf_counter()
    sync_res = adapter.sync_from_vault(v_path)
    sync_time = time.perf_counter() - t0

    # 10 warm-up lookups
    all_notes = adapter.find_notes(limit=200)
    for note in all_notes[:10]:
        _ = adapter.get_by_path(note.relative_path)

    # Measure 100 queries
    lookup_times_ms = []
    test_notes = all_notes[:100] if len(all_notes) >= 100 else (all_notes * (100 // max(1, len(all_notes)) + 1))[:100]
    for note in test_notes:
        t_start = time.perf_counter()
        _ = adapter.get_by_path(note.relative_path)
        lookup_times_ms.append((time.perf_counter() - t_start) * 1000)

    lookup_times_ms.sort()
    p50_ms = lookup_times_ms[len(lookup_times_ms) // 2] if lookup_times_ms else 0.0
    p95_ms = lookup_times_ms[int(len(lookup_times_ms) * 0.95)] if lookup_times_ms else 0.0
    avg_lookup_ms = (sum(lookup_times_ms) / len(lookup_times_ms)) if lookup_times_ms else 0.0

    return {
        "total_notes": sync_res["scanned"],
        "sync_time_seconds": round(sync_time, 4),
        "sync_throughput_notes_per_sec": round(sync_res["scanned"] / sync_time, 2) if sync_time > 0 else 0,
        "lookup_p50_ms": round(p50_ms, 4),
        "lookup_p95_ms": round(p95_ms, 4),
        "avg_point_lookup_ms": round(avg_lookup_ms, 4),
        "db_size_bytes": db_file.stat().st_size if db_file.exists() else 0,
    }


def benchmark_indexing(
    vault_dir: Path | str,
    chroma_dir: Optional[Path | str] = None,
    count_limit: Optional[int] = None,
) -> dict[str, Any]:
    """Measures incremental vector indexing performance using FakeEmbeddings."""
    import tracemalloc
    from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
    from tools.archivist.indexing_worker import FakeEmbeddings, IndexingWorker

    v_path = Path(vault_dir).resolve()
    c_dir = Path(chroma_dir).resolve() if chroma_dir else v_path / ".chroma_bench"
    if c_dir.exists():
        import shutil
        shutil.rmtree(c_dir, ignore_errors=True)

    db_file = v_path / ".system" / "benchmark_catalog.db"
    cat = SqliteNoteCatalogAdapter(db_path=db_file, vault_root=v_path)

    os.environ["VAULT_ALLOW_TEST_EMBEDDINGS"] = "1"
    worker = IndexingWorker(
        catalog=cat,
        vault_root=v_path,
        chroma_dir=c_dir,
        embeddings=FakeEmbeddings(size=384),
    )

    base_rss = get_process_rss_bytes()
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()

    t0 = time.perf_counter()
    res = worker.sync_index(batch_size=128, limit=count_limit)
    index_time = time.perf_counter() - t0

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_overhead = max(0, peak_memory - baseline_current)
    peak_rss = get_process_rss_bytes()
    peak_rss_overhead = max(0, peak_rss - base_rss)

    total_indexed = res.get("total_indexed", 0)

    return {
        "total_indexed": total_indexed,
        "indexing_time_seconds": round(index_time, 4),
        "throughput_notes_per_sec": round(total_indexed / index_time, 2) if index_time > 0 else 0,
        "peak_memory_overhead_bytes": peak_overhead,
        "peak_memory_overhead_mib": round(peak_overhead / (1024 * 1024), 2),
        "peak_rss_overhead_bytes": peak_rss_overhead,
        "peak_rss_overhead_mib": round(peak_rss_overhead / (1024 * 1024), 2),
    }


def run_scale_benchmarks(
    sizes: list[int],
    output_dir: Path | str,
    seed: int = 20260906,
) -> dict[str, Any]:
    """Runs end-to-end synthetic scale benchmarks across specified sizes."""
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for size in sizes:
        sub_vault = out_root / f"corpus_{size}"
        gen_meta = generate_corpus(sub_vault, count=size, seed=seed)
        cat_meta = benchmark_catalog(sub_vault)
        idx_meta = benchmark_indexing(sub_vault, count_limit=size)

        results[f"{size}_notes"] = {
            "generation": gen_meta,
            "catalog": cat_meta,
            "indexing": idx_meta,
        }

    # Write summary reports
    json_path = out_root / "scale-report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = out_root / "scale-report.md"
    lines = [
        "# Obsidian Vault V2 Scale Benchmark Report",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Hardware: {platform.processor()} ({os.cpu_count()} cores)",
        "",
        "| Corpus Size | Gen Time (s) | Gen RSS (MiB) | Catalog Sync (s) | Lookup p50 (ms) | Lookup p95 (ms) | Indexing (notes/s) | Indexing RSS (MiB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for size_key, data in results.items():
        gen = data["generation"]
        cat = data["catalog"]
        idx = data["indexing"]
        lines.append(
            f"| {gen['count']:,} | {gen['elapsed_seconds']}s | {gen.get('peak_rss_overhead_mib', gen['peak_memory_overhead_mib'])} MiB | "
            f"{cat['sync_time_seconds']}s | {cat.get('lookup_p50_ms', cat['avg_point_lookup_ms'])} ms | {cat.get('lookup_p95_ms', cat['avg_point_lookup_ms'])} ms | "
            f"{idx['throughput_notes_per_sec']} | {idx.get('peak_rss_overhead_mib', idx['peak_memory_overhead_mib'])} MiB |"
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return results

