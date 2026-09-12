import os
import shutil
import json
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("Migration")

def get_vault_path() -> Path:
    return Path(os.getenv("OBSIDIAN_VAULT_PATH", "C:/ChinoDoc/Projects/Claude/invest-agents/memories")).resolve()

def merge_directories(src: Path, dest: Path):
    """Move all contents from src to dest, without overwriting."""
    if not src.exists(): return
    dest.mkdir(parents=True, exist_ok=True)
    
    for item in src.iterdir():
        dest_item = dest / item.name
        if item.is_dir():
            merge_directories(item, dest_item)
        else:
            if dest_item.exists():
                log.warning(f"File collision: {dest_item} already exists. Skipping move of {item}.")
            else:
                shutil.move(str(item), str(dest_item))
                log.info(f"Moved {item} -> {dest_item}")
                
    # Delete src if empty
    try:
        src.rmdir()
        log.info(f"Removed empty directory {src}")
    except OSError:
        pass

def flatten_directory(root: Path):
    """Move files from subdirectories to root, excluding Inbox and Stocks, and avoiding collisions."""
    if not root.exists(): return
    
    for subdir in list(root.iterdir()):
        if not subdir.is_dir(): continue
        if subdir.name.lower() in ("inbox", "stocks"): continue
        
        # Move files from subdir up to root
        for file in list(subdir.rglob("*")):
            if file.is_dir(): continue
            dest_file = root / file.name
            
            if dest_file.exists():
                # Collision handling
                # Append subdir name as suffix
                stem = dest_file.stem
                suffix = dest_file.suffix
                new_name = f"{stem}_{subdir.name}{suffix}"
                dest_file = root / new_name
                
                if dest_file.exists():
                    log.warning(f"Collision still exists after suffix for {file}. Skipping.")
                    continue
            
            shutil.move(str(file), str(dest_file))
            log.info(f"Flattened {file} -> {dest_file}")
            
        # Try to remove empty directories
        for d in sorted(subdir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass
        try:
            subdir.rmdir()
        except OSError:
            pass

def update_canvas_references(vault_path: Path):
    """Rewrite internal JSON paths in .canvas files to reflect the new structure."""
    # Find all canvas files
    for canvas_file in vault_path.rglob("*.canvas"):
        try:
            content = canvas_file.read_text(encoding="utf-8")
            data = json.loads(content)
            modified = False
            for node in data.get("nodes", []):
                if node.get("type") == "file":
                    file_val = node.get("file", "")
                    
                    # 1. Update Macro Baselines
                    if file_val.startswith("40_Macro_Baselines/"):
                        new_val = file_val.replace("40_Macro_Baselines/", "30_Knowledge_Base/Macroeconomics/Baselines/")
                        node["file"] = new_val
                        modified = True
                        
                    # 2. Update Flattened paths (e.g. YouTube_Summaries/ChannelName/Video.md -> YouTube_Summaries/Video.md)
                    elif "Daily_Snapshots/" in file_val or "News/" in file_val or "YouTube_Summaries/" in file_val:
                        parts = file_val.split("/")
                        # If there's a subfolder between root (News, etc) and the filename, remove it
                        # Assuming structure: .../Category/Subfolder/Filename.md
                        # Except Inbox or Stocks
                        if len(parts) >= 3 and parts[-2].lower() not in ("inbox", "stocks"):
                            # This is a bit naive, but it catches most flattened paths
                            # Reconstruct without the subfolder part
                            new_val = "/".join(parts[:-2] + [parts[-1]])
                            node["file"] = new_val
                            modified = True
            
            if modified:
                canvas_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                log.info(f"Updated references in {canvas_file}")
                
        except Exception as e:
            log.error(f"Failed to process canvas {canvas_file}: {e}")

def main():
    vault_path = get_vault_path()
    log.info(f"Starting migration in vault: {vault_path}")
    
    # 1. Merge Daily Snapshots
    old_daily = vault_path / "30_Knowledge_Base" / "Daily_Snapshots"
    new_daily = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots"
    merge_directories(old_daily, new_daily)
    
    # 2. Move Baselines
    old_baselines = vault_path / "40_Macro_Baselines"
    new_baselines = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Baselines"
    merge_directories(old_baselines, new_baselines)
    
    # 3. Flatten directories (Daily Snapshots, News, YouTube)
    flatten_directory(new_daily)
    
    news_dir = vault_path / "30_Knowledge_Base" / "News"
    flatten_directory(news_dir)
    
    yt_dir = vault_path / "30_Knowledge_Base" / "YouTube_Summaries"
    flatten_directory(yt_dir)
    
    # 4. Update Canvas references
    update_canvas_references(vault_path)
    
    log.info("Migration completed successfully.")
    
    # Big warning print
    print("\n" + "="*80)
    print("!!! MIGRATION FINISHED !!!")
    print("PLEASE DELETE `.chroma_index` and `master_index.json` IN THE VAULT")
    print("AND PERFORM A FULL RE-INDEX OF THE SYSTEM.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
