import os
import re
from pathlib import Path

portfolio_dir = Path("c:/ChinoDoc/Projects/Claude/invest-agents/tools/portfolio")

# Pattern to match the block of constants
# It starts with VAULT_PATH = ... and ends with WATCHLIST_ITEMS_DIR = ...
# Some files might have GOALS_... definitions too. We'll handle goals separately if needed.
pattern = re.compile(
    r'VAULT_PATH\s*=\s*Path\(os\.getenv\("OBSIDIAN_VAULT_PATH"[^)]+\)\).*?'
    r'(WATCHLIST_ITEMS_DIR\s*=\s*VAULT_PATH\s*/\s*"20_Portfolio_Management/Current_Holdings/WatchlistItems"|'
    r'GOALS_ITEMS_DIR\s*=\s*VAULT_PATH\s*/\s*"20_Portfolio_Management/Goals/Items")',
    re.DOTALL
)

for file_path in portfolio_dir.glob("*.py"):
    if file_path.name == "constants.py":
        continue
    content = file_path.read_text(encoding="utf-8")
    
    # Check if there are GOALS constants to remove as well
    content = re.sub(
        r'GOALS_REL\s*=\s*os\.getenv\("GOALS_FILE",\s*"20_Portfolio_Management/Goals/Goals\.md"\)\s*\n'
        r'GOALS_PATH\s*=\s*VAULT_PATH\s*/\s*GOALS_REL\s*\n',
        '',
        content
    )
    
    if pattern.search(content):
        new_content = pattern.sub('from .constants import *', content)
        file_path.write_text(new_content, encoding="utf-8")
        print(f"Refactored {file_path.name}")
