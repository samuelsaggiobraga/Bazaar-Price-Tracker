import os
import sys
import json
from pathlib import Path

# Get the absolute path of the root directory
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add it to the sys.path so Python can find the Utils module
if repo_root not in sys.path:
    sys.path.append(repo_root)


from Utils.db import setup_timescaledb, drop_timescaledb_tables, populate_bazaar_items  # noqa: E402


if __name__ == "__main__":
    drop_timescaledb_tables()
    setup_timescaledb()

    root = Path(__file__).resolve().parent.parent
    file_path = os.path.join(
        root,
        "Backend",
        "all_bazaar_items.json",
    )
    with open(file_path, "r", encoding="utf-8") as file:
        # Load and parse the JSON content into a Python dictionary or list
        data = json.load(file)
        populate_bazaar_items(data)
