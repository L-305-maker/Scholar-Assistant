from __future__ import annotations

import argparse
import json
from pathlib import Path

from scholar_assistant.mcp.stdio_smoke import run_sync


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-path", type=Path, default=None)
    args = parser.parse_args()
    project_path = args.project_path.resolve() if args.project_path else None
    result = run_sync(project_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
