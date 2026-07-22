from __future__ import annotations

import json
import os
from pathlib import Path

from scholar_assistant.core.config import ScholarSettings
from scholar_assistant.retrieval.model_smoke import run_retrieval_smoke


def main() -> None:
    project_path = Path(os.environ.get("SCHOLAR_PROJECT_PATH", ".")).resolve()
    settings = ScholarSettings.load(project_path)
    result = run_retrieval_smoke(settings, allow_model_download=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
