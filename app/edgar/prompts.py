"""Load prompt templates from the top-level `prompts/` directory (`$name` placeholders).

`PROMPT_VERSION=v2` loads the snapshot in `prompts/versions/v2/` instead of the live prompts.
"""

import os
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def prompt_version() -> str:
    """The prompt version in use: `$PROMPT_VERSION`, or `live` for the top-level `prompts/` files."""
    return os.environ.get("PROMPT_VERSION") or "live"


def prompt_path(name: str) -> Path:
    version = prompt_version()
    return PROMPT_DIR / f"{name}.md" if version == "live" else PROMPT_DIR / "versions" / version / f"{name}.md"


def load_prompt(name: str, **values: str) -> str:
    """Read `prompts/<name>.md` and substitute `$placeholders`; a missing value raises KeyError."""
    return Template(prompt_path(name).read_text(encoding="utf-8")).substitute(values).strip()
