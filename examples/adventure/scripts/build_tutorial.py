"""Render the tutorial to static HTML files (no server needed).

    uv run python scripts/build_tutorial.py [output_dir]   # default: tutorial_html/
"""

from __future__ import annotations

import sys
from pathlib import Path

from adventure.tutorial import PAGES, render_index, render_page


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "tutorial_html"
    out.mkdir(parents=True, exist_ok=True)

    def link(slug: str) -> str:
        return f"{slug}.html"

    (out / "index.html").write_text(render_index(link), encoding="utf-8")
    for slug, _ in PAGES:
        (out / f"{slug}.html").write_text(render_page(slug, link), encoding="utf-8")
    print(f"Wrote {len(PAGES) + 1} pages to {out}")


if __name__ == "__main__":
    main()
