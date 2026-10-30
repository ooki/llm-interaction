"""Tutorial pages, with code snippets pulled live from the source files.

Mark a region in any source file with::

    # --8<-- [start:name]
    ...
    # --8<-- [end:name]

(or ``// --8<--`` in JS, ``{# --8<-- ... #}`` in Jinja), then embed it in a page
with ``{{ snippet("tools.py", "name") }}``. The snippet is read from disk on every
render, so the tutorial can never drift out of date with the code.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable
from pathlib import Path

import jinja2
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from markupsafe import Markup, escape

PACKAGE_DIR = Path(__file__).parent
TUTORIAL_DIR = PACKAGE_DIR / "templates" / "tutorial"

# (slug, title) -- the order is the reading order.
PAGES: list[tuple[str, str]] = [
    ("01-overview", "Overview & running it"),
    ("02-world-model", "The world model"),
    ("03-commands", "Commands without the LLM"),
    ("04-genesis", "Your first LLM calls: creating a world"),
    ("05-context", "Building the context"),
    ("06-tools", "Tools: giving the LLM hands"),
    ("07-invent-and-save", "Invent, then save"),
    ("08-agent-loop", "The agent loop"),
    ("09-dice", "Randomness belongs in the engine"),
    ("10-watching", "Watching the LLM work"),
    ("11-memory", "Memory without chat history"),
    ("12-going-crazy", "Going crazy, safely"),
    ("13-testing", "Testing without an LLM"),
    ("14-exercises", "Exercises"),
]

_MARKER = re.compile(r"--8<--\s*\[(start|end):([\w-]+)\]")
_LANGS = {".py": "python", ".js": "javascript", ".jinja": "jinja", ".html": "html",
          ".css": "css", ".json": "json", ".toml": "toml"}


class SnippetError(LookupError):
    pass


# --8<-- [start:extract]
def extract(path: str, name: str | None = None) -> tuple[str, int, int]:
    """Return (code, first_line, last_line) of a marked region (or the whole file)."""
    file = (PACKAGE_DIR / path).resolve()
    if PACKAGE_DIR.parent not in file.parents or not file.is_file():
        raise SnippetError(f"No such file: {path}")
    lines = file.read_text(encoding="utf-8").splitlines()

    if name is None:
        start, end = 0, len(lines)
    else:
        found = {(m.group(1), m.group(2)): i for i, line in enumerate(lines)
                 if (m := _MARKER.search(line))}
        if ("start", name) not in found or ("end", name) not in found:
            raise SnippetError(f"Snippet '{name}' not found in {path}")
        start, end = found[("start", name)] + 1, found[("end", name)]

    body = [line for line in lines[start:end] if not _MARKER.search(line)]
    return textwrap.dedent("\n".join(body)).strip("\n"), start + 1, end
# --8<-- [end:extract]


def snippet(path: str, name: str | None = None, caption: str | None = None) -> Markup:
    code, first, last = extract(path, name)
    lang = _LANGS.get(Path(path).suffix, "text")
    where = (PACKAGE_DIR / path).resolve().relative_to(PACKAGE_DIR.parent.parent.parent)
    label = caption or (f"{where} &middot; lines {first}&ndash;{last}" if name else where)
    return Markup(
        f'<figure class="snippet"><figcaption>{label}</figcaption>'
        f'<pre><code class="language-{lang}">{escape(code)}</code></pre></figure>'
    )


def _env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TUTORIAL_DIR),
        autoescape=jinja2.select_autoescape(["html", "jinja"]),
        undefined=jinja2.StrictUndefined,
    )
    env.globals["snippet"] = snippet
    return env


def render_page(slug: str, link: Callable[[str], str]) -> str:
    """Render one tutorial page. ``link(slug)`` builds URLs (live server vs static files)."""
    slugs = [s for s, _ in PAGES]
    if slug not in slugs:
        raise KeyError(slug)
    i = slugs.index(slug)
    return _env().get_template(f"{slug}.html.jinja").render(
        pages=PAGES,
        slug=slug,
        title=PAGES[i][1],
        number=i + 1,
        prev=PAGES[i - 1] if i > 0 else None,
        next=PAGES[i + 1] if i + 1 < len(PAGES) else None,
        link=link,
    )


def render_index(link: Callable[[str], str]) -> str:
    return _env().get_template("index.html.jinja").render(pages=PAGES, link=link)


router = APIRouter(prefix="/tutorial")


def _live_link(slug: str) -> str:
    return "/tutorial" if slug == "index" else f"/tutorial/{slug}"


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def tutorial_index() -> str:
    return render_index(_live_link)


@router.get("/{slug}", response_class=HTMLResponse)
def tutorial_page(slug: str) -> str:
    try:
        return render_page(slug, _live_link)
    except KeyError:
        raise HTTPException(404, f"No tutorial page '{slug}'") from None
