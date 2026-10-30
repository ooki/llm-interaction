"""Ask questions about a saved world from the terminal.

    uv run python scripts/peek.py "the-salt-road" "Who hates me, and why?"

Uses the *sync* API (``sync_query_template``), which is handy for scripts and
notebooks. The web game is async, so it uses the async methods instead.
"""

from __future__ import annotations

import sys

from dotenv import find_dotenv, load_dotenv
from llm_interaction import LLMInteraction

from adventure.app import DEFAULT_SAVES, PROMPT_DIR
from adventure.world import WorldStore


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    name, question = sys.argv[1], " ".join(sys.argv[2:])

    load_dotenv(find_dotenv(usecwd=True))
    world = WorldStore(DEFAULT_SAVES).load(name)
    llm = LLMInteraction(prompt_dir=PROMPT_DIR)

    response = llm.sync_query_template(
        "peek", {"world_json": world.model_dump_json(indent=1), "question": question}
    )
    print(response.text)


if __name__ == "__main__":
    main()
