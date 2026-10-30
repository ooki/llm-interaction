"""Step 2: Jinja templates fill in the catalog, and the reply is parsed as JSON."""

from pathlib import Path

from catalog import BOOKS
from dotenv import load_dotenv

from llm_interaction import LLMInteraction

load_dotenv()


def main() -> None:
    llm = LLMInteraction(prompt_dir=Path(__file__).parent / "prompts")

    variables = {
        "books": BOOKS,
        "patron": "Ada",
        "request": "I want something with dragons or wizards.",
    }

    # Renders prompts/librarian_system.jinja and prompts/librarian_user.jinja.
    response = llm.sync_query_template("librarian", variables)

    pick = response.json()
    book = next(b for b in BOOKS if b["id"] == pick["book_id"])
    print(f'Recommended: {book["title"]} by {book["author"]}')
    print(f'Why: {pick["reason"]}')


if __name__ == "__main__":
    main()
