"""Step 3: the model searches the catalog and checks out a book through tools."""

from pathlib import Path

from catalog import BOOKS
from dotenv import load_dotenv

from llm_interaction import LLMInteraction, tool

load_dotenv()

checked_out: set[str] = set()


@tool
def search_books(query: str) -> list[dict]:
    """Search the catalog by title, author or genre.

    Args:
        query: Text to look for, e.g. "Le Guin" or "mystery".
    """
    q = query.lower()
    return [
        {**b, "available": b["id"] not in checked_out}
        for b in BOOKS
        if q in b["title"].lower() or q in b["author"].lower() or q in b["genre"]
    ]


@tool
def check_out(book_id: str) -> str:
    """Check out a book.

    Args:
        book_id: The id of the book, e.g. "b1".
    """
    if book_id in checked_out:
        return f"{book_id} is already checked out."
    checked_out.add(book_id)
    return f"Checked out {book_id}."


@tool(stop=True)
def finish(message: str) -> str:
    """Tell the patron what happened. Call this last.

    Args:
        message: A short message to the patron.
    """
    return message


def print_tool_call(index: int, name: str, args: dict, result: object) -> None:
    print(f"  {index}. {name}({args}) -> {result}")


def main() -> None:
    llm = LLMInteraction(prompt_dir=Path(__file__).parent / "prompts")

    system = "You are a librarian. Use the tools to find and check out books."
    user = "I'd like to borrow a science fiction book by Ursula K. Le Guin."
    tools = [search_books, check_out, finish]

    result = llm.sync_agent_loop(
        system=system,
        user=user,
        tools=tools,
        on_tool_call=print_tool_call,
    )

    print(f"stop_reason: {result.stop_reason} after {result.tool_call_count} tool calls")
    print(f"checked out: {checked_out}")


if __name__ == "__main__":
    main()
