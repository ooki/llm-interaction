"""Step 4: the tools get the library and the patron through ToolContext instead of globals."""

from dataclasses import dataclass, field
from pathlib import Path

from catalog import BOOKS
from dotenv import load_dotenv

from llm_interaction import LLMInteraction, ToolContext, tool

load_dotenv()


@dataclass
class Library:
    books: list[dict]
    loans: dict[str, str] = field(default_factory=dict)  # book_id -> patron name

    def search(self, query: str) -> list[dict]:
        q = query.lower()
        return [
            {**b, "available": b["id"] not in self.loans}
            for b in self.books
            if q in b["title"].lower() or q in b["author"].lower() or q in b["genre"]
        ]

    def check_out(self, book_id: str, patron: str, limit: int) -> str:
        if book_id in self.loans:
            return f"{book_id} is already checked out."
        if sum(name == patron for name in self.loans.values()) >= limit:
            return f"{patron} has reached their limit of {limit} books."
        self.loans[book_id] = patron
        return f"Checked out {book_id} to {patron}."


@dataclass
class Patron:
    name: str
    limit: int = 1


# The ToolContext parameters are filled in from `context=[...]`;
# the model only sees `query` and `book_id`.
@tool
def search_books(library: ToolContext[Library], query: str) -> list[dict]:
    """Search the catalog by title, author or genre.

    Args:
        query: Text to look for, e.g. "Le Guin" or "mystery".
    """
    return library.search(query)


@tool
def check_out(library: ToolContext[Library], patron: ToolContext[Patron], book_id: str) -> str:
    """Check out a book to the current patron.

    Args:
        book_id: The id of the book, e.g. "b1".
    """
    return library.check_out(book_id, patron.name, patron.limit)


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
    library = Library(books=BOOKS, loans={"b4": "Grace"})
    patron = Patron(name="Ada", limit=1)

    system = "You are a librarian. Use the tools to find and check out books."
    user = "I'd like to borrow both of Ursula K. Le Guin's books."
    tools = [search_books, check_out, finish]
    context = [library, patron]

    result = llm.sync_agent_loop(
        system=system,
        user=user,
        tools=tools,
        context=context,
        on_tool_call=print_tool_call,
    )

    print(f"stop_reason: {result.stop_reason} after {result.tool_call_count} tool calls")
    print(f"loans:       {library.loans}")


if __name__ == "__main__":
    main()
