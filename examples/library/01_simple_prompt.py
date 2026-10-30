"""Step 1: a plain system + user prompt."""

from pathlib import Path

from dotenv import load_dotenv

from llm_interaction import LLMInteraction

load_dotenv()


def main() -> None:
    # Backend, model and key come from the LLM_INTERACTION_* env vars.
    llm = LLMInteraction(prompt_dir=Path(__file__).parent / "prompts")

    system = "You are a friendly librarian. Answer in two sentences."
    user = "Can you recommend a fantasy novel for a rainy weekend?"

    response = llm.sync_query(system=system, user=user)
    print(response.text)


if __name__ == "__main__":
    main()
