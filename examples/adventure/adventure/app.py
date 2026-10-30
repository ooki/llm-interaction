"""FastAPI app: one page, one JSON endpoint, plus the tutorial pages."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from llm_interaction import LLMInteraction
from pydantic import BaseModel

from adventure import tutorial
from adventure.game import Game
from adventure.world import WorldStore

PACKAGE_DIR = Path(__file__).parent
PROMPT_DIR = PACKAGE_DIR / "prompts"
STATIC_DIR = PACKAGE_DIR / "static"
DEFAULT_SAVES = PACKAGE_DIR.parent / "saves"


class CommandIn(BaseModel):
    text: str


# --8<-- [start:create_app]
def create_app(game: Game) -> FastAPI:
    app = FastAPI(title="Adventure")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(tutorial.router)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/command")
    async def api_command(body: CommandIn) -> dict[str, Any]:
        return await game.handle(body.text)

    @app.get("/api/state")
    def api_state() -> dict[str, Any]:
        return {"state": game.state(), "debug": game.last_debug}

    return app
# --8<-- [end:create_app]


# --8<-- [start:build_game]
def build_game() -> Game:
    # Reads LLM_INTERACTION_BACKEND / _API_KEY / _MODEL from the nearest .env file.
    load_dotenv(find_dotenv(usecwd=True))
    llm = LLMInteraction(prompt_dir=PROMPT_DIR)
    store = WorldStore(Path(os.getenv("ADVENTURE_SAVES", DEFAULT_SAVES)))
    game = Game(llm, store)
    game.load_latest()
    return game
# --8<-- [end:build_game]


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    host = os.getenv("ADVENTURE_HOST", "127.0.0.1")
    port = int(os.getenv("ADVENTURE_PORT", "8000"))
    uvicorn.run(create_app(build_game()), host=host, port=port)


if __name__ == "__main__":
    main()
