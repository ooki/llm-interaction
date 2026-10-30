# Adventure: an "anything goes" text adventure

A single-player, D&D-style text adventure where an LLM plays Dungeon Master.
You can try anything. The LLM invents places, items, characters and lore as the
story needs them, and saves every invention through tools, so the world stays
consistent from turn to turn.

It's built on [`llm-interaction`](../../README.md) and comes with a 14-page
HTML tutorial that explains every part. The snippets in the tutorial are read
live from the source code.

## Run it

```bash
cd examples/adventure
cp .env.example .env        # add your DeepSeek API key
uv run adventure            # http://127.0.0.1:8000
```

The game loads the first `.env` it finds, searching upwards from the current
directory, so the repo-level `.env` works too. To use another provider, change
`LLM_INTERACTION_BACKEND` (`azure`, `openrouter`, `local`, ...).

- Game: http://127.0.0.1:8000
- Tutorial: http://127.0.0.1:8000/tutorial
- Static tutorial (no server): `uv run python scripts/build_tutorial.py`, which writes to `tutorial_html/`

## Playing

Type anything. Input that starts with `/` is a command. Commands never call the LLM:

| Command | |
|---|---|
| `/new [premise]` | Start a new world (random premise if empty) |
| `/look` `/inv` `/me` `/map` | Read the world straight from the save file |
| `/journal` `/facts` | History and lore |
| `/roll 2d6+1` | Roll dice yourself |
| `/undo` | Undo the last Dungeon Master turn |
| `/worlds` `/load <name>` | Switch worlds |
| `/debug` | Show prompts, tool calls and the raw API call |

Worlds are saved as JSON in `saves/`.

## Ask questions about a save from the terminal

```bash
uv run python scripts/peek.py the-salt-road "Who in this world dislikes me?"
```

## Tests

```bash
uv run --extra dev pytest
```

None of the tests call a real LLM. A scripted fake client drives the real agent
loop, the real tools and the real save files.

## Layout

```
adventure/
  world.py       data model, search, save/load
  dice.py        dice notation parser (the engine rolls, not the LLM)
  tools.py       everything the LLM can do to the world (@tool)
  context.py     what the LLM sees each turn
  turn.py        one turn: context -> agent_loop_template -> journal
  genesis.py     /new: premise (.json()) -> WorldSeed (.parse()) -> World
  chronicler.py  every 10 turns: journal -> story_so_far + facts (.scratchpad_json())
  commands.py    slash commands
  game.py        routing, undo, saving
  app.py         FastAPI
  tutorial.py    tutorial pages + snippet extraction
  prompts/       Jinja prompt templates
  static/        index.html, game.js, style.css
  templates/tutorial/  the tutorial pages
```
