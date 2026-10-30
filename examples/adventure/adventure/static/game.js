// The whole client: send text, show the reply, refresh the side panels.
// Everything the LLM writes is inserted with textContent (never innerHTML),
// so model output can't inject HTML or scripts into the page.

const $ = (id) => document.getElementById(id);
const log = $("log");
const input = $("input");
const history = [];
let historyIndex = 0;

function addEntry(kind, text) {
  const div = document.createElement("div");
  div.className = `entry ${kind}`;
  div.textContent = text;
  log.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
  return div;
}

function addChanges(changes) {
  if (!changes || !changes.length) return;
  const ul = document.createElement("ul");
  ul.className = "entry changes";
  const feed = $("panel-changes");
  for (const change of changes) {
    const li = document.createElement("li");
    li.textContent = change;
    ul.appendChild(li);
    feed.prepend(li.cloneNode(true));
  }
  while (feed.children.length > 60) feed.lastChild.remove();
  log.appendChild(ul);
  ul.scrollIntoView({ behavior: "smooth", block: "end" });
}

function renderState(state) {
  if (!state || !state.world) {
    $("subtitle").textContent = "no world loaded";
    for (const id of ["look", "inventory", "me", "map"]) $(`panel-${id}`).textContent = "";
    return;
  }
  const c = state.counts;
  $("title").textContent = state.world;
  $("subtitle").textContent =
    `turn ${state.turn} · ${state.place} · ${c.places} places, ${c.things} things, ` +
    `${c.beings} beings, ${c.facts} facts`;
  $("panel-look").textContent = state.look;
  $("panel-inventory").textContent = state.inventory;
  $("panel-me").textContent = state.me;
  $("panel-map").textContent = state.map;
}

function details(summary, content, open = false) {
  const d = document.createElement("details");
  d.open = open;
  const s = document.createElement("summary");
  s.textContent = summary;
  const pre = document.createElement("pre");
  pre.textContent = typeof content === "string" ? content : JSON.stringify(content, null, 2);
  d.append(s, pre);
  return d;
}

// --8<-- [start:debug]
function renderDebug(debug) {
  const body = $("debug-body");
  if (!debug || !Object.keys(debug).length) return;
  body.replaceChildren();
  if (debug.stop_reason) {
    const p = document.createElement("p");
    p.textContent = `stop_reason: ${debug.stop_reason} · tool calls: ${debug.tool_calls}`;
    body.appendChild(p);
  }
  for (const call of debug.trace || []) {
    const failed = call.result && typeof call.result === "object" && "error" in call.result;
    const el = details(`${call.n}. ${call.tool}${failed ? "  ✗ error" : ""}`,
                       { args: call.args, result: call.result });
    if (failed) el.classList.add("failed");
    body.appendChild(el);
  }
  if (debug.prompt) {
    body.appendChild(details("Prompt: system (llm.render)", debug.prompt.system));
    body.appendChild(details("Prompt: user (llm.render)", debug.prompt.user));
  }
  if (debug.genesis) body.appendChild(details("Genesis seed (parsed WorldSeed)", debug.genesis, true));
  if (debug.chronicle) body.appendChild(details("Chronicle (scratchpad_json)", debug.chronicle, true));
  if (debug.last_call) body.appendChild(details("llm.last_call (raw request/response)", debug.last_call));
}
// --8<-- [end:debug]

// --8<-- [start:send]
async function send(text) {
  addEntry("player", `> ${text}`);
  const thinking = text.startsWith("/") ? null : addEntry("thinking", "The Dungeon Master considers...");
  input.disabled = $("send").disabled = true;
  try {
    const response = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const reply = await response.json();
    thinking?.remove();
    addEntry(reply.kind, reply.text);
    addChanges(reply.changes);
    renderState(reply.state);
    renderDebug(reply.debug);
    if (reply.client_action === "toggle_debug") $("debug").hidden = !$("debug").hidden;
  } catch (err) {
    thinking?.remove();
    addEntry("error", `Could not reach the server: ${err}`);
  } finally {
    input.disabled = $("send").disabled = false;
    input.focus();
  }
}
// --8<-- [end:send]

$("prompt").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  history.push(text);
  historyIndex = history.length;
  input.value = "";
  send(text);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "ArrowUp" && historyIndex > 0) {
    input.value = history[--historyIndex];
    event.preventDefault();
  } else if (event.key === "ArrowDown" && historyIndex < history.length) {
    input.value = history[++historyIndex] ?? "";
    event.preventDefault();
  }
});

for (const button of document.querySelectorAll("#quick button")) {
  button.addEventListener("click", () => send(button.dataset.cmd));
}
$("debug-toggle").addEventListener("click", () => { $("debug").hidden = !$("debug").hidden; });

(async function start() {
  const { state, debug } = await (await fetch("/api/state")).json();
  renderState(state);
  renderDebug(debug);
  if (state.world) {
    addEntry("command", `Welcome back to ${state.world}.\n\n${state.look}`);
  } else {
    addEntry("command",
      "Welcome, adventurer.\n\nType /new to begin a random adventure, or describe one:\n" +
      "  /new a heist in a city built on the back of a sleeping titan\n\n" +
      "Then do anything you like. /help lists the commands that skip the LLM.");
  }
})();
