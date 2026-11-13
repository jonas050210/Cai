"use strict";

const $ = (id) => document.getElementById(id);
const messages = $("messages");
const input = $("input");
const sendBtn = $("send");
const stopBtn = $("stop");
const statusEl = $("status");
const statsEl = $("stats");

let history = [];
let controller = null;
const STORE = "codeplan-chat-v1";

/* ---------------------------------------------------------------- settings */

function settings() {
  return {
    temperature: Number($("temperature").value),
    top_k: Number($("top-k").value),
    top_p: Number($("top-p").value),
    max_tokens: Number($("max-tokens").value),
    system: $("system").value,
  };
}

for (const id of ["temperature", "top-k", "top-p", "max-tokens"]) {
  const out = $(`${id}-out`);
  const sync = () => (out.textContent = $(id).value);
  $(id).addEventListener("input", sync);
  sync();
}

/* ------------------------------------------------------------- persistence */

function save() {
  localStorage.setItem(
    STORE,
    JSON.stringify({ history, system: $("system").value, model: $("model").value })
  );
}

function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) || "null");
    if (!saved) return;
    history = saved.history || [];
    if (saved.system) $("system").value = saved.system;
    for (const msg of history) addMessage(msg.role, msg.content);
  } catch {
    history = [];
  }
}

/* ------------------------------------------------------------------ render */

function addMessage(role, content) {
  const empty = messages.querySelector(".empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "du" : "modell";
  const body = document.createElement("div");
  body.className = "content";
  body.innerHTML = renderMarkdown(content);
  div.append(roleEl, body);
  messages.append(div);
  messages.scrollTop = messages.scrollHeight;
  return body;
}

function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}

/* tiny markdown: fenced code, inline code, bold, italics */
function renderMarkdown(text) {
  const parts = text.split(/```/);
  return parts
    .map((part, index) => {
      if (index % 2 === 1) {
        const body = part.replace(/^[a-zA-Z0-9_+-]*\n/, "");
        const code = `<pre><button class="ghost copy">copy</button><code>${escapeHtml(
          body.replace(/\n$/, "")
        )}</code></pre>`;
        return code;
      }
      return escapeHtml(part)
        .replace(/`([^`]+)`/g, "<p><code>$1</code></p>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>")
        .replace(/^### (.+)$/gm, "<strong>$1</strong>");
    })
    .join("");
}

messages.addEventListener("click", async (event) => {
  if (!event.target.classList.contains("copy")) return;
  const code = event.target.parentElement.querySelector("code").textContent;
  await navigator.clipboard.writeText(code);
  event.target.textContent = "kopiert";
  setTimeout(() => (event.target.textContent = "copy"), 1200);
});

/* --------------------------------------------------------------- streaming */

function setBusy(busy) {
  sendBtn.disabled = busy;
  stopBtn.hidden = !busy;
}

async function streamAnswer(userText) {
  const target = addMessage("assistant", "");
  target.classList.add("cursor");
  controller = new AbortController();
  setBusy(true);
  let answer = "";
  let started = performance.now();
  let firstToken = null;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history, ...settings(), stream: true }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      const detail = await response.text();
      throw new Error(detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.token !== undefined) {
          if (firstToken === null) firstToken = performance.now();
          answer += payload.token;
          target.innerHTML = renderMarkdown(answer);
          messages.scrollTop = messages.scrollHeight;
        } else if (payload.done) {
          const seconds = (performance.now() - started) / 1000;
          const tps = payload.completion_tokens / Math.max(seconds, 1e-3);
          statsEl.textContent = `${payload.completion_tokens} Tokens · ${tps.toFixed(1)} tok/s · ${
            payload.device
          }`;
        } else if (payload.error) {
          throw new Error(payload.error);
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      answer += answer ? " …" : "(abgebrochen)";
    } else {
      answer = `Fehler: ${err.message}`;
      statusEl.className = "status err";
      statusEl.textContent = "Serverfehler";
    }
    target.innerHTML = renderMarkdown(answer);
  } finally {
    target.classList.remove("cursor");
    controller = null;
    setBusy(false);
    if (answer) {
      history.push({ role: "assistant", content: answer });
      save();
    }
  }
}

/* -------------------------------------------------------------------- send */

async function submit() {
  const text = input.value.trim();
  if (!text || controller) return;
  input.value = "";
  input.style.height = "auto";
  history.push({ role: "user", content: text });
  addMessage("user", text);
  await streamAnswer(text);
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  submit();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
});

stopBtn.addEventListener("click", () => controller?.abort());

$("new").addEventListener("click", () => {
  history = [];
  messages.innerHTML = "";
  statsEl.textContent = "";
  addEmptyHint();
  save();
});

$("export").addEventListener("click", () => {
  const lines = [`# CodePlan-LM Gespräch\n`];
  for (const msg of history) {
    lines.push(`## ${msg.role === "user" ? "Du" : "Modell"}\n\n${msg.content}\n`);
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `codeplan-${new Date().toISOString().slice(0, 16)}.md`;
  link.click();
  URL.revokeObjectURL(link.href);
});

/* ------------------------------------------------------------------- setup */

function addEmptyHint() {
  const div = document.createElement("div");
  div.className = "empty";
  div.textContent = "Noch keine Nachricht. Frag etwas oder füge Code ein.";
  messages.append(div);
}

async function init() {
  restore();
  if (!history.length) addEmptyHint();

  try {
    const health = await (await fetch("/api/health")).json();
    statusEl.className = "status ok";
    statusEl.textContent = `${health.model} · ${(health.parameters / 1e6).toFixed(1)}M · ${
      health.device
    } · ctx ${health.context}`;
  } catch {
    statusEl.className = "status err";
    statusEl.textContent = "Server nicht erreichbar";
  }

  try {
    const models = await (await fetch("/api/checkpoints")).json();
    const select = $("model");
    select.innerHTML = models.checkpoints
      .map((name) => `<option value="${name}">${name}</option>`)
      .join("");
    if (models.current) select.value = models.current;
  } catch {
    /* ignore */
  }
}

$("load").addEventListener("click", async () => {
  const name = $("model").value;
  if (!name) return;
  statusEl.className = "status";
  statusEl.textContent = `lade ${name} …`;
  const response = await fetch("/api/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint: name }),
  });
  const data = await response.json();
  if (response.ok) {
    statusEl.className = "status ok";
    statusEl.textContent = `${data.model} · ${(data.parameters / 1e6).toFixed(1)}M · ${data.device}`;
    history = [];
    messages.innerHTML = "";
    addEmptyHint();
    save();
  } else {
    statusEl.className = "status err";
    statusEl.textContent = data.error || "Laden fehlgeschlagen";
  }
});

init();
