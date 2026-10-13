let DOCS = [];  // filled from /api/docs
const SUGGESTIONS = [
  "What are the seven AI principles in Malaysia's AI Code of Ethics?",
  "What are the four key pillars of the AI Code of Ethics?",
  "What governance considerations should boards take into account when adopting AI?",
  "What is AI Nation 2030?",
];
const thread = document.getElementById("thread");
const scroller = document.getElementById("scroll");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
let busy = false, msgCount = 0;
let history = [];  // [{role, content}] sent so follow-ups can be resolved

const esc = s => s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

// Minimal, safe markdown: escape first, then bold / italics / code / headings / lists / citations.
function render(md, msgId) {
  const inline = s => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d+)\]/g, (_, n) => `<button class="cite" data-msg="${msgId}" data-n="${n}">${n}</button>`);
  const out = []; let list = null, para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.tag}>${list.items.map(i => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; } };
  for (const raw of md.split("\n")) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) { flushPara(); flushList(); continue; }
    if ((m = line.match(/^#{1,6}\s+(.*)/))) { flushPara(); flushList(); out.push(`<h3>${inline(m[1])}</h3>`); continue; }
    if ((m = line.match(/^\s*[-*•]\s+(.*)/))) { flushPara(); if (!list || list.tag !== "ul") { flushList(); list = {tag: "ul", items: []}; } list.items.push(m[1]); continue; }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { flushPara(); if (!list || list.tag !== "ol") { flushList(); list = {tag: "ol", items: []}; } list.items.push(m[1]); continue; }
    flushList(); para.push(line);
  }
  flushPara(); flushList();
  return out.join("");
}

function scrollDown() { scroller.scrollTop = scroller.scrollHeight; }

function showEmpty() {
  thread.innerHTML = `<div class="empty">
    <h2>What would you like to know?</h2>
    <p>Ask about Malaysia's AI governance, ethics and strategy. Every answer cites the page it came from.</p>
    <div class="suggest">${SUGGESTIONS.map(s => `<button type="button">${esc(s)}</button>`).join("")}</div>
    <div class="docs">Sources: ${DOCS.map(esc).join(" · ")}</div></div>`;
  thread.querySelectorAll(".suggest button").forEach(b => b.onclick = () => ask(b.textContent));
}

function addUser(text) {
  thread.querySelector(".empty")?.remove();
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  thread.appendChild(el); scrollDown();
}

function addBot() {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `<div class="avatar">P</div><div class="body"><div class="typing"><span></span><span></span><span></span></div></div>`;
  thread.appendChild(el); scrollDown();
  return el.querySelector(".body");
}

function fillAnswer(body, data) {
  const id = "m" + (++msgCount);
  const searched = data.rewritten && data.standalone_question
    ? `<div class="rewrite">Searched for: <em>${esc(data.standalone_question)}</em></div>` : "";
  if (data.status !== "answered") {
    const sug = data.suggestion
      ? `<button class="suggestion" type="button"><span>Try asking</span>${esc(data.suggestion)}</button>` : "";
    body.innerHTML = `${searched}<div class="notice">${esc(data.message || "Sorry, something went wrong.")}</div>${sug}`;
    const b = body.querySelector(".suggestion");
    if (b) b.onclick = () => ask(data.suggestion);
    return;
  }
  const sources = data.sources.map(s => `
    <div class="src" id="${id}-s${s.n}">
      <div class="head"><span class="n">[${s.n}]</span><span>${esc(s.title)}, p.${s.page}</span>
        <button class="toggle" type="button">show passage</button>
        <span class="score">match ${s.score.toFixed(2)}</span></div>
      <div class="snippet">${esc(s.text)}</div>
    </div>`).join("");
  body.innerHTML = `${searched}<div class="text">${render(data.answer, id)}</div>
    <details class="sources"><summary>${data.sources.length} sources</summary>${sources}</details>`;
  body.querySelectorAll(".toggle").forEach(b => b.onclick = () => {
    const src = b.closest(".src"); src.classList.toggle("open");
    b.textContent = src.classList.contains("open") ? "hide passage" : "show passage";
  });
}

// Clicking a [n] citation opens the source list and highlights that source.
thread.addEventListener("click", e => {
  const c = e.target.closest(".cite"); if (!c) return;
  const src = document.getElementById(`${c.dataset.msg}-s${c.dataset.n}`); if (!src) return;
  src.closest("details").open = true;
  src.scrollIntoView({behavior: "smooth", block: "center"});
  src.classList.remove("flash"); void src.offsetWidth; src.classList.add("flash");
});

async function ask(text) {
  text = text.trim();
  if (!text || busy) return;
  busy = true; send.disabled = true; input.value = ""; autosize();
  addUser(text);
  const body = addBot();
  try {
    const r = await fetch("/api/ask", {method: "POST", headers: {"Content-Type": "application/json"},
                                       body: JSON.stringify({question: text, history: history.slice(-6)})});
    const data = await r.json().catch(() => ({}));
    fillAnswer(body, r.ok ? data : {status: "error", message: data.detail || "Sorry, something went wrong."});
    if (r.ok && data.status !== "error") {
      // Store the resolved question so chained follow-ups keep their subject.
      history.push({role: "user", content: data.standalone_question || text});
      history.push({role: "assistant", content: (data.status === "answered" ? data.answer : data.message) || ""});
    }
  } catch {
    fillAnswer(body, {status: "error", message: "Couldn't reach the server. Please check your connection and try again."});
  }
  busy = false; send.disabled = !input.value.trim(); scrollDown(); input.focus();
}

function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; }
input.addEventListener("input", () => { autosize(); send.disabled = busy || !input.value.trim(); });
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(input.value); }
});
form.addEventListener("submit", e => { e.preventDefault(); ask(input.value); });
document.getElementById("newchat").onclick = () => { if (!busy) { history = []; showEmpty(); input.focus(); } };

fetch("/api/docs").then(r => r.json()).then(d => { DOCS = d; }).catch(() => {})
  .finally(() => { showEmpty(); input.focus(); });
