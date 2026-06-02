const $ = (id) => document.getElementById(id);

const questionEl = $("question");
const dbEl = $("db_id");
const schemaEl = $("schema");
const schemaToggle = $("schema-toggle");
const schemaPanel = $("schema-panel");
const schemaStatus = $("schema-status");
const goBtn = $("go");
const outputEl = $("output");
const sqlEl = $("sql");
const errEl = $("error");
const copyBtn = $("copy");
const statusEl = $("status");
const statusText = $("status-text");

function setStatus(state) {
  statusEl.classList.remove("online", "offline");
  if (state === "online") {
    statusEl.classList.add("online");
    statusText.textContent = "Online";
  } else if (state === "offline") {
    statusEl.classList.add("offline");
    statusText.textContent = "Offline";
  } else {
    statusText.textContent = "Checking…";
  }
}

async function pingHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!res.ok) throw new Error("bad status");
    const data = await res.json();
    setStatus(data.status === "ok" ? "online" : "offline");
  } catch {
    setStatus("offline");
  }
}

function setSchemaPanelOpen(open) {
  schemaToggle.setAttribute("aria-expanded", open ? "true" : "false");
  schemaPanel.hidden = !open;
}

function updateSchemaStatus() {
  const hasSchema = schemaEl.value.trim().length > 0;
  schemaStatus.textContent = hasSchema ? "● set" : "";
}

async function restoreState() {
  const { lastQuestion, lastDb, lastSchema } = await chrome.storage.local.get([
    "lastQuestion",
    "lastDb",
    "lastSchema",
  ]);
  if (lastQuestion) questionEl.value = lastQuestion;
  if (lastDb) dbEl.value = lastDb;
  if (lastSchema) {
    schemaEl.value = lastSchema;
    setSchemaPanelOpen(true);
  }
  updateSchemaStatus();
}

function showResult(sql) {
  errEl.style.display = "none";
  sqlEl.style.display = "block";
  copyBtn.style.display = "inline-block";
  sqlEl.textContent = sql;
  outputEl.classList.add("show");
}

function showError(msg) {
  sqlEl.style.display = "none";
  copyBtn.style.display = "none";
  errEl.style.display = "block";
  errEl.textContent = msg;
  outputEl.classList.add("show");
}

async function submit() {
  const question = questionEl.value.trim();
  const db_id = dbEl.value.trim();
  const schema = schemaEl.value.trim();

  if (!question) {
    showError("Please enter a question.");
    return;
  }

  await chrome.storage.local.set({
    lastQuestion: question,
    lastDb: db_id,
    lastSchema: schema,
  });

  goBtn.disabled = true;
  const originalLabel = goBtn.textContent;
  goBtn.textContent = "Generating…";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, db_id, schema }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) {
      showError(data.error || `Request failed (${res.status}).`);
    } else if (data.sql) {
      showResult(data.sql);
    } else {
      showError("Unexpected response from backend.");
    }
  } catch {
    setStatus("offline");
    showError("Backend unreachable. Is the Flask server running?");
  } finally {
    goBtn.disabled = false;
    goBtn.textContent = originalLabel;
  }
}

schemaToggle.addEventListener("click", () => {
  const open = schemaToggle.getAttribute("aria-expanded") === "true";
  setSchemaPanelOpen(!open);
});

schemaEl.addEventListener("input", updateSchemaStatus);

goBtn.addEventListener("click", submit);

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    submit();
  }
});

copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(sqlEl.textContent);
    copyBtn.textContent = "Copied!";
    copyBtn.classList.add("copied");
    setTimeout(() => {
      copyBtn.textContent = "Copy";
      copyBtn.classList.remove("copied");
    }, 1200);
  } catch {
    copyBtn.textContent = "Failed";
    setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
  }
});

document.addEventListener("DOMContentLoaded", () => {
  restoreState();
  pingHealth();
});
