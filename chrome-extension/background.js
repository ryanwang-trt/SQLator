importScripts("config.js");

const MENU_ID = "t2s-convert";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: "Convert to SQL →",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID) return;
  if (!tab || tab.id === undefined) return;

  const question = (info.selectionText || "").trim();
  if (!question) return;

  try {
    await chrome.tabs.sendMessage(tab.id, {
      type: "T2S_SHOW_LOADING",
      question,
    });
  } catch {
    return;
  }

  const { lastDb, lastSchema } = await chrome.storage.local.get(["lastDb", "lastSchema"]);
  const db_id = (lastDb || "").trim() || "unknown";
  const schema = (lastSchema || "").trim();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, db_id, schema }),
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok || data.error) {
      await chrome.tabs.sendMessage(tab.id, {
        type: "T2S_SHOW_RESULT",
        error: data.error || `Request failed (${res.status}).`,
      });
    } else {
      await chrome.tabs.sendMessage(tab.id, {
        type: "T2S_SHOW_RESULT",
        sql: data.sql || "",
      });
    }
  } catch {
    try {
      await chrome.tabs.sendMessage(tab.id, {
        type: "T2S_SHOW_RESULT",
        error: "Backend unreachable. Is the Flask server running?",
      });
    } catch {}
  }
});
