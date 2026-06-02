(() => {
  if (window.__t2sInjected) return;
  window.__t2sInjected = true;

  let panel = null;

  function buildPanel() {
    const el = document.createElement("div");
    el.id = "t2s-panel";
    el.innerHTML = `
      <div class="t2s-bar">
        <div class="t2s-title">SQL<span>ator</span></div>
        <div class="t2s-actions">
          <button class="t2s-copy" type="button">Copy</button>
          <button class="t2s-close" type="button" aria-label="Close">✕</button>
        </div>
      </div>
      <div class="t2s-question"></div>
      <div class="t2s-body">
        <div class="t2s-spinner">
          <div class="t2s-dot"></div>
          <div class="t2s-dot"></div>
          <div class="t2s-dot"></div>
        </div>
        <pre class="t2s-sql"></pre>
        <div class="t2s-error"></div>
      </div>
    `;
    document.body.appendChild(el);

    el.querySelector(".t2s-close").addEventListener("click", hide);
    el.querySelector(".t2s-copy").addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      const sql = el.querySelector(".t2s-sql").textContent;
      if (!sql) return;
      try {
        await navigator.clipboard.writeText(sql);
        btn.textContent = "Copied!";
        btn.classList.add("t2s-copied");
        setTimeout(() => {
          btn.textContent = "Copy";
          btn.classList.remove("t2s-copied");
        }, 1200);
      } catch {
        btn.textContent = "Failed";
        setTimeout(() => (btn.textContent = "Copy"), 1200);
      }
    });

    return el;
  }

  function ensurePanel() {
    if (!panel || !document.body.contains(panel)) {
      panel = buildPanel();
    }
    return panel;
  }

  function show() {
    const el = ensurePanel();
    requestAnimationFrame(() => el.classList.add("t2s-show"));
  }

  function hide() {
    if (panel) panel.classList.remove("t2s-show");
  }

  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  function setLoading(question) {
    const el = ensurePanel();
    el.querySelector(".t2s-question").textContent = truncate(question, 140);
    el.querySelector(".t2s-spinner").style.display = "flex";
    el.querySelector(".t2s-sql").style.display = "none";
    el.querySelector(".t2s-sql").textContent = "";
    el.querySelector(".t2s-error").style.display = "none";
    el.querySelector(".t2s-error").textContent = "";
    el.querySelector(".t2s-copy").style.display = "none";
    show();
  }

  function setResult({ sql, error }) {
    const el = ensurePanel();
    el.querySelector(".t2s-spinner").style.display = "none";
    if (error) {
      el.querySelector(".t2s-sql").style.display = "none";
      el.querySelector(".t2s-copy").style.display = "none";
      const errEl = el.querySelector(".t2s-error");
      errEl.style.display = "block";
      errEl.textContent = error;
    } else {
      el.querySelector(".t2s-error").style.display = "none";
      const sqlEl = el.querySelector(".t2s-sql");
      sqlEl.style.display = "block";
      sqlEl.textContent = sql || "";
      el.querySelector(".t2s-copy").style.display = "inline-block";
    }
    show();
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (!msg || !msg.type) return;
    if (msg.type === "T2S_SHOW_LOADING") setLoading(msg.question || "");
    else if (msg.type === "T2S_SHOW_RESULT") setResult(msg);
  });
})();
