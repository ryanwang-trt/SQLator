# SQLator Chrome Extension

A thin Chrome wrapper around the SQLator Flask backend. Two ways to use it:

- **Popup** — click the extension icon, type a question (optionally a database name), hit *Generate SQL* (or `Ctrl`+`Enter`). The result lands in the popup with a copy button. The header shows whether the backend is online.
- **Context menu** — select any text on a webpage, right-click → *Convert to SQL →*. A floating panel slides in from the bottom-right with the generated SQL.

## Install (load unpacked)

1. Open `chrome://extensions` in Chrome.
2. Toggle **Developer mode** on (top-right).
3. Click **Load unpacked** and select this `chrome-extension/` folder.
4. The extension icon should appear in your toolbar.

## Add icons

The `icons/` folder is empty. Drop three PNGs in there before publishing:

- `icons/icon16.png` — 16×16
- `icons/icon48.png` — 48×48
- `icons/icon128.png` — 128×128

Chrome will still load the unpacked extension without them (it falls back to a default icon), but you'll see a warning.

## Run the backend

From the repository root:

```bash
pip install -r requirements.txt
python app.py
```

The extension talks to two endpoints on the Flask app:

- `GET /health` → `{ "status": "ok" }` (used to render the online/offline pill in the popup)
- `POST /predict` with JSON body `{ "question": "...", "db_id": "..." }` → `{ "sql": "..." }` or `{ "error": "..." }`

The existing form-based `/` route is untouched, so the standalone web demo still works.

## Deployment

The repo ships a `Dockerfile` + `.dockerignore` ready for **Hugging Face Spaces** (free CPU tier).

### 1. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. **Owner**: your username. **Name**: `sqlator`. **SDK**: **Docker** → *Blank*. **Hardware**: CPU Basic (free).
3. Click *Create Space*. HF generates a git repo with a default `README.md` containing the required frontmatter — keep that file.

### 2. Push the backend

Clone the new Space repo into a fresh directory and copy the backend files in:

```bash
git clone https://huggingface.co/spaces/ryanwang-trt/sqlator sqlator-space
cd sqlator-space

# copy the backend from your project (leave out chrome-extension/, data/, models/)
cp ../Text-to-Sql/{app.py,config.py,schema.py,requirements.txt,Dockerfile,.dockerignore} .

git add .
git commit -m "Deploy SQLator backend"
git push
```

The Space will build the Docker image (~5 min the first time) and start serving. The first `/predict` call after a cold start takes ~30s while the model downloads from the Hub.

### 3. Point the extension at the Space

Two files change — `chrome-extension/config.js` and `chrome-extension/manifest.json`:

1. **`config.js`** — set `API_BASE` to your Space URL:
   ```js
   const API_BASE = "https://ryanwang-trt-sqlator.hf.space";
   ```
2. **`manifest.json`** — update `host_permissions`:
   ```json
   "host_permissions": ["https://ryanwang-trt-sqlator.hf.space/*"]
   ```

Reload the extension at `chrome://extensions` (click the refresh icon on the SQLator card). The popup pill should show *Online*.

### Other hosts

The same `Dockerfile` works on **Render**, **Railway**, **Fly.io**, or any Docker host. Two things to know:
- The container listens on port `7860`. Map or override via your host's port config.
- The model downloads from HF Hub on first request (~30s cold start). Persistent storage means it's cached after that.

## Usage tips

- The popup remembers your last question and database — they're persisted to `chrome.storage.local`.
- `Ctrl`+`Enter` (or `Cmd`+`Enter` on macOS) inside the question field submits.
- The context-menu panel stays open until you dismiss it with the `✕` button.
- If the popup shows *Offline*, start the Flask server and reopen the popup.
