"""Run the extension for real: demo server + demo page + Chromium with the extension loaded.

Shared by the end-to-end test (tests/test_extension_e2e.py) and the store
screenshot tool (store_assets.py), so both drive exactly the same stack. Needs
``attestq[server]`` and Playwright with Chromium installed.
"""

from __future__ import annotations

import functools
import json
import shutil
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Optional

EXTENSION_DIR = Path(__file__).resolve().parents[1]
DEMO_PAGE = "vendor-review.html"


class BrowserUnavailable(RuntimeError):
    """Playwright's Chromium isn't installed (run `playwright install chromium`)."""


@dataclass
class Stack:
    api_url: str
    demo_url: str
    context: object  # playwright BrowserContext
    extension_id: str

    def extension_page(self, name: str = "options.html"):
        page = self.context.new_page()
        page.goto(f"chrome-extension://{self.extension_id}/{name}")
        return page


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def api_server(engine=None) -> Iterator[str]:
    """`attestq serve --demo --offline`, in-process. Yields its base URL."""
    import uvicorn

    from attestq import Engine, HashEmbedder
    from attestq.cli import offline_chat
    from attestq.demo import DEMO_DOCUMENTS, DEMO_NAMESPACE
    from attestq.server import create_app

    if engine is None:
        engine = Engine(chat=offline_chat, embed=HashEmbedder())
        engine.ingest(DEMO_DOCUMENTS, namespace=DEMO_NAMESPACE)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@contextmanager
def static_server(directory: Path) -> Iterator[str]:
    """Serve a directory over HTTP (extensions can't script file:// pages by default)."""
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(directory)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


@contextmanager
def browser_with_extension(playwright, headless: bool = True) -> Iterator[tuple]:
    """Chromium with the extension loaded. Yields (context, extension_id).

    activeTab is granted by a person clicking the toolbar button, which
    automation can't do, so the loaded copy gets host access instead. Nothing
    else about it differs from what ships.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ext = Path(tmp) / "attestq"
        shutil.copytree(EXTENSION_DIR, ext, ignore=shutil.ignore_patterns("demo", "tools", "__pycache__"))
        manifest = json.loads((ext / "manifest.json").read_text())
        manifest["host_permissions"] = ["<all_urls>"]
        (ext / "manifest.json").write_text(json.dumps(manifest))
        try:
            context = playwright.chromium.launch_persistent_context(
                str(Path(tmp) / "profile"),
                channel="chromium",
                headless=headless,
                args=[f"--disable-extensions-except={ext}", f"--load-extension={ext}"],
            )
        except Exception as exc:  # playwright raises its own Error for a missing binary
            raise BrowserUnavailable(f"Chromium unavailable: {exc}") from exc
        try:
            worker = (context.service_workers[0] if context.service_workers
                      else context.wait_for_event("serviceworker"))
            yield context, worker.url.split("/")[2]
        finally:
            context.close()


@contextmanager
def demo_stack(playwright, headless: bool = True, engine=None) -> Iterator[Stack]:
    """Everything at once, with the extension already pointed at the demo server."""
    with api_server(engine) as api_url, static_server(EXTENSION_DIR / "demo") as site, \
            browser_with_extension(playwright, headless) as (context, ext_id):
        stack = Stack(api_url=api_url, demo_url=f"{site}/{DEMO_PAGE}", context=context, extension_id=ext_id)
        page = stack.extension_page()
        page.evaluate("(url) => chrome.storage.sync.set({serverUrl: url})", api_url)
        page.close()
        yield stack


# --- driving a job, as the popup does -------------------------------------------------


def tab_id_for(ext_page, url: str) -> int:
    return ext_page.evaluate("async (url) => (await chrome.tabs.query({url}))[0].id", url)


def get_job(ext_page, tab_id: int) -> Optional[dict]:
    return ext_page.evaluate(
        "async (id) => (await chrome.storage.session.get(`job:${id}`))[`job:${id}`] || null", tab_id
    )


def wait_for_job(ext_page, tab_id: int, statuses, timeout: float = 60, until=None) -> dict:
    """Poll the stored job until its status is in `statuses` (and `until(job)`, if given)."""
    deadline = time.time() + timeout
    job = None
    while time.time() < deadline:
        job = get_job(ext_page, tab_id)
        if job and job["status"] in statuses and (until is None or until(job)):
            return job
        time.sleep(0.25)
    raise AssertionError(f"job never reached {statuses}; last: {job and job['status']}")


def analyze(ext_page, tab_id: int, namespace: str) -> dict:
    ext_page.evaluate(
        "([tabId, ns]) => chrome.runtime.sendMessage({type: 'analyze', tabId, namespace: ns})",
        [tab_id, namespace],
    )
    return wait_for_job(ext_page, tab_id, {"ready", "error"})


def edit_items(ext_page, tab_id: int, edits: dict) -> None:
    """Apply reviewer edits the way popup.js stores them: {item key: new text}, ticked."""
    ext_page.evaluate(
        """async ([tabId, edits]) => {
             const k = `job:${tabId}`;
             const job = (await chrome.storage.session.get(k))[k];
             for (const item of job.items) {
               if (item.key in edits) {
                 item.edit.text = edits[item.key];
                 item.checked = true;
               }
             }
             await chrome.storage.session.set({[k]: job});
           }""",
        [tab_id, edits],
    )


def commit(ext_page, tab_id: int) -> dict:
    ext_page.evaluate("(tabId) => chrome.runtime.sendMessage({type: 'commit', tabId})", tab_id)
    return wait_for_job(ext_page, tab_id, {"ready"}, until=lambda j: j["message"].startswith("Filled"))


def render_popup(stack: Stack, job: dict, path: Path, height: int = 900) -> Path:
    """Screenshot the popup showing `job` (the popup opened as a tab at its real width)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    page = stack.context.new_page()
    page.set_viewport_size({"width": 460, "height": height})
    page.goto(f"chrome-extension://{stack.extension_id}/popup.html")
    page.evaluate(
        """async (job) => {
             const me = await chrome.tabs.getCurrent();
             await chrome.storage.session.set({[`job:${me.id}`]: {...job, url: me.url}});
           }""",
        job,
    )
    page.reload()
    page.wait_for_selector(".item")
    page.screenshot(path=str(path))
    page.close()
    return path
