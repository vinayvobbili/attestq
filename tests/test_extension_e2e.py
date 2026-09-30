"""End-to-end test of the Chrome extension against a live `attestq serve`.

Loads chrome-extension/ unpacked into Playwright's Chromium, scans the bundled
demo portal page, answers from the demo corpus with the offline engine, fills
the page, and checks the page itself registered every change.

Skipped unless Playwright and its Chromium are installed:

    pip install playwright && playwright install chromium

Set ATTESTQ_E2E_SCREENSHOTS=DIR to also save screenshots of the review popup
and the filled page.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
sync_api = pytest.importorskip("playwright.sync_api")

from attestq import Engine, HashEmbedder  # noqa: E402
from attestq.cli import offline_chat  # noqa: E402
from attestq.demo import DEMO_DOCUMENTS, DEMO_NAMESPACE  # noqa: E402
from attestq.server import create_app  # noqa: E402

EXTENSION_DIR = Path(__file__).resolve().parents[1] / "chrome-extension"
DEMO_PAGE = "vendor-review.html"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def api_url():
    engine = Engine(chat=offline_chat, embed=HashEmbedder())
    engine.ingest(DEMO_DOCUMENTS, namespace=DEMO_NAMESPACE)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def demo_url():
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(EXTENSION_DIR / "demo"))
    handler.log_message = lambda *a, **k: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/{DEMO_PAGE}"
    httpd.shutdown()


@pytest.fixture(scope="module")
def browser(tmp_path_factory):
    # activeTab is granted by a user clicking the toolbar button, which automation
    # can't do; the test copy gets host access instead. Nothing else differs.
    ext = tmp_path_factory.mktemp("ext") / "attestq"
    shutil.copytree(EXTENSION_DIR, ext, ignore=shutil.ignore_patterns("demo"))
    manifest = json.loads((ext / "manifest.json").read_text())
    manifest["host_permissions"] = ["<all_urls>"]
    (ext / "manifest.json").write_text(json.dumps(manifest))

    with sync_api.sync_playwright() as pw:
        try:
            context = pw.chromium.launch_persistent_context(
                str(tmp_path_factory.mktemp("profile")),
                channel="chromium",
                headless=True,
                args=[f"--disable-extensions-except={ext}", f"--load-extension={ext}"],
            )
        except Exception as exc:  # browser binary missing
            pytest.skip(f"Chromium unavailable: {exc}")
        worker = context.service_workers[0] if context.service_workers else context.wait_for_event("serviceworker")
        context.extension_id = worker.url.split("/")[2]
        yield context
        context.close()


def _wait_for_job(ext_page, tab_id, statuses, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = ext_page.evaluate(
            "async (id) => (await chrome.storage.session.get(`job:${id}`))[`job:${id}`] || null", tab_id
        )
        if job and job["status"] in statuses:
            return job
        time.sleep(0.25)
    raise AssertionError(f"job never reached {statuses}; last: {job and job['status']}")


def test_scan_answer_review_fill(browser, api_url, demo_url):
    shots = os.environ.get("ATTESTQ_E2E_SCREENSHOTS")
    ext_id = browser.extension_id

    form = browser.new_page()
    form.goto(demo_url)

    ext = browser.new_page()
    ext.goto(f"chrome-extension://{ext_id}/options.html")
    ext.evaluate("(url) => chrome.storage.sync.set({serverUrl: url})", api_url)
    tab_id = ext.evaluate("async (url) => (await chrome.tabs.query({url}))[0].id", demo_url)

    # --- scan + answer ---
    ext.evaluate(
        "([tabId, ns]) => chrome.runtime.sendMessage({type: 'analyze', tabId, namespace: ns})",
        [tab_id, DEMO_NAMESPACE],
    )
    job = _wait_for_job(ext, tab_id, {"ready", "error"})
    assert job["status"] == "ready", job.get("error")
    assert not job["error"]

    # Eight questions across four layouts; vendor name, email and date are not questions.
    prompts = [it["prompt"] for it in job["items"]]
    found = "\n".join(f"{it['prompt'][:70]} | choices={it['choices']} text={it['hasText']}" for it in job["items"])
    assert len(prompts) == 8, found
    assert not any("Vendor name" in p or "email" in p.lower() for p in prompts)

    mfa = next(it for it in job["items"] if "multi-factor" in it["prompt"])
    assert mfa["choices"] == ["Yes", "No", "N/A"] and mfa["hasText"]
    enc = next(it for it in job["items"] if "encrypted" in it["prompt"])
    assert enc["choices"] == ["Met", "Partially Met", "Not Met", "Not Applicable"] and enc["hasText"]
    scanning = next(it for it in job["items"] if "vulnerability scanning" in it["prompt"])
    assert scanning["choices"] == ["Yes", "No", "N/A"] and scanning["hasText"]
    ir = next(it for it in job["items"] if it["prompt"].startswith("4.1"))
    assert ir["choices"] is None and ir["hasText"]
    rte = next(it for it in job["items"] if it["prompt"].startswith("4.2"))
    assert rte["choices"] is None

    # Answers came back; a question the demo evidence doesn't cover is gated, not guessed.
    assert all(it["answer"] and not it["answer"].get("error") for it in job["items"])
    logs = next(it for it in job["items"] if "log-retention" in it["prompt"])
    assert logs["answer"]["insufficient_evidence"] and not logs["checked"]
    assert mfa["checked"] and mfa["draft"]["choice"] == "Yes"

    if shots:
        _screenshot_popup(browser, ext_id, job, Path(shots) / "popup-review.png")

    # --- the reviewer edits a draft, and answers two gated questions by hand ---
    # (This is what popup.js writes: an item's `edit` and `checked`.)
    edits = {
        mfa["key"]: "Edited by reviewer: MFA is enforced via SSO.",
        ir["key"]: "Written by reviewer: see the Helios IR plan.",
        rte["key"]: "Written by reviewer: subcontractors are assessed annually.",
    }
    ext.evaluate(
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

    # --- fill ---
    ext.evaluate("(tabId) => chrome.runtime.sendMessage({type: 'commit', tabId})", tab_id)
    job = _wait_for_job(ext, tab_id, {"ready"})
    deadline = time.time() + 10
    while "Filled" not in job["message"] and time.time() < deadline:
        job = _wait_for_job(ext, tab_id, {"ready"})
    filled = [it for it in job["items"] if it["result"]]
    assert filled and all(it["result"]["ok"] for it in filled), [it["result"] for it in filled]

    assert form.locator("input[name=q11][value=yes]").is_checked()
    assert form.locator("#q11c").input_value() == "Edited by reviewer: MFA is enforced via SSO."
    assert form.locator("select[name=q21]").input_value() == "Met"
    assert form.locator("select[name=q33]").input_value() == ""  # gated and unticked: untouched
    assert form.locator("#q41").input_value() == edits[ir["key"]]
    assert form.locator(".rte").inner_text().strip() == edits[rte["key"]]  # rich-text editor
    change_log = form.locator("#log").inner_text()
    for field in ("q11", "q12", "q21", "q11c", "q41"):
        assert f"change {field}" in change_log, field
    assert "input  q11c" in change_log  # text fields fire input as well as change
    assert "q33" not in change_log

    if shots:
        form.screenshot(path=str(Path(shots) / "filled-form.png"), full_page=True)


def _screenshot_popup(context, ext_id, job, path):
    """Render the popup as a tab showing this job, for eyeballing the review UI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    page = context.new_page()
    page.set_viewport_size({"width": 460, "height": 900})
    page.goto(f"chrome-extension://{ext_id}/popup.html")
    page.evaluate(
        """async (job) => {
             const me = await chrome.tabs.getCurrent();
             job = {...job, url: me.url};
             await chrome.storage.session.set({[`job:${me.id}`]: job});
           }""",
        job,
    )
    page.reload()
    page.wait_for_selector(".item")
    page.screenshot(path=str(path), full_page=True)
    page.close()
