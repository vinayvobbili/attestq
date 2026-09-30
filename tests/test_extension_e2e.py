"""End-to-end test of the Chrome extension against a live `attestq serve`.

Loads chrome-extension/ unpacked into Playwright's Chromium, scans the bundled
demo portal page, answers from the demo corpus with the offline engine, fills
the page, and checks the page itself registered every change. The stack comes
from chrome-extension/tools/harness.py, which the store screenshot tool uses too.

Skipped unless Playwright and its Chromium are installed:

    pip install playwright && playwright install chromium

Set ATTESTQ_E2E_SCREENSHOTS=DIR to also save screenshots of the review popup
and the filled page.
"""

from __future__ import annotations

import contextlib
import os
import struct
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
sync_api = pytest.importorskip("playwright.sync_api")

TOOLS = Path(__file__).resolve().parents[1] / "chrome-extension" / "tools"
sys.path.insert(0, str(TOOLS))
import harness  # noqa: E402
import store_assets  # noqa: E402

from attestq.demo import DEMO_NAMESPACE  # noqa: E402


@pytest.fixture(scope="module")
def pw():
    with sync_api.sync_playwright() as playwright:
        yield playwright


@pytest.fixture(scope="module")
def stack(pw):
    with contextlib.ExitStack() as scope:
        try:
            demo = scope.enter_context(harness.demo_stack(pw))
        except harness.BrowserUnavailable as exc:
            pytest.skip(str(exc))
        yield demo


def test_scan_answer_review_fill(stack):
    shots = os.environ.get("ATTESTQ_E2E_SCREENSHOTS")
    form = stack.context.new_page()
    form.goto(stack.demo_url)
    ext = stack.extension_page()
    tab_id = harness.tab_id_for(ext, stack.demo_url)

    # --- scan + answer ---
    job = harness.analyze(ext, tab_id, DEMO_NAMESPACE)
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
        harness.render_popup(stack, job, Path(shots) / "popup-review.png")

    # --- the reviewer edits a draft, and answers two gated questions by hand ---
    edits = {
        mfa["key"]: "Edited by reviewer: MFA is enforced via SSO.",
        ir["key"]: "Written by reviewer: see the Helios IR plan.",
        rte["key"]: "Written by reviewer: subcontractors are assessed annually.",
    }
    harness.edit_items(ext, tab_id, edits)

    # --- fill ---
    job = harness.commit(ext, tab_id)
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


def _png_size(path: Path):
    return struct.unpack(">II", path.read_bytes()[16:24])


def test_store_assets_have_the_sizes_the_web_store_requires(tmp_path, pw):
    try:
        written = {p.name: p for p in store_assets.build(tmp_path, playwright=pw)}
    except harness.BrowserUnavailable as exc:
        pytest.skip(str(exc))
    assert set(written) == {
        "screenshot-1-review.png", "screenshot-2-filled.png", "screenshot-3-settings.png",
        "promo-small.png", "icon128.png",
    }
    for name, path in written.items():
        expected = {"promo-small.png": store_assets.PROMO, "icon128.png": (128, 128)}.get(name, store_assets.SCREEN)
        assert _png_size(path) == expected, name
