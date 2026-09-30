"""Generate the Chrome Web Store listing images from the real extension.

    python chrome-extension/tools/store_assets.py          # -> dist/store/

Runs the demo stack (see harness.py): the offline demo server, the demo portal
page, and Chromium with the extension loaded. It then captures what a reviewer
sees and composes the images the store asks for:

    screenshot-1-review.png    1280x800   the review list next to the questionnaire
    screenshot-2-filled.png    1280x800   the form after filling
    screenshot-3-settings.png  1280x800   the settings page
    promo-small.png            440x280    small promo tile
    promo-marquee.png          1400x560   marquee promo tile (used if the store features it)
    icon128.png                128x128    store icon

Needs ``attestq[server]`` and Playwright with Chromium.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import re
import shutil
import tempfile
from pathlib import Path

import harness

REPO_DIR = harness.EXTENSION_DIR.parent
SCREEN = (1280, 800)
PROMO = (440, 280)
MARQUEE = (1400, 560)
CAPTION_H = 96
PANEL_W = 460  # popup.css body width
EXAMPLE_SERVER = "https://attestq.example.internal"

# Reviewer answers for the questions the offline demo evidence gates, so the
# filled screenshot shows every kind of field being written.
REVIEWER_TEXT = {
    "4.1": "Our incident response plan is tested twice a year; see IR-Plan-2026.pdf.",
    "4.2": "Subcontractors are assessed annually against our vendor security standard.",
}

_WORD = re.compile(r"[a-z0-9-]{4,}")
_EXCERPT = re.compile(r"^\[(\d+)\] \(source: [^)]*\)\n(.+?)\n\n", re.MULTILINE | re.DOTALL)


def extractive_chat(prompt: str) -> str:
    """A model stand-in whose answers read like answers: the excerpt sentence that
    best matches the question, cited. Like ``offline_chat`` it takes the first
    allowed determination, so only use it against the demo evidence."""
    question = prompt.split("=== QUESTION ===\n", 1)[-1].split("\n\n", 1)[0]
    wanted = set(_WORD.findall(question.lower()))
    best = (-1, "1", "")
    for number, text in _EXCERPT.findall(prompt):
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
            score = len(wanted & set(_WORD.findall(sentence.lower())))
            if score > best[0] and not sentence.isupper():
                best = (score, number, sentence)
    allowed = re.search(r"DETERMINATION must be exactly one of: (.+)\.", prompt)
    determination = allowed.group(1).split(", ")[0] if allowed else "See evidence"
    return (f"DETERMINATION: {determination}\nEVIDENCE SUMMARY: {best[2]}\n"
            f"CITATIONS: {best[1]}\nNOTES: none")


def demo_engine():
    from attestq import Engine, HashEmbedder
    from attestq.demo import DEMO_DOCUMENTS, DEMO_NAMESPACE

    engine = Engine(chat=extractive_chat, embed=HashEmbedder())
    engine.ingest(DEMO_DOCUMENTS, namespace=DEMO_NAMESPACE)
    return engine


STAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
  html, body {{ margin: 0; width: {w}px; height: {h}px; overflow: hidden; }}
  body {{ background: #0969da; font: 15px/1.4 system-ui, sans-serif; color: #fff; }}
  header {{ height: {cap}px; display: flex; flex-direction: column; justify-content: center; padding: 0 40px; }}
  h1 {{ margin: 0; font-size: 28px; font-weight: 650; letter-spacing: -0.01em; }}
  p {{ margin: 4px 0 0; font-size: 16px; opacity: 0.9; }}
  .shots {{ display: flex; gap: 20px; padding: 0 40px; height: {body}px;
             align-items: flex-start; justify-content: center; }}
  img {{ display: block; border-radius: 10px; box-shadow: 0 8px 28px rgb(0 0 0 / 0.28); }}
</style></head><body>
  <header><h1>{title}</h1><p>{subtitle}</p></header>
  <div class="shots">{images}</div>
</body></html>"""

PROMO_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  html, body {{ margin: 0; width: {w}px; height: {h}px; overflow: hidden; }}
  body {{ background: linear-gradient(135deg, #0969da, #0550ae); color: #fff;
          font: 15px/1.35 system-ui, sans-serif; display: flex; flex-direction: column;
          justify-content: center; padding: 0 32px; box-sizing: border-box; }}
  .brand {{ display: flex; align-items: center; gap: 14px; }}
  img {{ width: 64px; height: 64px; border-radius: 14px; box-shadow: 0 0 0 2px rgb(255 255 255 / 0.5); }}
  h1 {{ margin: 0; font-size: 34px; font-weight: 700; }}
  p {{ margin: 18px 0 0; font-size: 17px; }}
</style></head><body>
  <div class="brand"><img src="{icon}"><h1>attestq</h1></div>
  <p>Answer security questionnaires in web forms from your own evidence, with every answer reviewed.</p>
</body></html>"""

MARQUEE_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  html, body {{ margin: 0; width: {w}px; height: {h}px; overflow: hidden; }}
  body {{ background: linear-gradient(135deg, #0969da, #0550ae); color: #fff;
          font: 15px/1.35 system-ui, sans-serif; display: flex; gap: 72px;
          padding: 0 88px; box-sizing: border-box; }}
  .copy {{ flex: 1; display: flex; flex-direction: column; justify-content: center; }}
  .brand {{ display: flex; align-items: center; gap: 22px; }}
  .brand img {{ width: 96px; height: 96px; border-radius: 20px; box-shadow: 0 0 0 3px rgb(255 255 255 / 0.5); }}
  h1 {{ margin: 0; font-size: 60px; font-weight: 700; }}
  p {{ margin: 30px 0 0; font-size: 28px; line-height: 1.3; max-width: 640px; }}
  .shot {{ margin-top: 64px; width: {panel}px; border-radius: 12px 12px 0 0;
           box-shadow: 0 12px 40px rgb(0 0 0 / 0.35); align-self: flex-start; }}
</style></head><body>
  <div class="copy">
    <div class="brand"><img src="{icon}"><h1>attestq</h1></div>
    <p>Answer security questionnaires in web forms from your own evidence, with every answer reviewed.</p>
  </div>
  <img class="shot" src="{shot}">
</body></html>"""


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _render(context, html: str, size, target: Path) -> Path:
    page = context.new_page()
    page.set_viewport_size({"width": size[0], "height": size[1]})
    page.set_content(html)
    page.screenshot(path=str(target))
    page.close()
    return target


def compose(context, title: str, subtitle: str, shots, target: Path) -> Path:
    """One 1280x800 store screenshot: a caption band over one or more captures."""
    images = "".join(f'<img src="{_data_uri(p)}">' for p in shots)
    html = STAGE.format(w=SCREEN[0], h=SCREEN[1], cap=CAPTION_H, body=SCREEN[1] - CAPTION_H,
                        title=title, subtitle=subtitle, images=images)
    return _render(context, html, SCREEN, target)


def _capture_form(page, target: Path, section: int = 1) -> Path:
    """Screenshot the demo form scrolled to its `section`-th section (0 = engagement details)."""
    page.evaluate("(i) => document.querySelectorAll('section')[i].scrollIntoView({block: 'start'})", section)
    page.screenshot(path=str(target))
    return target


def build(out_dir: Path, headless: bool = True, playwright=None) -> list:
    """Write every listing image into `out_dir`. Pass `playwright` to reuse a running one."""
    with contextlib.ExitStack() as scope:
        if playwright is None:
            from playwright.sync_api import sync_playwright

            playwright = scope.enter_context(sync_playwright())
        return _capture(playwright, out_dir, headless)


def _capture(pw, out_dir: Path, headless: bool) -> list:
    from attestq.demo import DEMO_NAMESPACE

    out_dir.mkdir(parents=True, exist_ok=True)
    form_size = {"width": SCREEN[0] - 80 - 20 - PANEL_W, "height": SCREEN[1] - CAPTION_H}
    written = []
    with tempfile.TemporaryDirectory() as tmp, \
            harness.demo_stack(pw, headless, engine=demo_engine()) as stack:
        tmp = Path(tmp)
        ctx = stack.context
        form = ctx.new_page()
        form.emulate_media(color_scheme="light")
        form.set_viewport_size(form_size)
        form.goto(stack.demo_url)
        ext = stack.extension_page()
        tab = harness.tab_id_for(ext, stack.demo_url)

        job = harness.analyze(ext, tab, DEMO_NAMESPACE)
        if job["status"] != "ready":
            raise SystemExit(f"demo run failed: {job.get('error')}")
        before = _capture_form(form, tmp / "form-before.png")
        review = harness.render_popup(stack, job, tmp / "popup-review.png", height=form_size["height"])
        written.append(compose(
            ctx, "Every answer is reviewed before it reaches the form",
            "Confident answers come pre-ticked with their sources; weak or missing evidence is flagged.",
            [before, review], out_dir / "screenshot-1-review.png"))

        edits = {it["key"]: REVIEWER_TEXT[p] for it in job["items"]
                 for p in REVIEWER_TEXT if it["prompt"].startswith(p)}
        harness.edit_items(ext, tab, edits)
        job = harness.commit(ext, tab)
        after = _capture_form(form, tmp / "form-after.png")
        done = harness.render_popup(stack, job, tmp / "popup-done.png", height=form_size["height"])
        written.append(compose(
            ctx, "Fills radios, dropdowns, comment boxes and rich-text editors",
            "Works with the events web frameworks listen for. You still save or submit the form yourself.",
            [after, done], out_dir / "screenshot-2-filled.png"))

        ext.evaluate("(url) => chrome.storage.sync.set({serverUrl: url, apiToken: ''})", EXAMPLE_SERVER)
        settings = ctx.new_page()
        settings.emulate_media(color_scheme="light")
        settings.set_viewport_size({"width": 640, "height": 200})
        settings.goto(f"chrome-extension://{stack.extension_id}/options.html")
        settings.wait_for_timeout(300)
        settings.screenshot(path=str(tmp / "settings.png"), full_page=True)
        written.append(compose(
            ctx, "Set up once, or not at all",
            "Point it at your organisation's attestq server, or let your administrator lock the settings by policy.",
            [tmp / "settings.png"], out_dir / "screenshot-3-settings.png"))

        icon = harness.EXTENSION_DIR / "icons" / "icon128.png"
        written.append(_render(ctx, PROMO_HTML.format(w=PROMO[0], h=PROMO[1], icon=_data_uri(icon)),
                               PROMO, out_dir / "promo-small.png"))
        written.append(_render(ctx, MARQUEE_HTML.format(w=MARQUEE[0], h=MARQUEE[1], panel=PANEL_W,
                                                        icon=_data_uri(icon), shot=_data_uri(review)),
                               MARQUEE, out_dir / "promo-marquee.png"))
        written.append(Path(shutil.copy(icon, out_dir / "icon128.png")))
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_DIR / "dist" / "store",
                        help="output directory (default: dist/store/)")
    parser.add_argument("--headed", action="store_true", help="show the browser while capturing")
    args = parser.parse_args(argv)
    for path in build(args.out, headless=not args.headed):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
