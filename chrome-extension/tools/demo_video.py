"""Record a narrated walkthrough of the extension as an MP4.

    pip install "slidecast[playwright]"     # and ffmpeg on PATH (or slidecast[ffmpeg])
    python chrome-extension/tools/demo_video.py              # -> dist/store/attestq-demo.mp4
    python chrome-extension/tools/demo_video.py --voice "Ava (Premium)"

Runs the demo stack (see harness.py) and drives the real popup the way a
reviewer would: pick the vendor, Scan & Answer, look through the drafts, write
an answer the evidence couldn't supply, and Fill. Each step is captured frame by
frame, with the form and the popup side by side under a caption, and encoded as
a clip. slidecast then narrates the clips (holding a clip's last frame while the
voice finishes), adds the end card and fades, and writes one 1280x800 MP4. That
is the shape the store's promo video (a YouTube link) and the README want.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import harness
import store_assets

FPS = 15
SIZE = store_assets.SCREEN
CAPTION_H = store_assets.CAPTION_H
PANEL_W = store_assets.PANEL_W
PANEL_H = 600  # popup.css caps the popup's height, as Chrome does
FORM_SIZE = (SIZE[0] - 80 - 20 - PANEL_W, SIZE[1] - CAPTION_H)
FORM_X, POPUP_X = 40, 40 + FORM_SIZE[0] + 20
REPO_URL = "github.com/vinayvobbili/attestq"
SAY_LIKE = {r"\battestq\b": "attest Q"}  # how the voice should pronounce the name

CAPTION_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  html, body {{ margin: 0; width: {w}px; height: {h}px; overflow: hidden; background: transparent; }}
  header {{ height: {cap}px; background: #0969da; color: #fff; font: 15px/1.4 system-ui, sans-serif;
            display: flex; flex-direction: column; justify-content: center; padding: 0 40px; }}
  h1 {{ margin: 0; font-size: 28px; font-weight: 650; letter-spacing: -0.01em; }}
  p {{ margin: 4px 0 0; font-size: 16px; opacity: 0.9; }}
</style></head><body><header><h1>{title}</h1><p>{subtitle}</p></header></body></html>"""

END_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  html, body {{ margin: 0; width: {w}px; height: {h}px; overflow: hidden; }}
  body {{ background: linear-gradient(135deg, #0969da, #0550ae); color: #fff; text-align: center;
          font: 15px/1.4 system-ui, sans-serif; display: flex; flex-direction: column;
          align-items: center; justify-content: center; }}
  .brand {{ display: flex; align-items: center; gap: 22px; }}
  img {{ width: 104px; height: 104px; border-radius: 22px; box-shadow: 0 0 0 3px rgb(255 255 255 / 0.5); }}
  h1 {{ margin: 0; font-size: 64px; font-weight: 700; }}
  p {{ margin: 28px 0 0; font-size: 28px; max-width: 900px; }}
  code {{ display: block; margin-top: 36px; font: 600 24px ui-monospace, monospace; opacity: 0.95; }}
</style></head><body>
  <div class="brand"><img src="{icon}"><h1>attestq</h1></div>
  <p>You check the page and submit it yourself.<br>Every answer is grounded in the vendor's evidence.</p>
  <code>{url}</code>
</body></html>"""

END_NARRATION = "Then you check the page, and submit it yourself. attestq is open source, on GitHub."

CURSOR_JS = """() => {
  if (document.getElementById('__demo_cursor')) return;
  const c = document.createElement('div');
  c.id = '__demo_cursor';
  c.style.cssText = 'position:fixed;z-index:2147483647;left:-40px;top:-40px;width:22px;height:22px;' +
    'margin:-11px 0 0 -11px;border-radius:50%;background:rgb(31 35 40 / 0.35);border:2px solid #fff;' +
    'box-shadow:0 1px 5px rgb(0 0 0 / 0.45);pointer-events:none;transition:transform .08s';
  document.documentElement.appendChild(c);
}"""


@dataclass
class Step:
    key: str
    narration: str
    first_frame: int
    frames: int = 0


class Recorder:
    """Captures the form and the popup together, one frame per call, split into steps."""

    def __init__(self, ctx, form, popup, work: Path):
        self.ctx, self.form, self.popup, self.work = ctx, form, popup, work
        self.frames = work / "frames"
        self.frames.mkdir()
        self.count = 0
        self.steps = []
        self.layer = None
        self.cursor = (PANEL_W / 2, PANEL_H + 40)

    def step(self, key: str, title: str, subtitle: str, narration: str) -> None:
        """Start a new step: a new caption on screen and a new narrated clip."""
        page = self.ctx.new_page()
        page.set_viewport_size({"width": SIZE[0], "height": SIZE[1]})
        page.set_content(CAPTION_HTML.format(w=SIZE[0], h=SIZE[1], cap=CAPTION_H, title=title, subtitle=subtitle))
        self.layer = self.work / f"caption-{key}.png"
        page.screenshot(path=str(self.layer), omit_background=True)
        page.close()
        self.steps.append(Step(key, narration, self.count))

    def frame(self) -> None:
        name = f"{self.count:05d}"
        self.form.screenshot(path=str(self.frames / f"form_{name}.jpg"), type="jpeg", quality=92)
        self.popup.screenshot(path=str(self.frames / f"popup_{name}.jpg"), type="jpeg", quality=92)
        os.link(self.layer, self.frames / f"layer_{name}.png")
        self.count += 1
        self.steps[-1].frames += 1

    def hold(self, seconds: float) -> None:
        for _ in range(round(seconds * FPS)):
            self.frame()

    def until(self, done, timeout: float = 60) -> None:
        for _ in range(round(timeout * FPS)):
            if done():
                return
            self.frame()
        raise AssertionError("the demo got stuck waiting")

    # --- the pointer, which headless screenshots don't show ------------------------------

    def _put_cursor(self, x: float, y: float, pressed: bool = False) -> None:
        self.popup.evaluate(
            "([x, y, p]) => { const c = document.getElementById('__demo_cursor');"
            " c.style.left = x + 'px'; c.style.top = y + 'px'; c.style.transform = p ? 'scale(.7)' : ''; }",
            [x, y, pressed],
        )
        self.cursor = (x, y)

    def point(self, locator, frames: int = 9) -> None:
        box = locator.bounding_box()
        x, y = box["x"] + min(box["width"] / 2, 60), box["y"] + box["height"] / 2
        x0, y0 = self.cursor
        for i in range(1, frames + 1):
            t = i / frames
            ease = t * t * (3 - 2 * t)
            self._put_cursor(x0 + (x - x0) * ease, y0 + (y - y0) * ease)
            self.frame()

    def click(self, locator) -> None:
        self.point(locator)
        self._put_cursor(*self.cursor, pressed=True)
        self.frame()
        locator.click()
        self._put_cursor(*self.cursor)
        self.frame()

    def type(self, locator, text: str, per_frame: int = 3) -> None:
        self.click(locator)
        for i in range(0, len(text), per_frame):
            locator.press_sequentially(text[i:i + per_frame])
            self.frame()

    def scroll_popup(self, dy: int, steps: int = 8) -> None:
        self.popup.mouse.move(*self.cursor)
        for _ in range(steps):
            self.popup.mouse.wheel(0, dy / steps)
            self.frame()

    def scroll_form(self, section: int, steps: int = 18) -> None:
        start = self.form.evaluate("() => window.scrollY")
        end = self.form.evaluate(
            "(i) => document.querySelectorAll('section')[i].getBoundingClientRect().top + window.scrollY - 8",
            section,
        )
        for i in range(1, steps + 1):
            t = i / steps
            self.form.evaluate("(y) => window.scrollTo(0, y)", start + (end - start) * t * t * (3 - 2 * t))
            self.frame()


def _item(popup, number: str):
    """The review card for question `number` (matched at the start of its text, not in answers)."""
    question = popup.locator(".question", has_text=re.compile(rf"^{re.escape(number)}\s"))
    return popup.locator(".item").filter(has=question)


def walkthrough(rec: Recorder) -> None:
    """What happens, in order, with what the caption shows and the voice says."""
    from attestq.demo import DEMO_NAMESPACE

    popup = rec.popup

    rec.step("intro", "A vendor security questionnaire, in the vendor's portal",
             "attestq for Chrome answers it from that vendor's own evidence.",
             "Security questionnaires often live in a vendor's web portal. "
             "attestq for Chrome answers them from that vendor's own evidence.")
    rec.hold(1.5)
    rec.scroll_form(1)
    rec.hold(0.5)

    rec.step("scan", "Pick the vendor, then Scan & Answer",
             "It reads the questions on the page and drafts a cited answer for each one.",
             "Pick the vendor, and click Scan and Answer. "
             "The extension reads every question on the page, and drafts a cited answer for each one.")
    rec.click(popup.locator("#namespace"))
    popup.locator("#namespace").select_option(DEMO_NAMESPACE)
    rec.hold(0.4)
    rec.click(popup.locator("#scan"))
    rec.until(lambda: popup.locator("#footer").is_visible() and popup.locator(".item").count() > 0)
    rec.hold(1.0)

    rec.step("review", "Review every answer before it touches the page",
             "Confident answers arrive ticked, with their sources. Gaps are flagged, never guessed.",
             "Nothing touches the page yet. Confident answers arrive ticked, with their sources. "
             "Where the evidence is thin, the answer is flagged, not guessed.")
    rec.click(popup.locator(".sources summary").first)
    rec.hold(1.5)
    gap = _item(popup, "4.1").locator("textarea")
    rec.point(_item(popup, "1.2"))
    while gap.bounding_box()["y"] + gap.bounding_box()["height"] > PANEL_H - 70:
        rec.scroll_popup(240)
    rec.hold(0.8)

    rec.step("gap", "Answer what the evidence doesn't cover",
             "Typing an answer ticks it. Nothing is filled until you say so.",
             "For a question the evidence doesn't cover, just type the answer. That ticks it.")
    rec.type(gap, store_assets.REVIEWER_TEXT["4.1"])
    rec.hold(1.0)

    rec.step("fill", "Fill selected answers",
             "Radio buttons, dropdowns, comment boxes and rich-text editors, with the events the page expects.",
             "Click Fill selected answers. Radio buttons, dropdowns, comment boxes and rich text editors "
             "are all filled, with the events the page expects.")
    rec.click(popup.locator("#commit"))
    rec.until(lambda: popup.locator(".result.ok").count() > 0, timeout=30)
    rec.hold(1.5)
    for section in (2, 3, 4):
        rec.scroll_form(section)
        rec.hold(1.0)


def encode_step(ffmpeg: str, frames: Path, step: Step, out: Path) -> Path:
    """One step's frames, composed: blue stage, form, popup, caption on top."""
    graph = (
        f"color=c=0x0969da:s={SIZE[0]}x{SIZE[1]}:r={FPS}[bg];"
        f"[bg][0:v]overlay={FORM_X}:{CAPTION_H}:shortest=1[a];"
        f"[a][1:v]overlay={POPUP_X}:{CAPTION_H}[b];"
        f"[b][2:v]overlay=0:0,format=yuv420p[v]"
    )
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    for layer in ("form_%05d.jpg", "popup_%05d.jpg", "layer_%05d.png"):
        cmd += ["-framerate", str(FPS), "-start_number", str(step.first_frame), "-i", str(frames / layer)]
    cmd += ["-filter_complex", graph, "-map", "[v]", "-frames:v", str(step.frames),
            "-c:v", "libx264", "-crf", "16", "-tune", "stillimage", str(out)]
    subprocess.run(cmd, check=True)
    return out


def make_tts(voice: str):
    import slidecast

    if voice == "silent":
        return slidecast.SilentTTS()
    if voice == "gtts":
        return slidecast.GTTSTTS(phonetic=SAY_LIKE)
    if shutil.which("say") is None:
        raise SystemExit(f"voice {voice!r} needs macOS `say`; use --voice gtts or --voice silent elsewhere")
    return slidecast.MacSayTTS(voice=voice, phonetic=SAY_LIKE)


def record(target: Path, voice: str = "Samantha", headless: bool = True, playwright=None) -> Path:
    """Record and narrate the walkthrough into `target` (.mp4). Pass `playwright` to reuse one."""
    import slidecast

    ffmpeg = slidecast.find_ffmpeg()
    target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as scope:
        if playwright is None:
            from playwright.sync_api import sync_playwright

            playwright = scope.enter_context(sync_playwright())
        work = Path(scope.enter_context(tempfile.TemporaryDirectory()))
        stack = scope.enter_context(harness.demo_stack(playwright, headless, engine=store_assets.demo_engine()))
        ctx = stack.context

        form = ctx.new_page()
        form.emulate_media(color_scheme="light")
        form.set_viewport_size({"width": FORM_SIZE[0], "height": FORM_SIZE[1]})
        form.goto(stack.demo_url)

        # Opened as a page, the popup acts on the active tab like the real one does,
        # so the form comes to the front before the popup loads.
        popup = ctx.new_page()
        popup.emulate_media(color_scheme="light")
        popup.set_viewport_size({"width": PANEL_W, "height": PANEL_H})
        form.bring_to_front()
        popup.goto(f"chrome-extension://{stack.extension_id}/popup.html")
        popup.wait_for_selector("#namespace option", state="attached")
        popup.evaluate(CURSOR_JS)

        rec = Recorder(ctx, form, popup, work)
        walkthrough(rec)

        # The end card is an HTML slide; slidecast renders it with a browser borrowed
        # from this Playwright, since a second sync Playwright can't start inside it.
        browser = playwright.chromium.launch()
        scope.callback(browser.close)
        reel = slidecast.Reel(width=SIZE[0], height=SIZE[1], fps=FPS, tts=make_tts(voice),
                              renderer=slidecast.PlaywrightRenderer(browser=browser))
        for step in rec.steps:
            clip = encode_step(ffmpeg, rec.frames, step, work / f"{step.key}.mp4")
            reel.add_clip(clip, step.narration, tail_pad=0.5)
        icon = store_assets._data_uri(harness.EXTENSION_DIR / "icons" / "icon128.png")
        reel.add(END_HTML.format(w=SIZE[0], h=SIZE[1], icon=icon, url=REPO_URL), END_NARRATION,
                 tail_pad=1.2, min_duration=4.0)
        return reel.render(target, ffmpeg=ffmpeg, fade_in=0.6, fade_out=1.0, make_poster=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=store_assets.REPO_DIR / "dist" / "store" / "attestq-demo.mp4",
                        help="output file (default: dist/store/attestq-demo.mp4)")
    parser.add_argument("--voice", default="Samantha",
                        help="a macOS voice from `say -v '?'` (default Samantha), 'gtts', or 'silent'")
    parser.add_argument("--headed", action="store_true", help="show the browser while recording")
    args = parser.parse_args(argv)
    print(f"Wrote {record(args.out, voice=args.voice, headless=not args.headed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
