"""Build the zip you upload to the Chrome Web Store (or host for self-managed installs).

    python chrome-extension/tools/package.py               # -> dist/attestq-extension-<version>.zip

Packs only what the browser loads — no demo page, tools, tests or docs — with
manifest.json at the zip root as the store requires. Refuses to build when the
manifest version and the attestq package version disagree, so the extension and
the server it talks to are released together.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

EXTENSION_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXTENSION_DIR.parent
EXCLUDE_DIRS = {"demo", "tools", "__pycache__"}
EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_SUFFIXES = {".md"}  # docs for people, not the browser


def package_version() -> str:
    text = (REPO_DIR / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("could not find the version in pyproject.toml")
    return match.group(1)


def extension_files(root: Path = EXTENSION_DIR):
    """Files the browser needs, as paths relative to the extension directory."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and not set(rel.parts[:-1]) & EXCLUDE_DIRS and rel.name not in EXCLUDE_NAMES \
                and rel.suffix not in EXCLUDE_SUFFIXES:
            yield rel


def check_manifest(root: Path = EXTENSION_DIR) -> dict:
    """Validate what the store would reject: missing files, icons, version mismatch."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    referenced = [
        manifest["background"]["service_worker"],
        manifest["action"]["default_popup"],
        manifest["options_page"],
        manifest["storage"]["managed_schema"],
        *manifest["icons"].values(),
    ]
    missing = [f for f in referenced if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"manifest references missing files: {', '.join(missing)}")
    if "128" not in manifest["icons"]:
        raise SystemExit("the Web Store needs a 128x128 icon")
    return manifest


def build(out_dir: Path, root: Path = EXTENSION_DIR, expected_version: str | None = None) -> Path:
    manifest = check_manifest(root)
    if expected_version and manifest["version"] != expected_version:
        raise SystemExit(
            f"manifest version {manifest['version']} != attestq {expected_version}; bump them together"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"attestq-extension-{manifest['version']}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in extension_files(root):
            zf.write(root / rel, rel.as_posix())
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_DIR / "dist", help="output directory (default: dist/)")
    args = parser.parse_args(argv)
    target = build(args.out, expected_version=package_version())
    with zipfile.ZipFile(target) as zf:
        names = zf.namelist()
    print(f"Wrote {target} ({len(names)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
