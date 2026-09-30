"""Tests for the Chrome extension's build tools (chrome-extension/tools/)."""

from __future__ import annotations

import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import pytest

from attestq import __version__

TOOLS = Path(__file__).resolve().parents[1] / "chrome-extension" / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ext_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package = _load("package")
make_icons = _load("make_icons")


def test_manifest_version_tracks_the_package():
    manifest = json.loads((package.EXTENSION_DIR / "manifest.json").read_text())
    assert manifest["version"] == __version__ == package.package_version()


def test_manifest_references_only_files_that_exist():
    package.check_manifest()  # raises SystemExit otherwise


def test_zip_holds_what_the_browser_loads_and_nothing_else(tmp_path):
    target = package.build(tmp_path, expected_version=__version__)
    names = set(zipfile.ZipFile(target).namelist())
    assert {"manifest.json", "background.js", "page.js", "popup.html", "icons/icon128.png"} <= names
    assert not any(n.startswith(("demo/", "tools/")) for n in names)


def test_version_mismatch_refuses_to_build(tmp_path):
    with pytest.raises(SystemExit, match="bump them together"):
        package.build(tmp_path, expected_version="0.0.0")


def test_missing_referenced_file_is_caught(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "background": {"service_worker": "background.js"},
        "action": {"default_popup": "popup.html"},
        "options_page": "options.html",
        "storage": {"managed_schema": "managed_schema.json"},
        "icons": {"128": "icons/icon128.png"},
    }))
    with pytest.raises(SystemExit, match="missing files"):
        package.check_manifest(tmp_path)


def test_icons_are_valid_rgba_pngs_of_the_requested_size():
    png = make_icons.render_png(32)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height, depth, colour = struct.unpack(">IIBB", png[16:26])
    assert (width, height, depth, colour) == (32, 32, 8, 6)


def test_committed_icons_match_the_generator():
    committed = (package.EXTENSION_DIR / "icons" / "icon48.png").read_bytes()
    assert committed == make_icons.render_png(48)
