# attestq for Chrome

Answer security questionnaires **in web forms** from your vendors' evidence. The
extension reads the questions on the page, asks an `attestq serve` server for
grounded, cited answers, lets the reviewer check every one, and then fills the
form. The reviewer still saves or submits the form themselves.

▶ [Watch the 45-second walkthrough](https://youtu.be/WpVs9-BO05c)

```
 questionnaire page ──scan──▶ extension ──/answer──▶ attestq serve ──▶ your evidence + LLM
        ▲                         │
        └────── fill ◀── review ◀─┘  (what the reviewer ships ──/feedback──▶ scorecard)
```

## For the person filling questionnaires

1. Open the questionnaire in Chrome and click the **attestq** icon in the toolbar.
2. Choose the vendor whose evidence to answer from, then click **Scan & Answer**.
   You can close the popup while it works.
3. Review the list:
   - **Green:** a confident answer, already ticked.
   - **Amber:** worth a look (the evidence match is weak, a detail couldn't be
     found in the evidence, or the page already has an answer). Not ticked.
   - **Red:** the evidence doesn't cover the question. Answer it yourself;
     typing an answer ticks it.
   - **Sources:** click to see which documents an answer came from.
4. Click **Fill selected answers**. Filled fields flash green on the page.
5. Check the page and save or submit it as usual. For forms that span several
   pages, repeat on each page.

## Try it locally (5 minutes)

```bash
pip install "attestq[server]"
attestq serve --demo --offline                           # sample vendor "helios", no LLM needed
python -m http.server 8080 -d chrome-extension/demo      # a demo portal page
```

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked**
   and choose the `chrome-extension/` folder.
2. Open <http://localhost:8080/vendor-review.html>, click the attestq icon, pick
   **helios** and click **Scan & Answer**.

`--offline` uses token-overlap retrieval and canned verdicts, so every moving part
runs without a model. For real answers, drop `--offline` and configure a provider
exactly as for `attestq run`, for example
`--provider openai --base-url https://your-gateway/v1`.

## Rolling it out in an organisation

Reviewers should only have to install the extension. Everything else is an
administrator's job and happens once.

### 1. Run the server

Run it next to the evidence store your team already uses:

```bash
attestq serve \
  --chroma /srv/attestq/store \
  --provider openai --base-url https://llm-gateway.internal/v1 \
  --verify \
  --feedback /srv/attestq/feedback.jsonl \
  --host 0.0.0.0 --port 8000 \
  --token "$ATTESTQ_API_TOKEN"
```

- `--chroma`: the same persistent store your other attestq tools ingest into.
  Every namespace in it shows up as a vendor.
- `--verify`: turns on the grounding and quality checks, so risky answers arrive
  flagged and unticked.
- `--feedback`: records what reviewers shipped against what was drafted.
  `GET /scorecard` returns acceptance rate, calibration and per-source trust.
- `--token`: every call except `/health` must carry it. Put the server behind
  your usual TLS reverse proxy.

The server only accepts browser calls from `chrome-extension://` origins, so
ordinary web pages can't read your evidence through it.

### 2. Push the extension and its settings through Chrome policy

- **Install:** add the extension's Web Store ID to `ExtensionInstallForcelist`
  (Intune, Group Policy, Jamf or the Google Admin console). It then appears,
  pinned, in every reviewer's Chrome. If your policy blocks unlisted extensions,
  add the ID to `ExtensionInstallAllowlist` as well.
- **Settings:** set them through the extension's managed-storage policy. Values
  set here are locked on the extension's settings page.

  ```json
  {
    "serverUrl": { "Value": "https://attestq.example.internal" },
    "apiToken": { "Value": "…" },
    "appendSources": { "Value": true }
  }
  ```

  On Windows these live under
  `HKLM\Software\Policies\Google\Chrome\3rdparty\extensions\<extension-id>\policy`.
  On macOS they go in the `com.google.Chrome.extensions.<extension-id>` managed
  preferences domain. The schema is in [`managed_schema.json`](managed_schema.json).

Without policy, each reviewer enters the server address once under **Settings**
and clicks **Test connection**.

### If the Chrome Web Store is blocked

Chrome installs extensions from outside the store only through policy, so this
is an administrator task.

1. **Pack a signed `.crx`.** Keep the generated `.pem` key: it fixes the
   extension ID, and every later version must be signed with it.

   ```bash
   "/path/to/Google Chrome" --pack-extension=chrome-extension --pack-extension-key=attestq.pem
   ```

   The first run, without `--pack-extension-key`, creates the key.
2. **Host the `.crx` next to an `updates.xml`** on an internal HTTPS server:

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <gupdate xmlns="http://www.google.com/update2/response" protocol="2.0">
     <app appid="EXTENSION_ID">
       <updatecheck codebase="https://intranet.example/attestq/attestq-0.5.0.crx" version="0.5.0" />
     </app>
   </gupdate>
   ```

3. **Force-install it:** add `EXTENSION_ID;https://intranet.example/attestq/updates.xml`
   to `ExtensionInstallForcelist`. If policy restricts install sources, also add
   `https://intranet.example/*` to `ExtensionInstallSources`.
4. **To update:** bump the version, re-pack with the same key, and update
   `updates.xml`. Chrome picks it up within a few hours.

**Load unpacked** (Developer mode) is for development only. Managed browsers
usually disable it, and it isn't how an organisation should distribute tools.

## Publishing to the Chrome Web Store

```bash
python chrome-extension/tools/package.py        # -> dist/attestq-extension-<version>.zip
python chrome-extension/tools/store_assets.py   # -> dist/store/ screenshots, promo tile, icon
```

- **`package.py`:** packs only what the browser loads, and refuses to build if
  `manifest.json` and the attestq package versions differ.
- **`store_assets.py`:** runs the real extension against the demo form and
  captures the listing images at the sizes the store requires.

To record the narrated walkthrough, a 1280×800 MP4 for the listing's promo
video and for the README, run:

```bash
pip install "slidecast[playwright]>=0.3"   # needs ffmpeg on PATH
python chrome-extension/tools/demo_video.py                          # -> dist/store/attestq-demo.mp4
python chrome-extension/tools/demo_video.py --voice "Ava (Premium)"  # any macOS voice; or gtts / silent
```

It drives the real popup through scan, review, a hand-written answer and fill,
then narrates each step.

[`STORE_LISTING.md`](STORE_LISTING.md) has every dashboard field ready to paste:
- the description;
- the single purpose;
- the permission justifications;
- the data-usage answers;
- the privacy policy URL;
- the visibility options.

To regenerate the icons after changing their colour or shape, run
`python chrome-extension/tools/make_icons.py`.

## How it decides what a question is

- **Choice questions:** a radio group or a dropdown with at least two real
  options. Its label comes from ARIA, a `<legend>`, or the nearest text before
  it in the same block. For example: `1.1 Is MFA enforced…?` with
  `Yes / No / N/A`.
- **Comment boxes:** a comments box that follows a choice question in the same
  block (or is labelled *Comments*, *Explanation*, *Details*…) holds that
  question's explanation. Both get filled.
- **Free-text questions:** a textarea or rich-text editor with its own
  question-like label.
- **Ignored:** names, emails, dates, numbers, checkboxes, multi-selects and file
  uploads. They aren't answerable from evidence.

attestq's `choices` constraint means a drafted verdict is always one of the
page's own options; anything else is shown to the reviewer, never filled.

## Limits

- **Frames:** forms inside cross-origin frames aren't reachable with `activeTab`
  alone; the extension reads the frames it can.
- **Shadow DOM:** controls inside closed shadow roots aren't found.
- **Uploads:** evidence-upload fields are left for the reviewer.
- **One page at a time:** scan each page or section as you reach it.

## Development

```bash
pip install -e ".[dev,server]" playwright && playwright install chromium
pytest tests/test_extension_e2e.py      # loads the extension into Chromium and fills the demo form
ATTESTQ_E2E_SCREENSHOTS=shots pytest tests/test_extension_e2e.py   # also saves screenshots
```

| File | Role |
|------|------|
| `page.js` | Injected into the page: finds questions (`scan`) and writes answers (`fill`). |
| `background.js` | Service worker: runs scan → answer → fill per tab; survives the popup closing. |
| `popup.*` | The review UI: a view of the job plus the reviewer's edits. |
| `options.*`, `managed_schema.json` | Settings, lockable by policy. |
| `lib.js` | Settings resolution and the server client, shared by all of the above. |
| `tools/` | Icon generator, Web Store packager, listing-image generator and demo-video recorder; `harness.py` runs the demo stack for them and the e2e test. |
| `demo/` | A sample portal page covering the common layouts. |
