# Chrome Web Store listing

Everything the [Developer Dashboard](https://chrome.google.com/webstore/devconsole)
asks for, in the order it asks. Build the upload and the images first:

```bash
python chrome-extension/tools/package.py        # dist/attestq-extension-<version>.zip
python chrome-extension/tools/store_assets.py   # dist/store/: screenshots, promo tile, icon
```

## Package

Upload `dist/attestq-extension-<version>.zip`. The name and the short
description come from `manifest.json`.

## Store listing

**Description**

> Answer security questionnaires in web forms from your own evidence, and
> review every answer before it reaches the page.
>
> Vendor security reviews, due-diligence forms and audit requests increasingly
> live in web portals rather than spreadsheets. attestq for Chrome reads the
> questions on the form, asks your organisation's attestq server for answers
> grounded in that vendor's evidence (SOC 2 reports, policies, prior
> questionnaires), and shows each answer with its sources before anything is
> filled.
>
> How it works
> • Click the attestq icon on a questionnaire page and choose the vendor.
> • Click Scan & Answer. Confident answers arrive ticked, with their sources.
>   Weak or missing evidence is flagged and left unticked. The extension never
>   guesses.
> • Edit, tick or skip each answer, then click Fill selected answers.
> • Check the page and save or submit it yourself. attestq never submits a form.
>
> What it fills
> • Yes/No/N/A radio groups and dropdowns, always with one of the form's own
>   options.
> • Comment and explanation boxes, including rich-text editors.
> • It works with modern web apps (React, Angular, Vue) by firing the events
>   they listen for.
>
> Built for organisations
> • It needs an attestq server, the open-source answer engine
>   (https://github.com/vinayvobbili/attestq), run by you or your
>   organisation. It works with the language model your organisation already
>   uses.
> • Administrators can pre-configure and lock the settings through Chrome
>   policy, so reviewers just install it and start.
> • It records what reviewers ship against what was drafted, which gives an
>   acceptance rate and a calibration scorecard.
>
> Privacy
> • It reads only the tab where you click the icon, and only when you click Scan.
> • Page content goes only to the server in your settings. The authors run no
>   server and receive nothing.
> • No analytics, tracking or advertising.

- **Category:** Productivity › Workflow & Planning
- **Language:** English
- **Store icon:** `dist/store/icon128.png`
- **Screenshots:** `dist/store/screenshot-1-review.png`,
  `screenshot-2-filled.png`, `screenshot-3-settings.png` (1280×800)
- **Small promo tile:** `dist/store/promo-small.png` (440×280)
- **Marquee promo tile** (optional; shown only if the store features the item):
  `dist/store/promo-marquee.png` (1400×560)
- **Global promo video** (optional): upload `dist/store/attestq-demo.mp4`
  (from `tools/demo_video.py`) to YouTube as Unlisted, then paste its URL.
- **Homepage URL:** https://github.com/vinayvobbili/attestq
- **Support URL:** https://github.com/vinayvobbili/attestq/issues
- **Mature content:** No

## Privacy practices

**Single purpose**

> Fill security-questionnaire web forms with answers drafted from the
> organisation's own evidence by its attestq server, after the user reviews
> each one.

**Permission justifications**

| Permission | Justification |
|---|---|
| `activeTab` | Reads the questionnaire's questions, and fills the reviewed answers, only on the tab where the user clicked the extension's icon. |
| `scripting` | Injects the script that finds the form's questions and writes the reviewed answers into that tab. It runs only after the user clicks Scan & Answer or Fill. |
| `storage` | Saves the user's settings (server address, access token, preferences) and the in-progress review for the current tab, so closing the popup loses nothing. |

No host permissions are requested.

**Remote code:** No, I am not using remote code. All JavaScript ships in the
package. The server returns JSON data only.

**Data usage.** Tick only **Website content**. The form's question text and
options are sent to the server in the user's settings. Leave every other type
unticked:
- personally identifiable information;
- health information;
- financial and payment information;
- authentication information: the optional access token is the user's own
  credential for their own server, stored locally, and never collected by the
  developer;
- personal communications;
- location;
- web history;
- user activity.

Certify all three:
- I do not sell or transfer user data to third parties, outside of the approved
  use cases.
- I do not use or transfer user data for purposes that are unrelated to my
  item's single purpose.
- I do not use or transfer user data to determine creditworthiness or for
  lending purposes.

**Privacy policy URL:**
https://github.com/vinayvobbili/attestq/blob/main/chrome-extension/PRIVACY.md

## Distribution

- **Payment:** Free.
- **Visibility:** choose one:
  - **Public:** anyone can find it;
  - **Unlisted:** only people with the link can find it;
  - **Private:** limited to a Google Workspace domain or named testers.
- **Regions:** All regions.

## Account (one-time)

- Register at the dashboard: a one-off US$5 fee, and the Google account needs
  2-Step Verification.
- Verify the contact email.
- Complete the trader / non-trader declaration.
- Set the publisher name shown on the listing.

The first review usually takes from a few days to a couple of weeks. Once it's
approved, the item's ID is fixed. Administrators use that ID in
`ExtensionInstallForcelist` and in the managed-settings policy path (see
[README](README.md#rolling-it-out-in-an-organisation)).
