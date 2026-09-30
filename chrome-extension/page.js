/**
 * Page-side half of the extension: find questionnaire questions in a web form, and
 * later write reviewed answers back into them.
 *
 * Injected with chrome.scripting into each frame of the active tab. It defines
 * `globalThis.__attestq = { scan, fill }` in the extension's isolated world, which
 * the service worker then calls. It runs in pages we don't control, so it touches
 * nothing but the controls it tags and never throws out of `scan` or `fill`.
 *
 * A "question" pairs up to two controls:
 *   choice - a radio group or single <select> (Yes / No / N/A, Met / Not Met, ...)
 *   text   - a textarea, rich-text editor or text input for the explanation
 * Portals usually render a control question as a choice plus a comments box, so
 * a choice claims the text control that follows it when both sit in the same
 * question block. Standalone text boxes with a question-like label become
 * free-text questions. Contact details, dates, checkboxes and uploads are left
 * alone: they aren't answerable from evidence.
 */
(() => {
  if (globalThis.__attestq) return;

  const Q_ATTR = 'data-attestq-q';
  const ROLE_ATTR = 'data-attestq-role';
  const CONTROL_SEL = 'input, select, textarea, [contenteditable=""], [contenteditable="true"]';
  const TEXT_INPUT_TYPES = new Set(['text', '']);
  const COMMENT_RE = /\b(comment|explain|explanation|justif|detail|evidence|remark|note|response|additional|clarif)/i;
  const PLACEHOLDER_OPTION_RE = /^(-+|select|choose|please select|pick)\b/i;
  const NUMBERING_RE = /^[\s\d.)(:-]*$|^[a-z]{0,4}[-.]?\d+(\.\d+)*[.)]?$/i;
  const MAX_LABEL = 1000;

  // --- text helpers ----------------------------------------------------------

  const clean = (s) => String(s ?? '').replace(/\s+/g, ' ').trim();
  const trimLabel = (s) => clean(s).replace(/\s*\*+\s*$/, '').replace(/\s*\(required\)$/i, '').slice(0, MAX_LABEL);
  const textOf = (node) => trimLabel(node ? node.innerText ?? node.textContent : '');
  const norm = (s) => clean(s).toLowerCase();
  const isQuestionLike = (label) => /\?\s*$/.test(label) || label.length > 60;

  function isShown(el) {
    if (!el || !el.getClientRects().length) return false;
    const cs = getComputedStyle(el);
    return cs.visibility !== 'hidden' && cs.display !== 'none';
  }

  function isEditable(el) {
    return !el.disabled && !el.readOnly && el.getAttribute('aria-disabled') !== 'true';
  }

  /** Text of the labels attached to a control: <label for>, wrapping <label>, ARIA. */
  function ownLabel(el) {
    const ids = el.getAttribute('aria-labelledby');
    if (ids) {
      const t = trimLabel(ids.split(/\s+/).map((id) => textOf(document.getElementById(id))).join(' '));
      if (t) return t;
    }
    const byLabel = trimLabel([...(el.labels || [])].map(textOf).join(' '));
    return byLabel || trimLabel(el.getAttribute('aria-label') || el.title || '');
  }

  /**
   * The nearest text that precedes `el` inside its own question block. Walks
   * previous siblings, then climbs; stops at siblings holding another field,
   * since text there belongs to that field. `skip` holds nodes to look past
   * (a radio group's own option labels).
   */
  function precedingText(el, skip = new Set()) {
    let node = el;
    for (let depth = 0; node && node !== document.body && depth < 8; depth++) {
      for (let sib = node.previousSibling; sib; sib = sib.previousSibling) {
        if (sib.nodeType === Node.TEXT_NODE) {
          const t = trimLabel(sib.textContent);
          if (t && !NUMBERING_RE.test(t)) return t;
          continue;
        }
        if (sib.nodeType !== Node.ELEMENT_NODE || skip.has(sib)) continue;
        if (!isShown(sib) || sib.matches('script, style, template')) continue;
        if (sib.matches(CONTROL_SEL) || sib.querySelector(CONTROL_SEL)) {
          if ([...skip].some((s) => sib.contains(s))) continue;
          return '';
        }
        const t = textOf(sib);
        if (t && !NUMBERING_RE.test(t)) return t;
      }
      node = node.parentElement;
      if (node && node.matches('fieldset')) {
        const legend = node.querySelector(':scope > legend');
        if (legend && textOf(legend)) return textOf(legend);
      }
    }
    return '';
  }

  /** Question text for a radio group: its group container's label, else the text before it. */
  function groupLabel(radios) {
    const optionLabels = new Set(radios.flatMap((r) => [...(r.labels || [])]));
    const group = radios[0].closest('[role="radiogroup"], [role="group"], fieldset');
    if (group && radios.every((r) => group.contains(r))) {
      const aria = ownLabel(group);
      if (aria) return aria;
      const legend = group.matches('fieldset') && group.querySelector(':scope > legend');
      if (legend && textOf(legend)) return textOf(legend);
    }
    return precedingText(radios[0], optionLabels);
  }

  const optionText = (radio) => ownLabel(radio) || clean(radio.value);

  // --- scan ------------------------------------------------------------------

  /** Every usable control in document order, radio buttons folded into groups. */
  function collectControls() {
    const controls = [];
    const radioGroups = new Map();
    for (const el of document.querySelectorAll(CONTROL_SEL)) {
      if (el.isContentEditable && el.parentElement && el.parentElement.isContentEditable) continue;
      if (!isEditable(el)) continue;
      const tag = el.tagName.toLowerCase();
      const type = tag === 'input' ? (el.getAttribute('type') || 'text').toLowerCase() : tag;

      if (type === 'radio') {
        const labelsShown = [...(el.labels || [])].some(isShown);
        if (!isShown(el) && !labelsShown) continue;
        const key = el.name ? `${el.form ? [...document.forms].indexOf(el.form) : -1}:${el.name}` : el;
        let ctl = radioGroups.get(key);
        if (!ctl) {
          ctl = { kind: 'choice', els: [] };
          radioGroups.set(key, ctl);
          controls.push(ctl);
        }
        ctl.els.push(el);
        continue;
      }
      if (!isShown(el)) continue;
      if (tag === 'select' && !el.multiple) {
        controls.push({ kind: 'choice', els: [el] });
      } else if (tag === 'textarea' || el.isContentEditable) {
        controls.push({ kind: 'long', els: [el] });
      } else if (tag === 'input' && TEXT_INPUT_TYPES.has(type)) {
        controls.push({ kind: 'short', els: [el] });
      }
    }

    for (const ctl of controls) {
      const el = ctl.els[0];
      if (ctl.kind === 'choice' && el.type === 'radio') {
        ctl.label = groupLabel(ctl.els);
        ctl.options = ctl.els.map(optionText);
      } else if (ctl.kind === 'choice') {
        ctl.label = ownLabel(el) || precedingText(el);
        ctl.options = [...el.options]
          .filter((o) => !o.disabled && o.value !== '' && !PLACEHOLDER_OPTION_RE.test(clean(o.text)))
          .map((o) => clean(o.text));
      } else {
        ctl.label = ownLabel(el) || precedingText(el) || trimLabel(el.getAttribute('placeholder'));
      }
    }
    return controls.filter((c) => c.kind !== 'choice' || c.options.length >= 2);
  }

  /** Smallest element containing both nodes. */
  function commonAncestor(a, b) {
    for (let node = a; node; node = node.parentElement) if (node.contains(b)) return node;
    return document.documentElement;
  }

  /** Does `textCtl` explain `choiceCtl`, rather than being a question of its own? */
  function isCompanion(choiceCtl, textCtl, choices) {
    const label = textCtl.label;
    if (label && COMMENT_RE.test(label) && !isQuestionLike(label)) return true;
    if (label && label !== choiceCtl.label && isQuestionLike(label)) return false;
    // Unlabelled or briefly labelled: pair when nothing else competes for the block.
    const block = commonAncestor(choiceCtl.els[0], textCtl.els[0]);
    return choices.filter((c) => c.els.some((el) => block.contains(el))).length === 1;
  }

  function currentChoice(ctl) {
    const el = ctl.els[0];
    if (el.type === 'radio') {
      const checked = ctl.els.find((r) => r.checked);
      return checked ? optionText(checked) : '';
    }
    return el.value === '' ? '' : clean(el.selectedOptions[0]?.text);
  }

  const currentText = (el) => (el.isContentEditable ? el.innerText : el.value) || '';

  function scan() {
    try {
      document.querySelectorAll(`[${Q_ATTR}]`).forEach((n) => {
        n.removeAttribute(Q_ATTR);
        n.removeAttribute(ROLE_ATTR);
      });
      const controls = collectControls();
      const choices = controls.filter((c) => c.kind === 'choice');
      const questions = [];

      const tag = (ctl, id, role) =>
        ctl.els.forEach((el) => {
          el.setAttribute(Q_ATTR, id);
          el.setAttribute(ROLE_ATTR, role);
        });

      for (let i = 0; i < controls.length; i++) {
        const ctl = controls[i];
        const id = `q${questions.length}`;
        if (ctl.kind === 'choice') {
          if (!ctl.label) continue;
          const next = controls[i + 1];
          const companion = next && next.kind !== 'choice' && isCompanion(ctl, next, choices) ? next : null;
          tag(ctl, id, 'choice');
          if (companion) {
            tag(companion, id, 'text');
            i++;
          }
          questions.push({
            id,
            prompt: ctl.label,
            choices: ctl.options,
            hasText: Boolean(companion),
            currentChoice: currentChoice(ctl),
            currentText: companion ? currentText(companion.els[0]) : '',
          });
        } else if (ctl.label && (ctl.kind === 'long' || isQuestionLike(ctl.label))) {
          tag(ctl, id, 'text');
          questions.push({
            id,
            prompt: ctl.label,
            choices: null,
            hasText: true,
            currentChoice: '',
            currentText: currentText(ctl.els[0]),
          });
        }
      }
      return { ok: true, questions };
    } catch (err) {
      return { ok: false, error: String(err && err.message ? err.message : err), questions: [] };
    }
  }

  // --- fill ------------------------------------------------------------------

  /**
   * Set a value the way frameworks notice. React and friends track the value
   * through the element's own setter, so assign through the prototype's native
   * setter, then fire the events a user's typing would.
   */
  function setNativeValue(el, value) {
    const proto = Object.getPrototypeOf(el);
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (setter) setter.call(el, value);
    else el.value = value;
  }

  function fireEdit(el) {
    el.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    el.dispatchEvent(new FocusEvent('blur'));
    el.dispatchEvent(new FocusEvent('focusout', { bubbles: true }));
  }

  /** Exact (case-insensitive) match first; else a unique containment match. */
  function pickOption(options, wanted) {
    const w = norm(wanted);
    if (!w) return null;
    const exact = options.find((o) => o.texts.some((t) => norm(t) === w));
    if (exact) return exact.el;
    const loose = options.filter((o) => o.texts.some((t) => norm(t) && (norm(t).includes(w) || w.includes(norm(t)))));
    return loose.length === 1 ? loose[0].el : null;
  }

  function setChoice(els, value) {
    if (els[0].type === 'radio') {
      const radio = pickOption(els.map((r) => ({ el: r, texts: [optionText(r), r.value] })), value);
      if (!radio) return `"${value}" isn't one of the options`;
      // A real click runs the page's own handlers and fires input + change natively.
      if (!radio.checked) radio.click();
      return radio.checked ? null : 'the page did not accept the selection';
    }
    const select = els[0];
    const option = pickOption([...select.options].map((o) => ({ el: o, texts: [o.text, o.value] })), value);
    if (!option) return `"${value}" isn't one of the options`;
    setNativeValue(select, option.value);
    fireEdit(select);
    return select.value === option.value ? null : 'the page did not accept the selection';
  }

  function setText(el, value) {
    if (el.isContentEditable) {
      el.focus();
      const range = document.createRange();
      range.selectNodeContents(el);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      // insertText goes through the editor's own input pipeline (Quill, CKEditor,
      // ProseMirror...), so its internal model updates, not just the DOM.
      if (!document.execCommand('insertText', false, value)) {
        el.textContent = value;
        el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
      }
      el.blur();
      return null;
    }
    setNativeValue(el, value);
    fireEdit(el);
    return el.value === value ? null : 'the page changed or rejected the text';
  }

  function flash(el) {
    const target = el.type === 'radio' ? el.closest('label') || el : el;
    const previous = target.style.outline;
    target.style.outline = '2px solid #1a7f37';
    setTimeout(() => (target.style.outline = previous), 1600);
  }

  /** Write answers back. items: [{id, choice?, text?}] -> [{id, ok, message}] */
  function fill(items) {
    return items.map(({ id, choice, text }) => {
      try {
        const els = [...document.querySelectorAll(`[${Q_ATTR}="${CSS.escape(id)}"]`)];
        if (!els.length) return { id, ok: false, message: 'Field is no longer on the page. Scan again.' };
        const choiceEls = els.filter((el) => el.getAttribute(ROLE_ATTR) === 'choice');
        const textEl = els.find((el) => el.getAttribute(ROLE_ATTR) === 'text');
        const problems = [];
        if (choiceEls.length && choice) {
          const problem = setChoice(choiceEls, choice);
          if (problem) problems.push(problem);
          else flash(choiceEls.find((r) => r.checked) || choiceEls[0]);
        }
        if (textEl && typeof text === 'string') {
          const problem = setText(textEl, text);
          if (problem) problems.push(problem);
          else flash(textEl);
        }
        return problems.length
          ? { id, ok: false, message: problems.join('; ') }
          : { id, ok: true, message: 'Filled' };
      } catch (err) {
        return { id, ok: false, message: String(err && err.message ? err.message : err) };
      }
    });
  }

  globalThis.__attestq = { scan, fill };
})();
