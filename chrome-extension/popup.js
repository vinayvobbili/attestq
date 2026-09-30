/**
 * Popup: pick a vendor, start a scan, review the drafted answers, fill the page.
 *
 * The service worker (background.js) does the work and keeps the job in
 * chrome.storage.session; this file renders that job and writes back only what
 * the reviewer changes: each item's `edit` and `checked`.
 */
import { ApiError, api, getSettings } from './lib.js';

const $ = (id) => document.getElementById(id);
const ui = {
  namespace: $('namespace'),
  scan: $('scan'),
  status: $('status'),
  progress: $('progress'),
  error: $('error'),
  review: $('review'),
  items: $('items'),
  toggleAll: $('toggle-all'),
  footer: $('footer'),
  commit: $('commit'),
};

let tab = null;
let job = null;
let renderedRev = -1;
let saveTimer = null;
const BUSY = new Set(['scanning', 'answering', 'filling']);
const jobKey = () => `job:${tab.id}`;

// --- small DOM helpers -----------------------------------------------------------

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const child of children) if (child != null) node.append(child);
  return node;
}

function showError(message, withSettingsLink = false) {
  ui.error.replaceChildren(message);
  if (withSettingsLink) {
    const link = el('a', { href: '#', textContent: 'Open Settings' });
    link.addEventListener('click', (e) => {
      e.preventDefault();
      chrome.runtime.openOptionsPage();
    });
    ui.error.append(' ', link);
  }
  ui.error.hidden = !message;
}

// --- vendors ---------------------------------------------------------------------

async function loadNamespaces() {
  const settings = await getSettings();
  const { lastNamespace } = await chrome.storage.local.get('lastNamespace');
  try {
    const { namespaces } = await api('/namespaces', { settings });
    ui.namespace.replaceChildren(
      ...namespaces.map((n) => el('option', { value: n.name, textContent: n.name })),
    );
    if (!namespaces.length) {
      ui.namespace.append(el('option', { value: '', textContent: 'No vendors loaded on the server' }));
    }
    const preferred = [job && job.namespace, lastNamespace, settings.defaultNamespace].find(
      (n) => n && namespaces.some((x) => x.name === n),
    );
    if (preferred) ui.namespace.value = preferred;
  } catch (err) {
    ui.namespace.replaceChildren(el('option', { value: '', textContent: 'Server unavailable' }));
    showError(err instanceof ApiError ? err.message : String(err), true);
  }
  updateButtons();
}

// --- rendering -------------------------------------------------------------------

function itemState(item) {
  const a = item.answer;
  if (!a) return 'pending';
  if (a.error) return 'error';
  if (a.insufficient_evidence) return 'none';
  if (item.checked || (item.result && item.result.ok)) return 'ready';
  return 'check';
}

function badges(item) {
  const a = item.answer;
  const out = [];
  if (!a) return [el('span', { className: 'badge', textContent: 'Waiting…' })];
  if (a.error) return [el('span', { className: 'badge bad', textContent: 'Could not answer' })];
  if (a.insufficient_evidence) {
    return [el('span', { className: 'badge bad', textContent: 'No supporting evidence' })];
  }
  const pct = Math.round((a.confidence || 0) * 100);
  out.push(el('span', { className: `badge ${pct >= 60 ? 'ok' : ''}`, textContent: `Evidence match ${pct}%` }));
  if (a.needs_review) out.push(el('span', { className: 'badge warn', textContent: 'Check this answer' }));
  if (item.choices && !item.draft.choice) {
    out.push(el('span', { className: 'badge warn', textContent: 'Pick an option' }));
  }
  if (item.currentChoice || (item.currentText || '').trim()) {
    out.push(el('span', { className: 'badge warn', textContent: 'Page already has an answer' }));
  }
  return out;
}

function notes(item) {
  const a = item.answer;
  if (!a) return null;
  let text = '';
  if (a.error) text = `The server couldn't answer this one (${a.error}). Answer it yourself.`;
  else if (a.insufficient_evidence) text = "The vendor's evidence doesn't cover this. Answer it yourself.";
  else if (item.choices && !item.draft.choice && a.determination) {
    text = `Suggested "${a.determination}", which isn't one of the page's options.`;
  } else if (a.review_notes && a.review_notes.length) text = a.review_notes.join(' ');
  return text ? el('p', { className: 'note', textContent: text }) : null;
}

function sources(item) {
  const cites = (item.answer && item.answer.citations) || [];
  if (!cites.length) return null;
  return el(
    'details',
    { className: 'sources' },
    el('summary', { textContent: `${cites.length} source${cites.length > 1 ? 's' : ''}` }),
    el(
      'ul',
      {},
      ...cites.map((c) => el('li', {}, el('b', { textContent: c.source }), ` — ${c.snippet}`)),
    ),
  );
}

function renderItem(item, editable) {
  const li = el('li', { className: `item state-${itemState(item)}${item.checked ? '' : ' unticked'}` });
  li.dataset.key = item.key;

  const pick = el('input', { type: 'checkbox', checked: item.checked, disabled: !editable || !item.answer });
  pick.setAttribute('aria-label', 'Fill this answer');
  pick.addEventListener('change', () => {
    item.checked = pick.checked;
    li.classList.toggle('unticked', !item.checked);
    queueSave();
  });

  const question = el('div', { className: 'question', textContent: item.prompt, title: 'Click to expand' });
  question.addEventListener('click', () => question.classList.toggle('expanded'));

  const markTicked = () => {
    item.checked = true;
    pick.checked = true;
    li.classList.remove('unticked');
  };

  const fields = el('div', { className: 'fields' });
  if (item.choices) {
    const select = el('select', { disabled: !editable || !item.answer });
    select.setAttribute('aria-label', 'Answer');
    select.append(el('option', { value: '', textContent: '— choose —' }));
    for (const choice of item.choices) select.append(el('option', { value: choice, textContent: choice }));
    select.value = item.edit.choice;
    select.addEventListener('change', () => {
      item.edit.choice = select.value;
      if (select.value) markTicked();
      queueSave();
    });
    fields.append(select);
  }
  if (item.hasText) {
    const area = el('textarea', { value: item.edit.text, disabled: !editable || !item.answer, rows: 3 });
    area.setAttribute('aria-label', 'Explanation');
    area.placeholder = item.choices ? 'Explanation (optional)' : 'Answer';
    area.addEventListener('input', () => {
      item.edit.text = area.value;
      if (area.value.trim()) markTicked();
      queueSave();
    });
    fields.append(area);
  }

  li.append(
    el('div', { className: 'item-top' }, pick, question),
    el('div', { className: 'badges' }, ...badges(item)),
    fields,
  );
  const note = notes(item);
  if (note) li.append(note);
  const src = sources(item);
  if (src) li.append(src);
  if (item.result) {
    li.append(
      el('p', {
        className: `result ${item.result.ok ? 'ok' : 'fail'}`,
        textContent: item.result.ok ? '✓ Filled on the page' : `✗ ${item.result.message}`,
      }),
    );
  }
  return li;
}

function render() {
  const busy = job && BUSY.has(job.status);
  ui.status.hidden = !job;
  ui.status.textContent = job ? job.message : '';
  ui.progress.hidden = !(job && job.status === 'answering');
  if (job && job.progress.total) {
    ui.progress.firstElementChild.style.width = `${(100 * job.progress.done) / job.progress.total}%`;
  }
  if (job && job.error) showError(job.error);
  else if (job) showError('');

  const hasItems = Boolean(job && job.items.length);
  ui.review.hidden = !hasItems;
  ui.footer.hidden = !hasItems;
  if (hasItems) {
    const editable = job.status === 'ready';
    ui.items.replaceChildren(...job.items.map((item) => renderItem(item, editable)));
  }
  renderedRev = job ? job.rev : -1;
  ui.scan.textContent = busy ? 'Working…' : job ? 'Scan again' : 'Scan & Answer';
  updateButtons();
}

function updateButtons() {
  const busy = job && BUSY.has(job.status);
  ui.scan.disabled = Boolean(busy) || !ui.namespace.value;
  const count = job ? job.items.filter((i) => i.checked).length : 0;
  ui.commit.disabled = !job || job.status !== 'ready' || count === 0;
  ui.commit.textContent = count ? `Fill ${count} selected answer${count > 1 ? 's' : ''}` : 'Select answers to fill';
  const fillable = job ? job.items.filter((i) => i.answer && (i.edit.choice || i.edit.text.trim())) : [];
  ui.toggleAll.disabled = !job || job.status !== 'ready' || !fillable.length;
  ui.toggleAll.checked = fillable.length > 0 && fillable.every((i) => i.checked);
}

// --- persisting the reviewer's edits ---------------------------------------------------

function queueSave() {
  updateButtons();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveEdits, 250);
}

/** Merge edits into the stored job; the worker owns everything else in the record. */
async function saveEdits() {
  clearTimeout(saveTimer);
  if (!job) return;
  const stored = (await chrome.storage.session.get(jobKey()))[jobKey()];
  if (!stored || stored.runId !== job.runId || stored.status !== 'ready') return;
  const mine = new Map(job.items.map((i) => [i.key, i]));
  for (const item of stored.items) {
    const local = mine.get(item.key);
    if (local) {
      item.edit = local.edit;
      item.checked = local.checked;
    }
  }
  await chrome.storage.session.set({ [jobKey()]: stored });
}

// --- actions ---------------------------------------------------------------------

ui.scan.addEventListener('click', async () => {
  const namespace = ui.namespace.value;
  if (!namespace) return;
  showError('');
  await chrome.storage.local.set({ lastNamespace: namespace });
  ui.scan.disabled = true;
  chrome.runtime.sendMessage({ type: 'analyze', tabId: tab.id, namespace });
});

ui.commit.addEventListener('click', async () => {
  ui.commit.disabled = true;
  await saveEdits();
  chrome.runtime.sendMessage({ type: 'commit', tabId: tab.id });
});

ui.toggleAll.addEventListener('change', () => {
  for (const item of job.items) {
    if (item.answer && (item.edit.choice || item.edit.text.trim())) item.checked = ui.toggleAll.checked;
  }
  render();
  queueSave();
});

ui.namespace.addEventListener('change', updateButtons);
$('settings').addEventListener('click', () => chrome.runtime.openOptionsPage());

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'session' || !tab || !changes[jobKey()]) return;
  const next = changes[jobKey()].newValue || null;
  if (next && next.rev === renderedRev) return; // our own edit echoing back
  job = next;
  render();
});

// --- start -----------------------------------------------------------------------

async function init() {
  [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const stored = (await chrome.storage.session.get(jobKey()))[jobKey()] || null;
  // A job from before the tab navigated describes fields that no longer exist.
  job = stored && stored.url === tab.url ? stored : null;
  render();
  await loadNamespaces();
}

init();
