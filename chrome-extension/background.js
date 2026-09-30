/**
 * Service worker: runs the scan -> answer -> fill job for a tab.
 *
 * The work lives here rather than in the popup because the popup closes the moment
 * the user clicks the page, and a 60-question form takes a minute or two to
 * answer. Each tab's job is kept in chrome.storage.session under `job:<tabId>`;
 * the popup is only a view of that record plus the reviewer's edits.
 *
 * Job record:
 *   { runId, status: 'scanning' | 'answering' | 'ready' | 'filling' | 'error',
 *     url, title, namespace, message, error, progress: {done, total}, items: [Item] }
 * Item:
 *   { key, frameId, localId, prompt, choices, hasText, currentChoice, currentText,
 *     answer, draft: {choice, text}, edit: {choice, text}, checked, result }
 */
import { ApiError, api, getSettings } from './lib.js';

const BATCH_SIZE = 8;
const KEEPALIVE_MS = 20_000;
const jobKey = (tabId) => `job:${tabId}`;
const activeRuns = new Map(); // tabId -> runId of the job allowed to write

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === 'analyze') {
    analyze(msg.tabId, msg.namespace);
    sendResponse({ started: true });
  } else if (msg?.type === 'commit') {
    commit(msg.tabId);
    sendResponse({ started: true });
  }
  return false;
});

chrome.tabs.onRemoved.addListener((tabId) => {
  activeRuns.delete(tabId);
  chrome.storage.session.remove(jobKey(tabId));
});

// --- storage -----------------------------------------------------------------

async function loadJob(tabId) {
  return (await chrome.storage.session.get(jobKey(tabId)))[jobKey(tabId)] || null;
}

async function saveJob(tabId, job) {
  if (activeRuns.get(tabId) !== job.runId) return false; // superseded by a newer scan
  // `rev` counts worker writes only, so the popup redraws for progress but not
  // for its own edits (which would reset the textarea the reviewer is typing in).
  job.rev = (job.rev || 0) + 1;
  await chrome.storage.session.set({ [jobKey(tabId)]: job });
  return true;
}

/** Long jobs outlive the worker's 30s idle timer; an extension API call resets it. */
function keepAlive() {
  const timer = setInterval(() => chrome.runtime.getPlatformInfo(), KEEPALIVE_MS);
  return () => clearInterval(timer);
}

const friendly = (err) =>
  err instanceof ApiError ? err.message : `Something went wrong: ${err && err.message ? err.message : err}`;

// --- page access ---------------------------------------------------------------

/** Inject page.js and run `func` in the tab; every frame we may access, else the top frame. */
async function runInPage(target, func, args = []) {
  const run = async (t) => {
    await chrome.scripting.executeScript({ target: t, files: ['page.js'] });
    return chrome.scripting.executeScript({ target: t, func, args });
  };
  try {
    return await run(target);
  } catch (err) {
    if (!target.allFrames) throw err;
    return run({ tabId: target.tabId }); // a cross-origin frame refused; settle for the page itself
  }
}

async function scanTab(tabId) {
  let results;
  try {
    results = await runInPage({ tabId, allFrames: true }, () =>
      globalThis.__attestq ? globalThis.__attestq.scan() : null,
    );
  } catch (err) {
    throw new ApiError(
      /cannot access|cannot be scripted|extensions gallery/i.test(String(err && err.message))
        ? "Chrome doesn't let extensions read this page. Open the questionnaire in a normal tab."
        : `Couldn't read the page: ${err.message}`,
    );
  }
  const items = [];
  for (const { frameId, result } of results) {
    if (!result || !result.ok) continue;
    for (const q of result.questions) {
      items.push({
        key: `${frameId}:${q.id}`,
        frameId,
        localId: q.id,
        prompt: q.prompt,
        choices: q.choices && q.choices.length ? q.choices : null,
        hasText: q.hasText,
        currentChoice: q.currentChoice,
        currentText: q.currentText,
        answer: null,
        draft: { choice: '', text: '' },
        edit: { choice: '', text: '' },
        checked: false,
        result: null,
      });
    }
  }
  return items;
}

// --- answering -----------------------------------------------------------------

function matchChoice(choices, determination) {
  if (!choices) return '';
  const d = String(determination || '').trim().toLowerCase();
  return choices.find((c) => c.toLowerCase() === d) || '';
}

function applyAnswer(item, answer, settings) {
  item.answer = answer;
  const usable = !answer.error && !answer.insufficient_evidence;
  const choice = usable ? matchChoice(item.choices, answer.determination) : '';
  let text = '';
  if (item.hasText && usable) {
    text = answer.summary || '';
    const sources = [...new Set((answer.citations || []).map((c) => c.source))];
    if (settings.appendSources && sources.length) text += `\n\nSources: ${sources.join(', ')}`;
  }
  item.draft = { choice, text };
  item.edit = { ...item.draft };
  // Pre-tick only drafts a reviewer can trust at a glance, and never over an answer
  // already on the page; everything else waits for a human.
  const alreadyAnswered = Boolean(item.currentChoice || item.currentText.trim());
  item.checked =
    usable && !answer.needs_review && !alreadyAnswered && (item.choices ? Boolean(choice) : Boolean(text));
}

function summarizeJob(job) {
  const answered = job.items.filter((i) => i.answer && !i.answer.error);
  const ready = job.items.filter((i) => i.checked).length;
  const noEvidence = answered.filter((i) => i.answer.insufficient_evidence).length;
  const review = job.items.length - ready - noEvidence;
  const parts = [`${ready} ready to fill`];
  if (review) parts.push(`${review} to check`);
  if (noEvidence) parts.push(`${noEvidence} with no evidence`);
  return `Found ${job.items.length} questions: ${parts.join(', ')}.`;
}

async function analyze(tabId, namespace) {
  const runId = crypto.randomUUID();
  activeRuns.set(tabId, runId);
  const stop = keepAlive();
  const tab = await chrome.tabs.get(tabId);
  const job = {
    runId,
    status: 'scanning',
    url: tab.url,
    title: tab.title,
    namespace,
    message: 'Reading the questions on this page…',
    error: '',
    progress: { done: 0, total: 0 },
    items: [],
  };
  try {
    await saveJob(tabId, job);
    const settings = await getSettings();
    job.items = await scanTab(tabId);
    if (!job.items.length) {
      throw new ApiError(
        'No questions found on this page. Open the page that shows the questions and try again.',
      );
    }
    job.status = 'answering';
    job.progress.total = job.items.length;
    job.message = `Answering ${job.items.length} questions from ${namespace}'s evidence…`;
    if (!(await saveJob(tabId, job))) return;

    for (let i = 0; i < job.items.length; i += BATCH_SIZE) {
      const batch = job.items.slice(i, i + BATCH_SIZE);
      const res = await api('/answer', {
        method: 'POST',
        settings,
        body: {
          namespace,
          questions: batch.map((it) => ({ id: it.key, prompt: it.prompt, choices: it.choices })),
        },
      });
      const byId = new Map(res.answers.map((a) => [a.question_id, a]));
      for (const it of batch) {
        applyAnswer(it, byId.get(it.key) || { error: 'No answer returned' }, settings);
      }
      job.progress.done += batch.length;
      job.message = `Answered ${job.progress.done} of ${job.progress.total}…`;
      if (!(await saveJob(tabId, job))) return;
    }
    job.status = 'ready';
    job.message = summarizeJob(job);
  } catch (err) {
    job.error = friendly(err);
    // Keep whatever was answered before the failure; it is still reviewable.
    job.status = job.items.some((i) => i.answer) ? 'ready' : 'error';
    if (job.status === 'ready') job.message = summarizeJob(job);
  } finally {
    stop();
    await saveJob(tabId, job);
  }
}

// --- filling -------------------------------------------------------------------

async function commit(tabId) {
  const job = await loadJob(tabId);
  if (!job || job.status !== 'ready') return;
  activeRuns.set(tabId, job.runId);
  const chosen = job.items.filter((i) => i.checked);
  job.status = 'filling';
  job.error = '';
  job.message = `Filling ${chosen.length} answers…`;
  await saveJob(tabId, job);

  const byFrame = new Map();
  for (const it of chosen) {
    if (!byFrame.has(it.frameId)) byFrame.set(it.frameId, []);
    byFrame.get(it.frameId).push(it);
  }
  for (const [frameId, items] of byFrame) {
    const payload = items.map((it) => ({
      id: it.localId,
      choice: it.choices ? it.edit.choice : undefined,
      text: it.hasText ? it.edit.text : undefined,
    }));
    let results = null;
    try {
      const [res] = await runInPage(
        { tabId, frameIds: [frameId] },
        (entries) => (globalThis.__attestq ? globalThis.__attestq.fill(entries) : null),
        [payload],
      );
      results = res && res.result;
    } catch {
      results = null;
    }
    const byId = new Map((results || []).map((r) => [r.id, r]));
    for (const it of items) {
      it.result = byId.get(it.localId) || { ok: false, message: 'The page changed. Scan again.' };
    }
  }

  const filled = chosen.filter((i) => i.result && i.result.ok);
  job.status = 'ready';
  job.message =
    `Filled ${filled.length} of ${chosen.length}. ` +
    'Check the page, then save or submit it the way you normally do.';
  if (filled.length < chosen.length) {
    job.error = `${chosen.length - filled.length} could not be filled; see the items marked below.`;
  }
  for (const it of filled) it.checked = false; // done; a second click won't re-fill them
  await saveJob(tabId, job);
  sendFeedback(job, filled);
}

/** Tell the server what reviewers shipped vs. what was drafted. Best effort. */
async function sendFeedback(job, filled) {
  if (!filled.length) return;
  const asText = (v) => [v.choice, v.text].filter(Boolean).join('\n').trim();
  const items = filled.map((it) => ({
    question_id: it.key,
    prompt: it.prompt,
    draft: asText(it.draft) || null,
    final: asText(it.edit),
    confidence: it.answer ? it.answer.confidence : null,
    insufficient_evidence: Boolean(it.answer && it.answer.insufficient_evidence),
    sources: it.answer ? [...new Set((it.answer.citations || []).map((c) => c.source))] : [],
  }));
  try {
    await api('/feedback', { method: 'POST', body: { namespace: job.namespace, items } });
  } catch (err) {
    console.info('attestq: feedback not recorded:', err.message);
  }
}
