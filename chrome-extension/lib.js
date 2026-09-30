/**
 * Settings and server calls shared by the service worker, popup and options page.
 *
 * Settings resolve in three layers: built-in defaults, then what the user saved on
 * the options page (chrome.storage.sync), then anything an administrator pushed
 * through Chrome policy (chrome.storage.managed). Policy always wins, so an
 * organisation can pre-configure the server and users never see a setup step.
 */

export const DEFAULTS = Object.freeze({
  serverUrl: 'http://127.0.0.1:8000',
  apiToken: '',
  appendSources: true,
  defaultNamespace: '',
});

const REQUEST_TIMEOUT_MS = 180_000;

/** Policy values, or {} when none are set (or the platform has no managed storage). */
export async function getManagedSettings() {
  try {
    return (await chrome.storage.managed.get(null)) || {};
  } catch {
    return {};
  }
}

export async function getSettings() {
  const [saved, managed] = await Promise.all([chrome.storage.sync.get(null), getManagedSettings()]);
  const pick = (source) =>
    Object.fromEntries(Object.entries(source || {}).filter(([k]) => k in DEFAULTS));
  const settings = { ...DEFAULTS, ...pick(saved), ...pick(managed) };
  settings.serverUrl = String(settings.serverUrl).trim().replace(/\/+$/, '');
  return settings;
}

/** An error whose message is safe to show a non-technical user as-is. */
export class ApiError extends Error {}

/**
 * Call the attestq server. Resolves with parsed JSON; rejects with an ApiError
 * that says what went wrong in plain words.
 */
export async function api(path, { method = 'GET', body, settings } = {}) {
  const cfg = settings || (await getSettings());
  const headers = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (cfg.apiToken) headers.Authorization = `Bearer ${cfg.apiToken}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(cfg.serverUrl + path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') throw new ApiError('The answer server took too long to respond.');
    throw new ApiError(`Can't reach the answer server at ${cfg.serverUrl}. Check Settings, or ask your admin.`);
  } finally {
    clearTimeout(timer);
  }

  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON body; handled below */
  }
  if (!res.ok) {
    if (res.status === 401) throw new ApiError('The answer server rejected the access token. Check Settings.');
    const detail = data && typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`;
    throw new ApiError(`The answer server returned an error: ${detail}`);
  }
  return data;
}
