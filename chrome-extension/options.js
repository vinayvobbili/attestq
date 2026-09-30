/** Settings page. Values pushed by Chrome policy are shown but locked. */
import { DEFAULTS, api, getManagedSettings, getSettings } from './lib.js';

const form = document.getElementById('form');
const result = document.getElementById('result');
const fields = Object.keys(DEFAULTS).map((name) => document.getElementById(name));

function read(input) {
  return input.type === 'checkbox' ? input.checked : input.value.trim();
}

function write(input, value) {
  if (input.type === 'checkbox') input.checked = Boolean(value);
  else input.value = value ?? '';
}

function say(message, ok) {
  result.textContent = message;
  result.className = ok ? 'ok' : 'bad';
}

async function load() {
  const [settings, managed] = await Promise.all([getSettings(), getManagedSettings()]);
  for (const input of fields) {
    write(input, settings[input.id]);
    if (input.id in managed) {
      input.disabled = true;
      const hint = document.createElement('div');
      hint.className = 'hint managed';
      hint.textContent = 'Set by your organisation.';
      input.closest('.field').append(hint);
    }
  }
}

async function save() {
  const values = {};
  for (const input of fields) if (!input.disabled) values[input.id] = read(input);
  await chrome.storage.sync.set(values);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  await save();
  say('Saved.', true);
});

document.getElementById('test').addEventListener('click', async () => {
  say('Checking…', true);
  await save();
  try {
    const health = await api('/health');
    const { namespaces } = await api('/namespaces');
    say(
      `Connected to attestq ${health.version}. ${namespaces.length} vendor${namespaces.length === 1 ? '' : 's'} available.`,
      true,
    );
  } catch (err) {
    say(err.message, false);
  }
});

load();
