// Browser demo: loads the exported models and streams generations locally, one token per frame.

import { generateTokens } from "./generate.js";
import { createModel } from "./models.js";
import { BPETokenizer } from "./tokenizer.js";

const LABELS = { transformer: "Transformer", lstm: "LSTM", rnn: "RNN" };
const STRATEGIES = [
  ["out-greedy", { greedy: true }],
  ["out-temperature", { temperature: 0.8 }],
  ["out-nucleus", { temperature: 1.0, topP: 0.9 }],
];

const $ = (id) => document.getElementById(id);
const state = { tokenizer: null, models: new Map(), selected: null, run: null };
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

async function fetchOk(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response;
}

async function loadModels() {
  const [index, tokenizerData] = await Promise.all([
    fetchOk("models/index.json").then((r) => r.json()),
    fetchOk("models/tokenizer.json").then((r) => r.json()),
  ]);
  state.tokenizer = new BPETokenizer(tokenizerData);
  const entries = await Promise.all(
    index.models.map(async (name) => {
      const manifest = await fetchOk(`models/${name}.json`).then((r) => r.json());
      const buffer = await fetchOk(`models/${manifest.weights}`).then((r) => r.arrayBuffer());
      return [name, createModel(manifest, buffer)];
    }),
  );
  for (const [name, model] of entries) state.models.set(name, model);
  return index.default;
}

function renderModelPicker() {
  const container = $("models");
  for (const [name, model] of state.models) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-option";
    button.setAttribute("role", "radio");
    button.dataset.model = name;
    const title = document.createElement("span");
    title.className = "model-name";
    title.textContent = LABELS[model.type] ?? name;
    const meta = document.createElement("span");
    meta.className = "model-meta";
    meta.textContent = `${model.params.toLocaleString("en-US")} parameters`;
    button.append(title, meta);
    button.addEventListener("click", () => selectModel(name));
    container.append(button);
  }
  // Arrow keys move between models, as in a native radio group.
  container.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const names = [...state.models.keys()];
    const next = names[(names.indexOf(state.selected) + step + names.length) % names.length];
    selectModel(next);
    container.querySelector(`[data-model="${next}"]`).focus();
  });
}

function selectModel(name) {
  cancelRun();
  state.selected = name;
  for (const button of $("models").children) {
    const checked = button.dataset.model === name;
    button.setAttribute("aria-checked", String(checked));
    button.tabIndex = checked ? 0 : -1;
  }
}

function readOptions() {
  const topK = Number($("top-k").value);
  const topP = Number($("top-p").value);
  // Same rules as the python demo: 0 turns top-k off, and top-p is off at 0 or 1.
  return {
    maxNewTokens: Number($("max-tokens").value),
    temperature: Number($("temperature").value),
    topK: topK > 0 ? topK : null,
    topP: topP > 0 && topP < 1 ? topP : null,
    greedy: $("greedy").checked,
  };
}

function renderText(element, promptText, text) {
  const promptSpan = document.createElement("span");
  promptSpan.className = "prompt-text";
  const generated = document.createElement("span");
  if (text.startsWith(promptText)) {
    promptSpan.textContent = promptText;
    generated.textContent = text.slice(promptText.length);
  } else {
    generated.textContent = text;
  }
  element.replaceChildren(promptSpan, generated);
}

function renderMessage(element, message) {
  const span = document.createElement("span");
  span.className = "message";
  span.textContent = message;
  element.replaceChildren(span);
}

function startRun(button, idleLabel) {
  cancelRun();
  const run = { cancelled: false, button, idleLabel };
  state.run = run;
  button.textContent = "Stop";
  button.classList.add("running");
  return run;
}

function finishRun(run) {
  run.button.textContent = run.idleLabel;
  run.button.classList.remove("running");
  if (state.run === run) state.run = null;
}

function cancelRun() {
  if (state.run) state.run.cancelled = true;
}

// Steps each stream once per frame until all finish, returning how much compute time they used.
async function stream(run, streams) {
  let compute = 0;
  while (!run.cancelled && streams.some((s) => !s.done)) {
    const start = performance.now();
    for (const s of streams.filter((s) => !s.done)) {
      const { value, done } = s.steps.next();
      if (done) {
        s.done = true;
      } else {
        s.text = value;
        s.tokens += 1;
      }
    }
    compute += performance.now() - start;
    for (const s of streams) renderText(s.element, s.promptText, s.text);
    await nextFrame();
  }
  return compute;
}

async function runGenerate() {
  if (state.run?.button === $("generate")) return cancelRun();
  const { tokenizer } = state;
  const model = state.models.get(state.selected);
  const prompt = $("prompt").value;
  const output = $("output");
  const promptIds = tokenizer.encode(prompt);
  if (promptIds.length === 0) {
    $("stats").textContent = "";
    return renderMessage(output, "Type a prompt first.");
  }

  const run = startRun($("generate"), "Generate");
  const promptText = tokenizer.decode(promptIds);
  const s = { steps: generateTokens(model, tokenizer, prompt, readOptions()), element: output, promptText };
  Object.assign(s, { text: promptText, tokens: 0, done: false });
  output.classList.add("streaming");
  $("stats").textContent = "Generating…";

  const compute = await stream(run, [s]);

  output.classList.remove("streaming");
  const count = `${s.tokens} token${s.tokens === 1 ? "" : "s"}`;
  const perToken = s.tokens ? ` · ${(compute / s.tokens).toFixed(1)} ms/token` : "";
  const ending = run.cancelled ? " · stopped" : "";
  $("stats").textContent = `${count}${perToken}${ending}`;
  finishRun(run);
}

async function runCompare() {
  if (state.run?.button === $("compare")) return cancelRun();
  const { tokenizer } = state;
  const model = state.models.get(state.selected);
  const prompt = $("compare-prompt").value;
  const promptIds = tokenizer.encode(prompt);
  if (promptIds.length === 0) {
    for (const [id] of STRATEGIES) renderMessage($(id), "Type a prompt first.");
    return;
  }

  const run = startRun($("compare"), "Compare");
  const promptText = tokenizer.decode(promptIds);
  const maxNewTokens = Number($("compare-tokens").value);
  const streams = STRATEGIES.map(([id, options]) => ({
    steps: generateTokens(model, tokenizer, prompt, { ...options, maxNewTokens }),
    element: $(id),
    promptText,
    text: promptText,
    tokens: 0,
    done: false,
  }));
  for (const s of streams) s.element.classList.add("streaming");
  await stream(run, streams);
  for (const s of streams) s.element.classList.remove("streaming");
  finishRun(run);
}

function bindSlider(id, format = String) {
  const input = $(id);
  const show = () => {
    $(`${id}-value`).textContent = format(Number(input.value));
  };
  input.addEventListener("input", show);
  show();
}

function selectTab(tab) {
  for (const other of document.querySelectorAll('[role="tab"]')) {
    const selected = other === tab;
    other.setAttribute("aria-selected", String(selected));
    other.tabIndex = selected ? 0 : -1;
    $(other.getAttribute("aria-controls")).hidden = !selected;
  }
}

function bindUi() {
  bindSlider("max-tokens");
  bindSlider("temperature", (v) => v.toFixed(1));
  bindSlider("top-k", (v) => (v > 0 ? String(v) : "off"));
  bindSlider("top-p", (v) => (v > 0 && v < 1 ? v.toFixed(2) : "off"));
  bindSlider("compare-tokens");

  const syncGreedy = () => {
    for (const id of ["temperature", "top-k", "top-p"]) $(id).disabled = $("greedy").checked;
  };
  $("greedy").addEventListener("change", syncGreedy);
  syncGreedy();

  for (const chip of document.querySelectorAll("[data-example]")) {
    chip.addEventListener("click", () => {
      $("prompt").value = chip.dataset.example;
      $("prompt").focus();
    });
  }

  $("generate").addEventListener("click", runGenerate);
  $("compare").addEventListener("click", runCompare);
  for (const [field, action] of [["prompt", runGenerate], ["compare-prompt", runCompare]]) {
    $(field).addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey || field === "compare-prompt")) {
        event.preventDefault();
        if (!state.run) action();
      }
    });
  }

  const tabs = [...document.querySelectorAll('[role="tab"]')];
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      const next = tabs[(i + step + tabs.length) % tabs.length];
      selectTab(next);
      next.focus();
    });
  });
}

async function main() {
  bindUi();
  const status = $("status");
  try {
    const defaultModel = await loadModels();
    renderModelPicker();
    selectModel(defaultModel);
    status.textContent = "Ready, runs in your browser";
    status.classList.add("ready");
    $("generate").disabled = false;
    $("compare").disabled = false;
  } catch (error) {
    status.textContent = "Could not load the models. Please refresh the page.";
    status.classList.add("error");
    console.error(error);
  }
}

main();
