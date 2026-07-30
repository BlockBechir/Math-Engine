"use strict";

const MIN_DIM = 1;
const MAX_DIM = 12;

const state = {
  pyodide: null,
  ready: false,
  matrices: [],   // { id, name, rows, cols, values, activeOp, node }
  lines: [],      // { id }
  nextId: 1,
};

function uid() { return state.nextId++; }

function debounce(fn, wait) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

function emptyGrid(rows, cols, fill = "0") {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => fill));
}

function resizeGrid(values, rows, cols) {
  const out = emptyGrid(rows, cols, "0");
  for (let r = 0; r < Math.min(rows, values.length); r++) {
    for (let c = 0; c < Math.min(cols, values[r].length); c++) {
      out[r][c] = values[r][c];
    }
  }
  return out;
}

function nextFreeName() {
  const used = new Set(state.matrices.map(m => m.name));
  for (let code = 65; code <= 90; code++) {
    const letter = String.fromCharCode(code);
    if (!used.has(letter)) return letter;
  }
  let n = 1;
  while (used.has("M" + n)) n++;
  return "M" + n;
}

const RESERVED = new Set([
  "Identity", "I", "x", "i", "j", "det", "rank", "inv", "transpose", "trace", "rref",
  "kernel", "image", "charpoly", "sqrt", "cbrt", "nthrt", "Integer", "Rational", "Rationalize",
  "Matrix", "Vector", "Constant", "pi", "e", "and", "or", "not", "ans",
]);

function isValidName(name, ownerId) {
  if (!/^[A-Za-z_][A-Za-z_0-9]*$/.test(name)) return false;
  if (RESERVED.has(name)) return false;
  return !state.matrices.some(m => m.id !== ownerId && m.name === name);
}

const debouncedRunNotebook = debounce(() => runNotebook(), 350);

/* =========================================================
   MATRIX BLOCKS
   ========================================================= */
const matrixList = document.getElementById("matrixList");
const matrixTpl = document.getElementById("matrixBlockTemplate");

function addMatrix(preset) {
  const m = {
    id: uid(),
    name: nextFreeName(),
    rows: preset ? preset.length : 3,
    cols: preset ? preset[0].length : 3,
    values: preset || [["1", "0", "2"], ["0", "3", "1"], ["2", "1", "0"]],
    activeOp: null,
  };
  state.matrices.push(m);
  const node = matrixTpl.content.firstElementChild.cloneNode(true);
  m.node = node;
  node.dataset.id = m.id;
  matrixList.appendChild(node);
  wireMatrixBlock(m, node);
  renderMatrixGrid(m, node);
  return m;
}

function wireMatrixBlock(m, node) {
  const nameInput = node.querySelector(".matrix-name");
  nameInput.value = m.name;
  nameInput.addEventListener("input", () => {
    const candidate = nameInput.value.trim();
    if (candidate && isValidName(candidate, m.id)) {
      m.name = candidate;
      renderMatrixGrid(m, node);
      debouncedRunNotebook();
    }
  });

  node.querySelectorAll(".stepper").forEach(btn => {
    btn.addEventListener("click", () => {
      const [axis, dir] = btn.dataset.action.split("-");
      const delta = dir === "inc" ? 1 : -1;
      const next = Math.max(MIN_DIM, Math.min(MAX_DIM, m[axis] + delta));
      if (next === m[axis]) return;
      m[axis] = next;
      m.values = resizeGrid(m.values, m.rows, m.cols);
      renderMatrixGrid(m, node);
      replayActiveOp(m, node);
      debouncedRunNotebook();
    });
  });

  node.querySelector(".remove-btn").addEventListener("click", () => {
    state.matrices = state.matrices.filter(x => x.id !== m.id);
    node.remove();
    debouncedRunNotebook();
  });

  node.querySelectorAll(".ops-row [data-op]").forEach(btn => {
    btn.addEventListener("click", () => runMatrixOp(m, node, btn.dataset.op, btn));
  });

  node.querySelector(".power-input").addEventListener("input", () => {
    if (m.activeOp === "power") replayActiveOp(m, node);
  });
}

function renderMatrixGrid(m, node) {
  const grid = node.querySelector(".matrix-grid");
  grid.style.gridTemplateColumns = `repeat(${m.cols}, minmax(0, 1fr))`;
  grid.innerHTML = "";
  for (let r = 0; r < m.rows; r++) {
    for (let c = 0; c < m.cols; c++) {
      const input = document.createElement("input");
      input.type = "text";
      input.className = "cell-input";
      input.autocomplete = "off";
      input.spellcheck = false;
      input.value = m.values[r][c];
      input.setAttribute("aria-label", `Matrix ${m.name} row ${r + 1} column ${c + 1}`);
      input.addEventListener("input", (e) => {
        m.values[r][c] = e.target.value;
        replayActiveOp(m, node);
        debouncedRunNotebook();
      });
      grid.appendChild(input);
    }
  }
  node.querySelectorAll('[data-dim="rows"]').forEach(el => el.textContent = m.rows);
  node.querySelectorAll('[data-dim="cols"]').forEach(el => el.textContent = m.cols);
  node.querySelectorAll('[data-dim-label="rows"]').forEach(el => el.textContent = m.rows);
  node.querySelectorAll('[data-dim-label="cols"]').forEach(el => el.textContent = m.cols);
}

const OP_LABELS = {
  transpose: "transpose", det: "det", rank: "rank", inv: "inverse", rref: "rref",
  kernel: "kernel basis", image: "image basis", trace: "trace",
  charpoly: "characteristic polynomial", power: "power", properties: "properties",
};

const debouncedReplay = debounce((m, node) => computeMatrixOp(m, node), 200);

function replayActiveOp(m, node) {
  if (!m.activeOp || !state.ready) return;
  debouncedReplay(m, node);
}

function runMatrixOp(m, node, op, btnEl) {
  if (!state.ready) return;
  node.querySelectorAll(".op-btn").forEach(b => b.classList.remove("is-active"));
  btnEl?.closest(".op-btn")?.classList.add("is-active");
  m.activeOp = op;
  computeMatrixOp(m, node);
}

function computeMatrixOp(m, node) {
  const op = m.activeOp;
  if (!op || !state.ready) return;
  const outWrap = node.querySelector(".block-output");
  const outLabel = node.querySelector(".block-output__label");
  const outMath = node.querySelector(".block-output__math");
  outWrap.hidden = false;
  outLabel.textContent = `${m.name} — ${OP_LABELS[op] || op}`;
  outMath.classList.remove("is-error");

  try {
    const aJson = JSON.stringify(m.values);
    const k = node.querySelector(".power-input").value || "0";
    const raw = state.pyodide.globals.get("js_compute")(op, aJson, k);
    const result = JSON.parse(raw);
    renderBlockResult(result, outMath);
  } catch (err) {
    console.error(err);
    outMath.classList.add("is-error");
    outMath.textContent = String(err);
  }
}

function renderBlockResult(result, container) {
  container.innerHTML = "";
  if (!result.ok) {
    container.classList.add("is-error");
    container.textContent = result.error || "Something went wrong.";
    return;
  }
  if (result.type === "properties") {
    const LABELS = { square: "square", symmetric: "symmetric", antisymmetric: "antisymmetric", triangular: "triangular", diagonal: "diagonal" };
    const chips = document.createElement("div");
    chips.className = "chips";
    Object.entries(result.value).forEach(([k, v]) => {
      const chip = document.createElement("span");
      chip.className = "prop-chip" + (v ? " is-true" : "");
      chip.textContent = `${LABELS[k] || k}: ${v ? "yes" : "no"}`;
      chips.appendChild(chip);
    });
    container.appendChild(chips);
  } else if (result.type === "vectors") {
    if (result.vectors.length === 0) {
      container.textContent = "Only the zero vector — trivial basis.";
      return;
    }
    const wrap = document.createElement("div");
    wrap.className = "vector-basis";
    result.vectors.forEach((latex, i) => {
      const item = document.createElement("div");
      item.className = "vector-basis__item";
      const cap = document.createElement("span");
      cap.className = "vector-basis__caption";
      cap.textContent = `v${i + 1}`;
      const math = document.createElement("div");
      renderKatex(latex, math, true);
      item.appendChild(cap);
      item.appendChild(math);
      wrap.appendChild(item);
    });
    container.appendChild(wrap);
  } else if (result.type === "matrix") {
    renderKatex(result.latex, container, true);
  } else {
    renderKatex(result.latex, container, false);
  }
}

function renderKatex(latex, el, displayMode) {
  try {
    katex.render(latex, el, { throwOnError: false, displayMode: !!displayMode });
  } catch (err) {
    el.textContent = latex;
  }
}

document.getElementById("addMatrixBtn").addEventListener("click", () => addMatrix());

/* =========================================================
   NOTEBOOK (MathLive math-field inputs)
   ========================================================= */
const notebookLinesEl = document.getElementById("notebookLines");
const lineTpl = document.getElementById("notebookLineTemplate");

function addLine(latex) {
  const line = { id: uid() };
  state.lines.push(line);
  const node = lineTpl.content.firstElementChild.cloneNode(true);
  node.dataset.id = line.id;
  notebookLinesEl.appendChild(node);
  const mf = node.querySelector(".nb-line__input");
  if (latex) mf.value = latex;
  wireLine(line, node, mf);
  return node;
}

function wireLine(line, node, mf) {
  mf.addEventListener("input", () => debouncedRunNotebook());
  mf.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      runNotebook();
    }
  });
  node.querySelector(".nb-line__remove").addEventListener("click", () => {
    state.lines = state.lines.filter(x => x.id !== line.id);
    node.remove();
    debouncedRunNotebook();
  });
}

document.getElementById("addLineBtn").addEventListener("click", () => {
  const node = addLine("");
  node.querySelector(".nb-line__input").focus();
});
document.getElementById("runAllBtn").addEventListener("click", runNotebook);

function runNotebook() {
  if (!state.ready) return;
  const matricesObj = {};
  state.matrices.forEach(m => { matricesObj[m.name] = m.values; });

  const nodes = [...notebookLinesEl.querySelectorAll(".nb-line")];
  const lineTexts = nodes.map(node => {
    const mf = node.querySelector(".nb-line__input");
    try {
      return mf.getValue ? mf.getValue("ascii-math") : "";
    } catch (err) {
      return "";
    }
  });

  let raw;
  try {
    raw = state.pyodide.globals.get("js_run_notebook")(
      JSON.stringify(matricesObj), JSON.stringify(lineTexts)
    );
  } catch (err) {
    console.error(err);
    return;
  }
  const results = JSON.parse(raw);
  results.forEach((res, i) => {
    const node = nodes[i];
    if (!node) return;
    const out = node.querySelector(".nb-line__output");
    out.classList.remove("is-error");
    out.innerHTML = "";
    if (res.empty || !lineTexts[i]) return;
    if (!res.ok) {
      out.classList.add("is-error");
      out.textContent = res.error || "Error";
      return;
    }
    if (res.name) {
      const prefix = document.createElement("span");
      prefix.textContent = `${res.name} = `;
      prefix.style.color = "var(--text-dim)";
      out.appendChild(prefix);
    }
    const isBlock = /\\begin\{bmatrix\}/.test(res.latex);
    const math = document.createElement("span");
    renderKatex(res.latex, math, isBlock);
    out.appendChild(math);
  });
}

/* =========================================================
   PYODIDE BOOTSTRAP
   ========================================================= */
const statusEl = document.getElementById("engineStatus");
const notebookPanel = document.getElementById("notebookPanel");

function setStatus(mode, text) {
  statusEl.className = `status status--${mode}`;
  statusEl.innerHTML = `<span class="status__dot"></span>${text}`;
}

async function boot() {
  try {
    setStatus("loading", "booting runtime");
    const pyodide = await loadPyodide();
    setStatus("loading", "loading engine");
    const [engineSrc, bridgeSrc] = await Promise.all([
      fetch("engine.py").then(r => r.text()),
      fetch("bridge.py").then(r => r.text()),
    ]);
    await pyodide.runPythonAsync(engineSrc);
    await pyodide.runPythonAsync(bridgeSrc);
    state.pyodide = pyodide;
    state.ready = true;
    setStatus("ready", "ready");
    notebookPanel.setAttribute("aria-busy", "false");
    runNotebook();
  } catch (err) {
    console.error(err);
    setStatus("error", "failed to load — see console. Are you serving this over http:// (not file://)?");
  }
}
boot();

/* =========================================================
   INITIAL CONTENT
   ========================================================= */
addMatrix([["1", "2"], ["3", "4"]]);
addMatrix([["5", "6"], ["7", "8"]]);
addLine("P=x^2+1");
addLine("Q=x+1");
addLine("\\frac{PQ}{P+Q}");
addLine("A^2+B+Identity");
