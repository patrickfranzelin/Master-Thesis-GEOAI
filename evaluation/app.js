let samples = [];
let index = 0;
let results = {};

const ERROR_TYPES = [
  "SHAPE_MISMATCH",
  "MISSING_PARTS",
  "EXTRA_PARTS",
  "OVERSIMPLIFIED",
  "SHIFTED"
];

async function loadData() {
  try {
    const res = await fetch("data/samples.json");
    if (!res.ok) throw new Error(`HTTP ${res.status} – data/samples.json not found`);
    samples = await res.json();
    console.log("Loaded samples:", samples.length);
    showSample();
  } catch (err) {
    console.error("loadData failed:", err);
    document.getElementById("title").innerText = "ERROR: " + err.message;
  }
}

function getKey(sample) {
  return `${sample.building_id}_${sample.sam_id}`;
}

// -------------------------
// SAMPLE DISPLAY
// -------------------------
function showSample() {
  const sample = samples[index];
  const key = getKey(sample);

  document.getElementById("patch").src = sample.image;
  document.getElementById("title").innerText =
    `Building ${sample.building_id} (${index + 1}/${samples.length})`;

  if (!results[key]) {
    results[key] = {
      building_id: sample.building_id,
      original: null,
      original_errors: [],
      post_vs_sam: null,
      post: null,
      post_errors: []
    };
  }

  renderUI();
}

function renderUI() {
  const key = getKey(samples[index]);
  const r = results[key];

  // Restore button active states
  restoreButtons("original-group", r.original);
  restoreButtons("post-vs-sam-group", r.post_vs_sam);
  restoreButtons("post-group", r.post);

  // Restore error boxes
  renderErrorBoxIfNeeded("original", r.original);
  renderErrorBoxIfNeeded("post", r.post);
}

function restoreButtons(groupId, value) {
  const group = document.getElementById(groupId);
  if (!group) return;
  group.querySelectorAll("button").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.value === value);
  });
}

function renderErrorBoxIfNeeded(type, value) {
  const box = document.getElementById(`${type}-error-box`);

  const shouldShow =
    value === "good" ||
    value === "ok" ||
    value === "bad";

  if (shouldShow) {
    box.classList.remove("hidden");
    renderErrorBox(type);
  } else {
    box.classList.add("hidden");
    box.innerHTML = "";
  }
}
function jumpTo() {
  const input = document.getElementById("jump-input").value;

  if (!input) return;

  const idx = parseInt(input, 10) - 1; // user enters 1-based

  if (isNaN(idx) || idx < 0 || idx >= samples.length) {
    alert(`Enter a number between 1 and ${samples.length}`);
    return;
  }

  index = idx;
  showSample();
  updateProgress();
}
// -------------------------
// ANSWER HANDLING
// -------------------------
function answer(type, value, btn) {
  const key = getKey(samples[index]);
  results[key][type] = value;

  const group = btn.parentElement;
  group.querySelectorAll("button").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  if (type === "original" || type === "post") {
    renderErrorBoxIfNeeded(type, value);
  }
}

// -------------------------
// ERROR BOX
// -------------------------
function renderErrorBox(type) {
  const key = getKey(samples[index]);
  const listName = type === "original" ? "original_errors" : "post_errors";
  const selected = results[key][listName] || [];
  const box = document.getElementById(`${type}-error-box`);

    const shortcutMap = type === "original"
      ? ["Q", "W", "E", "R", "T"]
      : ["A", "S", "D", "F", "G"];

  box.innerHTML = `
    <p><b>Error types:</b></p>
    ${ERROR_TYPES.map((e, i) => `
      <label class="${selected.includes(e) ? "checked" : ""}">
        <input type="checkbox"
          ${selected.includes(e) ? "checked" : ""}
          onchange="toggleError('${type}', '${e}')">
        <kbd>${shortcutMap[i]}</kbd> ${e}
      </label>
    `).join("")}
  `;
}

function toggleError(type, error) {
  const key = getKey(samples[index]);
  const listName = type === "original" ? "original_errors" : "post_errors";

  if (!results[key][listName]) results[key][listName] = [];
  const arr = results[key][listName];

  if (arr.includes(error)) {
    results[key][listName] = arr.filter(e => e !== error);
  } else {
    arr.push(error);
  }

  renderErrorBox(type);
}

// -------------------------
// KEYBOARD CONTROL
// -------------------------
document.addEventListener("keydown", (e) => {
  // Ignore if typing in an input
  if (e.target.tagName === "INPUT") return;

  const key = e.key.toLowerCase();

  // ORIGINAL quality
  if (key === "1") clickByGroup("original-group", 0);
  if (key === "2") clickByGroup("original-group", 1);
  if (key === "3") clickByGroup("original-group", 2);
  if (key === "4") clickByGroup("original-group", 3);

  // POST VS SAM
  if (key === "n") clickByGroup("post-vs-sam-group", 0);
  if (key === "y") clickByGroup("post-vs-sam-group", 1);

  // FINAL POST quality
  if (key === "5") clickByGroup("post-group", 0);
  if (key === "6") clickByGroup("post-group", 1);
  if (key === "7") clickByGroup("post-group", 2);
  if (key === "8") clickByGroup("post-group", 3);

  // ORIGINAL ERRORS — only if error box is visible
  const origBox = document.getElementById("original-error-box");
  if (!origBox.classList.contains("hidden")) {
    if (key === "q") toggleError("original", ERROR_TYPES[0]);
    if (key === "w") toggleError("original", ERROR_TYPES[1]);
    if (key === "e") toggleError("original", ERROR_TYPES[2]);
    if (key === "r") toggleError("original", ERROR_TYPES[3]);
    if (key === "t") toggleError("original", ERROR_TYPES[4]);

  }

  // POST ERRORS — only if error box is visible
  const postBox = document.getElementById("post-error-box");
  if (!postBox.classList.contains("hidden")) {
    if (key === "a") toggleError("post", ERROR_TYPES[0]);
    if (key === "s") toggleError("post", ERROR_TYPES[1]);
    if (key === "d") toggleError("post", ERROR_TYPES[2]);
    if (key === "f") toggleError("post", ERROR_TYPES[3]);
    if (key === "g") toggleError("post", ERROR_TYPES[4]);
  }

  // NEXT
  if (key === "enter") nextSample();
});

function clickByGroup(groupId, btnIndex) {
  const group = document.getElementById(groupId);
  if (!group) return;
  const buttons = group.querySelectorAll("button");
  if (buttons[btnIndex]) buttons[btnIndex].click();
}

// -------------------------
// PROGRESS
// -------------------------
function updateProgress() {
  const bar = document.getElementById("progress-bar");
  const label = document.getElementById("progress-label");
  const pct = ((index) / samples.length) * 100;
  bar.style.width = pct + "%";
  label.innerText = `${index} / ${samples.length} done`;
}

// -------------------------
// SAVE + NEXT
// -------------------------
async function nextSample() {
  const sample = samples[index];
  const key = getKey(sample);
  const r = results[key];

  if (!r?.original || !r?.post_vs_sam || !r?.post) {
    showValidationError("Please fill in all three quality ratings before continuing.");
    return;
  }

  clearValidationError();

  await fetch("http://localhost:8001/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      building_id: sample.building_id,
      original: r.original,
      original_error: r.original_errors,
      post_vs_sam: r.post_vs_sam,
      post: r.post,
      post_error: r.post_errors,
      has_post: sample.has_post
    })
  }).catch(() => {
    // Silently continue if server unavailable (dev mode)
  });

  index++;
  updateProgress();

  if (index >= samples.length) {
    document.getElementById("app").innerHTML = `
      <div id="done-screen">
        <div class="done-icon">✓</div>
        <h2>Evaluation Complete</h2>
        <p>All ${samples.length} samples have been labelled.</p>
      </div>
    `;
    return;
  }

  showSample();
}

function showValidationError(msg) {
  const el = document.getElementById("validation-error");
  el.textContent = msg;
  el.classList.remove("hidden");
}

function clearValidationError() {
  document.getElementById("validation-error").classList.add("hidden");
}

loadData();