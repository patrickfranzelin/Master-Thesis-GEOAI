let samples = [];
let index = 0;
let results = {};

async function loadData() {
  try {
    const res = await fetch("semantic_data/semantic_samples.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}: semantic sample file not found`);
    samples = await res.json();
    showSample();
  } catch (err) {
    document.getElementById("title").innerText = "ERROR: " + err.message;
  }
}

function getKey(sample) {
  return String(sample.building_id);
}

function showSample() {
  const sample = samples[index];
  const key = getKey(sample);

  document.getElementById("patch").src = sample.image;
  document.getElementById("sentence").innerText = sample.error_description || "";
  document.getElementById("title").innerText =
    `Building ${sample.building_id} (${index + 1}/${samples.length})`;

  if (!results[key]) {
    results[key] = {
      building_id: sample.building_id,
      sentence_quality: null
    };
  }

  restoreButtons("quality-group", results[key].sentence_quality);
  clearValidationError();
}

function restoreButtons(groupId, value) {
  const group = document.getElementById(groupId);
  group.querySelectorAll("button").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.value === value);
  });
}

function answer(field, value, btn) {
  const key = getKey(samples[index]);
  results[key][field] = value;
  btn.parentElement.querySelectorAll("button").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
}

function jumpTo() {
  const input = document.getElementById("jump-input").value;
  const idx = parseInt(input, 10) - 1;
  if (isNaN(idx) || idx < 0 || idx >= samples.length) {
    showValidationError(`Enter a number between 1 and ${samples.length}`);
    return;
  }
  index = idx;
  updateProgress();
  showSample();
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const key = e.key.toLowerCase();

  if (key === "1") clickByGroup("quality-group", 0);
  if (key === "2") clickByGroup("quality-group", 1);
  if (key === "3") clickByGroup("quality-group", 2);
  if (key === "4") clickByGroup("quality-group", 3);
  if (key === "enter") nextSample();
});

function clickByGroup(groupId, btnIndex) {
  const buttons = document.getElementById(groupId).querySelectorAll("button");
  if (buttons[btnIndex]) buttons[btnIndex].click();
}

function updateProgress() {
  const pct = samples.length ? (index / samples.length) * 100 : 0;
  document.getElementById("progress-bar").style.width = `${pct}%`;
  document.getElementById("progress-label").innerText = `${index} / ${samples.length} done`;
}

async function nextSample() {
  const sample = samples[index];
  const key = getKey(sample);
  const r = results[key];

  if (!r.sentence_quality) {
    showValidationError("Please evaluate the sentence before continuing.");
    return;
  }

  await fetch("http://localhost:8002/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(r)
  }).catch(() => {});

  index++;
  updateProgress();

  if (index >= samples.length) {
    document.getElementById("app").innerHTML = `
      <div id="done-screen">
        <div class="done-icon">OK</div>
        <h2>Evaluation Complete</h2>
        <p>All ${samples.length} semantic descriptions have been labelled.</p>
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
