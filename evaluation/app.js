let samples = [];
let index = 0;
let results = {};

async function loadData() {
  const res = await fetch("data/samples.json");
  samples = await res.json();
  showSample();
}

function getKey(sample) {
  return `${sample.building_id}_${sample.sam_id}`;
}

function preloadImage(src) {
  const img = new Image();
  img.src = src;
}

function showSample() {
  const sample = samples[index];

  document.getElementById("patch").src = sample.image;
  document.getElementById("title").innerText =
    `Building ${sample.building_id} (${index + 1}/${samples.length})`;

  const key = getKey(sample);

  if (!results[key]) {
    results[key] = {
      building_id: sample.building_id
    };
  }

  // preload next
  if (samples[index + 1]) preloadImage(samples[index + 1].image);
  if (samples[index + 2]) preloadImage(samples[index + 2].image);

  resetUI();
}

function resetUI() {
  document.querySelectorAll("button").forEach(b => b.classList.remove("active"));
  document.getElementById("error-section").style.display = "none";
}

function answer(type, value, btn = null) {
  const sample = samples[index];
  const key = getKey(sample);

  if (!results[key]) {
    results[key] = {
      building_id: sample.building_id
    };
  }

  results[key][type] = value;

  // highlight clicked button
  if (btn) {
    const group = btn.parentElement;
    const buttons = group.querySelectorAll("button");
    buttons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  }

  // show error dropdown if BAD
  if (type === "quality") {
    const errorSection = document.getElementById("error-section");
    errorSection.style.display = value === "bad" ? "block" : "none";
  }

  console.log("Updated:", results[key]);
}

async function nextSample() {
  const sample = samples[index];
  const key = getKey(sample);
  const r = results[key];

  // optional guard
  if (!r?.quality) {
    alert("Please rate quality first");
    return;
  }

  await fetch("http://localhost:8001/save", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      building_id: sample.building_id,
      quality: r?.quality ?? null,
      postprocessing: r?.postprocessing ?? null,
      best: r?.best ?? null,
      error_type: r?.error_type ?? null,
      has_post: sample.has_post
    })
  });

  console.log("Saved:", r);

  index++;

  if (index >= samples.length) {
    alert("Done!");
    return;
  }

  showSample();
}

loadData();