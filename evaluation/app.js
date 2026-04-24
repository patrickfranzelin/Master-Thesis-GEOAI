let samples = [];
let index = 0;
let results = {};

function getKey(sample) {
  return `${sample.building_id}_${sample.sam_id}`;
}

async function loadData() {
  const res = await fetch("data/samples.json");
  samples = await res.json();
  showSample();
}

function showSample() {
  const sample = samples[index];
  const key = getKey(sample);

  document.getElementById("patch").src = sample.image;
  document.getElementById("title").innerText =
    `Building ${sample.building_id} (${index + 1}/${samples.length})`;

  if (!results[key]) {
    results[key] = {
      building_id: sample.building_id,
      tags: []
    };
  }

  resetUI();
}

function resetUI() {
  document.querySelectorAll("button").forEach(b => b.classList.remove("active"));
  document.getElementById("tag-section").style.display = "none";
}

function answer(type, value, btn) {
  const sample = samples[index];
  const key = getKey(sample);

  results[key][type] = value;

  // highlight
  const group = btn.parentElement;
  group.querySelectorAll("button").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  // 🔥 LOGIC
  if (type === "original") {
    if (value === "partial" || value === "wrong") {
      showTags();
    }
  }

  if (type === "post") {
    if (value === "ok" || value === "bad") {
      showTags();
    }
  }

  console.log("Updated:", results[key]);
}

function showTags() {
  document.getElementById("tag-section").style.display = "block";
}

function toggleTag(tag, btn) {
  const sample = samples[index];
  const key = getKey(sample);

  let tags = results[key].tags || [];

  if (tags.includes(tag)) {
    tags = tags.filter(t => t !== tag);
    btn.classList.remove("active");
  } else {
    tags.push(tag);
    btn.classList.add("active");
  }

  results[key].tags = tags;
}

async function nextSample() {
  const sample = samples[index];
  const key = getKey(sample);
  const r = results[key];

  if (!r?.original || !r?.post) {
    alert("Please evaluate Original and Post");
    return;
  }

  await fetch("http://localhost:8001/save", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      building_id: sample.building_id,
      original: r.original,
      sam: r.sam ?? null,
      post: r.post,
      tags: r.tags ?? [],
      has_post: sample.has_post
    })
  });

  index++;

  if (index >= samples.length) {
    alert("Done!");
    return;
  }

  showSample();
}

loadData();