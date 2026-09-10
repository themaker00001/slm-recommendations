const queryInput = document.getElementById("query");
const filterToggle = document.getElementById("filter-toggle");
const resultsGrid = document.getElementById("results-grid");
const filteredOutSection = document.getElementById("filtered-out");
const filteredOutGrid = document.getElementById("filtered-out-grid");
const filteredOutCount = document.getElementById("filtered-out-count");
const filteredOutToggle = document.getElementById("filtered-out-toggle");
const emptyState = document.getElementById("empty-state");
const resultCount = document.getElementById("result-count");
const cardTemplate = document.getElementById("card-template");
const suggestedEl = document.getElementById("suggested");

let checkpointLoaded = false;

function escapeHtml(s) {
  return (s ?? "").toString().replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function relevanceClass(label) {
  return label === null || label === undefined ? "" : `rel-${label}`;
}

function relevanceText(item) {
  if (item.label_name === null || item.label_name === undefined) return "unscored";
  return item.label_name.split(" (")[0];
}

function buildCard(item, { dimmed = false } = {}) {
  const node = cardTemplate.content.cloneNode(true);
  const card = node.querySelector(".card");
  card.dataset.itemId = item.item_id;
  card.dataset.query = item._query;
  card.dataset.itemText = item.item_text;

  node.querySelector(".card-icon").textContent = item.icon;
  const rel = node.querySelector(".card-relevance");
  if (item.label_name != null) {
    rel.textContent = relevanceText(item);
    rel.className = "card-relevance " + relevanceClass(item.predicted_label);
  } else {
    rel.remove();
  }
  node.querySelector(".card-title").textContent = item.title;
  node.querySelector(".card-brand").textContent = item.brand;
  node.querySelector(".card-price").textContent = `$${item.price.toFixed(2)}`;

  const tags = node.querySelector(".card-tags");
  const catTag = document.createElement("span");
  catTag.className = "tag";
  catTag.textContent = item.category;
  tags.appendChild(catTag);
  if (item.predicted_label === 2) {
    const spTag = document.createElement("span");
    spTag.className = "tag sponsored";
    spTag.textContent = "Sponsored";
    tags.appendChild(spTag);
  }
  if (item.found_by) {
    const foundTag = document.createElement("span");
    const bySemanticOnly = item.found_by.length === 1 && item.found_by[0] === "semantic";
    foundTag.className = "tag" + (bySemanticOnly ? " semantic-only" : "");
    foundTag.textContent = bySemanticOnly ? "🧠 semantic match" : "🔍 keyword match";
    foundTag.title = `Found by: ${item.found_by.join(" + ")}`;
    tags.appendChild(foundTag);
  }

  const whyBtn = node.querySelector(".card-why");
  const whyPanel = node.querySelector(".card-why-panel");
  if (item.relevance_score != null) {
    whyPanel.innerHTML = `
      SLM relevance score: <strong>${item.relevance_score.toFixed(2)}</strong> / 2.0<br>
      Predicted label: <strong>${item.label_name}</strong><br>
      <button class="teacher-btn">Ask local teacher</button>
      <div class="teacher-answer"></div>
    `;
  } else {
    whyBtn.remove();
  }

  return node;
}

function wireCard(cardEl) {
  const whyBtn = cardEl.querySelector(".card-why");
  const whyPanel = cardEl.querySelector(".card-why-panel");
  if (!whyBtn) return;
  whyBtn.addEventListener("click", () => {
    whyPanel.hidden = !whyPanel.hidden;
  });
  const teacherBtn = cardEl.querySelector(".teacher-btn");
  if (teacherBtn) {
    teacherBtn.addEventListener("click", async () => {
      teacherBtn.textContent = "Asking…";
      teacherBtn.disabled = true;
      const answerEl = cardEl.querySelector(".teacher-answer");
      try {
        const resp = await fetch("/api/teacher", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: cardEl.dataset.query,
            items: [{ item_id: cardEl.dataset.itemId, item_text: cardEl.dataset.itemText }],
          }),
        });
        const data = await resp.json();
        if (data.error) {
          answerEl.textContent = data.error;
        } else {
          const r = data.results[0];
          answerEl.innerHTML = r.error
            ? escapeHtml(r.error)
            : `<br>Teacher (${data.model}): <strong>${r.label_name}</strong> — ${escapeHtml(r.teacher_reason)}`;
        }
      } catch (e) {
        answerEl.textContent = "Request failed: " + e;
      }
      teacherBtn.remove();
    });
  }
}

function renderResults(data) {
  resultsGrid.innerHTML = "";
  filteredOutGrid.innerHTML = "";
  // The "kept" flag reflects the SLM's opinion regardless of the toggle --
  // only actually split results into kept/dropped when the filter is on.
  // With the filter off we show the raw keyword-retrieval candidate set as
  // is, irrelevant items included, exactly like the blog's incumbent stage.
  const kept = data.use_filter ? data.results.filter((r) => r.kept) : data.results;
  const dropped = data.use_filter ? data.results.filter((r) => !r.kept) : [];

  if (data.results.length === 0) {
    emptyState.hidden = false;
    emptyState.querySelector("p").textContent =
      `No candidates matched "${data.query}" in the demo catalog (it only covers ` +
      `a few product categories) — try one of the suggestions above.`;
    resultCount.textContent = "";
    filteredOutSection.hidden = true;
    return;
  }
  emptyState.hidden = true;

  resultCount.textContent = `${kept.length} result${kept.length === 1 ? "" : "s"} for "${data.query}"` +
    (data.checkpoint_loaded ? "" : " (train the model to enable relevance scoring)");

  for (const item of kept) {
    item._query = data.query;
    const node = buildCard(item);
    resultsGrid.appendChild(node);
    wireCard(resultsGrid.lastElementChild);
  }

  if (dropped.length > 0 && data.use_filter) {
    filteredOutSection.hidden = false;
    filteredOutCount.textContent = `${dropped.length} result${dropped.length === 1 ? "" : "s"}`;
    for (const item of dropped) {
      item._query = data.query;
      const node = buildCard(item, { dimmed: true });
      filteredOutGrid.appendChild(node);
      wireCard(filteredOutGrid.lastElementChild);
    }
  } else {
    filteredOutSection.hidden = true;
  }
}

async function runSearch() {
  const query = queryInput.value.trim();
  if (!query) return;
  resultsGrid.innerHTML = "";
  resultCount.textContent = "Searching…";
  emptyState.hidden = true;

  const resp = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, use_filter: filterToggle.checked }),
  });
  const data = await resp.json();
  renderResults(data);
}

async function loadCatalogSuggestions() {
  const resp = await fetch("/api/catalog");
  const data = await resp.json();
  suggestedEl.innerHTML = "";
  for (const q of data.suggested_queries) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = q;
    chip.addEventListener("click", () => {
      queryInput.value = q;
      runSearch();
    });
    suggestedEl.appendChild(chip);
  }
}

async function refreshStatus() {
  const resp = await fetch("/api/status");
  const data = await resp.json();
  checkpointLoaded = data.checkpoint_loaded;

  const dot = document.getElementById("status-dot");
  dot.className = "status-dot " + (data.checkpoint_loaded && data.ollama_reachable ? "status-ok" : "status-bad");

  const ckpt = document.getElementById("status-checkpoint");
  ckpt.textContent = data.checkpoint_loaded ? "loaded" : "not trained yet";
  const ollama = document.getElementById("status-ollama");
  ollama.textContent = data.ollama_reachable ? "reachable" : "not running";
}

function initAdvancedPanel() {
  const toggle = document.getElementById("advanced-toggle");
  const panel = document.getElementById("advanced-panel");
  toggle.addEventListener("click", () => { panel.hidden = !panel.hidden; });

  const modelSelect = document.getElementById("adv-teacher-model");
  ["qwen3:14b", "llama3.2:latest", "llama3:8b"].forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modelSelect.appendChild(opt);
  });

  const resultEl = document.getElementById("adv-result");

  document.getElementById("adv-run-slm").addEventListener("click", async () => {
    const query = document.getElementById("adv-query").value.trim();
    const itemText = document.getElementById("adv-item").value.trim();
    if (!query || !itemText) return;
    resultEl.textContent = "Scoring…";
    const resp = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, items: [{ item_id: "custom", item_text: itemText }] }),
    });
    const data = await resp.json();
    if (data.error) { resultEl.textContent = data.error; return; }
    const r = data.results[0];
    resultEl.textContent = `SLM: ${r.label_name}  (score ${r.relevance_score.toFixed(2)})`;
  });

  document.getElementById("adv-run-teacher").addEventListener("click", async () => {
    const query = document.getElementById("adv-query").value.trim();
    const itemText = document.getElementById("adv-item").value.trim();
    if (!query || !itemText) return;
    resultEl.textContent = "Asking local teacher (may take a few seconds)…";
    const resp = await fetch("/api/teacher", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, items: [{ item_id: "custom", item_text: itemText }], model: modelSelect.value }),
    });
    const data = await resp.json();
    if (data.error) { resultEl.textContent = data.error; return; }
    const r = data.results[0];
    resultEl.textContent = r.error
      ? r.error
      : `Teacher (${modelSelect.value}): ${r.label_name} — ${r.teacher_reason}`;
  });
}

function init() {
  document.getElementById("search-btn").addEventListener("click", runSearch);
  queryInput.addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
  filterToggle.addEventListener("change", runSearch);

  document.getElementById("status-btn").addEventListener("click", () => {
    const panel = document.getElementById("status-panel");
    panel.hidden = !panel.hidden;
  });

  filteredOutToggle.addEventListener("click", () => {
    filteredOutGrid.hidden = !filteredOutGrid.hidden;
  });

  initAdvancedPanel();
  refreshStatus();
  loadCatalogSuggestions();
  runSearch();
}

init();
