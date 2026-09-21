const $ = (selector) => document.querySelector(selector);
const state = { latestTraceId: null };

function showToast(message, error = false) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.style.background = error ? '#a44f3b' : '#17221f';
  toast.classList.add('show');
  window.setTimeout(() => toast.classList.remove('show'), 2800);
}

function setView(view) {
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === view));
  document.querySelectorAll('.view').forEach((section) => section.classList.toggle('active', section.id === `view-${view}`));
  const labels = { ask: ['MEMORY LAYER / QUERY', 'Ask your memory'], ingest: ['MEMORY LAYER / INGESTION', 'Ingest a source'], observability: ['MEMORY LAYER / OBSERVABILITY', 'See what Memora remembers'] };
  $('#view-kicker').textContent = labels[view][0];
  $('#view-title').textContent = labels[view][1];
  if (view === 'observability') loadMetrics();
}

document.querySelectorAll('.nav-item').forEach((item) => item.addEventListener('click', () => setView(item.dataset.view)));

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

async function checkHealth() {
  try {
    await api('/health');
    $('#health-dot').classList.remove('offline');
    $('#health-label').textContent = 'Local engine connected';
  } catch (error) {
    $('#health-dot').classList.add('offline');
    $('#health-label').textContent = 'Engine unavailable';
  }
}

function renderSources(sources) {
  $('#source-count').textContent = sources.length;
  $('#source-list').innerHTML = sources.length ? sources.map((source) => `
    <div class="source-item"><span class="source-file">${escapeHtml(source.source)} <span class="source-score">${source.score.toFixed(3)}</span></span>
    <div class="source-text">${escapeHtml(source.text)}</div></div>`).join('') : '<p class="muted">No evidence was returned for this question.</p>';
}

async function loadTrace(traceId) {
  const trace = await api(`/observability/traces/${encodeURIComponent(traceId)}`);
  $('#trace-state').textContent = 'LOADED';
  $('#trace-content').innerHTML = [['QUERY', trace.query], ['RETRIEVED', `${trace.retrieved.length} candidates`], ['RERANKED', `${trace.reranked.length} candidates`], ['CONFLICTS', `${trace.conflicts.length} detected`], ['CONTEXT', `${trace.context.length.toLocaleString()} characters`], ['TRACE ID', trace.trace_id]].map(([label, value]) => `<div class="trace-row"><b>${label}</b><span>${escapeHtml(String(value))}</span></div>`).join('');
}

$('#query-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = $('#query-input').value.trim();
  if (!query) return;
  $('#answer-state').textContent = 'WORKING';
  $('#answer-empty').classList.remove('hidden');
  $('#answer-empty').querySelector('p').textContent = 'Retrieving, reranking, and composing grounded context...';
  $('#answer-body').classList.add('hidden');
  try {
    const result = await api('/query/', { method: 'POST', body: JSON.stringify({ query }) });
    state.latestTraceId = result.trace_id;
    $('#answer-state').textContent = 'GROUNDED';
    $('#answer-empty').classList.add('hidden');
    $('#answer-body').textContent = result.answer;
    $('#answer-body').classList.remove('hidden');
    $('#answer-meta').textContent = `TRACE ${result.trace_id}  /  ${result.sources.length} SOURCES CITED`;
    $('#answer-meta').classList.remove('hidden');
    renderSources(result.sources);
    await loadTrace(result.trace_id);
  } catch (error) {
    $('#answer-state').textContent = 'ERROR';
    $('#answer-empty').querySelector('p').textContent = error.message;
    showToast(error.message, true);
  }
});

$('#ingest-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = $('#ingest-message');
  message.className = 'form-message';
  message.textContent = 'Processing source...';
  try {
    const result = await api('/ingest/', { method: 'POST', body: JSON.stringify({ path: $('#path-input').value.trim() }) });
    message.textContent = `Added ${result.chunk_count} chunks from ${result.source}`;
    showToast('Source added to memory');
  } catch (error) {
    message.className = 'form-message error';
    message.textContent = error.message;
    showToast(error.message, true);
  }
});

async function loadMetrics() {
  try {
    const metrics = await api('/observability/metrics');
    const cards = [metrics.trace_count, `${(metrics.answer_grounding_rate * 100).toFixed(0)}%`, Math.round(metrics.avg_context_tokens), metrics.unique_sources_cited];
    document.querySelectorAll('.metric-card strong').forEach((card, index) => { card.textContent = cards[index]; });
  } catch (error) { showToast(error.message, true); }
}

$('#evaluate-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const result = $('#evaluation-result');
  result.textContent = 'Evaluation running. This deliberately calls the local model...';
  try {
    const data = await api('/observability/evaluate', { method: 'POST', body: JSON.stringify({ top_k: Number($('#top-k').value), sample_size: Number($('#sample-size').value) }) });
    result.textContent = `Recall@${data.top_k}: ${(data.recall_at_k * 100).toFixed(1)}%  /  ${data.sample_size} sampled chunks`;
  } catch (error) { result.textContent = error.message; showToast(error.message, true); }
});

$('#refresh-metrics').addEventListener('click', () => { checkHealth(); loadMetrics(); showToast('Signals refreshed'); });
function escapeHtml(value) { return value.replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character])); }
checkHealth();
