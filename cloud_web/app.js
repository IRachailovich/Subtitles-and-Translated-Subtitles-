const state = {
  apiBaseUrl: String(window.SUBGEN_RUNTIME_CONFIG?.apiBaseUrl || '').replace(/\/$/, ''),
  publicConfig: null,
  token: localStorage.getItem('subgen_access_token') || '',
  refreshToken: localStorage.getItem('subgen_refresh_token') || '',
  devUser: localStorage.getItem('subgen_dev_user') || '',
  user: null,
  settings: { revision: 0, config: {} },
  videoId: null,
  currentReviewJob: null,
  currentReviewStatus: null,
  currentReview: null,
  jobs: [],
  videos: [],
};

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

function message(element, text, kind = '') {
  element.textContent = text || '';
  element.className = `message ${kind}`.trim();
}

function authHeaders(json = true) {
  const headers = {};
  if (json) headers['Content-Type'] = 'application/json';
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (state.publicConfig?.development_auth && state.devUser) headers['X-SubGen-Dev-User'] = state.devUser;
  return headers;
}

async function refreshAccessToken() {
  if (!state.refreshToken || !state.publicConfig?.auth_url) return false;
  const response = await fetch(`${state.publicConfig.auth_url}/auth/v1/token?grant_type=refresh_token`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json', apikey: state.publicConfig.auth_public_key},
    body: JSON.stringify({refresh_token: state.refreshToken}),
  });
  if (!response.ok) return false;
  const session = await response.json();
  saveSession(session);
  return true;
}

async function api(path, options = {}, retry = true) {
  const response = await fetch(`${state.apiBaseUrl}${path}`, {...options, headers: {...authHeaders(options.body !== undefined), ...(options.headers || {})}});
  if (response.status === 401 && retry && await refreshAccessToken()) return api(path, options, false);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

function saveSession(session) {
  state.token = session.access_token || '';
  state.refreshToken = session.refresh_token || state.refreshToken;
  localStorage.setItem('subgen_access_token', state.token);
  if (state.refreshToken) localStorage.setItem('subgen_refresh_token', state.refreshToken);
}

function clearSession() {
  state.token = ''; state.refreshToken = ''; state.devUser = ''; state.user = null;
  ['subgen_access_token', 'subgen_refresh_token', 'subgen_dev_user'].forEach(key => localStorage.removeItem(key));
}

function readAuthCallback() {
  const values = new URLSearchParams(location.hash.slice(1));
  if (values.get('access_token')) {
    saveSession({access_token: values.get('access_token'), refresh_token: values.get('refresh_token')});
    history.replaceState({}, '', location.pathname);
  }
}

async function login(email) {
  if (state.publicConfig.development_auth) {
    state.devUser = email;
    localStorage.setItem('subgen_dev_user', email);
    await openWorkspace();
    return;
  }
  if (!state.publicConfig.auth_url || !state.publicConfig.auth_public_key) throw new Error('Authentication is not configured on this deployment.');
  const response = await fetch(`${state.publicConfig.auth_url}/auth/v1/otp`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json', apikey: state.publicConfig.auth_public_key},
    body: JSON.stringify({email, options: {emailRedirectTo: location.origin}}),
  });
  if (!response.ok) throw new Error('The sign-in service could not send the email.');
  message($('#login-message'), 'Check your email and open the secure sign-in link.', 'success');
}

async function openWorkspace() {
  state.user = await api('/v1/me');
  $('#account-email').textContent = state.user.email || state.user.subject;
  $('#auth-view').hidden = true;
  $('#app-view').hidden = false;
  await Promise.all([loadSettings(), loadCredentials(), loadJobs(), loadVideos()]);
}

const viewMeta = {
  create: ['Create subtitles', 'Upload a video and choose its target language.'],
  jobs: ['Jobs & review', 'Progress, subtitle review, and completed downloads.'],
  settings: ['API settings', 'Encrypted credentials and account-wide pipeline defaults.'],
};

function showView(name) {
  $$('.nav-item').forEach(button => button.classList.toggle('active', button.dataset.view === name));
  $$('.view').forEach(view => view.classList.toggle('active', view.id === `view-${name}`));
  [$('#view-title').textContent, $('#view-subtitle').textContent] = viewMeta[name];
  $('.sidebar').classList.remove('open');
  if (name === 'jobs') loadJobs();
}

async function loadSettings() {
  state.settings = await api('/v1/settings');
  const config = state.settings.config || {};
  $('#default-transcription').value = config.transcription_provider || 'google';
  $('#default-timing').value = config.timing_anchor_provider || 'openai';
  $('#default-translation').value = config.translation_provider || 'openai';
}

async function saveSettings() {
  const config = {
    ...state.settings.config,
    transcription_provider: $('#default-transcription').value,
    timing_anchor_provider: $('#default-timing').value,
    translation_provider: $('#default-translation').value,
  };
  state.settings = await api('/v1/settings', {method: 'PUT', body: JSON.stringify({config, revision: state.settings.revision})});
  message($('#settings-message'), 'Defaults saved and available on every signed-in device.', 'success');
}

async function loadCredentials() {
  const credentials = await api('/v1/credentials');
  $('#credential-list').innerHTML = credentials.length ? credentials.map(item => `
    <div class="credential"><span>${escapeHtml(item.provider)} · ${escapeHtml(item.profile)}</span><strong>Configured</strong></div>
  `).join('') : '<p class="message">No cloud credentials saved yet.</p>';
}

async function loadVideos() {
  state.videos = await api('/v1/videos');
  const select = $('#existing-video');
  select.innerHTML = '<option value="">Choose an existing video…</option>' + state.videos.map(video =>
    `<option value="${video.id}">${escapeHtml(video.name)} · ${prettyBytes(video.size_bytes)}</option>`
  ).join('');
}

async function saveCredential() {
  const payload = {provider: $('#credential-provider').value, profile: 'default', api_key: $('#credential-key').value};
  await api('/v1/credentials', {method: 'PUT', body: JSON.stringify(payload)});
  $('#credential-key').value = '';
  message($('#credential-message'), 'Key encrypted and saved. It will never be displayed again.', 'success');
  await loadCredentials();
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

function prettyBytes(bytes) {
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

async function uploadFile(file) {
  const progressWrap = $('#upload-progress-wrap');
  progressWrap.hidden = false;
  $('#upload-progress').style.width = '0%';
  $('#upload-status').textContent = 'Creating resumable upload…';
  const initiated = await api('/v1/uploads/initiate', {method: 'POST', body: JSON.stringify({filename: file.name, size_bytes: file.size})});
  if (initiated.reused) return initiated.video_id;
  const parts = [];
  for (let partNumber = 1; partNumber <= initiated.part_count; partNumber += 1) {
    const start = (partNumber - 1) * initiated.part_size;
    const end = Math.min(file.size, start + initiated.part_size);
    const signed = await api(`/v1/uploads/${initiated.upload_id}/part-url`, {method: 'POST', body: JSON.stringify({part_number: partNumber})});
    const sameOrigin = new URL(signed.url, location.href).origin === location.origin;
    const response = await fetch(signed.url, {
      method: 'PUT',
      headers: sameOrigin ? authHeaders(false) : {},
      body: file.slice(start, end),
    });
    if (!response.ok) throw new Error(`Upload part ${partNumber} failed.`);
    const etag = response.headers.get('ETag');
    if (!etag) throw new Error('Object storage did not expose the ETag header. Configure bucket CORS to expose ETag.');
    parts.push({part_number: partNumber, etag});
    const percent = Math.round((partNumber / initiated.part_count) * 100);
    $('#upload-progress').style.width = `${percent}%`;
    $('#upload-status').textContent = `Uploaded ${percent}%`;
  }
  await api(`/v1/uploads/${initiated.upload_id}/complete`, {method: 'POST', body: JSON.stringify({parts})});
  $('#upload-status').textContent = 'Upload complete';
  return initiated.video_id;
}

async function createJob() {
  const file = $('#video-file').files[0];
  const existingVideoId = $('#existing-video').value;
  if (!file && !existingVideoId) throw new Error('Choose a new or existing video first.');
  if (file && file.size > state.publicConfig.max_upload_bytes) throw new Error('This file exceeds the deployment upload limit.');
  const button = $('#start-button');
  button.disabled = true;
  const videoId = file ? await uploadFile(file) : existingVideoId;
  const pipelineConfig = {
    ...state.settings.config,
    tiktok_style: $('#tiktok-style').checked,
  };
  const job = await api('/v1/jobs', {method: 'POST', body: JSON.stringify({
    video_id: videoId,
    target_language: $('#target-language').value.trim() || null,
    pipeline_config: pipelineConfig,
    style_config: {},
    cache_action: $('#cache-action').value,
  })});
  message($('#create-message'), `Job ${job.id.slice(0, 8)} queued. You may close this browser.`, 'success');
  if (job.reused) message($('#create-message'), `Compatible completed output reused from job ${job.id.slice(0, 8)}.`, 'success');
  await loadVideos();
  showView('jobs');
  button.disabled = false;
}

function renderJobs() {
  $('#job-count').textContent = state.jobs.filter(job => !['completed', 'failed', 'cancelled'].includes(job.status)).length;
  $('#jobs-list').innerHTML = state.jobs.length ? state.jobs.map(job => `
    <article class="job-row" data-job-id="${job.id}">
      <div><strong>${escapeHtml(job.target_language || 'Original language')}</strong><small>${escapeHtml(job.id.slice(0, 8))} · ${new Date(job.created_at).toLocaleString()}</small></div>
      <span class="job-status">${escapeHtml(job.status.replaceAll('_', ' '))}</span>
      <span>${job.progress}%</span>
      <div><button class="secondary job-open">${['waiting_for_review','needs_attention','in_review','stale_after_edit','approved','burn_failed'].includes(job.status) ? 'Review' : 'Details'}</button></div>
    </article>
  `).join('') : '<p class="message">No jobs yet.</p>';
  $$('.job-open').forEach(button => button.addEventListener('click', () => openJob(button.closest('.job-row').dataset.jobId)));
}

async function loadJobs() {
  if (!state.user) return;
  state.jobs = await api('/v1/jobs');
  renderJobs();
}

async function openJob(jobId) {
  const job = await api(`/v1/jobs/${jobId}`);
  if (['waiting_for_review','needs_attention','in_review','stale_after_edit','approved','burn_failed'].includes(job.status)) return openReview(jobId);
  if (job.status === 'completed' && job.artifacts.length) {
    const output = job.artifacts.find(item => item.kind === 'output_video');
    if (output) await downloadArtifact(output.id);
    return;
  }
  if (job.error_message) alert(job.error_message);
}

async function openReview(jobId) {
  const review = await api(`/v1/jobs/${jobId}/review`);
  state.currentReviewJob = jobId;
  state.currentReviewStatus = review.status;
  state.currentReview = review.review;
  $('#review-list').innerHTML = review.segments.map((segment, index) => `
    <div class="cue" data-index="${index}" data-cue-id="${escapeHtml(segment.id || '')}">
      <span class="cue-index">${index + 1}</span>
      <input class="cue-select" type="checkbox" aria-label="Select cue ${index + 1} for retranslation">
      <input class="cue-start" value="${escapeHtml(segment.start)}" aria-label="Cue ${index + 1} start">
      <input class="cue-end" value="${escapeHtml(segment.end)}" aria-label="Cue ${index + 1} end">
      <label class="cue-field">Source<textarea class="cue-text" aria-label="Cue ${index + 1} source text">${escapeHtml(segment.text)}</textarea></label>
      <label class="cue-field">Translation<textarea class="cue-translation" aria-label="Cue ${index + 1} translation">${escapeHtml(segment.translation || '')}</textarea></label>
    </div>
  `).join('');
  $('#review-issues').innerHTML = (review.issues || []).length ? (review.issues || []).map(issue => `
    <div class="review-issue ${escapeHtml(issue.severity || 'warning')}" data-start="${escapeHtml(issue.start_seconds ?? '')}" data-cue-id="${escapeHtml((issue.affected_cue_ids || [])[0] || '')}">
      <button type="button" class="issue-jump">${escapeHtml(issue.severity || 'warning')}: ${escapeHtml(issue.message || issue.code)}</button>
      ${issue.status === 'unresolved' ? `<button type="button" class="issue-resolve" data-issue-id="${escapeHtml(issue.id)}" data-severity="${escapeHtml(issue.severity || 'warning')}">${issue.blocking ? 'Mark corrected' : 'Accept warning'}</button>` : `<span>${escapeHtml(issue.status)}</span>`}
    </div>`).join('') : '<p class="message">No review issues.</p>';
  $$('.review-issue .issue-jump').forEach(button => button.addEventListener('click', () => {
    const issue = button.closest('.review-issue');
    const cueId = issue.dataset.cueId;
    const cue = cueId ? document.querySelector(`.cue[data-cue-id="${CSS.escape(cueId)}"]`) : null;
    (cue || $('#review-list')).scrollIntoView({behavior: 'smooth', block: 'center'});
    cue?.querySelector('textarea')?.focus();
  }));
  $$('.review-issue .issue-resolve').forEach(button => button.addEventListener('click', () => resolveCloudIssue(button.dataset.issueId, button.dataset.severity).catch(error => message($('#review-message'), error.message, 'error'))));
  $('#approve-button').hidden = review.status === 'approved';
  $('#save-review-button').hidden = review.status === 'approved';
  $('#burn-button').hidden = review.status !== 'approved' && review.status !== 'burn_failed';
  $('#review-panel').hidden = false;
  $('#review-panel').scrollIntoView({behavior: 'smooth'});
}

function collectReview() {
  return $$('.cue').map((cue, index) => ({
    id: cue.dataset.cueId || undefined,
    index: index + 1,
    start: cue.querySelector('.cue-start').value.trim(),
    end: cue.querySelector('.cue-end').value.trim(),
    text: cue.querySelector('.cue-text').value,
    translation: cue.querySelector('.cue-translation').value,
  }));
}

async function saveReview() {
  const segments = collectReview();
  await api(`/v1/jobs/${state.currentReviewJob}/review`, {method: 'PUT', body: JSON.stringify({segments, translation_confirmed: true})});
  message($('#review-message'), 'Draft saved. Any previous approval was invalidated.', 'success');
}

async function retranslateSelected() {
  const cueIds = $$('.cue').filter(cue => cue.querySelector('.cue-select')?.checked).map(cue => cue.dataset.cueId);
  if (!cueIds.length) throw new Error('Select at least one source cue to retranslate.');
  const result = await api(`/v1/jobs/${state.currentReviewJob}/review/retranslate`, {
    method: 'POST', body: JSON.stringify({cue_ids: cueIds}),
  });
  message($('#review-message'), `Retranslated ${result.retranslation.translated_cue_count} selected cue(s). Previous translations remain in history.`, 'success');
  await openReview(state.currentReviewJob);
}

async function approveReview() {
  await saveReview();
  await api(`/v1/jobs/${state.currentReviewJob}/approve`, {method: 'POST', body: JSON.stringify({translation_confirmed: true})});
  message($('#review-message'), 'Exact draft approved. Burn remains a separate action.', 'success');
  await openReview(state.currentReviewJob);
  await loadJobs();
}

async function burnApprovedReview() {
  await api(`/v1/jobs/${state.currentReviewJob}/burn`, {method: 'POST', body: JSON.stringify({})});
  message($('#review-message'), 'Approved draft queued for burn.', 'success');
  $('#review-panel').hidden = true;
  await loadJobs();
}

async function resolveCloudIssue(issueId, severity) {
  const corrected = severity === 'critical';
  const reason = corrected ? window.prompt('Describe the correction made:') : 'Explicitly accepted during review';
  if (corrected && !reason) return;
  await api(`/v1/jobs/${state.currentReviewJob}/issues/${issueId}/resolve`, {
    method: 'POST', body: JSON.stringify({status: corrected ? 'corrected' : 'accepted', reason}),
  });
  await openReview(state.currentReviewJob);
}

async function downloadArtifact(artifactId) {
  const link = await api(`/v1/artifacts/${artifactId}/download`);
  const sameOrigin = new URL(link.url, location.href).origin === location.origin;
  if (!sameOrigin) { location.href = link.url; return; }
  const response = await fetch(link.url, {headers: authHeaders(false)});
  if (!response.ok) throw new Error('Download failed.');
  const blob = await response.blob();
  const anchor = document.createElement('a');
  anchor.href = URL.createObjectURL(blob); anchor.download = 'subgen-output'; anchor.click();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
}

async function bootstrap() {
  readAuthCallback();
  const response = await fetch(`${state.apiBaseUrl}/v1/public-config`);
  if (!response.ok) throw new Error('The cloud API is unavailable. Check the deployment configuration.');
  state.publicConfig = await response.json();
  $('#upload-limit').textContent = `Up to ${prettyBytes(state.publicConfig.max_upload_bytes)}`;
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('/service-worker.js').catch(() => {});
  if (state.token || (state.publicConfig.development_auth && state.devUser)) {
    try { await openWorkspace(); } catch (_) { clearSession(); }
  }
}

$('#login-form').addEventListener('submit', event => {event.preventDefault(); login($('#login-email').value).catch(error => message($('#login-message'), error.message, 'error'));});
$('#logout-button').addEventListener('click', () => {clearSession(); location.reload();});
$$('.nav-item').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));
$('#menu-button').addEventListener('click', () => $('.sidebar').classList.toggle('open'));
$('#video-file').addEventListener('change', () => {const file=$('#video-file').files[0]; if(file) $('#existing-video').value=''; $('#file-label').textContent=file?`${file.name} · ${prettyBytes(file.size)}`:'MP4, MOV, MKV, AVI, or WebM'; $('#start-button').disabled=!file&&!$('#existing-video').value;});
$('#existing-video').addEventListener('change', () => {if($('#existing-video').value){$('#video-file').value='';$('#file-label').textContent='MP4, MOV, MKV, AVI, or WebM';} $('#start-button').disabled=!$('#existing-video').value&&!$('#video-file').files[0];});
$('#job-form').addEventListener('submit', event => {event.preventDefault(); createJob().catch(error => {$('#start-button').disabled=false; message($('#create-message'), error.message, 'error');});});
$('#refresh-jobs').addEventListener('click', () => loadJobs().catch(console.error));
$('#close-review').addEventListener('click', () => {$('#review-panel').hidden=true;});
$('#save-review-button').addEventListener('click', () => saveReview().catch(error => message($('#review-message'), error.message, 'error')));
$('#retranslate-button').addEventListener('click', () => retranslateSelected().catch(error => message($('#review-message'), error.message, 'error')));
$('#approve-button').addEventListener('click', () => approveReview().catch(error => message($('#review-message'), error.message, 'error')));
$('#burn-button').addEventListener('click', () => burnApprovedReview().catch(error => message($('#review-message'), error.message, 'error')));
$('#credential-form').addEventListener('submit', event => {event.preventDefault(); saveCredential().catch(error => message($('#credential-message'), error.message, 'error'));});
$('#pipeline-settings-form').addEventListener('submit', event => {event.preventDefault(); saveSettings().catch(error => message($('#settings-message'), error.message, 'error'));});
window.addEventListener('online', () => $('#connection-state').textContent='Online');
window.addEventListener('offline', () => $('#connection-state').textContent='Offline');
setInterval(() => { if (state.user) loadJobs().catch(() => {}); }, 5000);
bootstrap().catch(error => message($('#login-message'), error.message, 'error'));
