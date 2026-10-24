// CT Auto-Dubber Frontend Logic

// State
let videoQueue = [];
let keysPool = [];
let isProcessing = false;
let ws = null;

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const dragOverlay = document.getElementById('drag-overlay');
const emptyState = document.getElementById('empty-state');
const queueTable = document.getElementById('queue-table');
const queueTbody = document.getElementById('queue-tbody');
const selectAllCheckbox = document.getElementById('selectAllVideos');
const logsConsole = document.getElementById('logs-console');

// Controls
const btnAddFiles = document.getElementById('btn-add-files');
const btnAddFolder = document.getElementById('btn-add-folder');

const btnRetryErrors = document.getElementById('btn-retry-errors');
const btnRemoveCompleted = document.getElementById('btn-remove-completed');
const btnClearAll = document.getElementById('btn-clear-all');
const btnClearLogs = document.getElementById('btn-clear-logs');
const btnVerifyApi = document.getElementById('btn-verify-api');
const btnSaveSettings = document.getElementById('btn-save-settings');
const btnManageKeys = document.getElementById('btn-manage-keys');
const btnConfigApi = document.getElementById('btn-config-api');
const configModal = document.getElementById('config-modal');
const btnCloseConfigModal = document.getElementById('btn-close-config-modal');
const btnCancelConfig = document.getElementById('btn-cancel-config');
const btnDoneConfig = document.getElementById('btn-done-config');
const btnSelectOutput = document.getElementById('btn-select-output');
const btnStartProcessing = document.getElementById('btn-start-processing');
const btnCancelProcessing = document.getElementById('btn-cancel-processing');

// System Updates Elements
const btnOpenUpdates = document.getElementById('btn-open-updates');
const updatesModal = document.getElementById('updates-modal');
const btnCloseUpdatesModal = document.getElementById('btn-close-updates-modal');
const btnCloseUpdatesDone = document.getElementById('btn-close-updates-done');
const btnUpdateEdgeTts = document.getElementById('btn-update-edge-tts');
const btnCheckGithubUpdate = document.getElementById('btn-check-github-update');
const btnCheckAllUpdates = document.getElementById('btn-check-all-updates');
const btnSaveGhRepo = document.getElementById('btn-save-gh-repo');
const inputGithubRepo = document.getElementById('input-github-repo');
const edgeUpdateStatus = document.getElementById('edge-update-status');
const githubUpdateStatus = document.getElementById('github-update-status');
const updateLogBox = document.getElementById('update-log-box');
const updateAppVerDisplay = document.getElementById('update-app-ver-display');
const updateEdgeVerDisplay = document.getElementById('update-edge-ver-display');
const updateAppBadge = document.getElementById('update-app-badge');
const updateEdgeBadge = document.getElementById('update-edge-badge');

// Settings Fields
const apiTypeRadios = document.getElementsByName('api-type');
const outputTypeRadios = document.getElementsByName('output-type');
const customPathContainer = document.getElementById('custom-path-container');
const outputPathText = document.getElementById('output-path-text');
const selectModel = document.getElementById('select-model');

const selectTargetLang = document.getElementById('select-target-lang');
const selectGender = document.getElementById('select-gender');
const selectTtsEngine = document.getElementById('select-tts-engine');
const inputHfSpace = document.getElementById('input-hf-space');
const hfSpaceContainer = document.getElementById('hf-space-container');
const inputHfToken = document.getElementById('input-hf-token');
const hfTokenContainer = document.getElementById('hf-token-container');
const inputVoxcpmCfg = document.getElementById('input-voxcpm-cfg');
const chkVoxcpmUltimateClone = document.getElementById('chk-voxcpm-ultimate-clone');
const inputMaxSpeechSpeed = document.getElementById('input-max-speech-speed');
const maleVoicePathText = document.getElementById('male-voice-path-text');
const femaleVoicePathText = document.getElementById('female-voice-path-text');
const narratorMaleVoicePathText = document.getElementById('narrator-male-voice-path-text');
const narratorFemaleVoicePathText = document.getElementById('narrator-female-voice-path-text');
const youngMaleVoicePathText = document.getElementById('young-male-voice-path-text');
const youngFemaleVoicePathText = document.getElementById('young-female-voice-path-text');
const oldMaleVoicePathText = document.getElementById('old-male-voice-path-text');
const oldFemaleVoicePathText = document.getElementById('old-female-voice-path-text');
const btnSelectMaleVoice = document.getElementById('btn-select-male-voice');
const btnSelectFemaleVoice = document.getElementById('btn-select-female-voice');
const btnSelectNarratorMaleVoice = document.getElementById('btn-select-narrator-male-voice');
const btnSelectNarratorFemaleVoice = document.getElementById('btn-select-narrator-female-voice');
const btnSelectYoungMaleVoice = document.getElementById('btn-select-young-male-voice');
const btnSelectYoungFemaleVoice = document.getElementById('btn-select-young-female-voice');
const btnSelectOldMaleVoice = document.getElementById('btn-select-old-male-voice');
const btnSelectOldFemaleVoice = document.getElementById('btn-select-old-female-voice');
const btnClearMaleVoice = document.getElementById('btn-clear-male-voice');
const btnClearFemaleVoice = document.getElementById('btn-clear-female-voice');
const btnClearNarratorMaleVoice = document.getElementById('btn-clear-narrator-male-voice');
const btnClearNarratorFemaleVoice = document.getElementById('btn-clear-narrator-female-voice');
const btnClearYoungMaleVoice = document.getElementById('btn-clear-young-male-voice');
const btnClearYoungFemaleVoice = document.getElementById('btn-clear-young-female-voice');
const btnClearOldMaleVoice = document.getElementById('btn-clear-old-male-voice');
const btnClearOldFemaleVoice = document.getElementById('btn-clear-old-female-voice');
const selectGpu = document.getElementById('select-gpu');
const selectThreads = document.getElementById('select-threads');
const selectSyncOffset = document.getElementById('select-sync-offset');
const mixerDrums = document.getElementById('mixer-drums');
const mixerBass = document.getElementById('mixer-bass');
const mixerMelody = document.getElementById('mixer-melody');
const mixerDrumsVol = document.getElementById('mixer-drums-vol');
const mixerBassVol = document.getElementById('mixer-bass-vol');
const mixerMelodyVol = document.getElementById('mixer-melody-vol');
const drumsVolBadge = document.getElementById('drums-vol-badge');
const bassVolBadge = document.getElementById('bass-vol-badge');
const melodyVolBadge = document.getElementById('melody-vol-badge');
const chkAutoShutdown = document.getElementById('chk-auto-shutdown');
const chkDeleteOriginal = document.getElementById('chk-delete-original');

// Enhance Video Elements
const chkMovieRecap = document.getElementById('chk-movie-recap');
const chkVideoSpeedAdjust = document.getElementById('chk-video-speed-adjust');
const videoSpeedOptions = document.getElementById('video-speed-options');
const inputVideoSpeed = document.getElementById('input-video-speed');
const videoSpeedBadge = document.getElementById('video-speed-badge');
const chkBurnSubtitles = document.getElementById('chk-burn-subtitles');
const chkSubtitleBox = document.getElementById('chk-subtitle-box');
const inputSubMargin = document.getElementById('input-sub-margin');
const chkColorCorrection = document.getElementById('chk-color-correction');

// YouTube Multi-Audio Elements
const chkYoutubeMultiAudio = document.getElementById('chk-youtube-multi-audio');
const youtubeLanguagesPanel = document.getElementById('youtube-languages-panel');
const ytDropdownWrapper = document.getElementById('yt-dropdown-wrapper');
const ytDropdownTrigger = document.getElementById('yt-dropdown-trigger');
const ytDropdownMenu = document.getElementById('yt-dropdown-menu');
const ytDropdownLabel = document.getElementById('yt-dropdown-label');
const ytSelectedCountBadge = document.getElementById('yt-selected-count-badge');
const ytLangSearch = document.getElementById('yt-lang-search');
const btnYtSelectAll = document.getElementById('btn-yt-select-all');
const btnYtClearAll = document.getElementById('btn-yt-clear-all');

// Key Modal Elements
const keysModal = document.getElementById('keys-modal');
const btnCloseModal = document.getElementById('btn-close-modal');
const btnCancelKeys = document.getElementById('btn-cancel-keys');
const btnSaveKeys = document.getElementById('btn-save-keys');
const apiKeysTextarea = document.getElementById('api-keys-textarea');
const apiStatusText = document.getElementById('api-status-text');
const apiStatusDot = document.getElementById('api-status-dot');

// Filter Counters
const filterError = document.getElementById('filter-error');
const filterCompleted = document.getElementById('filter-completed');
const filterTotal = document.getElementById('filter-total');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    loadSettings();
    setupEventListeners();
    refreshQueueUI();
});

// Setup WebSocket Connection
function initWebSocket() {
    const wsUrl = `ws://${window.location.host}/ws`;
    addLog('Connecting to local engine...', 'info');

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        addLog('[SUCCESS] CONNECTED TO DENZIN ENGINE.', 'success');
        fetchQueue();
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWsMessage(data);
        } catch (e) {
            console.error('Error parsing WS message:', e);
        }
    };

    ws.onerror = (error) => {
        console.error('WS Error:', error);
    };

    ws.onclose = () => {
        addLog('[WARNING] Disconnected from engine. Retrying in 3 seconds...', 'warning');
        setTimeout(initWebSocket, 3000);
    };
}

// Handle WebSocket Messages
function handleWsMessage(data) {
    switch (data.type) {
        case 'log':
            addLog(data.message, data.level || 'info');
            break;
        case 'queue_update':
            videoQueue = data.queue;
            refreshQueueUI();
            break;
        case 'progress_update':
            updateVideoProgress(data.id, data.progress, data.status);
            break;
        case 'api_status':
            updateApiStatus(data.keys_count, data.status);
            break;
        case 'status_update':
            isProcessing = data.is_processing;
            toggleProcessingUI(isProcessing);
            break;
    }
}

// Fetch Queue from Backend
async function fetchQueue() {
    try {
        const response = await fetch('/api/queue');
        const data = await response.json();
        videoQueue = data.queue;
        refreshQueueUI();
    } catch (e) {
        console.error('Failed to fetch queue', e);
    }
}

// Add Log Line to Console
function addLog(message, level = 'info') {
    const line = document.createElement('div');
    line.className = `log-line text-${level}`;

    // Add timestamp
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    line.textContent = `[${timeStr}] ${message}`;

    logsConsole.appendChild(line);
    logsConsole.scrollTop = logsConsole.scrollHeight;
}

// Event Listeners
function setupEventListeners() {
    // Add Files / Folders (via Native File Dialog on server side)
    btnAddFiles.addEventListener('click', () => browseNative('files'));
    btnAddFolder.addEventListener('click', () => browseNative('folder'));

    // API Type selection
    apiTypeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'groq') {
                if (selectModel && selectModel.value.startsWith('gemini-')) {
                    selectModel.value = 'openai/gpt-oss-120b';
                }
            } else if (e.target.value === 'free' || e.target.value === 'vertex') {
                if (selectModel && (selectModel.value.includes('oss') || selectModel.value.includes('qwen') || selectModel.value.includes('llama'))) {
                    selectModel.value = 'gemini-2.5-flash';
                }
            }
            saveSettings();
            verifyApiKey();
        });
    });

    // Output Destination selection
    outputTypeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'custom') {
                customPathContainer.style.display = 'flex';
            } else {
                customPathContainer.style.display = 'none';
            }
        });
    });
    btnSelectOutput.addEventListener('click', () => browseNative('output'));

    // Voice Cloning reference audio buttons
    if (btnSelectMaleVoice) btnSelectMaleVoice.addEventListener('click', () => browseNative('male_voice'));
    if (btnSelectFemaleVoice) btnSelectFemaleVoice.addEventListener('click', () => browseNative('female_voice'));
    if (btnSelectNarratorMaleVoice) btnSelectNarratorMaleVoice.addEventListener('click', () => browseNative('narrator_male_voice'));
    if (btnSelectNarratorFemaleVoice) btnSelectNarratorFemaleVoice.addEventListener('click', () => browseNative('narrator_female_voice'));
    if (btnSelectYoungMaleVoice) btnSelectYoungMaleVoice.addEventListener('click', () => browseNative('young_male_voice'));
    if (btnSelectYoungFemaleVoice) btnSelectYoungFemaleVoice.addEventListener('click', () => browseNative('young_female_voice'));
    if (btnSelectOldMaleVoice) btnSelectOldMaleVoice.addEventListener('click', () => browseNative('old_male_voice'));
    if (btnSelectOldFemaleVoice) btnSelectOldFemaleVoice.addEventListener('click', () => browseNative('old_female_voice'));
    if (btnClearMaleVoice) btnClearMaleVoice.addEventListener('click', () => {
        maleVoicePathText.textContent = 'Default Male Voice';
        maleVoicePathText.title = 'Default Model Voice';
        saveSettings();
    });
    if (btnClearFemaleVoice) btnClearFemaleVoice.addEventListener('click', () => {
        femaleVoicePathText.textContent = 'Default Female Voice';
        femaleVoicePathText.title = 'Default Model Voice';
        saveSettings();
    });
    if (btnClearNarratorMaleVoice) btnClearNarratorMaleVoice.addEventListener('click', () => {
        narratorMaleVoicePathText.textContent = 'Default Narrator Male';
        narratorMaleVoicePathText.title = 'Default Model Voice';
        saveSettings();
    });
    if (btnClearNarratorFemaleVoice) btnClearNarratorFemaleVoice.addEventListener('click', () => {
        narratorFemaleVoicePathText.textContent = 'Default Narrator Female';
        narratorFemaleVoicePathText.title = 'Default Model Voice';
        saveSettings();
    });
    if (btnClearYoungMaleVoice) btnClearYoungMaleVoice.addEventListener('click', () => {
        youngMaleVoicePathText.textContent = 'Default Child Male Voice';
        youngMaleVoicePathText.title = 'Default Model Voice';
        saveSettings();
    });
    if (btnClearYoungFemaleVoice) btnClearYoungFemaleVoice.addEventListener('click', () => {
        youngFemaleVoicePathText.textContent = 'Default Child Female Voice';
        youngFemaleVoicePathText.title = 'Default Model Voice';
        saveSettings();
    });
    if (btnClearOldMaleVoice) btnClearOldMaleVoice.addEventListener('click', () => {
        oldMaleVoicePathText.textContent = 'Default Elderly Male Voice';
        oldMaleVoicePathText.title = 'Default Model Voice';
        saveSettings();
    });
    if (btnClearOldFemaleVoice) btnClearOldFemaleVoice.addEventListener('click', () => {
        oldFemaleVoicePathText.textContent = 'Default Elderly Female Voice';
        oldFemaleVoicePathText.title = 'Default Model Voice';
        saveSettings();
    });

    // Global settings changes
    if (chkVoxcpmUltimateClone) chkVoxcpmUltimateClone.addEventListener('change', saveSettings);
    if (inputVoxcpmCfg) {
        inputVoxcpmCfg.addEventListener('input', (e) => {
            const badge = document.getElementById('voxcpm-cfg-badge');
            if (badge) badge.textContent = parseFloat(e.target.value).toFixed(1);
        });
        inputVoxcpmCfg.addEventListener('change', saveSettings);
    }
    if (inputMaxSpeechSpeed) {
        inputMaxSpeechSpeed.addEventListener('input', (e) => {
            const badge = document.getElementById('speech-speed-badge');
            if (badge) badge.textContent = `${parseFloat(e.target.value).toFixed(2)}x`;
        });
        inputMaxSpeechSpeed.addEventListener('change', saveSettings);
    }

    document.querySelectorAll('input[name="voice-clone-mode"]').forEach(radio => {
        radio.addEventListener('change', saveSettings);
    });

    // Modal Keys management
    btnManageKeys.addEventListener('click', () => {
        apiKeysTextarea.value = keysPool.join('\n');
        keysModal.style.display = 'flex';
        updateKeyInspector();
    });

    // Config API management
    btnConfigApi.addEventListener('click', () => {
        configModal.style.display = 'flex';
    });

    const closeConfigModal = () => configModal.style.display = 'none';
    btnCloseConfigModal.addEventListener('click', closeConfigModal);
    btnCancelConfig.addEventListener('click', closeConfigModal);
    btnDoneConfig.addEventListener('click', () => {
        saveSettings();
        closeConfigModal();
    });

    // --- System Updates Modal & Logic ---
    const closeUpdatesModal = () => { if (updatesModal) updatesModal.style.display = 'none'; };
    if (btnOpenUpdates) {
        btnOpenUpdates.addEventListener('click', () => {
            if (updatesModal) {
                updatesModal.style.display = 'flex';
                checkSystemUpdates();
            }
        });
    }
    if (btnCloseUpdatesModal) btnCloseUpdatesModal.addEventListener('click', closeUpdatesModal);
    if (btnCloseUpdatesDone) btnCloseUpdatesDone.addEventListener('click', closeUpdatesModal);

    function logUpdate(msg) {
        if (!updateLogBox) return;
        updateLogBox.style.display = 'block';
        const d = new Date().toLocaleTimeString();
        updateLogBox.innerHTML += `<div>[${d}] ${msg}</div>`;
        updateLogBox.scrollTop = updateLogBox.scrollHeight;
    }

    async function checkSystemUpdates() {
        if (!updatesModal) return;
        try {
            logUpdate("Checking for updates via PyPI and GitHub...");
            if (edgeUpdateStatus) {
                edgeUpdateStatus.style.display = 'block';
                edgeUpdateStatus.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Checking PyPI repository...';
            }
            if (githubUpdateStatus) {
                githubUpdateStatus.style.display = 'block';
                githubUpdateStatus.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Checking GitHub releases...';
            }

            const res = await fetch('/api/system/check-updates');
            const data = await res.json();

            if (updateAppVerDisplay) updateAppVerDisplay.textContent = `v${data.current_app_version || '2.1.0'}`;
            if (updateEdgeVerDisplay) updateEdgeVerDisplay.textContent = `v${data.current_edge_tts || 'unknown'}`;
            if (inputGithubRepo && data.github?.repo) inputGithubRepo.value = data.github.repo;

            // Edge-TTS Status
            if (data.edge_tts?.has_update) {
                if (updateEdgeBadge) updateEdgeBadge.innerHTML = `<span class="badge badge-warning" style="font-size: 11px;">Update Available: v${data.edge_tts.latest_version}</span>`;
                if (edgeUpdateStatus) edgeUpdateStatus.innerHTML = `<span style="color: #f6c177;"><i class="fa-solid fa-circle-exclamation"></i> New version v${data.edge_tts.latest_version} available on PyPI! Click "Update TTS Now" below.</span>`;
            } else if (data.edge_tts?.error) {
                if (edgeUpdateStatus) edgeUpdateStatus.innerHTML = `<span style="color: #eb6f92;"><i class="fa-solid fa-triangle-exclamation"></i> Check failed: ${data.edge_tts.error}</span>`;
            } else {
                if (updateEdgeBadge) updateEdgeBadge.innerHTML = `<span class="badge badge-success" style="font-size: 11px;">Up to date (v${data.current_edge_tts})</span>`;
                if (edgeUpdateStatus) edgeUpdateStatus.innerHTML = `<span style="color: #9ccfd8;"><i class="fa-solid fa-check"></i> Latest version installed (v${data.current_edge_tts}). WebSocket handshake verified.</span>`;
            }

            // GitHub Status
            if (data.github?.has_update) {
                if (updateAppBadge) updateAppBadge.innerHTML = `<span class="badge badge-warning" style="font-size: 11px;">Update Available: v${data.github.latest_version}</span>`;
                if (githubUpdateStatus) {
                    githubUpdateStatus.innerHTML = `
                        <div style="color: #f6c177; margin-bottom: 8px;"><i class="fa-solid fa-circle-exclamation"></i> New release <b>v${data.github.latest_version}</b> available!</div>
                        <button class="btn btn-sm btn-primary" id="btn-apply-github-patch"><i class="fa-solid fa-download"></i> Apply Patch Now</button>
                    `;
                    const btnApply = document.getElementById('btn-apply-github-patch');
                    if (btnApply) btnApply.addEventListener('click', () => applyGitHubPatch(data.github.download_url));
                }
            } else if (data.github?.error) {
                if (githubUpdateStatus) githubUpdateStatus.innerHTML = `<span style="color: #aaa;"><i class="fa-solid fa-info-circle"></i> ${data.github.error}</span>`;
            } else if (data.github?.repo) {
                if (updateAppBadge) updateAppBadge.innerHTML = `<span class="badge badge-success" style="font-size: 11px;">Up to date</span>`;
                if (githubUpdateStatus) githubUpdateStatus.innerHTML = `<span style="color: #9ccfd8;"><i class="fa-solid fa-check"></i> Core engine is up to date (v${data.current_app_version}).</span>`;
            } else {
                if (githubUpdateStatus) githubUpdateStatus.innerHTML = `<span style="color: #888;">Enter your GitHub repository (e.g. username/repo) to check for app code updates.</span>`;
            }

            logUpdate("Update check complete.");
        } catch (e) {
            logUpdate(`Update check failed: ${e.message}`);
        }
    }

    if (btnCheckAllUpdates) btnCheckAllUpdates.addEventListener('click', checkSystemUpdates);
    if (btnCheckGithubUpdate) btnCheckGithubUpdate.addEventListener('click', checkSystemUpdates);

    if (btnSaveGhRepo && inputGithubRepo) {
        btnSaveGhRepo.addEventListener('click', async () => {
            const repo = inputGithubRepo.value.trim();
            try {
                const sRes = await fetch('/api/settings');
                const sData = await sRes.json();
                sData.settings.github_update_repo = repo;
                await fetch('/api/settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(sData.settings)
                });
                logUpdate(`Saved GitHub update repository: ${repo}`);
                checkSystemUpdates();
            } catch (err) {
                logUpdate(`Failed to save repository: ${err.message}`);
            }
        });
    }

    if (btnUpdateEdgeTts) {
        btnUpdateEdgeTts.addEventListener('click', async () => {
            btnUpdateEdgeTts.disabled = true;
            btnUpdateEdgeTts.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Patching...';
            logUpdate("Contacting PyPI to download latest Edge-TTS wheel...");
            try {
                const res = await fetch('/api/system/update-edge-tts', { method: 'POST' });
                const d = await res.json();
                if (res.ok) {
                    logUpdate(`SUCCESS: Edge-TTS updated to v${d.new_version}! Runtime hot-patched.`);
                    if (updateEdgeVerDisplay) updateEdgeVerDisplay.textContent = `v${d.new_version}`;
                    if (updateEdgeBadge) updateEdgeBadge.innerHTML = `<span class="badge badge-success" style="font-size: 11px;">Up to date (v${d.new_version})</span>`;
                    if (edgeUpdateStatus) edgeUpdateStatus.innerHTML = `<span style="color: #38ef7d;"><i class="fa-solid fa-circle-check"></i> Successfully updated to v${d.new_version}! No recompile needed.</span>`;
                } else {
                    logUpdate(`ERROR: ${d.detail || 'Update failed'}`);
                }
            } catch (e) {
                logUpdate(`ERROR: ${e.message}`);
            } finally {
                btnUpdateEdgeTts.disabled = false;
                btnUpdateEdgeTts.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Update TTS Now';
            }
        });
    }

    async function applyGitHubPatch(downloadUrl) {
        logUpdate("Downloading patch archive from GitHub...");
        try {
            const res = await fetch('/api/system/apply-github-update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ download_url: downloadUrl })
            });
            const d = await res.json();
            if (res.ok) {
                logUpdate(`SUCCESS: Extracted ${d.extracted_files} updated files. Refreshing UI in 2 seconds...`);
                setTimeout(() => window.location.reload(), 2000);
            } else {
                logUpdate(`ERROR: ${d.detail || 'Patch failed'}`);
            }
        } catch (e) {
            logUpdate(`ERROR: ${e.message}`);
        }
    }

    apiKeysTextarea.addEventListener('input', updateKeyInspector);

    const closeModal = () => keysModal.style.display = 'none';
    btnCloseModal.addEventListener('click', closeModal);
    btnCancelKeys.addEventListener('click', closeModal);

    btnSaveKeys.addEventListener('click', async () => {
        const keysText = apiKeysTextarea.value;
        keysPool = keysText.split('\n').map(k => k.trim()).filter(k => k.length > 0);
        closeModal();
        await saveApiKeys();
    });

    // Verify API
    btnVerifyApi.addEventListener('click', verifyApiKey);

    // Save settings
    btnSaveSettings.addEventListener('click', saveSettings);
    selectModel.addEventListener('change', saveSettings);
    selectTargetLang.addEventListener('change', saveSettings);
    selectGender.addEventListener('change', saveSettings);
    selectThreads.addEventListener('change', saveSettings);
    if (selectSyncOffset) selectSyncOffset.addEventListener('change', saveSettings);
    if (chkAutoShutdown) chkAutoShutdown.addEventListener('change', saveSettings);
    if (chkDeleteOriginal) chkDeleteOriginal.addEventListener('change', saveSettings);

    if (chkMovieRecap) chkMovieRecap.addEventListener('change', saveSettings);
    if (chkVideoSpeedAdjust && videoSpeedOptions) {
        chkVideoSpeedAdjust.addEventListener('change', () => {
            videoSpeedOptions.style.display = chkVideoSpeedAdjust.checked ? 'block' : 'none';
            saveSettings();
        });
    }
    if (inputVideoSpeed && videoSpeedBadge) {
        inputVideoSpeed.addEventListener('input', (e) => {
            videoSpeedBadge.textContent = `${parseFloat(e.target.value).toFixed(2)}x`;
        });
        inputVideoSpeed.addEventListener('change', saveSettings);
    }
    if (chkBurnSubtitles) chkBurnSubtitles.addEventListener('change', saveSettings);
    if (chkSubtitleBox) chkSubtitleBox.addEventListener('change', saveSettings);
    if (inputSubMargin) inputSubMargin.addEventListener('change', saveSettings);
    if (chkColorCorrection) chkColorCorrection.addEventListener('change', saveSettings);

    // YouTube Multi-Audio Tracks
    if (chkYoutubeMultiAudio && youtubeLanguagesPanel) {
        chkYoutubeMultiAudio.addEventListener('change', () => {
            youtubeLanguagesPanel.style.display = chkYoutubeMultiAudio.checked ? 'block' : 'none';
            if (!chkYoutubeMultiAudio.checked && ytDropdownWrapper) {
                ytDropdownWrapper.classList.remove('open');
                if (ytDropdownTrigger) ytDropdownTrigger.setAttribute('aria-expanded', 'false');
            }
            saveSettings();
        });
    }

    // Dropdown toggle
    if (ytDropdownTrigger && ytDropdownWrapper) {
        ytDropdownTrigger.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = ytDropdownWrapper.classList.toggle('open');
            ytDropdownTrigger.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            if (isOpen && ytLangSearch) {
                setTimeout(() => ytLangSearch.focus(), 50);
            }
        });
    }

    if (ytDropdownMenu) {
        ytDropdownMenu.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    document.addEventListener('click', (e) => {
        if (ytDropdownWrapper && !ytDropdownWrapper.contains(e.target)) {
            ytDropdownWrapper.classList.remove('open');
            if (ytDropdownTrigger) ytDropdownTrigger.setAttribute('aria-expanded', 'false');
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && ytDropdownWrapper && ytDropdownWrapper.classList.contains('open')) {
            ytDropdownWrapper.classList.remove('open');
            if (ytDropdownTrigger) ytDropdownTrigger.setAttribute('aria-expanded', 'false');
        }
    });

    // Real-time language search
    if (ytLangSearch) {
        ytLangSearch.addEventListener('input', (e) => {
            const q = (e.target.value || '').toLowerCase().trim();
            document.querySelectorAll('.yt-lang-item').forEach(item => {
                const text = (item.textContent || '').toLowerCase();
                const name = (item.getAttribute('data-lang-name') || '').toLowerCase();
                const native = (item.getAttribute('data-lang-native') || '').toLowerCase();
                const matches = !q || text.includes(q) || name.includes(q) || native.includes(q);
                item.style.display = matches ? 'flex' : 'none';
            });
        });
    }

    if (btnYtSelectAll) {
        btnYtSelectAll.addEventListener('click', () => {
            document.querySelectorAll('.chk-yt-lang').forEach(cb => { cb.checked = true; });
            updateYtDropdownSummary();
            saveSettings();
        });
    }
    if (btnYtClearAll) {
        btnYtClearAll.addEventListener('click', () => {
            document.querySelectorAll('.chk-yt-lang').forEach(cb => { cb.checked = false; });
            updateYtDropdownSummary();
            saveSettings();
        });
    }
    document.querySelectorAll('.chk-yt-lang').forEach(cb => {
        cb.addEventListener('change', () => {
            updateYtDropdownSummary();
            saveSettings();
        });
    });

    // Mixer Checkboxes & Volume Sliders
    [mixerDrums, mixerBass, mixerMelody].forEach(cb => {
        if (cb) cb.addEventListener('change', saveSettings);
    });
    const bindStemVol = (slider, badge) => {
        if (!slider || !badge) return;
        const update = () => {
            badge.textContent = `${slider.value}%`;
        };
        slider.addEventListener('input', update);
        slider.addEventListener('change', saveSettings);
    };
    bindStemVol(mixerDrumsVol, drumsVolBadge);
    bindStemVol(mixerBassVol, bassVolBadge);
    bindStemVol(mixerMelodyVol, melodyVolBadge);

    // Queue actions

    btnRetryErrors.addEventListener('click', () => queueAction('retry-errors'));
    btnRemoveCompleted.addEventListener('click', () => queueAction('remove-completed'));
    btnClearAll.addEventListener('click', () => queueAction('clear-all'));

    btnClearLogs.addEventListener('click', () => {
        logsConsole.innerHTML = '';
        addLog('Logs cleared.', 'info');
    });

    // Start / Stop processing
    btnStartProcessing.addEventListener('click', startProcessing);
    btnCancelProcessing.addEventListener('click', stopProcessing);

    // Select All Checkbox
    const selectAllCheckbox = document.getElementById('select-all-videos');
    if (selectAllCheckbox) {
        selectAllCheckbox.addEventListener('change', (e) => {
            const checked = e.target.checked;
            document.querySelectorAll('.video-checkbox').forEach(cb => {
                cb.checked = checked;
            });
        });
    }

    // Drag and drop event handlers
    window.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dragOverlay.classList.add('active');
    });

    dragOverlay.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    dragOverlay.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragOverlay.classList.remove('active');
    });

    dragOverlay.addEventListener('drop', async (e) => {
        e.preventDefault();
        dragOverlay.classList.remove('active');

        // Since we are inside a browser environment, standard file drop only gives us files.
        // We will send file names/sizes to the backend. For local server paths, native browsing is preferred.
        // But we will handle basic drag and drop metadata to allow local mocking.
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            addLog(`Dropped ${files.length} file(s). Uploading metadata to server...`, 'info');

            // Call API to add dropped files
            const fileData = files.map(f => ({ name: f.name, size: f.size, path: f.path || "" }));
            try {
                const response = await fetch('/api/queue/add-dropped', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ files: fileData })
                });
                const data = await response.json();
                videoQueue = data.queue;
                refreshQueueUI();
            } catch (err) {
                addLog(`Error adding dropped files: ${err.message}`, 'error');
            }
        }
    });
}

// Request Native Dialog browse
async function browseNative(type) {
    addLog(`Opening file explorer dialog for ${type}...`, 'info');
    try {
        const response = await fetch(`/api/browse?type=${type}`);
        const data = await response.json();

        if (data.cancelled) {
            addLog('Selection cancelled.', 'warning');
            return;
        }

        if (type === 'output') {
            outputPathText.textContent = `Path: ${data.path}`;
            outputPathText.title = data.path;
            addLog(`Output path set to: ${data.path}`, 'success');
            saveSettings();
        } else if (type === 'male_voice') {
            maleVoicePathText.textContent = data.path;
            maleVoicePathText.title = data.path;
            addLog(`Male actor voice reference set: ${data.path}`, 'success');
            saveSettings();
        } else if (type === 'female_voice') {
            femaleVoicePathText.textContent = data.path;
            femaleVoicePathText.title = data.path;
            addLog(`Female actor voice reference set: ${data.path}`, 'success');
            saveSettings();
        } else if (type === 'narrator_male_voice') {
            narratorMaleVoicePathText.textContent = data.path;
            narratorMaleVoicePathText.title = data.path;
            addLog(`Narrator Male voice reference set: ${data.path}`, 'success');
            saveSettings();
        } else if (type === 'narrator_female_voice') {
            narratorFemaleVoicePathText.textContent = data.path;
            narratorFemaleVoicePathText.title = data.path;
            addLog(`Narrator Female voice reference set: ${data.path}`, 'success');
            saveSettings();
        } else if (type === 'young_male_voice') {
            youngMaleVoicePathText.textContent = data.path;
            youngMaleVoicePathText.title = data.path;
            addLog(`Child Male voice reference set: ${data.path}`, 'success');
            saveSettings();
        } else if (type === 'young_female_voice') {
            youngFemaleVoicePathText.textContent = data.path;
            youngFemaleVoicePathText.title = data.path;
            addLog(`Child Female voice reference set: ${data.path}`, 'success');
            saveSettings();
        } else if (type === 'old_male_voice') {
            oldMaleVoicePathText.textContent = data.path;
            oldMaleVoicePathText.title = data.path;
            addLog(`Elderly Male voice reference set: ${data.path}`, 'success');
            saveSettings();
        } else if (type === 'old_female_voice') {
            oldFemaleVoicePathText.textContent = data.path;
            oldFemaleVoicePathText.title = data.path;
            addLog(`Elderly Female voice reference set: ${data.path}`, 'success');
            saveSettings();
        } else {
            addLog(`Added ${data.added_count} items to the queue.`, 'success');
            videoQueue = data.queue;
            refreshQueueUI();
        }
    } catch (e) {
        addLog(`Failed to open browse dialog: ${e.message}`, 'error');
    }
}

// Refresh Queue UI
function refreshQueueUI() {
    if (videoQueue.length === 0) {
        emptyState.style.display = 'flex';
        queueTable.style.display = 'none';
    } else {
        emptyState.style.display = 'none';
        queueTable.style.display = 'table';
    }

    queueTbody.innerHTML = '';

    let errorCount = 0;
    let completedCount = 0;

    videoQueue.forEach((video, index) => {
        if (video.status === 'Error') errorCount++;
        if (video.status === 'Completed') completedCount++;

        const tr = document.createElement('tr');
        if (video.status === 'Processing') tr.className = 'processing';

        tr.innerHTML = `
            <td><input type="checkbox" class="video-checkbox" data-id="${video.id}"></td>
            <td class="video-name-cell" title="${video.name}">${index + 1}. 🎬 ${video.name}</td>
            <td class="video-path-cell" title="${video.path}">${video.path || 'Local Upload'}</td>
            <td>
                <div class="progress-container">
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: ${video.progress}%"></div>
                    </div>
                    <div class="progress-text">${video.progress}%</div>
                </div>
            </td>
            <td>
                <span class="status-badge status-${video.status.toLowerCase().replace('..', '').trim()}" style="display: inline-flex; align-items: center; gap: 4px;">
                    <i class="fa-solid ${getStatusIcon(video.status)}"></i>
                    ${video.status}
                </span>
                ${video.status === 'Completed' ? `
                <button class="btn-edit-row" onclick="openSubtitleEditor('${video.id}', '${video.name.replace(/'/g, "\\'")}', '${video.path.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}')" style="margin-left: 6px;">
                    <i class="fa-solid fa-pen-to-square"></i> Edit
                </button>` : ''}
            </td>
            <td>
                <button class="btn-delete-row" onclick="deleteVideo('${video.id}')">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </td>
        `;
        queueTbody.appendChild(tr);
    });

    filterError.textContent = `ERROR: ${errorCount}`;
    filterCompleted.textContent = `COMPLETED: ${completedCount}`;
    filterTotal.textContent = `TOTAL: ${videoQueue.length}`;
}

function getStatusIcon(status) {
    status = status.toLowerCase();
    if (status.includes('queued')) return 'fa-clock';
    if (status.includes('completed')) return 'fa-circle-check';
    if (status.includes('error')) return 'fa-triangle-exclamation';
    if (status.includes('cancelled')) return 'fa-ban';
    return 'fa-spinner fa-spin'; // Processing / Isolating / Dubbing
}

// Update specific video progress in UI directly for smoothness
function updateVideoProgress(id, progress, status) {
    const video = videoQueue.find(v => v.id === id);
    if (video) {
        video.progress = progress;
        video.status = status;
        refreshQueueUI();
    }
}

// Delete video from queue
async function deleteVideo(id) {
    try {
        const response = await fetch(`/api/queue/delete/${id}`, { method: 'DELETE' });
        const data = await response.json();
        videoQueue = data.queue;
        refreshQueueUI();
    } catch (e) {
        addLog(`Failed to delete item: ${e.message}`, 'error');
    }
}

// Queue Action buttons
async function queueAction(action) {
    try {
        const response = await fetch(`/api/queue/${action}`, { method: 'POST' });
        const data = await response.json();
        videoQueue = data.queue;
        refreshQueueUI();
        addLog(`Executed action: ${action.replace('-', ' ')}`, 'info');
    } catch (e) {
        addLog(`Failed to execute ${action}: ${e.message}`, 'error');
    }
}

// Client-side Key Format Identifier
function identifyKeyFormat(key) {
    if (typeof key !== 'string') key = key ? String(key) : '';
    let keyClean = key.trim().replace(/^["']|["']$/g, '');
    if (keyClean.startsWith('GEMINI_API_KEY=')) {
        keyClean = keyClean.replace('GEMINI_API_KEY=', '').trim();
    }

    if (!keyClean) {
        return {
            raw: key, key: '', masked: '', type: 'empty',
            label: 'Empty Key', badge: 'EMPTY', badgeClass: 'badge-gray',
            description: 'No key string provided.', length: 0
        };
    }

    const len = keyClean.length;
    const masked = len > 10 ? `${keyClean.slice(0, 6)}...${keyClean.slice(-4)}` : '***';

    if (keyClean.startsWith('AQ.') || keyClean.startsWith('AQ')) {
        return {
            raw: key, key: keyClean, masked: masked, type: 'gemini_new_aq',
            label: 'New Gemini API Key (AQ....)', badge: 'NEW AQ FORMAT', badgeClass: 'badge-cyan',
            description: 'Modern high-security Gemini developer / enterprise key format.', length: len
        };
    } else if (keyClean.startsWith('AIza')) {
        return {
            raw: key, key: keyClean, masked: masked, type: 'gemini_legacy_aiza',
            label: 'Legacy Google AI Studio Key (AIza...)', badge: 'AIZA LEGACY', badgeClass: 'badge-gold',
            description: 'Standard Google AI Studio developer API key format.', length: len
        };
    } else if (keyClean.startsWith('ya29.')) {
        return {
            raw: key, key: keyClean, masked: masked, type: 'google_oauth',
            label: 'Google OAuth Access Token (ya29...)', badge: 'OAUTH TOKEN', badgeClass: 'badge-purple',
            description: 'Short-lived Google Cloud / Vertex AI OAuth token.', length: len
        };
    } else if (keyClean.startsWith('gsk_')) {
        return {
            raw: key, key: keyClean, masked: masked, type: 'groq_api_key',
            label: 'Groq Cloud API Key (gsk_...)', badge: 'GROQ CLOUD', badgeClass: 'badge-orange',
            description: 'Ultra-fast Groq LPU API key for Llama 3.3 70B & 3.1 8B.', length: len
        };
    } else {
        return {
            raw: key, key: keyClean, masked: masked, type: 'unknown',
            label: 'Custom / Unknown Format', badge: 'CUSTOM KEY', badgeClass: 'badge-blue',
            description: 'Custom key format or proxy credential.', length: len
        };
    }
}

// Render Key Format Inspector in Modal
function updateKeyInspector() {
    const textarea = document.getElementById('api-keys-textarea');
    const listEl = document.getElementById('inspector-list');
    const summaryEl = document.getElementById('inspector-summary');
    if (!textarea || !listEl || !summaryEl) return;

    const lines = textarea.value.split('\n');
    const items = lines.map(l => identifyKeyFormat(l)).filter(item => item.key.length > 0);

    if (items.length === 0) {
        summaryEl.textContent = '0 Keys Detected';
        listEl.innerHTML = `<div class="inspector-empty">Paste or type keys above to inspect format and validity...</div>`;
        return;
    }

    let counts = { gemini_new_aq: 0, gemini_legacy_aiza: 0, google_oauth: 0, groq_api_key: 0, unknown: 0 };
    items.forEach(item => counts[item.type] = (counts[item.type] || 0) + 1);

    let summaryParts = [];
    if (counts.groq_api_key > 0) summaryParts.push(`${counts.groq_api_key} Groq Key${counts.groq_api_key > 1 ? 's' : ''}`);
    if (counts.gemini_new_aq > 0) summaryParts.push(`${counts.gemini_new_aq} New AQ Key${counts.gemini_new_aq > 1 ? 's' : ''}`);
    if (counts.gemini_legacy_aiza > 0) summaryParts.push(`${counts.gemini_legacy_aiza} Legacy Key${counts.gemini_legacy_aiza > 1 ? 's' : ''}`);
    if (counts.google_oauth > 0) summaryParts.push(`${counts.google_oauth} OAuth Token${counts.google_oauth > 1 ? 's' : ''}`);
    if (counts.unknown > 0) summaryParts.push(`${counts.unknown} Custom Key${counts.unknown > 1 ? 's' : ''}`);

    summaryEl.textContent = summaryParts.join(', ') || `${items.length} Keys Detected`;

    listEl.innerHTML = items.map((item, idx) => `
        <div class="key-chip">
            <div class="key-chip-left">
                <span class="key-chip-num">#${idx + 1}</span>
                <span class="key-badge ${item.badgeClass}">${item.badge}</span>
                <span class="key-chip-masked" title="${item.key}">${item.masked}</span>
            </div>
            <div class="key-chip-right">
                <span class="key-chip-len">${item.length} chars</span>
            </div>
        </div>
    `).join('');
}

// Verify API Key
async function verifyApiKey() {
    const apiType = document.querySelector('input[name="api-type"]:checked')?.value || 'free';
    const currentKeys = apiKeysTextarea ? apiKeysTextarea.value.trim() : '';
    const apiTypeLabel = apiType === 'groq' ? 'Groq Cloud LPU' : (apiType === 'free' ? 'Gemini Free Pool' : 'Vertex AI');
    addLog(`Verifying API status (${apiTypeLabel})...`, 'info');

    try {
        const payload = { api_type: apiType };
        if (currentKeys) {
            payload.keys = currentKeys.split('\n').map(k => k.trim()).filter(k => k.length > 0);
        }
        const response = await fetch('/api/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (data.valid) {
            addLog(`[SUCCESS] API Verification Passed: ${data.message}`, 'success');
            updateApiStatus(data.keys_count, 'Valid', data.breakdown);
        } else {
            addLog(`[ERROR] API Verification Failed: ${data.message}`, 'error');
            updateApiStatus(data.keys_count, 'Invalid', data.breakdown);
        }
    } catch (e) {
        addLog(`API Verification Error: ${e.message}`, 'error');
    }
}

// Update API Status Label
function updateApiStatus(keysCount, status, breakdown) {
    let extraStr = '';
    if (breakdown && breakdown.summary) {
        extraStr = ` • ${breakdown.summary}`;
    }
    apiStatusText.textContent = `Pool Status: ${keysCount} Keys Loaded (${status}${extraStr})`;
    if (status === 'Valid' || (status === 'Loaded' && keysCount > 0)) {
        apiStatusDot.className = 'status-dot green';
    } else {
        apiStatusDot.className = 'status-dot red';
    }
}

// Save API Keys Pool
async function saveApiKeys() {
    try {
        const response = await fetch('/api/settings/keys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keys: keysPool })
        });
        const data = await response.json();
        updateApiStatus(data.keys_count, 'Loaded', data.breakdown);
        addLog(`Keys pool updated. Loaded ${data.keys_count} keys (${data.breakdown ? data.breakdown.summary : 'Active'}).`, 'success');
    } catch (e) {
        addLog(`Failed to save keys: ${e.message}`, 'error');
    }
}

// Get Engine Settings from form
function getSettings() {
    const apiType = document.querySelector('input[name="api-type"]:checked').value;
    const outputType = document.querySelector('input[name="output-type"]:checked').value;

    return {
        api_type: apiType,
        output_type: outputType,
        custom_output_path: outputPathText.textContent.replace('Path: ', ''),
        model: selectModel.value,

        target_lang: selectTargetLang.value,
        voice_gender: selectGender.value,
        tts_engine: document.querySelector('input[name="tts-engine-radio"]:checked')?.value || 'voxcpm2',
        hf_voxcpm_space: inputHfSpace ? inputHfSpace.value.trim() : 'openbmb/VoxCPM2',
        hf_token: inputHfToken ? inputHfToken.value.trim() : '',
        github_update_repo: inputGithubRepo ? inputGithubRepo.value.trim() : '',
        voxcpm_cfg: inputVoxcpmCfg ? parseFloat(inputVoxcpmCfg.value) : 3.0,
        voxcpm_ultimate_clone: chkVoxcpmUltimateClone ? chkVoxcpmUltimateClone.checked : false,
        max_speech_speed: inputMaxSpeechSpeed ? parseFloat(inputMaxSpeechSpeed.value) : 1.3,
        custom_male_voice_path: maleVoicePathText && maleVoicePathText.textContent !== 'Default Male Voice' ? maleVoicePathText.textContent : '',
        custom_female_voice_path: femaleVoicePathText && femaleVoicePathText.textContent !== 'Default Female Voice' ? femaleVoicePathText.textContent : '',
        custom_narrator_male_voice_path: narratorMaleVoicePathText && narratorMaleVoicePathText.textContent !== 'Default Narrator Male' ? narratorMaleVoicePathText.textContent : '',
        custom_narrator_female_voice_path: narratorFemaleVoicePathText && narratorFemaleVoicePathText.textContent !== 'Default Narrator Female' ? narratorFemaleVoicePathText.textContent : '',
        custom_young_male_voice_path: youngMaleVoicePathText && youngMaleVoicePathText.textContent !== 'Default Child Male Voice' ? youngMaleVoicePathText.textContent : '',
        custom_young_female_voice_path: youngFemaleVoicePathText && youngFemaleVoicePathText.textContent !== 'Default Child Female Voice' ? youngFemaleVoicePathText.textContent : '',
        custom_old_male_voice_path: oldMaleVoicePathText && oldMaleVoicePathText.textContent !== 'Default Elderly Male Voice' ? oldMaleVoicePathText.textContent : '',
        custom_old_female_voice_path: oldFemaleVoicePathText && oldFemaleVoicePathText.textContent !== 'Default Elderly Female Voice' ? oldFemaleVoicePathText.textContent : '',
        gpu_mode: selectGpu.value,
        concurrent_threads: parseInt(selectThreads.value, 10),
        audio_sync_offset_ms: selectSyncOffset ? parseFloat(selectSyncOffset.value) : 0,
        auto_shutdown_pc: chkAutoShutdown ? chkAutoShutdown.checked : false,
        delete_original_file: chkDeleteOriginal ? chkDeleteOriginal.checked : false,
        burn_subtitles: chkBurnSubtitles ? chkBurnSubtitles.checked : true,
        subtitle_box: chkSubtitleBox ? chkSubtitleBox.checked : true,
        sub_margin_v: inputSubMargin ? parseInt(inputSubMargin.value, 10) : 30,
        color_correction: chkColorCorrection ? chkColorCorrection.checked : false,
        movie_recap_mode: chkMovieRecap ? chkMovieRecap.checked : false,
        video_speed: (chkVideoSpeedAdjust && chkVideoSpeedAdjust.checked && inputVideoSpeed) ? parseFloat(inputVideoSpeed.value) : 1.0,
        youtube_multi_audio: chkYoutubeMultiAudio ? chkYoutubeMultiAudio.checked : false,
        youtube_extra_languages: Array.from(document.querySelectorAll('.chk-yt-lang:checked')).map(cb => cb.value),
        mixer: {
            drums: mixerDrums ? mixerDrums.checked : true,
            drums_vol: mixerDrumsVol ? parseInt(mixerDrumsVol.value, 10) : 100,
            bass: mixerBass ? mixerBass.checked : true,
            bass_vol: mixerBassVol ? parseInt(mixerBassVol.value, 10) : 100,
            melody: mixerMelody ? mixerMelody.checked : true,
            melody_vol: mixerMelodyVol ? parseInt(mixerMelodyVol.value, 10) : 100
        }
    };
}

// Save system configuration settings
async function saveSettings() {
    const settings = getSettings();
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        const data = await response.json();
        if (data.status === 'success') {
            addLog('System settings saved successfully.', 'success');
        }
    } catch (e) {
        addLog(`Failed to save settings: ${e.message}`, 'error');
    }
}

// Load Configuration Settings
async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        const data = await response.json();

        if (data.settings) {
            const s = data.settings;
            // Set values
            document.querySelector(`input[name="api-type"][value="${s.api_type}"]`).checked = true;
            document.querySelector(`input[name="output-type"][value="${s.output_type}"]`).checked = true;
            if (s.output_type === 'custom') {
                customPathContainer.style.display = 'flex';
                outputPathText.textContent = `Path: ${s.custom_output_path || 'Not Selected'}`;
            }
            selectModel.value = s.model;

            selectTargetLang.value = s.target_lang;
            selectGender.value = s.voice_gender;
            if (s.tts_engine) {
                const ttsRadio = document.querySelector(`input[name="tts-engine-radio"][value="${s.tts_engine}"]`);
                if (ttsRadio) ttsRadio.checked = true;
            }
            if (inputHfSpace && s.hf_voxcpm_space) {
                inputHfSpace.value = s.hf_voxcpm_space;
            }
            if (inputHfToken && s.hf_token) {
                inputHfToken.value = s.hf_token;
            }
            if (inputGithubRepo && s.github_update_repo) {
                inputGithubRepo.value = s.github_update_repo;
            }
            if (inputVoxcpmCfg && s.voxcpm_cfg !== undefined) {
                inputVoxcpmCfg.value = s.voxcpm_cfg;
                const badge = document.getElementById('voxcpm-cfg-badge');
                if (badge) badge.textContent = parseFloat(s.voxcpm_cfg).toFixed(1);
            }
            if (chkVoxcpmUltimateClone && s.voxcpm_ultimate_clone !== undefined) {
                chkVoxcpmUltimateClone.checked = s.voxcpm_ultimate_clone;
            }
            if (inputMaxSpeechSpeed && s.max_speech_speed !== undefined) {
                inputMaxSpeechSpeed.value = s.max_speech_speed;
                const badge = document.getElementById('speech-speed-badge');
                if (badge) badge.textContent = `${parseFloat(s.max_speech_speed).toFixed(2)}x`;
            }

            if (maleVoicePathText && s.custom_male_voice_path) {
                maleVoicePathText.textContent = s.custom_male_voice_path;
                maleVoicePathText.title = s.custom_male_voice_path;
            }
            if (femaleVoicePathText && s.custom_female_voice_path) {
                femaleVoicePathText.textContent = s.custom_female_voice_path;
                femaleVoicePathText.title = s.custom_female_voice_path;
            }
            if (narratorMaleVoicePathText && s.custom_narrator_male_voice_path) {
                narratorMaleVoicePathText.textContent = s.custom_narrator_male_voice_path;
                narratorMaleVoicePathText.title = s.custom_narrator_male_voice_path;
            }
            if (narratorFemaleVoicePathText && s.custom_narrator_female_voice_path) {
                narratorFemaleVoicePathText.textContent = s.custom_narrator_female_voice_path;
                narratorFemaleVoicePathText.title = s.custom_narrator_female_voice_path;
            }
            if (youngMaleVoicePathText && s.custom_young_male_voice_path) {
                youngMaleVoicePathText.textContent = s.custom_young_male_voice_path;
                youngMaleVoicePathText.title = s.custom_young_male_voice_path;
            }
            if (youngFemaleVoicePathText && s.custom_young_female_voice_path) {
                youngFemaleVoicePathText.textContent = s.custom_young_female_voice_path;
                youngFemaleVoicePathText.title = s.custom_young_female_voice_path;
            }
            if (oldMaleVoicePathText && s.custom_old_male_voice_path) {
                oldMaleVoicePathText.textContent = s.custom_old_male_voice_path;
                oldMaleVoicePathText.title = s.custom_old_male_voice_path;
            }
            if (oldFemaleVoicePathText && s.custom_old_female_voice_path) {
                oldFemaleVoicePathText.textContent = s.custom_old_female_voice_path;
                oldFemaleVoicePathText.title = s.custom_old_female_voice_path;
            }
            if (hfSpaceContainer) {
                const ttsRadios = document.querySelectorAll('input[name="tts-engine-radio"]');
                const updateVoxFields = () => {
                    const isVox = document.querySelector('input[name="tts-engine-radio"]:checked')?.value === 'voxcpm2';
                    hfSpaceContainer.style.display = isVox ? 'flex' : 'none';
                    if (hfTokenContainer) hfTokenContainer.style.display = isVox ? 'flex' : 'none';
                };
                updateVoxFields();
                ttsRadios.forEach(r => r.addEventListener('change', () => {
                    updateVoxFields();
                    saveSettings();
                }));
            }
            selectGpu.value = s.gpu_mode;
            selectThreads.value = s.concurrent_threads;
            if (selectSyncOffset && s.audio_sync_offset_ms !== undefined) {
                selectSyncOffset.value = String(s.audio_sync_offset_ms);
            }
            if (chkAutoShutdown && s.auto_shutdown_pc !== undefined) {
                chkAutoShutdown.checked = s.auto_shutdown_pc;
            }
            if (chkDeleteOriginal && s.delete_original_file !== undefined) {
                chkDeleteOriginal.checked = s.delete_original_file;
            }
            if (chkBurnSubtitles && s.burn_subtitles !== undefined) {
                chkBurnSubtitles.checked = s.burn_subtitles;
            }
            if (chkSubtitleBox && s.subtitle_box !== undefined) {
                chkSubtitleBox.checked = s.subtitle_box;
            }
            if (inputSubMargin && s.sub_margin_v !== undefined) {
                inputSubMargin.value = s.sub_margin_v;
            }
            if (chkColorCorrection && s.color_correction !== undefined) {
                chkColorCorrection.checked = s.color_correction;
            }
            if (chkMovieRecap && s.movie_recap_mode !== undefined) {
                chkMovieRecap.checked = s.movie_recap_mode;
            }
            if (chkVideoSpeedAdjust && s.video_speed !== undefined) {
                if (s.video_speed !== 1.0) {
                    chkVideoSpeedAdjust.checked = true;
                    if (videoSpeedOptions) videoSpeedOptions.style.display = 'block';
                    if (inputVideoSpeed) {
                        inputVideoSpeed.value = s.video_speed;
                        if (videoSpeedBadge) videoSpeedBadge.textContent = `${parseFloat(s.video_speed).toFixed(2)}x`;
                    }
                } else {
                    chkVideoSpeedAdjust.checked = false;
                    if (videoSpeedOptions) videoSpeedOptions.style.display = 'none';
                }
            }

            if (chkYoutubeMultiAudio && s.youtube_multi_audio !== undefined) {
                chkYoutubeMultiAudio.checked = s.youtube_multi_audio;
                if (youtubeLanguagesPanel) {
                    youtubeLanguagesPanel.style.display = s.youtube_multi_audio ? 'block' : 'none';
                }
            }
            if (Array.isArray(s.youtube_extra_languages)) {
                document.querySelectorAll('.chk-yt-lang').forEach(cb => {
                    cb.checked = s.youtube_extra_languages.includes(cb.value);
                });
            }
            updateYtDropdownSummary();

            if (s.mixer) {
                if (mixerDrums) mixerDrums.checked = s.mixer.drums !== undefined ? s.mixer.drums : true;
                if (mixerBass) mixerBass.checked = s.mixer.bass !== undefined ? s.mixer.bass : true;
                if (mixerMelody) mixerMelody.checked = s.mixer.melody !== undefined ? s.mixer.melody : true;

                if (mixerDrumsVol && s.mixer.drums_vol !== undefined) {
                    mixerDrumsVol.value = s.mixer.drums_vol;
                    if (drumsVolBadge) drumsVolBadge.textContent = `${s.mixer.drums_vol}%`;
                }
                if (mixerBassVol && s.mixer.bass_vol !== undefined) {
                    mixerBassVol.value = s.mixer.bass_vol;
                    if (bassVolBadge) bassVolBadge.textContent = `${s.mixer.bass_vol}%`;
                }
                if (mixerMelodyVol && s.mixer.melody_vol !== undefined) {
                    mixerMelodyVol.value = s.mixer.melody_vol;
                    if (melodyVolBadge) melodyVolBadge.textContent = `${s.mixer.melody_vol}%`;
                }
            }
        }

        if (data.keys) {
            keysPool = data.keys;
            updateApiStatus(keysPool.length, 'Loaded', data.breakdown);
        }
    } catch (e) {
        console.error('Failed to load settings', e);
    }
}

// Update YouTube Multi-Audio Dropdown Summary Text & Counter
function updateYtDropdownSummary() {
    const checkedBoxes = Array.from(document.querySelectorAll('.chk-yt-lang:checked'));
    const allBoxes = Array.from(document.querySelectorAll('.chk-yt-lang'));
    const count = checkedBoxes.length;

    // Toggle .selected class on parent .yt-lang-item
    allBoxes.forEach(cb => {
        const item = cb.closest('.yt-lang-item');
        if (item) {
            if (cb.checked) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        }
    });

    if (ytSelectedCountBadge) {
        if (count > 0) {
            ytSelectedCountBadge.textContent = count;
            ytSelectedCountBadge.style.display = 'inline-block';
        } else {
            ytSelectedCountBadge.style.display = 'none';
        }
    }

    if (ytDropdownLabel) {
        if (count === 0) {
            ytDropdownLabel.textContent = 'ជ្រើសរើសភាសាបន្ថែម (0 ភាសាត្រូវបានជ្រើសរើស / None Selected)';
        } else if (count === allBoxes.length) {
            ytDropdownLabel.textContent = `🌟 បានជ្រើសរើសគ្រប់ភាសាទាំងអស់ (${count} Languages Selected)`;
        } else if (count <= 2) {
            const names = checkedBoxes.map(cb => {
                const item = cb.closest('.yt-lang-item');
                return item ? item.querySelector('span:last-child')?.textContent?.trim() : cb.value;
            });
            ytDropdownLabel.textContent = `${names.join(', ')} (${count} Selected)`;
        } else {
            const firstTwo = checkedBoxes.slice(0, 2).map(cb => {
                const item = cb.closest('.yt-lang-item');
                return item ? item.querySelector('span:last-child')?.textContent?.trim() : cb.value;
            });
            ytDropdownLabel.textContent = `${firstTwo.join(', ')} +${count - 2} more (${count} Languages Selected)`;
        }
    }
}


// Toggle Processing UI state
function toggleProcessingUI(processing) {
    isProcessing = processing;

    if (processing) {
        btnStartProcessing.style.display = 'none';
        btnCancelProcessing.style.display = 'block';
        btnCancelProcessing.disabled = false;
    } else {
        btnStartProcessing.style.display = 'block';
        btnStartProcessing.innerHTML = `<i class="fa-solid fa-play"></i> START PROCESSING`;
        btnStartProcessing.className = 'btn btn-gold btn-lg btn-process';
        btnStartProcessing.disabled = false;

        btnCancelProcessing.style.display = 'none';
        btnCancelProcessing.disabled = true;
    }
}

// Start Processing Queue
async function startProcessing() {
    // First save settings
    await saveSettings();

    addLog('Starting auto-dubbing queue processing...', 'info');
    try {
        const response = await fetch('/api/process/start', { method: 'POST' });
        const data = await response.json();
        if (data.status === 'success') {
            toggleProcessingUI(true);
        } else {
            addLog(`Failed to start: ${data.message}`, 'error');
        }
    } catch (e) {
        addLog(`Start processing error: ${e.message}`, 'error');
    }
}

// Cancel / Stop Processing Queue
async function stopProcessing() {
    addLog('Cancelling processing queue...', 'warning');
    try {
        const response = await fetch('/api/process/stop', { method: 'POST' });
        const data = await response.json();
        if (data.status === 'success') {
            toggleProcessingUI(false);
            addLog('Processing cancelled by user.', 'warning');
        }
    } catch (e) {
        addLog(`Stop processing error: ${e.message}`, 'error');
    }
}

// ==========================================
// SUBTITLE & TIMELINE EDITOR CONTROLS
// ==========================================

let activeEditorVideoId = null;
let activeEditorVideoPath = null;
let activeSubtitles = [];
let timelineScale = 60; // Pixels per second (60px = 1s)
const segmentAudioPlayer = new Audio();

const editorModal = document.getElementById('editor-modal');
const editorVideo = document.getElementById('editor-video');
const editorVideoTitle = document.getElementById('editor-video-title');
const editorCurrentTime = document.getElementById('editor-current-time');
const editorDurationText = document.getElementById('editor-duration');
const editorTbody = document.getElementById('editor-tbody');
const trackSubtitlesContent = document.getElementById('track-subtitles-content');
const trackAudioContent = document.getElementById('track-audio-content');
const timelinePlayhead = document.getElementById('timeline-playhead');
const timelineRuler = document.getElementById('timeline-ruler');
const timelineWorkspace = document.getElementById('timeline-workspace');

// Format seconds into MM:SS.cc
function formatTimeMs(secs) {
    if (isNaN(secs)) return "00:00.00";
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = Math.floor(secs % 60).toString().padStart(2, '0');
    const c = Math.floor((secs % 1) * 100).toString().padStart(2, '0');
    return `${m}:${s}.${c}`;
}

// Sanitize Subtitles Timestamps (Prevents UI Table & Timeline Overlaps)
function sanitizeSubtitlesTimestamps(subtitles) {
    if (!subtitles || !Array.isArray(subtitles) || subtitles.length <= 1) return subtitles;

    subtitles.sort((a, b) => (parseFloat(a.start) || 0) - (parseFloat(b.start) || 0));

    for (let i = 0; i < subtitles.length - 1; i++) {
        let currStart = parseFloat(subtitles[i].start) || 0;
        let currEnd = parseFloat(subtitles[i].end) || 0;
        let nextStart = parseFloat(subtitles[i + 1].start) || 0;

        if (currEnd >= nextStart) {
            let maxCurrEnd = parseFloat((nextStart - 0.05).toFixed(2));
            let currDur = maxCurrEnd - currStart;

            if (currDur >= 0.6) {
                subtitles[i].end = maxCurrEnd;
            } else {
                subtitles[i].end = parseFloat((currStart + Math.max(0.6, currDur)).toFixed(2));
                subtitles[i + 1].start = parseFloat((subtitles[i].end + 0.05).toFixed(2));

                let nextEnd = parseFloat(subtitles[i + 1].end) || 0;
                if (parseFloat(subtitles[i + 1].start) >= nextEnd) {
                    subtitles[i + 1].end = parseFloat((parseFloat(subtitles[i + 1].start) + 1.2).toFixed(2));
                }
            }
        }
    }
    return subtitles;
}

// Open Editor Modal and Load Subtitles
async function openSubtitleEditor(videoId, name, path) {
    activeEditorVideoId = videoId;
    activeEditorVideoPath = path;
    activeSubtitles = [];

    editorVideoTitle.textContent = name;

    // Set video src to our stream endpoint (requesting dubbed audio if available)
    editorVideo.src = `/api/video/stream?path=${encodeURIComponent(path)}&dubbed=true&t=${Date.now()}`;
    editorVideo.load();

    // Show modal
    editorModal.style.display = 'flex';
    document.body.style.overflow = 'hidden'; // Disable scroll on main body

    addLog(`Loading subtitle data for: ${name}...`, 'info');

    try {
        const res = await fetch(`/api/subtitles/${videoId}`);
        const data = await res.json();

        if (data.subtitles) {
            activeSubtitles = data.subtitles;
            // Ensure each segment has ID
            activeSubtitles.forEach((s, idx) => {
                s.id = s.id || (idx + 1);
            });
            sanitizeSubtitlesTimestamps(activeSubtitles);
        }

        renderDialogueTable();

        // Wait for video metadata to render timeline ruler
        editorVideo.onloadedmetadata = () => {
            editorDurationText.textContent = formatTimeMs(editorVideo.duration);
            renderTimeline();
        };

        // If metadata is already loaded
        if (editorVideo.readyState >= 1) {
            editorDurationText.textContent = formatTimeMs(editorVideo.duration);
            renderTimeline();
        }

    } catch (err) {
        addLog(`Failed to load subtitles: ${err.message}`, 'error');
    }
}

// Close Editor Modal
function closeSubtitleEditor() {
    editorModal.style.display = 'none';
    document.body.style.overflow = 'auto';
    editorVideo.pause();
    editorVideo.src = '';
    activeEditorVideoId = null;
    activeEditorVideoPath = null;
    activeSubtitles = [];
}

// Render Dialogue Table
function renderDialogueTable() {
    editorTbody.innerHTML = '';

    if (activeSubtitles.length === 0) {
        editorTbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No subtitle dialogue found. Click "Add Row" to create new dialogue.</td></tr>';
        return;
    }

    // Sort activeSubtitles by start time
    activeSubtitles.sort((a, b) => parseFloat(a.start) - parseFloat(b.start));

    activeSubtitles.forEach((seg, index) => {
        const tr = document.createElement('tr');
        tr.id = `row-seg-${seg.id}`;
        tr.dataset.id = seg.id;

        let selectBg = 'rgba(0,0,0,0.3)';
        let selectColor = '#fff';
        if (seg.gender === 'male') {
            selectBg = '#add8e6'; // Light blue
            selectColor = '#000';
        } else if (seg.gender === 'female') {
            selectBg = '#ffb6c1'; // Light pink
            selectColor = '#000';
        }

        tr.innerHTML = `
            <td><input type="checkbox" class="seg-checkbox" data-id="${seg.id}"></td>
            <td style="color: var(--text-secondary); font-weight:bold;">${index + 1}</td>
            <td>
                <input type="number" step="0.01" min="0" class="time-input seg-start" data-id="${seg.id}" value="${parseFloat(seg.start).toFixed(2)}" style="width:75px;">
            </td>
            <td>
                <input type="number" step="0.01" min="0" class="time-input seg-end" data-id="${seg.id}" value="${parseFloat(seg.end).toFixed(2)}" style="width:75px;">
            </td>
            <td>
                <textarea rows="1" class="seg-translation" data-id="${seg.id}" style="resize: vertical; min-height: 28px;">${seg.translation || ''}</textarea>
            </td>
            <td class="audio-control-cell">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <select class="seg-voice-select" data-id="${seg.id}" style="width: 110px; padding: 4px 6px; font-size:13px; height:28px; background: ${selectBg}; border:1px solid var(--border-color); color: ${selectColor}; border-radius:3px; cursor:pointer; outline:none;">
                        <option value="female_lead" ${seg.gender === 'female' && seg.speaker_role !== 'narrator' && (!seg.age_group || seg.age_group === 'adult') ? 'selected' : ''}>ស្រី</option>
                        <option value="male_lead" ${seg.gender === 'male' && seg.speaker_role !== 'narrator' && (!seg.age_group || seg.age_group === 'adult') ? 'selected' : ''}>ប្រុស</option>
                        <option value="female_narrator" ${seg.gender === 'female' && seg.speaker_role === 'narrator' ? 'selected' : ''}>ស្រី (រៀបរាប់)</option>
                        <option value="male_narrator" ${seg.gender === 'male' && seg.speaker_role === 'narrator' ? 'selected' : ''}>ប្រុស (រៀបរាប់)</option>
                        <option value="young_female" ${seg.gender === 'female' && seg.age_group === 'young' ? 'selected' : ''}>ក្មេងស្រី</option>
                        <option value="young_male" ${seg.gender === 'male' && seg.age_group === 'young' ? 'selected' : ''}>ក្មេងប្រុស</option>
                        <option value="old_female" ${seg.gender === 'female' && seg.age_group === 'old' ? 'selected' : ''}>ចាស់ស្រី</option>
                        <option value="old_male" ${seg.gender === 'male' && seg.age_group === 'old' ? 'selected' : ''}>ចាស់ប្រុស</option>
                    </select>
                    <button class="btn-play-seg btn-xs" data-id="${seg.id}" title="Play preview" style="background:#28a745; color:#fff; border:none; padding:4px 8px; border-radius:3px; cursor:pointer; display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px;">
                        <i class="fa-solid fa-play"></i>
                    </button>
                    <button class="btn-redub-seg btn-xs" data-id="${seg.id}" title="Re-dub this line" style="background:#6c757d; color:#fff; border:none; padding:4px 8px; border-radius:3px; cursor:pointer; display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px;">
                        <i class="fa-solid fa-arrows-rotate"></i>
                    </button>
                    <div class="seg-loading-spinner" id="spinner-${seg.id}" style="display:none; color: var(--accent-gold); font-size:12px; margin-left:4px;">
                        <i class="fa-solid fa-circle-notch fa-spin"></i>
                    </div>
                </div>
            </td>
        `;

        // Add event listeners to input changes
        tr.querySelector('.seg-start').addEventListener('change', (e) => {
            const val = parseFloat(e.target.value) || 0;
            const s = activeSubtitles.find(x => x.id === seg.id);
            if (s) {
                s.start = val;
                renderTimeline();
            }
        });

        tr.querySelector('.seg-end').addEventListener('change', (e) => {
            const val = parseFloat(e.target.value) || 0;
            const s = activeSubtitles.find(x => x.id === seg.id);
            if (s) {
                s.end = val;
                renderTimeline();
            }
        });

        tr.querySelector('.seg-translation').addEventListener('input', (e) => {
            const s = activeSubtitles.find(x => x.id === seg.id);
            if (s) s.translation = e.target.value;
            const block = document.querySelector(`.timeline-block[data-id="${seg.id}"]`);
            if (block) block.textContent = e.target.value || s.text || '...';
        });

        // Voice select listener
        tr.querySelector('.seg-voice-select').addEventListener('change', (e) => {
            const val = e.target.value;
            const s = activeSubtitles.find(x => x.id === seg.id);

            if (val.includes('female')) {
                e.target.style.backgroundColor = '#ffb6c1';
                e.target.style.color = '#000';
            } else if (val.includes('male')) {
                e.target.style.backgroundColor = '#add8e6';
                e.target.style.color = '#000';
            }

            if (s) {
                if (val === 'female_lead') { s.gender = 'female'; s.speaker_role = 'lead_actor'; s.age_group = 'adult'; }
                else if (val === 'male_lead') { s.gender = 'male'; s.speaker_role = 'lead_actor'; s.age_group = 'adult'; }
                else if (val === 'female_narrator') { s.gender = 'female'; s.speaker_role = 'narrator'; s.age_group = 'adult'; }
                else if (val === 'male_narrator') { s.gender = 'male'; s.speaker_role = 'narrator'; s.age_group = 'adult'; }
                else if (val === 'young_female') { s.gender = 'female'; s.speaker_role = 'lead_actor'; s.age_group = 'young'; }
                else if (val === 'young_male') { s.gender = 'male'; s.speaker_role = 'lead_actor'; s.age_group = 'young'; }
                else if (val === 'old_female') { s.gender = 'female'; s.speaker_role = 'lead_actor'; s.age_group = 'old'; }
                else if (val === 'old_male') { s.gender = 'male'; s.speaker_role = 'lead_actor'; s.age_group = 'old'; }
            }
        });

        // Play segment listener
        tr.querySelector('.btn-play-seg').addEventListener('click', (e) => {
            e.stopPropagation();

            // Save in the background to not block the immediate user interaction needed for .play()
            saveTableInputsToMemory();
            saveSubtitlesData().then(saveOk => {
                if (!saveOk) {
                    addLog("Warning: Subtitle changes could not be saved to disk before preview.", "warning");
                }
            }).catch(console.error);

            try {
                const audioUrl = `/api/subtitles/${activeEditorVideoId}/audio/${seg.id}?t=${Date.now()}`;
                segmentAudioPlayer.src = audioUrl;
                segmentAudioPlayer.load();

                // Play immediately to satisfy browser autoplay policies
                const playPromise = segmentAudioPlayer.play();
                if (playPromise !== undefined) {
                    playPromise.catch(error => {
                        console.error("Audio playback error:", error);
                        addLog("No audio preview found for this segment. Please click the Re-dub (Refresh) button to generate it.", "warning");
                    });
                }

                addLog(`Playing segment ${index + 1} audio preview...`, 'info');
            } catch (err) {
                console.error(err);
                addLog("No audio preview found for this segment.", "warning");
            }
        });

        // Re-dub segment listener
        tr.querySelector('.btn-redub-seg').addEventListener('click', async (e) => {
            e.stopPropagation();
            saveTableInputsToMemory();

            const btnPlay = tr.querySelector('.btn-play-seg');
            const btnRedub = tr.querySelector('.btn-redub-seg');
            const spinner = tr.querySelector('.seg-loading-spinner');

            btnPlay.style.display = 'none';
            btnRedub.style.display = 'none';
            spinner.style.display = 'inline-block';

            addLog(`Re-dubbing segment ${index + 1} with new text and voice...`, 'info');

            try {
                const response = await fetch(`/api/subtitles/${activeEditorVideoId}/re-dub/${seg.id}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ segments: activeSubtitles })
                });
                const data = await response.json();

                if (data.status === 'success') {
                    addLog(`Segment ${index + 1} re-dubbed successfully!`, 'success');

                    // Update duration and end time dynamically in UI
                    if (data.audio_duration) {
                        seg.audio_duration = data.audio_duration;
                        seg.end = seg.start + data.audio_duration;
                        renderDialogueTable();
                        renderTimeline();
                    }

                    const audioUrl = `/api/subtitles/${activeEditorVideoId}/audio/${seg.id}?t=${Date.now()}`;
                    segmentAudioPlayer.src = audioUrl;
                    segmentAudioPlayer.load();
                    segmentAudioPlayer.play();
                } else {
                    addLog(`Failed to re-dub segment ${index + 1}.`, 'error');
                }
            } catch (err) {
                console.error(err);
                addLog(`Re-dub error: ${err.message}`, 'error');
            } finally {
                btnPlay.style.display = 'inline-flex';
                btnRedub.style.display = 'inline-flex';
                spinner.style.display = 'none';
            }
        });

        // Click row to jump video to start time
        tr.addEventListener('click', (e) => {
            if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA' && e.target.tagName !== 'SELECT' && !e.target.closest('button')) {
                const vs = getEditorVideoSpeed();
                editorVideo.currentTime = parseFloat(seg.start) / vs;
            }
        });

        editorTbody.appendChild(tr);
    });
}

function getEditorVideoSpeed() {
    const chk = document.getElementById('chk-video-speed-adjust');
    const inp = document.getElementById('input-video-speed');
    if (chk && chk.checked && inp) {
        return parseFloat(inp.value) || 1.0;
    }
    return 1.0;
}

// Render horizontal Timeline (ruler + blocks + mockup waveform)
function renderTimeline() {
    const vs = getEditorVideoSpeed();
    const duration = (editorVideo.duration || 60) * vs;
    const timelineWidth = Math.max(timelineWorkspace.clientWidth, duration * timelineScale);

    // Set width on tracks and ruler
    timelineRuler.style.width = `${timelineWidth}px`;
    document.querySelectorAll('.timeline-tracks, .timeline-track').forEach(el => {
        el.style.width = `${timelineWidth}px`;
    });

    // 1. Render Ruler Ticks (offset by 40px to align with tracks)
    timelineRuler.innerHTML = '';
    const majorInterval = 5; // Major ticks every 5 seconds
    const minorInterval = 1; // Minor ticks every 1 second

    for (let t = 0; t <= duration; t += minorInterval) {
        const isMajor = t % majorInterval === 0;
        const left = (t * timelineScale) + 40;

        const tick = document.createElement('div');
        tick.className = `timeline-tick ${isMajor ? 'major' : ''}`;
        tick.style.left = `${left}px`;
        timelineRuler.appendChild(tick);

        if (isMajor) {
            const label = document.createElement('div');
            label.className = 'timeline-tick-label';
            label.style.left = `${left}px`;
            label.textContent = formatTimeMs(t).split('.')[0]; // Only MM:SS
            timelineRuler.appendChild(label);
        }
    }

    // 2. Render Subtitle Blocks T1
    trackSubtitlesContent.innerHTML = '';
    activeSubtitles.forEach(seg => {
        const start = parseFloat(seg.start);
        const end = parseFloat(seg.end);
        const left = start * timelineScale;
        const width = (end - start) * timelineScale;

        const block = document.createElement('div');
        block.className = 'timeline-block';
        block.style.left = `${left}px`;
        block.style.width = `${Math.max(15, width)}px`;
        block.dataset.id = seg.id;
        block.textContent = seg.translation || seg.text || '...';
        block.title = `[${start.toFixed(2)}s - ${end.toFixed(2)}s] ${seg.translation || seg.text || ''}`;

        block.addEventListener('click', () => {
            editorVideo.currentTime = start / vs;
            highlightActiveSegment(seg.id);
        });

        trackSubtitlesContent.appendChild(block);
    });

    // 3. Render Mock Waveform A1 (offset by 40px wrapper)
    trackAudioContent.innerHTML = '';
    const waveformDiv = document.createElement('div');
    waveformDiv.className = 'timeline-waveform';
    waveformDiv.style.width = `${timelineWidth - 40}px`;
    waveformDiv.style.position = 'relative';
    waveformDiv.style.display = 'block';
    waveformDiv.style.height = '100%';

    const barWidth = 2;
    const barGap = 2;
    const stepPx = barWidth + barGap; // 4px steps
    const dt = stepPx / timelineScale; // time step per bar

    for (let t = 0; t < duration; t += dt) {
        const left = t * timelineScale;
        const bar = document.createElement('div');
        bar.className = 'waveform-bar';
        bar.style.position = 'absolute';
        bar.style.left = `${left}px`;
        bar.style.bottom = '4px';

        // Check if there is active dialogue at time t (using exact audio duration if available)
        const hasVoice = activeSubtitles.some(s => {
            const d = s.audio_duration || (parseFloat(s.end) - parseFloat(s.start));
            return t >= parseFloat(s.start) && t <= parseFloat(s.start) + d;
        });

        let h = 2; // Flat silent line by default

        if (hasVoice) {
            const i = Math.floor(left / stepPx);
            // Audio wave peak where sound is active
            h = 10 + Math.sin(i * 0.5) * 8 + Math.cos(i * 0.1) * 4;
            h = Math.max(6, Math.min(22, h)); // Fit nicely in compact 30px tracks
            bar.classList.add('active-wave');
        }
        bar.style.height = `${h}px`;
        waveformDiv.appendChild(bar);
    }
    trackAudioContent.appendChild(waveformDiv);

    // Refresh Playhead position
    updatePlayhead();
}

// Highlight the active subtitle segment in table and timeline
function highlightActiveSegment(activeId) {
    // Revert previous highlights
    document.querySelectorAll('.editor-table tr').forEach(r => r.classList.remove('active-row'));
    document.querySelectorAll('.timeline-block').forEach(b => b.classList.remove('active-block'));

    if (!activeId) return;

    const activeRow = document.getElementById(`row-seg-${activeId}`);
    if (activeRow) {
        activeRow.classList.add('active-row');
        // Scroll row into view smoothly if not visible
        activeRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    const activeBlock = document.querySelector(`.timeline-block[data-id="${activeId}"]`);
    if (activeBlock) {
        activeBlock.classList.add('active-block');
    }
}

// Update Playhead based on video time
function updatePlayhead() {
    if (!activeEditorVideoId) return;

    const vs = getEditorVideoSpeed();
    const curTimeOriginal = editorVideo.currentTime * vs;

    editorCurrentTime.textContent = formatTimeMs(curTimeOriginal);

    const left = curTimeOriginal * timelineScale;
    timelinePlayhead.style.left = `${left}px`;

    // Auto-scroll timeline to keep playhead in center
    const workspaceWidth = timelineWorkspace.clientWidth;
    const scrollLeft = left - (workspaceWidth / 2);
    if (editorVideo.paused === false || Math.abs(timelineWorkspace.scrollLeft - scrollLeft) > 150) {
        timelineWorkspace.scrollLeft = Math.max(0, scrollLeft);
    }

    // Find active subtitle segment
    const activeSeg = activeSubtitles.find(s => curTimeOriginal >= parseFloat(s.start) && curTimeOriginal <= parseFloat(s.end));
    if (activeSeg) {
        highlightActiveSegment(activeSeg.id);
    } else {
        highlightActiveSegment(null);
    }
}

// Sync video playhead loop
editorVideo.addEventListener('timeupdate', updatePlayhead);

// Timeline scrub functionality (click timeline to jump video)
timelineRuler.addEventListener('click', (e) => {
    const vs = getEditorVideoSpeed();
    const rect = timelineRuler.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickedSec = clickX / timelineScale;
    editorVideo.currentTime = Math.max(0, Math.min(editorVideo.duration, clickedSec / vs));
});

// Make video preview clickable to toggle play/pause
editorVideo.addEventListener('click', () => {
    if (editorVideo.paused) editorVideo.play();
    else editorVideo.pause();
});

// Close buttons
document.getElementById('btn-close-editor').addEventListener('click', closeSubtitleEditor);
document.getElementById('btn-cancel-editor').addEventListener('click', closeSubtitleEditor);

// Keyboard shortcuts inside modal
window.addEventListener('keydown', (e) => {
    if (editorModal.style.display === 'flex') {
        if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();
            if (editorVideo.paused) editorVideo.play();
            else editorVideo.pause();
        }
        if (e.code === 'Escape') {
            closeSubtitleEditor();
        }
    }
});

// Select All Checkbox
document.getElementById('editor-select-all').addEventListener('change', (e) => {
    const checked = e.target.checked;
    document.querySelectorAll('.seg-checkbox').forEach(cb => cb.checked = checked);
});

// Add Row Button
document.getElementById('editor-btn-add-row').addEventListener('click', () => {
    const nextId = activeSubtitles.length > 0 ? Math.max(...activeSubtitles.map(s => s.id)) + 1 : 1;
    const vs = getEditorVideoSpeed();
    const curTimeOriginal = editorVideo.currentTime * vs;

    const newSeg = {
        id: nextId,
        start: curTimeOriginal,
        end: curTimeOriginal + 2.5,
        text: 'New segment',
        translation: 'អត្ថបទបកប្រែថ្មី',
        gender: 'female',
        speaker_role: 'narrator',
        age_group: 'adult',
        emotion: 'dramatic'
    };

    activeSubtitles.push(newSeg);
    renderDialogueTable();
    renderTimeline();

    // Focus new row's translation
    setTimeout(() => {
        const el = document.querySelector(`.seg-translation[data-id="${nextId}"]`);
        if (el) el.focus();
    }, 100);
});

// Delete Selected Button
document.getElementById('editor-btn-delete-selected').addEventListener('click', () => {
    const checkedBoxes = document.querySelectorAll('.seg-checkbox:checked');
    if (checkedBoxes.length === 0) {
        alert('Please select at least one subtitle segment to delete.');
        return;
    }

    if (confirm(`Are you sure you want to delete the ${checkedBoxes.length} selected segment(s)?`)) {
        const idsToDelete = Array.from(checkedBoxes).map(cb => parseInt(cb.dataset.id));
        activeSubtitles = activeSubtitles.filter(s => !idsToDelete.includes(s.id));
        renderDialogueTable();
        renderTimeline();
        document.getElementById('editor-select-all').checked = false;
    }
});

// Save table fields into activeSubtitles memory array
function saveTableInputsToMemory() {
    const rows = document.querySelectorAll('#editor-tbody tr');
    for (let r of rows) {
        const id = parseInt(r.dataset.id);
        const originalSeg = activeSubtitles.find(x => x.id === id);
        if (!originalSeg) continue;

        originalSeg.start = parseFloat(r.querySelector('.seg-start').value) || 0;
        originalSeg.end = parseFloat(r.querySelector('.seg-end').value) || 0;
        originalSeg.translation = r.querySelector('.seg-translation').value;

        const voiceSelect = r.querySelector('.seg-voice-select');
        if (voiceSelect) {
            const val = voiceSelect.value;
            if (val === 'female_lead') { originalSeg.gender = 'female'; originalSeg.speaker_role = 'lead_actor'; originalSeg.age_group = 'adult'; }
            else if (val === 'male_lead') { originalSeg.gender = 'male'; originalSeg.speaker_role = 'lead_actor'; originalSeg.age_group = 'adult'; }
            else if (val === 'female_narrator') { originalSeg.gender = 'female'; originalSeg.speaker_role = 'narrator'; originalSeg.age_group = 'adult'; }
            else if (val === 'male_narrator') { originalSeg.gender = 'male'; originalSeg.speaker_role = 'narrator'; originalSeg.age_group = 'adult'; }
            else if (val === 'young_female') { originalSeg.gender = 'female'; originalSeg.speaker_role = 'lead_actor'; originalSeg.age_group = 'young'; }
            else if (val === 'young_male') { originalSeg.gender = 'male'; originalSeg.speaker_role = 'lead_actor'; originalSeg.age_group = 'young'; }
            else if (val === 'old_female') { originalSeg.gender = 'female'; originalSeg.speaker_role = 'lead_actor'; originalSeg.age_group = 'old'; }
            else if (val === 'old_male') { originalSeg.gender = 'male'; originalSeg.speaker_role = 'lead_actor'; originalSeg.age_group = 'old'; }
        }
    }
}

// Save Subtitles API call helper
async function saveSubtitlesData() {
    saveTableInputsToMemory();
    try {
        const res = await fetch(`/api/subtitles/${activeEditorVideoId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subtitles: activeSubtitles })
        });
        const data = await res.json();
        return data.status === 'success';
    } catch (err) {
        console.error(err);
        return false;
    }
}



// Click Save & Re-Dub button
document.getElementById('btn-save-redub-editor').addEventListener('click', async () => {
    addLog('Saving subtitle changes and scheduling re-dubbing...', 'info');
    const ok = await saveSubtitlesData();
    if (!ok) {
        addLog('Failed to save subtitles before re-dub.', 'error');
        alert('Failed to save subtitles. Cannot trigger re-dub.');
        return;
    }

    try {
        const res = await fetch(`/api/process/re-dub/${activeEditorVideoId}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            addLog('Fast Re-Dub pipeline queued successfully!', 'success');
            closeSubtitleEditor();
            // Refresh queue list UI
            await getQueue();
        } else {
            addLog('Failed to start re-dubbing pipeline.', 'error');
        }
    } catch (err) {
        addLog(`Re-dubbing trigger error: ${err.message}`, 'error');
    }
});


// --- KeyAuth Subscription Logic ---
document.addEventListener('DOMContentLoaded', () => {
    const activationModal = document.getElementById('activation-modal');
    const licenseInput = document.getElementById('license-key-input');
    const btnActivate = document.getElementById('btn-activate');
    const errorContainer = document.getElementById('activation-error');
    const errorText = document.getElementById('activation-error-text');

    // License protection removed - Ensure modal remains hidden
    if (activationModal) {
        activationModal.style.display = 'none';
    }
    const badge = document.getElementById('license-status-badge');
    if (badge) {
        badge.style.display = 'none';
    }

    btnActivate.addEventListener('click', () => {
        const key = licenseInput.value.trim();
        if (!key) {
            showError('Please enter a license key.');
            return;
        }

        btnActivate.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Verifying...';
        btnActivate.disabled = true;
        errorContainer.style.display = 'none';

        fetch('/api/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ license_key: key })
        })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    activationModal.style.display = 'none';

                    // Show badge
                    const badge = document.getElementById('license-status-badge');
                    const daysText = document.getElementById('license-days-text');
                    if (badge && daysText) {
                        badge.style.display = 'inline-block';
                        daysText.textContent = `License: ${data.days_left} Days Left`;
                    }

                    // Assumes a showToast function exists in app.js
                    if (typeof showToast !== 'undefined') showToast(data.message, 'success');
                } else {
                    showError(data.message || 'Invalid license key.');
                }
            })
            .catch(err => {
                showError('Network error connecting to server.');
            })
            .finally(() => {
                btnActivate.innerHTML = '<i class="fa-solid fa-unlock"></i> Activate Now';
                btnActivate.disabled = false;
            });
    });

    function showError(msg) {
        errorText.textContent = msg;
        errorContainer.style.display = 'flex';
    }
});
