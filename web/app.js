document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide Icons safely
    function safeReplaceLucide() {
        if (typeof lucide !== 'undefined') {
            try {
                if (typeof lucide.createIcons === 'function') {
                    lucide.createIcons();
                } else if (typeof lucide.replace === 'function') {
                    lucide.replace();
                }
            } catch (e) {
                console.error("Lucide replace error:", e);
            }
        }
    }
    safeReplaceLucide();

    // App State
    const state = {
        videoFile: null,
        videoPath: null,
        uploadJobId: null,
        videoObjectUrl: null,
        videoDuration: 0,
        videoHash: null,
        detectedLanguage: null,
        targetLanguage: 'es',
        status: 'idle', // idle, processing, waiting_for_review, burning, completed, error
        progress: 0,
        logs: [],
        segments: [],
        activeSegmentIndex: -1,
        cacheAction: 'reuse_all',
        editorUndo: [],
        editorRedo: [],
        editorIssues: { errors: [], warnings: [] },
        reviewIssues: [],
        reviewApproval: null,
        reviewFieldState: {},
        waveformPeaks: [],
        setupChecked: false,
        session: { isLocal: true, lanEnabled: false, clientPlatform: 'other', clientBrowser: 'other' },
        mobileAccessUrls: [],
        driveBatchId: null,
        driveReviewContext: null,
        driveInspection: null,
        driveConnected: false,
        isTauriDesktop: Boolean(window.__TAURI__?.dialog?.open),
        styleConfig: {
            font_name: 'Arial',
            font_size: 28,
            primary_color: '#FFFFFF',
            outline_color: '#000000',
            back_color: '#000000',
            bg_opacity: 70,
            outline_width: 1.5,
            shadow: 1,
            border_style: 1, // 1 = Outline, 3 = Opaque Box, 4 = Outline + Shadow
            alignment: 2, // 2 = Center, 1 = Left, 3 = Right
            margin_v: 30,
        }
    };

    // DOM Elements
    const tabButtons = document.querySelectorAll('.menu-item');
    const tabPanels = document.querySelectorAll('.tab-panel');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const themeToggleBtn = document.getElementById('btn-theme-toggle');
    
    // Upload Elements
    const dropzone = document.getElementById('video-dropzone');
    const videoInput = document.getElementById('video-input');
    const localPathInput = document.getElementById('local-path-input');
    const btnLoadLocalPath = document.getElementById('btn-load-local-path');
    const localPathDivider = document.getElementById('local-path-divider');
    const localPathGroup = document.getElementById('local-path-group');
    const fileDetails = document.getElementById('file-details');
    const detailName = document.getElementById('detail-name');
    const detailSize = document.getElementById('detail-size');
    const removeFileBtn = document.getElementById('btn-remove-file');
    const uploadPreviewContainer = document.getElementById('upload-preview-container');
    const uploadVideoPreview = document.getElementById('upload-video-preview');
    const btnStartProcess = document.getElementById('btn-start-process');
    const configForm = document.getElementById('config-form');
    const cacheActionModal = document.getElementById('cache-action-modal');
    const setupBanner = document.getElementById('setup-banner');
    const navConfig = document.getElementById('nav-config');
    const outputDirGroup = document.getElementById('output-dir-group');
    const btnMobileAccess = document.getElementById('btn-mobile-access');
    const mobileAccessModal = document.getElementById('mobile-access-modal');
    const mobileAccessQr = document.getElementById('mobile-access-qr');
    const mobileAccessUrl = document.getElementById('mobile-access-url');
    const mobileAccessState = document.getElementById('mobile-access-state');
    const mobileAccessIndicator = mobileAccessState.previousElementSibling;
    const mobileAccessQrFrame = mobileAccessQr.closest('.mobile-qr-frame');
    const mobileAccessIssues = document.getElementById('mobile-access-issues');
    const mobileAccessUnavailable = document.getElementById('mobile-access-unavailable');
    const mobileNetworkSelect = document.getElementById('mobile-network-select');
    const mobileNetworkLabel = document.getElementById('mobile-network-label');
    const btnCopyMobileUrl = document.getElementById('btn-copy-mobile-url');
    const btnRefreshMobileAccess = document.getElementById('btn-refresh-mobile-access');
    const btnRepairMobileAccess = document.getElementById('btn-repair-mobile-access');
    const mobilePlatformOptions = Array.from(document.querySelectorAll('[data-mobile-platform]'));
    const mobilePlatformHelp = document.getElementById('mobile-platform-help');
    const mobileDeviceStatus = document.getElementById('mobile-device-status');

    // Google Drive Batch Elements
    const driveConnectionState = document.getElementById('drive-connection-state');
    const driveAuthPanel = document.getElementById('drive-auth-panel');
    const driveClientPathGroup = document.getElementById('drive-client-path-group');
    const driveClientJsonPath = document.getElementById('drive-client-json-path');
    const btnDriveBrowseClient = document.getElementById('btn-drive-browse-client');
    const btnDriveConfigure = document.getElementById('btn-drive-configure');
    const btnDriveConnect = document.getElementById('btn-drive-connect');
    const btnDriveDisconnect = document.getElementById('btn-drive-disconnect');
    const driveAuthMessage = document.getElementById('drive-auth-message');
    const driveSourceFolder = document.getElementById('drive-source-folder');
    const driveDestinationFolder = document.getElementById('drive-destination-folder');
    const driveBatchTargetLanguage = document.getElementById('drive-batch-target-language');
    const driveBatchPipelineSummary = document.getElementById('drive-batch-pipeline-summary');
    const btnDriveInspect = document.getElementById('btn-drive-inspect');
    const btnDriveStart = document.getElementById('btn-drive-start');
    const driveFolderMessage = document.getElementById('drive-folder-message');
    const driveBatchTitle = document.getElementById('drive-batch-title');
    const driveBatchCounts = document.getElementById('drive-batch-counts');
    const driveBatchProgress = document.getElementById('drive-batch-progress');
    const driveBatchQueue = document.getElementById('drive-batch-queue');
    const driveOutputLink = document.getElementById('drive-output-link');
    const btnDriveResume = document.getElementById('btn-drive-resume');
    const btnDriveStop = document.getElementById('btn-drive-stop');

    // Style Editor Elements
    const stylePresets = document.getElementById('style-presets');
    const styleFont = document.getElementById('style-font');
    const styleSize = document.getElementById('style-size');
    const styleSizeVal = document.getElementById('style-size-val');
    const styleColorText = document.getElementById('style-color-text');
    const styleColorTextHex = document.getElementById('style-color-text-hex');
    const styleColorOutline = document.getElementById('style-color-outline');
    const styleColorOutlineHex = document.getElementById('style-color-outline-hex');
    const styleOutlineWidth = document.getElementById('style-outline-width');
    const styleOutlineWidthVal = document.getElementById('style-outline-width-val');
    const styleBorderStyle = document.getElementById('style-border-style');
    const groupColorBg = document.getElementById('group-color-bg');
    const styleColorBg = document.getElementById('style-color-bg');
    const styleColorBgHex = document.getElementById('style-color-bg-hex');
    const styleBgOpacity = document.getElementById('style-bg-opacity');
    const styleBgOpacityVal = document.getElementById('style-bg-opacity-val');
    const styleMarginV = document.getElementById('style-margin-v');
    const styleMarginVVal = document.getElementById('style-margin-v-val');
    const styleShadow = document.getElementById('style-shadow');
    const styleShadowVal = document.getElementById('style-shadow-val');
    const alignButtons = document.querySelectorAll('[data-align]');
    
    // Live Preview Elements
    const previewViewport = document.getElementById('preview-viewport');
    const previewVideo = document.getElementById('preview-video');
    const subtitlePreviewContainer = document.getElementById('subtitle-preview-container');
    const subtitleTextPreview = document.getElementById('subtitle-text-preview');
    const previewBtnPlay = document.getElementById('preview-btn-play');
    const previewPlaybackProgress = document.getElementById('preview-playback-progress');
    const previewProgressContainer = document.querySelector('.preview-progress-bar-container');

    // Progress Elements
    const progressPercentage = document.getElementById('progress-percentage');
    const progressStatusLabel = document.getElementById('progress-status-label');
    const radialProgressBar = document.getElementById('radial-progress-bar');
    const runningBadge = document.getElementById('running-badge');
    const logConsole = document.getElementById('log-console');
    const btnCopyLogs = document.getElementById('btn-copy-logs');
    
    // Editor Elements
    const editorVideo = document.getElementById('editor-video');
    const editorSubtitleOverlay = document.getElementById('editor-subtitle-overlay');
    const editorSubtitleOverlayText = document.getElementById('editor-subtitle-overlay-text');
    const segmentsContainer = document.getElementById('segments-container');
    const segmentCountText = document.getElementById('segment-count');
    const editorSearch = document.getElementById('editor-search');
    const btnBurnFinal = document.getElementById('btn-burn-final');
    const btnApproveDraft = document.getElementById('btn-approve-draft');
    const editorIssueFilter = document.getElementById('editor-issue-filter');
    const navEditor = document.getElementById('nav-editor');
    const navExport = document.getElementById('nav-export');
    const btnEditorUndo = document.getElementById('btn-editor-undo');
    const btnEditorRedo = document.getElementById('btn-editor-redo');
    const btnEditorInsert = document.getElementById('btn-editor-insert');
    const btnRetranslateSelected = document.getElementById('btn-retranslate-selected');
    const editorSaveState = document.getElementById('editor-save-state');
    const editorQualitySummary = document.getElementById('editor-quality-summary');
    const editorQualityIssues = document.getElementById('editor-quality-issues');
    const editorWaveform = document.getElementById('editor-waveform');
    const waveformStage = document.getElementById('waveform-stage');
    const timelineCues = document.getElementById('timeline-cues');
    const timelinePlayhead = document.getElementById('timeline-playhead');
    const timelineCurrentTime = document.getElementById('timeline-current-time');
    const timelineDuration = document.getElementById('timeline-duration');

    // Export Elements
    const finalVideo = document.getElementById('final-video');
    const btnDownloadVideo = document.getElementById('btn-download-video');
    const btnDownloadSRT = document.getElementById('btn-download-srt');
    const btnRestart = document.getElementById('btn-restart');

    // Polling Interval ID
    let pollIntervalId = null;

    // ==========================================
    // THEME TOGGLE
    // ==========================================
    themeToggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-theme');
        document.body.classList.toggle('light-theme');
        
        const isDark = document.body.classList.contains('dark-theme');
        document.getElementById('theme-icon-dark').style.display = isDark ? 'block' : 'none';
        document.getElementById('theme-icon-light').style.display = isDark ? 'none' : 'block';
    });

    // ==========================================
    // TAB NAVIGATION
    // ==========================================
    function switchTab(tabId) {
        tabButtons.forEach(btn => {
            if (btn.getAttribute('data-tab') === tabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        tabPanels.forEach(panel => {
            if (panel.id === `panel-${tabId}`) {
                panel.classList.add('active');
            } else {
                panel.classList.remove('active');
            }
        });

        // Update Header Title based on Tab
        const titles = {
            'upload': { title: 'Upload & Configuration', sub: 'Upload your video and configure transcription and translation settings.' },
            'style-editor': { title: 'Subtitle Style Editor', sub: 'Customize the visual appearance of your burned-in subtitles with live preview.' },
            'config': { title: 'API Settings & Keys', sub: 'Configure your API keys and select/add OpenAI profiles.' },
            'batch': { title: 'Google Drive Batch', sub: 'Process a Drive folder through the selected subtitle pipeline.' },
            'progress': { title: 'Pipeline Progress & Logs', sub: 'Track the real-time status of your subtitle generation process.' },
            'editor': { title: 'Review & Edit Subtitles', sub: 'Review the translated segments, adjust timings, and edit text before burning.' },
            'export': { title: 'Export & Download', sub: 'Your video is ready! Download the subtitled video or subtitle files.' }
        };

        if (titles[tabId]) {
            pageTitle.textContent = titles[tabId].title;
            pageSubtitle.textContent = titles[tabId].sub;
        }

        // Trigger layout resize/update for video elements if needed
        if (tabId === 'style-editor' || tabId === 'editor') {
            syncPreviewStyles();
        }
        if (tabId === 'batch') {
            renderDrivePipelineSummary();
            refreshDriveStatus();
        }
    }

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (!button.disabled) {
                switchTab(button.getAttribute('data-tab'));
            }
        });
    });

    function revokeVideoObjectUrl() {
        if (!state.videoObjectUrl) return;
        URL.revokeObjectURL(state.videoObjectUrl);
        state.videoObjectUrl = null;
    }

    function releaseUploadedVideoSession({ beacon = false } = {}) {
        if (!state.videoPath || !state.uploadJobId) return;
        const payload = JSON.stringify({
            video_path: state.videoPath,
            job_id: state.uploadJobId,
        });
        state.uploadJobId = null;

        if (beacon && navigator.sendBeacon) {
            navigator.sendBeacon(
                '/api/session/release-video',
                new Blob([payload], { type: 'application/json' }),
            );
            return;
        }
        fetch('/api/session/release-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: payload,
            keepalive: true,
        }).catch(error => console.warn('Could not release video session:', error));
    }

    function endCurrentVideoSession(options = {}) {
        releaseUploadedVideoSession(options);
        revokeVideoObjectUrl();
    }

    window.addEventListener('pagehide', event => {
        if (!event.persisted) endCurrentVideoSession({ beacon: true });
    });

    // ==========================================
    // DRAG AND DROP / UPLOAD FILE
    // ==========================================
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleSelectedFile(files[0]);
        }
    });

    videoInput.addEventListener('click', async (event) => {
        if (!state.isTauriDesktop) return;
        event.preventDefault();
        event.stopPropagation();
        try {
            const selectedPath = await window.__TAURI__.dialog.open({
                multiple: false,
                directory: false,
                title: 'Select a video',
                filters: [{
                    name: 'Video files',
                    extensions: ['mp4', 'mkv', 'mov', 'avi', 'webm', 'm4v'],
                }],
            });
            if (typeof selectedPath === 'string' && selectedPath) {
                await loadLocalVideoPath(selectedPath);
            }
        } catch (error) {
            console.error(error);
            alert('The desktop file picker could not open the selected video.');
        }
    }, true);

    videoInput.addEventListener('change', (e) => {
        if (state.isTauriDesktop) return;
        if (videoInput.files.length) {
            handleSelectedFile(videoInput.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please select a valid video file.');
            return;
        }

        endCurrentVideoSession();
        state.driveReviewContext = null;
        state.videoFile = file;
        state.videoPath = null;
        state.videoHash = null;
        
        // Show details
        detailName.textContent = file.name;
        detailSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
        dropzone.style.display = 'none';
        if (localPathDivider) localPathDivider.style.display = 'none';
        if (localPathGroup) localPathGroup.style.display = 'none';
        fileDetails.style.display = 'flex';

        // Load into previews
        const objectURL = URL.createObjectURL(file);
        state.videoObjectUrl = objectURL;
        
        uploadVideoPreview.src = objectURL;
        uploadPreviewContainer.style.display = 'block';
        
        previewVideo.src = objectURL;
        previewVideo.style.display = 'block';
        document.getElementById('preview-overlay-bg').style.display = 'none';
        
        editorVideo.src = objectURL;

        btnStartProcess.disabled = false;
        
        addLog(`[SYSTEM] Loaded video file: ${file.name} (${detailSize.textContent})`, 'info');
    }

    async function loadLocalVideoPath(pathVal) {
        pathVal = String(pathVal || '').trim();
        if (!pathVal) {
            alert('Please enter a valid local file path.');
            return;
        }

        try {
            const response = await fetch(`/api/check-local-path?path=${encodeURIComponent(pathVal)}`);
            const data = await response.json();
            if (!data.exists) {
                alert('File not found at the specified path. Make sure the path is correct.');
                return;
            }

            endCurrentVideoSession();
            // A desktop/local-path selection retains the original source path.
            // It must not be uploaded into the temporary C-drive job sandbox.
            state.driveReviewContext = null;
            state.videoFile = null;
            state.videoPath = data.filepath;

            // Show details
            detailName.textContent = data.filename;
            detailSize.textContent = (data.size_bytes / (1024 * 1024)).toFixed(2) + ' MB';

            dropzone.style.display = 'none';
            if (localPathDivider) localPathDivider.style.display = 'none';
            if (localPathGroup) localPathGroup.style.display = 'none';
            fileDetails.style.display = 'flex';

            // Load into video preview elements using streaming endpoint
            const videoUrl = `/api/video?path=${encodeURIComponent(data.filepath)}`;
            uploadVideoPreview.src = videoUrl;
            uploadPreviewContainer.style.display = 'block';

            previewVideo.src = videoUrl;
            previewVideo.style.display = 'block';
            document.getElementById('preview-overlay-bg').style.display = 'none';

            editorVideo.src = videoUrl;

            btnStartProcess.disabled = false;

            addLog(`[SYSTEM] Loaded local video file: ${data.filepath} (${detailSize.textContent})`, 'info');
        } catch (error) {
            console.error(error);
            alert('Error checking file path.');
        }
    }

    btnLoadLocalPath.addEventListener('click', () => {
        loadLocalVideoPath(localPathInput.value);
    });

    removeFileBtn.addEventListener('click', () => {
        endCurrentVideoSession();
        state.videoFile = null;
        state.videoPath = null;
        state.videoHash = null;
        videoInput.value = '';
        localPathInput.value = '';
        
        dropzone.style.display = 'block';
        if (localPathDivider) localPathDivider.style.display = 'flex';
        if (localPathGroup) localPathGroup.style.display = 'block';
        fileDetails.style.display = 'none';
        uploadPreviewContainer.style.display = 'none';
        uploadVideoPreview.src = '';
        
        previewVideo.src = '';
        previewVideo.style.display = 'none';
        document.getElementById('preview-overlay-bg').style.display = 'block';
        
        editorVideo.src = '';
        btnStartProcess.disabled = true;
        
        addLog(`[SYSTEM] Removed video file.`, 'info');
    });

    // ==========================================
    // STYLE EDITOR INTERACTIVE PREVIEW
    // ==========================================
    const presets = {
        'modern-white': {
            font_name: 'Outfit',
            font_size: 28,
            primary_color: '#FFFFFF',
            outline_color: '#000000',
            outline_width: 1.5,
            border_style: 1,
            shadow: 1,
            alignment: 2,
            margin_v: 30
        },
        'tiktok': {
            font_name: 'Impact',
            font_size: 38,
            primary_color: '#FFEA00',
            outline_color: '#000000',
            outline_width: 3.0,
            border_style: 4,
            shadow: 2,
            alignment: 2,
            margin_v: 60
        },
        'netflix': {
            font_name: 'Arial',
            font_size: 26,
            primary_color: '#FFFFFF',
            outline_color: '#000000',
            outline_width: 1.0,
            border_style: 4,
            shadow: 1,
            alignment: 2,
            margin_v: 24
        },
        'glassmorphic': {
            font_name: 'Outfit',
            font_size: 28,
            primary_color: '#FFFFFF',
            outline_color: '#8b5cf6',
            outline_width: 0.0,
            border_style: 3,
            back_color: '#0b0f19',
            bg_opacity: 50,
            shadow: 0,
            alignment: 2,
            margin_v: 30
        },
        'retro-yellow': {
            font_name: 'Trebuchet MS',
            font_size: 30,
            primary_color: '#FFFF00',
            outline_color: '#000000',
            outline_width: 2.0,
            border_style: 1,
            shadow: 2,
            alignment: 2,
            margin_v: 30
        }
    };

    function applyPreset(presetName) {
        if (presetName === 'custom' || !presets[presetName]) return;
        
        const preset = presets[presetName];
        Object.keys(preset).forEach(key => {
            state.styleConfig[key] = preset[key];
        });
        
        // Update form controls to match state
        styleFont.value = state.styleConfig.font_name;
        styleSize.value = state.styleConfig.font_size;
        styleSizeVal.textContent = state.styleConfig.font_size + 'px';
        
        styleColorText.value = state.styleConfig.primary_color;
        styleColorTextHex.value = state.styleConfig.primary_color;
        
        styleColorOutline.value = state.styleConfig.outline_color;
        styleColorOutlineHex.value = state.styleConfig.outline_color;
        
        styleOutlineWidth.value = state.styleConfig.outline_width;
        styleOutlineWidthVal.textContent = state.styleConfig.outline_width + 'px';
        
        styleBorderStyle.value = state.styleConfig.border_style;
        
        if (state.styleConfig.back_color) {
            styleColorBg.value = state.styleConfig.back_color;
            styleColorBgHex.value = state.styleConfig.back_color;
        }
        
        if (state.styleConfig.bg_opacity !== undefined) {
            styleBgOpacity.value = state.styleConfig.bg_opacity;
            styleBgOpacityVal.textContent = state.styleConfig.bg_opacity + '%';
        }
        
        styleMarginV.value = state.styleConfig.margin_v;
        styleMarginVVal.textContent = state.styleConfig.margin_v + 'px';
        
        styleShadow.value = state.styleConfig.shadow;
        styleShadowVal.textContent = state.styleConfig.shadow + 'px';

        alignButtons.forEach(btn => {
            if (parseInt(btn.getAttribute('data-align')) === state.styleConfig.alignment) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        syncPreviewStyles();
    }

    stylePresets.addEventListener('change', (e) => {
        applyPreset(e.target.value);
    });

    // Helper to convert Hex to RGBA for preview styling
    function hexToRgba(hex, opacityPercent) {
        let c;
        if(/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)){
            c= hex.substring(1).split('');
            if(c.length== 3){
                c= [c[0], c[0], c[1], c[1], c[2], c[2]];
            }
            c= '0x' + c.join('');
            return 'rgba('+[(c>>16)&255, (c>>8)&255, c&255].join(',')+','+(opacityPercent/100)+')';
        }
        return hex;
    }

    function applyStylesToOverlay(containerEl, textEl, config) {
        if (!containerEl || !textEl || !config) return;

        // Apply Font Family & Size
        textEl.style.fontFamily = `'${config.font_name}', sans-serif`;
        textEl.style.fontSize = `${config.font_size}px`;
        
        // Apply Text Color
        textEl.style.color = config.primary_color;
        
        // Apply Vertical Margin
        containerEl.style.bottom = `${config.margin_v}px`;
        
        // Reset classes and styles
        textEl.className = '';
        textEl.style.textShadow = 'none';
        textEl.style.backgroundColor = 'transparent';
        textEl.style.padding = '0';
        textEl.style.borderRadius = '0';

        // Apply Alignment
        if (config.alignment === 1) {
            containerEl.style.justifyContent = 'flex-start';
            textEl.style.textAlign = 'left';
        } else if (config.alignment === 3) {
            containerEl.style.justifyContent = 'flex-end';
            textEl.style.textAlign = 'right';
        } else {
            containerEl.style.justifyContent = 'center';
            textEl.style.textAlign = 'center';
        }

        // Border Style simulation
        if (config.border_style === 1) {
            // Outline Only
            textEl.classList.add('ass-style-1');
            const outlineColor = config.outline_color;
            const w = config.outline_width;
            if (w > 0) {
                textEl.style.textShadow = `
                    -${w}px -${w}px 0 ${outlineColor},  
                     ${w}px -${w}px 0 ${outlineColor},
                    -${w}px  ${w}px 0 ${outlineColor},
                     ${w}px  ${w}px 0 ${outlineColor},
                    -${w}px  0px   0 ${outlineColor},
                     ${w}px  0px   0 ${outlineColor},
                     0px   -${w}px 0 ${outlineColor},
                     0px    ${w}px 0 ${outlineColor}
                `;
            }
        } else if (config.border_style === 3) {
            // Opaque Box
            textEl.classList.add('ass-style-3');
            textEl.style.backgroundColor = hexToRgba(config.back_color, config.bg_opacity);
            textEl.style.padding = '6px 16px';
            textEl.style.borderRadius = '8px';
        } else if (config.border_style === 4) {
            // Outline & Shadow
            textEl.classList.add('ass-style-4');
            textEl.style.backgroundColor = hexToRgba(config.back_color, config.bg_opacity);
            textEl.style.padding = '6px 16px';
            textEl.style.borderRadius = '8px';
            const outlineColor = config.outline_color;
            const w = config.outline_width;
            const s = config.shadow;
            let shadowStr = '';
            if (w > 0) {
                shadowStr = `
                    -${w}px -${w}px 0 ${outlineColor},  
                     ${w}px -${w}px 0 ${outlineColor},
                    -${w}px  ${w}px 0 ${outlineColor},
                     ${w}px  ${w}px 0 ${outlineColor}
                `;
            }
            if (s > 0) {
                if (shadowStr) shadowStr += ',';
                shadowStr += ` ${s}px ${s}px ${s}px rgba(0,0,0,0.8)`;
            }
            textEl.style.textShadow = shadowStr || 'none';
        }
    }

    function syncPreviewStyles() {
        const config = state.styleConfig;
        
        // Apply to Style Customizer Preview
        applyStylesToOverlay(subtitlePreviewContainer, subtitleTextPreview, config);
        
        // Apply to Editor Video Sync Player Overlay
        applyStylesToOverlay(editorSubtitleOverlay, editorSubtitleOverlayText, config);

        // Show/hide background color group based on border style
        if (config.border_style === 3 || config.border_style === 4) {
            groupColorBg.style.display = 'block';
        } else {
            groupColorBg.style.display = 'none';
        }
    }

    // Bind Controls Events
    styleFont.addEventListener('change', (e) => { state.styleConfig.font_name = e.target.value; syncPreviewStyles(); });
    
    styleSize.addEventListener('input', (e) => {
        state.styleConfig.font_size = parseInt(e.target.value);
        styleSizeVal.textContent = e.target.value + 'px';
        syncPreviewStyles();
    });

    // Text Color Pickers
    styleColorText.addEventListener('input', (e) => {
        state.styleConfig.primary_color = e.target.value;
        styleColorTextHex.value = e.target.value;
        syncPreviewStyles();
    });
    styleColorTextHex.addEventListener('input', (e) => {
        if (e.target.value.startsWith('#') && e.target.value.length === 7) {
            state.styleConfig.primary_color = e.target.value;
            styleColorText.value = e.target.value;
            syncPreviewStyles();
        }
    });

    // Outline Color Pickers
    styleColorOutline.addEventListener('input', (e) => {
        state.styleConfig.outline_color = e.target.value;
        styleColorOutlineHex.value = e.target.value;
        syncPreviewStyles();
    });
    styleColorOutlineHex.addEventListener('input', (e) => {
        if (e.target.value.startsWith('#') && e.target.value.length === 7) {
            state.styleConfig.outline_color = e.target.value;
            styleColorOutline.value = e.target.value;
            syncPreviewStyles();
        }
    });

    // Outline Width
    styleOutlineWidth.addEventListener('input', (e) => {
        state.styleConfig.outline_width = parseFloat(e.target.value);
        styleOutlineWidthVal.textContent = e.target.value + 'px';
        syncPreviewStyles();
    });

    // Border Style Selection
    styleBorderStyle.addEventListener('change', (e) => {
        state.styleConfig.border_style = parseInt(e.target.value);
        syncPreviewStyles();
    });

    // Background Box Color Pickers
    styleColorBg.addEventListener('input', (e) => {
        state.styleConfig.back_color = e.target.value;
        styleColorBgHex.value = e.target.value;
        syncPreviewStyles();
    });
    styleColorBgHex.addEventListener('input', (e) => {
        if (e.target.value.startsWith('#') && e.target.value.length === 7) {
            state.styleConfig.back_color = e.target.value;
            styleColorBg.value = e.target.value;
            syncPreviewStyles();
        }
    });

    // Background Opacity
    styleBgOpacity.addEventListener('input', (e) => {
        state.styleConfig.bg_opacity = parseInt(e.target.value);
        styleBgOpacityVal.textContent = e.target.value + '%';
        syncPreviewStyles();
    });

    // Margin V
    styleMarginV.addEventListener('input', (e) => {
        state.styleConfig.margin_v = parseInt(e.target.value);
        styleMarginVVal.textContent = e.target.value + 'px';
        syncPreviewStyles();
    });

    // Shadow Depth
    styleShadow.addEventListener('input', (e) => {
        state.styleConfig.shadow = parseInt(e.target.value);
        styleShadowVal.textContent = e.target.value + 'px';
        syncPreviewStyles();
    });

    // Alignment
    alignButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            alignButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.styleConfig.alignment = parseInt(btn.getAttribute('data-align'));
            syncPreviewStyles();
        });
    });

    // Live Preview Playback Simulation
    let isPreviewPlaying = false;
    previewBtnPlay.addEventListener('click', () => {
        if (!state.videoFile) {
            alert('Please upload a video file first to play the preview.');
            return;
        }

        if (isPreviewPlaying) {
            previewVideo.pause();
            previewBtnPlay.innerHTML = '<i data-lucide="play" class="w-5 h-5"></i>';
        } else {
            previewVideo.play().catch(err => {
                console.error("Error playing video:", err);
            });
            previewBtnPlay.innerHTML = '<i data-lucide="pause" class="w-5 h-5"></i>';
        }
        isPreviewPlaying = !isPreviewPlaying;
        safeReplaceLucide();
    });

    previewVideo.addEventListener('timeupdate', () => {
        if (previewVideo.duration) {
            const pct = (previewVideo.currentTime / previewVideo.duration) * 100;
            previewPlaybackProgress.style.width = pct + '%';
            
            // Update time display
            const min = Math.floor(previewVideo.currentTime / 60).toString().padStart(2, '0');
            const sec = (previewVideo.currentTime % 60).toFixed(3).padStart(6, '0');
            document.querySelector('.preview-time').textContent = `${min}:${sec}`;
        }
    });

    previewProgressContainer.addEventListener('click', (e) => {
        if (previewVideo.duration) {
            const rect = previewProgressContainer.getBoundingClientRect();
            const clickPos = (e.clientX - rect.left) / rect.width;
            previewVideo.currentTime = clickPos * previewVideo.duration;
        }
    });

    // Initial styles sync
    applyPreset('modern-white');

    // ==========================================
    // PIPELINE API CALLS (REAL LOGIC)
    // ==========================================
    function addLog(message, type = 'info') {
        const line = document.createElement('div');
        line.className = `log-line ${type}`;
        
        // Format time
        const now = new Date();
        const timeStr = `[${now.toLocaleTimeString()}] `;
        
        line.textContent = timeStr + message;
        logConsole.appendChild(line);
        logConsole.scrollTop = logConsole.scrollHeight;
        
        state.logs.push({ time: now, text: message, type });
    }

    btnCopyLogs.addEventListener('click', () => {
        const logText = state.logs.map(l => `[${l.time.toLocaleTimeString()}] ${l.text}`).join('\n');
        navigator.clipboard.writeText(logText).then(() => {
            alert('Logs copied to clipboard!');
        });
    });

    function chooseCacheAction() {
        return new Promise(resolve => {
            const actionButtons = cacheActionModal.querySelectorAll('[data-cache-action]');
            const cancelButtons = cacheActionModal.querySelectorAll('[data-cache-action-cancel]');

            const finish = action => {
                cacheActionModal.hidden = true;
                actionButtons.forEach(button => button.removeEventListener('click', selectAction));
                cancelButtons.forEach(button => button.removeEventListener('click', cancel));
                document.removeEventListener('keydown', keydown);
                resolve(action);
            };
            const selectAction = event => finish(event.currentTarget.dataset.cacheAction);
            const cancel = () => finish(null);
            const keydown = event => {
                if (event.key === 'Escape') cancel();
            };

            actionButtons.forEach(button => button.addEventListener('click', selectAction));
            cancelButtons.forEach(button => button.addEventListener('click', cancel));
            document.addEventListener('keydown', keydown);
            cacheActionModal.hidden = false;
            actionButtons[0]?.focus();
            safeReplaceLucide();
        });
    }

    function startPipelineProcess() {
        if (state.pipelinePlanValid === false) {
            addLog('[ERROR] Resolve the pipeline configuration warning before starting.', 'error');
            return;
        }
        // Get form values
        const formDataObj = new FormData(configForm);
        const requestData = {
            video_path: state.videoPath,
            source_language: formDataObj.get('source_language') || null,
            target_language: formDataObj.get('target_language') || null,
            transcription_provider: formDataObj.get('transcription_provider'),
            transcription_model: formDataObj.get('transcription_model') || null,
            model_size: formDataObj.get('model_size'),
            timing_anchor_provider: formDataObj.get('timing_anchor_provider'),
            api_transcript_timing_mode: formDataObj.get('api_transcript_timing_mode'),
            translation_provider: formDataObj.get('translation_provider'),
            translation_model: formDataObj.get('translation_model') || null,
            subtitle_mode: formDataObj.get('subtitle_mode') || 'auto',
            tiktok_style: formDataObj.get('subtitle_mode') === 'tiktok',
            last_output_dir: formDataObj.get('last_output_dir'),
            job_id: state.uploadJobId,
            cache_action: 'reuse_all'
        };
        
        state.targetLanguage = requestData.target_language;
        
        // Before starting, check if the video has a cached transcription in the database
        fetch(`/api/video/check-cache?path=${encodeURIComponent(state.videoPath)}`)
            .then(res => {
                if (!res.ok) return { exists: false };
                return res.json();
            })
            .then(async cacheData => {
                state.videoHash = cacheData.hash || null;
                if (cacheData.exists) {
                    const cacheAction = await chooseCacheAction();
                    if (!cacheAction) {
                        btnStartProcess.disabled = false;
                        runningBadge.style.display = 'none';
                        switchTab('upload');
                        return;
                    }
                    requestData.cache_action = cacheAction;
                }
                state.cacheAction = requestData.cache_action;
                submitPipelineJob(requestData);
            })
            .catch(err => {
                console.error("Error checking video cache:", err);
                submitPipelineJob(requestData);
            });
    }

    function submitPipelineJob(requestData) {
        addLog(`[PIPELINE] Submitting generation job: Source=${requestData.source_language || 'Auto'}, Target=${requestData.target_language || 'None'}`, 'system');
        
        // Trigger pipeline start
        fetch('/api/process/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        })
        .then(res => {
            if (!res.ok) throw new Error('Failed to start pipeline processing');
            return res.json();
        })
        .then(data => {
            addLog(`[PIPELINE] Job started successfully. Monitoring progress...`, 'success');
            // Start Polling Status
            startPollingStatus();
        })
        .catch(err => {
            addLog(`[ERROR] ${err.message}`, 'error');
            updateProgress(0, 'Error');
            runningBadge.style.display = 'none';
            btnStartProcess.disabled = false;
        });
    }

    function uploadVideoFile(file) {
        return new Promise((resolve, reject) => {
            const request = new XMLHttpRequest();
            request.open('POST', '/api/upload');
            request.setRequestHeader('Content-Type', 'application/octet-stream');
            request.setRequestHeader('X-SubGen-Filename', encodeURIComponent(file.name));
            request.upload.addEventListener('progress', event => {
                if (!event.lengthComputable) return;
                const percent = Math.min(99, Math.round((event.loaded / event.total) * 100));
                updateProgress(percent, `Uploading video... ${percent}%`);
            });
            request.addEventListener('load', () => {
                let payload = {};
                try { payload = JSON.parse(request.responseText || '{}'); } catch (error) {}
                if (request.status >= 200 && request.status < 300) {
                    resolve(payload);
                } else {
                    reject(new Error(payload.error || `Upload failed (${request.status})`));
                }
            });
            request.addEventListener('error', () => reject(new Error('The upload connection was interrupted.')));
            request.addEventListener('abort', () => reject(new Error('Upload was cancelled.')));
            request.send(file);
        });
    }

    btnStartProcess.addEventListener('click', () => {
        if (!state.videoFile && !state.videoPath) return;
        
        // Switch to progress tab
        switchTab('progress');
        runningBadge.style.display = 'block';
        
        // Reset progress UI
        updateProgress(0, 'Initializing...');
        resetStagesUI();
        
        // Disable start button
        btnStartProcess.disabled = true;

        if (state.videoFile && !state.videoPath) {
            addLog(`[SYSTEM] Starting subtitle generation process...`, 'system');
            addLog(`[SYSTEM] Uploading video to local server...`, 'info');
            
            uploadVideoFile(state.videoFile)
            .then(data => {
                state.videoPath = data.filepath;
                state.uploadJobId = data.job_id;
                const uploadedLocation = state.session.isLocal ? data.filepath : data.filename;
                addLog(`[SYSTEM] Upload completed: ${uploadedLocation}`, 'success');
                startPipelineProcess();
            })
            .catch(err => {
                addLog(`[ERROR] ${err.message}`, 'error');
                updateProgress(0, 'Error');
                runningBadge.style.display = 'none';
                btnStartProcess.disabled = false;
            });
        } else {
            // The desktop original or current browser/mobile session copy is still available.
            addLog(`[SYSTEM] Starting subtitle generation process...`, 'system');
            addLog(`[SYSTEM] Using the active video session source (skipping duplicate upload)...`, 'success');
            startPipelineProcess();
        }
    });

    function updateProgress(percent, label) {
        state.progress = percent;
        progressPercentage.textContent = `${percent}%`;
        progressStatusLabel.textContent = label;
        
        // Radial progress calculations
        // Circumference is 2 * PI * r = 2 * 3.14159 * 54 = 339.292
        const circumference = 339.292;
        const offset = circumference - (percent / 100) * circumference;
        radialProgressBar.style.strokeDashoffset = offset;
    }

    function resetStagesUI() {
        const stages = ['stage-transcription', 'stage-alignment', 'stage-translation', 'stage-burning'];
        stages.forEach(id => {
            const el = document.getElementById(id);
            el.className = 'stage-item pending';
            el.querySelector('.icon-pending').style.display = 'block';
            el.querySelector('.icon-running').style.display = 'none';
            el.querySelector('.icon-success').style.display = 'none';
        });
    }

    function setStageState(stageId, status) {
        // status: pending, running, completed
        const el = document.getElementById(stageId);
        if (!el) return;
        
        el.className = `stage-item ${status}`;
        
        const iconPending = el.querySelector('.icon-pending');
        const iconRunning = el.querySelector('.icon-running');
        const iconSuccess = el.querySelector('.icon-success');
        
        if (status === 'pending') {
            iconPending.style.display = 'block';
            iconRunning.style.display = 'none';
            iconSuccess.style.display = 'none';
        } else if (status === 'running') {
            iconPending.style.display = 'none';
            iconRunning.style.display = 'block';
            iconSuccess.style.display = 'none';
        } else if (status === 'completed') {
            iconPending.style.display = 'none';
            iconRunning.style.display = 'none';
            iconSuccess.style.display = 'block';
        }
    }

    function startPollingStatus() {
        if (pollIntervalId) clearInterval(pollIntervalId);
        
        pollIntervalId = setInterval(() => {
            fetch('/api/process/status')
                .then(res => res.json())
                .then(data => {
                    // Update state
                    state.status = data.status;
                    
                    // Add new logs
                    if (data.new_logs && data.new_logs.length > 0) {
                        data.new_logs.forEach(log => {
                            addLog(log.text, log.type);
                        });
                    }
                    
                    // Update stages & progress based on server state
                    updateProgress(data.progress, data.status_label || data.status);
                    
                    // Stage highlights
                    if (data.stage === 'transcription') {
                        setStageState('stage-transcription', 'running');
                    } else if (data.stage === 'alignment') {
                        setStageState('stage-transcription', 'completed');
                        setStageState('stage-alignment', 'running');
                    } else if (data.stage === 'translation') {
                        setStageState('stage-transcription', 'completed');
                        setStageState('stage-alignment', 'completed');
                        setStageState('stage-translation', 'running');
                    } else if (data.stage === 'burning') {
                        setStageState('stage-transcription', 'completed');
                        setStageState('stage-alignment', 'completed');
                        setStageState('stage-translation', 'completed');
                        setStageState('stage-burning', 'running');
                    }
                    
                    // Handle transitions
                    if (data.status === 'waiting_for_review') {
                        clearInterval(pollIntervalId);
                        runningBadge.style.display = 'none';
                        addLog(`[SYSTEM] Subtitles generated and translated. Ready for review!`, 'success');
                        
                        setStageState('stage-transcription', 'completed');
                        setStageState('stage-alignment', 'completed');
                        setStageState('stage-translation', 'completed');
                        
                        state.segments = data.segments;
                        initializeEditor();
                        
                        // Enable review tab
                        navEditor.disabled = false;
                        switchTab('editor');
                    } else if (data.status === 'completed') {
                        clearInterval(pollIntervalId);
                        runningBadge.style.display = 'none';
                        addLog(`[SYSTEM] Burning complete! Video rendered successfully.`, 'success');
                        
                        setStageState('stage-burning', 'completed');
                        
                        // Set up export screen
                        setupExportScreen(data.output_video, data.final_srt);
                        
                        // Enable export tab
                        navExport.disabled = false;
                        switchTab('export');
                    } else if (data.status === 'error') {
                        clearInterval(pollIntervalId);
                        runningBadge.style.display = 'none';
                        btnStartProcess.disabled = false;
                        addLog(`[ERROR] Process terminated due to errors.`, 'error');
                    }
                })
                .catch(err => {
                    console.error('Error polling status:', err);
                });
        }, 1000);
    }

    // ==========================================
    // INTERACTIVE SUBTITLE SEGMENT EDITOR
    // ==========================================
    function parseTimeToSeconds(timeStr) {
        // Formats: "00:00:12,400" or "00:12.400"
        const parts = timeStr.replace(',', '.').split(':');
        let secs = 0;
        if (parts.length === 3) {
            secs = parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseFloat(parts[2]);
        } else if (parts.length === 2) {
            secs = parseInt(parts[0]) * 60 + parseFloat(parts[1]);
        } else {
            secs = parseFloat(parts[0]);
        }
        return secs;
    }

    function formatSecondsToTime(seconds) {
        const hrs = Math.floor(seconds / 3600).toString().padStart(2, '0');
        const mins = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
        const secs = (seconds % 60).toFixed(3).padStart(6, '0').replace('.', ',');
        return `${hrs}:${mins}:${secs}`;
    }

    function escapeHtml(text) {
        if (!text) return '';
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    let editorAutosaveTimer = null;

    function cloneSegments() {
        return JSON.parse(JSON.stringify(state.segments));
    }

    function normalizeEditorSegments() {
        state.segments.sort((a, b) => parseTimeToSeconds(a.start) - parseTimeToSeconds(b.start));
        state.segments.forEach((segment, index) => {
            segment.index = index + 1;
            segment.translation = segment.translation ?? segment.text ?? '';
            segment.text = segment.text ?? '';
        });
    }

    function snapshotEditor() {
        return JSON.stringify(state.segments);
    }

    function pushUndoSnapshot(snapshot = snapshotEditor()) {
        if (state.editorUndo[state.editorUndo.length - 1] !== snapshot) {
            state.editorUndo.push(snapshot);
            if (state.editorUndo.length > 100) state.editorUndo.shift();
        }
        state.editorRedo = [];
        updateEditorHistoryButtons();
    }

    function restoreEditorSnapshot(snapshot) {
        state.segments = JSON.parse(snapshot);
        normalizeEditorSegments();
        renderEditor();
        scheduleEditorAutosave();
    }

    function updateEditorHistoryButtons() {
        btnEditorUndo.disabled = state.editorUndo.length === 0;
        btnEditorRedo.disabled = state.editorRedo.length === 0;
    }

    function mutateEditor(mutator) {
        pushUndoSnapshot();
        mutator();
        state.reviewApproval = null;
        normalizeEditorSegments();
        renderEditor();
        scheduleEditorAutosave();
    }

    function validateEditorSegments() {
        const errors = [];
        const warnings = [];
        let previousEnd = 0;
        state.segments.forEach((segment, position) => {
            const start = parseTimeToSeconds(segment.start);
            const end = parseTimeToSeconds(segment.end);
            const cue = position + 1;
            if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
                errors.push({ index: cue, code: 'invalid_range', message: 'End must be after start.' });
                return;
            }
            if (position > 0 && start < previousEnd - 0.001) {
                errors.push({ index: cue, code: 'overlap', message: 'Overlaps the previous cue.' });
            }
            const duration = end - start;
            const text = (segment.translation || segment.text || '').trim();
            if (!text) warnings.push({ index: cue, code: 'empty_text', message: 'Subtitle text is empty.' });
            if (duration < 0.35) warnings.push({ index: cue, code: 'short_duration', message: 'Visible for less than 350 ms.' });
            if (duration > 8) warnings.push({ index: cue, code: 'long_duration', message: 'Visible for more than 8 seconds.' });
            if (text && text.length / Math.max(duration, 0.001) > 24) {
                warnings.push({ index: cue, code: 'reading_speed', message: 'Reading speed is high.' });
            }
            previousEnd = end;
        });
        state.editorIssues = { errors, warnings };
        return state.editorIssues;
    }

    function renderEditorQuality() {
        const { errors, warnings } = validateEditorSegments();
        const filter = editorIssueFilter?.value || 'all';
        const reviewIssues = (state.reviewIssues || []).filter(issue =>
            filter === 'all'
            || (filter === 'clean' ? false : issue.severity === filter)
        );
        const blocking = reviewIssues.filter(issue => issue.blocking && !['corrected', 'accepted', 'dismissed_with_reason'].includes(issue.status));
        btnApproveDraft.disabled = errors.length > 0 || blocking.length > 0 || state.segments.length === 0;
        btnBurnFinal.disabled = !state.reviewApproval || errors.length > 0 || blocking.length > 0 || state.segments.length === 0;
        editorQualitySummary.textContent = blocking.length
            ? `${blocking.length} blocking review ${blocking.length === 1 ? 'issue' : 'issues'}`
            : errors.length
            ? `${errors.length} timing ${errors.length === 1 ? 'error' : 'errors'}`
            : reviewIssues.length
                ? `${reviewIssues.length} review ${reviewIssues.length === 1 ? 'issue' : 'issues'}`
            : warnings.length
                ? `${warnings.length} ${warnings.length === 1 ? 'warning' : 'warnings'}`
                : 'No timing errors';
        const deterministic = [
            ...errors.map(issue => ({...issue, deterministic: true, severity: 'critical'})),
            ...warnings.map(issue => ({...issue, deterministic: true, severity: 'warning'})),
        ];
        editorQualityIssues.innerHTML = [...reviewIssues, ...deterministic].slice(0, 30).map(issue => {
            const cueIndex = issue.index || state.segments.findIndex(segment => (issue.affected_cue_ids || []).includes(segment.id)) + 1;
            const severity = issue.severity || (errors.includes(issue) ? 'critical' : 'warning');
            const action = issue.id && issue.status === 'unresolved'
                ? `<button type="button" class="issue-resolve" data-issue-id="${escapeHtml(issue.id)}" data-issue-severity="${severity}">${severity === 'critical' ? 'Mark corrected' : 'Accept'}</button>`
                : '';
            return `<div class="quality-issue ${severity === 'critical' ? 'error' : ''}" data-cue-index="${cueIndex > 0 ? cueIndex : ''}" data-start-seconds="${issue.start_seconds ?? ''}">` +
                `<button type="button" class="issue-jump"><span>${severity}</span>${escapeHtml(issue.message)}</button>${action}</div>`;
        }).join('');
        editorQualityIssues.querySelectorAll('[data-cue-index]').forEach(button => {
            button.querySelector('.issue-jump')?.addEventListener('click', () => {
                const cueIndex = Number(button.dataset.cueIndex);
                const segment = state.segments[cueIndex - 1];
                if (segment) {
                    editorVideo.currentTime = parseTimeToSeconds(segment.start);
                    highlightActiveSegment(segment.index);
                } else if (button.dataset.startSeconds) {
                    editorVideo.currentTime = Number(button.dataset.startSeconds);
                }
            });
        });
        editorQualityIssues.querySelectorAll('.issue-resolve').forEach(button => {
            button.addEventListener('click', () => resolveReviewIssue(button.dataset.issueId, button.dataset.issueSeverity));
        });
    }

    function renderSegments(filterQuery = editorSearch.value || '') {
        segmentsContainer.innerHTML = '';
        const query = filterQuery.toLowerCase();
        const filtered = state.segments.filter(segment =>
            (segment.text || '').toLowerCase().includes(query)
            || (segment.translation || '').toLowerCase().includes(query)
        );
        segmentCountText.textContent = state.segments.length;
        if (!filtered.length) {
            segmentsContainer.innerHTML = `<div class="empty-state py-12 text-center"><h4>No cues match</h4></div>`;
            return;
        }

        filtered.forEach(segment => {
            const card = document.createElement('article');
            card.className = `segment-card${state.activeSegmentIndex === segment.index ? ' active' : ''}`;
            card.dataset.index = segment.index;
            const issues = [...state.editorIssues.errors, ...state.editorIssues.warnings]
                .filter(issue => issue.index === segment.index);
            card.innerHTML = `
                <div class="segment-header">
                    <span class="segment-index"><input class="segment-retranslate-select" type="checkbox" aria-label="Select cue ${segment.index} for retranslation"> #${segment.index}</span>
                    <div class="segment-actions">
                        <button class="segment-btn-play" title="Play cue" type="button"><i data-lucide="play"></i></button>
                        <button class="segment-btn-split" title="Split cue" type="button"><i data-lucide="scissors"></i></button>
                        <button class="segment-btn-merge" title="Merge with next cue" type="button"><i data-lucide="combine"></i></button>
                        <button class="segment-btn-delete" title="Delete cue" type="button"><i data-lucide="trash-2"></i></button>
                    </div>
                </div>
                <div class="segment-time-editor">
                    <button type="button" class="time-nudge" data-edge="start" data-delta="-0.1" title="Start 100 ms earlier"><i data-lucide="chevron-left"></i></button>
                    <input class="segment-time-input" data-edge="start" value="${escapeHtml(segment.start)}" aria-label="Cue ${segment.index} start time">
                    <button type="button" class="time-nudge" data-edge="start" data-delta="0.1" title="Start 100 ms later"><i data-lucide="chevron-right"></i></button>
                    <span class="time-separator">to</span>
                    <button type="button" class="time-nudge" data-edge="end" data-delta="-0.1" title="End 100 ms earlier"><i data-lucide="chevron-left"></i></button>
                    <input class="segment-time-input" data-edge="end" value="${escapeHtml(segment.end)}" aria-label="Cue ${segment.index} end time">
                    <button type="button" class="time-nudge" data-edge="end" data-delta="0.1" title="End 100 ms later"><i data-lucide="chevron-right"></i></button>
                </div>
                <div class="segment-language-grid">
                    <label><span>Source ${state.reviewFieldState?.translation_stale ? '<em>translation stale</em>' : ''}</span><textarea class="segment-text-source-input" dir="auto" rows="3">${escapeHtml(segment.text || '')}</textarea></label>
                    <label><span>Translation</span><textarea class="segment-text-translated-input" dir="auto" rows="3">${escapeHtml(segment.translation || '')}</textarea></label>
                </div>
                ${segment.provenance ? `<details class="segment-provenance"><summary>Provenance</summary><pre>${escapeHtml(JSON.stringify(segment.provenance, null, 2))}</pre></details>` : ''}
                ${issues.length ? `<div class="segment-issues">${issues.map(issue => escapeHtml(issue.message)).join(' ')}</div>` : ''}
            `;

            const initialTextSnapshot = { value: null };
            const sourceTextarea = card.querySelector('.segment-text-source-input');
            sourceTextarea.addEventListener('focus', () => { initialTextSnapshot.value = snapshotEditor(); });
            sourceTextarea.addEventListener('input', event => {
                segment.text = event.target.value;
                state.reviewApproval = null;
                state.reviewFieldState.translation_stale = Boolean(state.targetLanguage && state.targetLanguage !== state.detectedLanguage);
                renderEditorQuality();
                scheduleEditorAutosave();
            });
            sourceTextarea.addEventListener('blur', () => {
                if (initialTextSnapshot.value && initialTextSnapshot.value !== snapshotEditor()) pushUndoSnapshot(initialTextSnapshot.value);
            });
            const textarea = card.querySelector('.segment-text-translated-input');
            textarea.addEventListener('focus', () => { initialTextSnapshot.value = snapshotEditor(); });
            textarea.addEventListener('input', event => {
                segment.translation = event.target.value;
                state.reviewApproval = null;
                if (state.activeSegmentIndex === segment.index) {
                    editorSubtitleOverlayText.textContent = segment.translation || segment.text;
                }
                renderEditorQuality();
                scheduleEditorAutosave();
            });
            textarea.addEventListener('blur', () => {
                if (initialTextSnapshot.value && initialTextSnapshot.value !== snapshotEditor()) {
                    pushUndoSnapshot(initialTextSnapshot.value);
                }
            });

            card.querySelectorAll('.segment-time-input').forEach(input => {
                let beforeEdit = null;
                input.addEventListener('focus', () => { beforeEdit = snapshotEditor(); });
                input.addEventListener('change', event => {
                    const seconds = parseTimeToSeconds(event.target.value);
                    if (Number.isFinite(seconds)) segment[event.target.dataset.edge] = formatSecondsToTime(Math.max(0, seconds));
                    if (beforeEdit && beforeEdit !== snapshotEditor()) pushUndoSnapshot(beforeEdit);
                    renderEditor();
                    scheduleEditorAutosave();
                });
            });

            card.querySelectorAll('.time-nudge').forEach(button => {
                button.addEventListener('click', () => mutateEditor(() => {
                    const edge = button.dataset.edge;
                    const value = parseTimeToSeconds(segment[edge]) + Number(button.dataset.delta);
                    segment[edge] = formatSecondsToTime(Math.max(0, value));
                }));
            });
            card.querySelector('.segment-btn-play').addEventListener('click', () => playEditorSegment(segment));
            card.querySelector('.segment-btn-split').addEventListener('click', () => splitEditorSegment(segment.index));
            card.querySelector('.segment-btn-merge').addEventListener('click', () => mergeEditorSegment(segment.index));
            card.querySelector('.segment-btn-delete').addEventListener('click', () => deleteEditorSegment(segment.index));
            card.addEventListener('click', event => {
                if (event.target.closest('button, input, textarea')) return;
                editorVideo.currentTime = parseTimeToSeconds(segment.start);
                highlightActiveSegment(segment.index);
            });
            segmentsContainer.appendChild(card);
        });
        safeReplaceLucide();
    }

    function playEditorSegment(segment) {
        const end = parseTimeToSeconds(segment.end);
        editorVideo.currentTime = parseTimeToSeconds(segment.start);
        editorVideo.play();
        const stop = () => {
            if (editorVideo.currentTime >= end) {
                editorVideo.pause();
                editorVideo.removeEventListener('timeupdate', stop);
            }
        };
        editorVideo.addEventListener('timeupdate', stop);
    }

    function splitTextAtMiddle(text) {
        const words = String(text || '').trim().split(/\s+/).filter(Boolean);
        const middle = Math.ceil(words.length / 2);
        return [words.slice(0, middle).join(' '), words.slice(middle).join(' ')];
    }

    function splitEditorSegment(index) {
        const position = state.segments.findIndex(segment => segment.index === index);
        if (position < 0) return;
        mutateEditor(() => {
            const segment = state.segments[position];
            const start = parseTimeToSeconds(segment.start);
            const end = parseTimeToSeconds(segment.end);
            const playhead = editorVideo.currentTime;
            const split = playhead > start + 0.2 && playhead < end - 0.2 ? playhead : (start + end) / 2;
            const [sourceLeft, sourceRight] = splitTextAtMiddle(segment.text);
            const [targetLeft, targetRight] = splitTextAtMiddle(segment.translation);
            segment.end = formatSecondsToTime(split);
            segment.text = sourceLeft;
            segment.translation = targetLeft;
            state.segments.splice(position + 1, 0, {
                index: index + 1,
                start: formatSecondsToTime(split),
                end: formatSecondsToTime(end),
                text: sourceRight,
                translation: targetRight,
            });
        });
    }

    function mergeEditorSegment(index) {
        const position = state.segments.findIndex(segment => segment.index === index);
        if (position < 0 || position >= state.segments.length - 1) return;
        mutateEditor(() => {
            const current = state.segments[position];
            const next = state.segments[position + 1];
            current.end = next.end;
            current.text = `${current.text || ''} ${next.text || ''}`.trim();
            current.translation = `${current.translation || ''} ${next.translation || ''}`.trim();
            state.segments.splice(position + 1, 1);
        });
    }

    function deleteEditorSegment(index) {
        mutateEditor(() => {
            state.segments = state.segments.filter(segment => segment.index !== index);
            if (state.activeSegmentIndex === index) state.activeSegmentIndex = -1;
        });
    }

    function insertEditorSegment() {
        mutateEditor(() => {
            const start = Math.max(0, editorVideo.currentTime || 0);
            const insertAt = state.segments.findIndex(segment => parseTimeToSeconds(segment.start) > start);
            const position = insertAt < 0 ? state.segments.length : insertAt;
            const nextStart = position < state.segments.length ? parseTimeToSeconds(state.segments[position].start) : start + 2;
            const end = Math.max(start + 0.35, Math.min(start + 2, nextStart));
            state.segments.splice(position, 0, {
                index: position + 1,
                start: formatSecondsToTime(start),
                end: formatSecondsToTime(end),
                text: '',
                translation: '',
            });
        });
    }

    function editorDuration() {
        return editorVideo.duration || Math.max(0, ...state.segments.map(segment => parseTimeToSeconds(segment.end)));
    }

    function drawWaveform() {
        const width = Math.max(1, waveformStage.clientWidth);
        const height = Math.max(1, editorWaveform.clientHeight || 120);
        const scale = window.devicePixelRatio || 1;
        editorWaveform.width = Math.round(width * scale);
        editorWaveform.height = Math.round(height * scale);
        editorWaveform.style.width = `${width}px`;
        const context = editorWaveform.getContext('2d');
        context.scale(scale, scale);
        context.clearRect(0, 0, width, height);
        context.fillStyle = 'rgba(34, 211, 238, 0.55)';
        const peaks = state.waveformPeaks.length ? state.waveformPeaks : new Array(300).fill(0.08);
        const barWidth = width / peaks.length;
        peaks.forEach((peak, index) => {
            const barHeight = Math.max(1, peak * (height - 18));
            context.fillRect(index * barWidth, (height - barHeight) / 2, Math.max(1, barWidth - 1), barHeight);
        });
    }

    function renderTimeline() {
        const duration = editorDuration();
        timelineDuration.textContent = formatSecondsToTime(duration || 0);
        timelineCues.innerHTML = '';
        if (!duration) return;
        state.segments.forEach(segment => {
            const start = parseTimeToSeconds(segment.start);
            const end = parseTimeToSeconds(segment.end);
            const cue = document.createElement('button');
            cue.type = 'button';
            cue.className = `timeline-cue${state.activeSegmentIndex === segment.index ? ' active' : ''}`;
            cue.dataset.index = segment.index;
            cue.style.left = `${start / duration * 100}%`;
            cue.style.width = `${Math.max(0.25, (end - start) / duration * 100)}%`;
            cue.innerHTML = `<span class="timeline-handle start" data-edge="start"></span><span>${segment.index}</span><span class="timeline-handle end" data-edge="end"></span>`;
            cue.addEventListener('click', event => {
                if (event.target.classList.contains('timeline-handle')) return;
                editorVideo.currentTime = start;
                highlightActiveSegment(segment.index);
            });
            cue.querySelectorAll('.timeline-handle').forEach(handle => enableTimelineDrag(handle, segment, duration));
            timelineCues.appendChild(cue);
        });
    }

    function enableTimelineDrag(handle, segment, duration) {
        handle.addEventListener('pointerdown', event => {
            event.preventDefault();
            handle.setPointerCapture(event.pointerId);
            const before = snapshotEditor();
            const edge = handle.dataset.edge;
            const cue = handle.closest('.timeline-cue');
            const move = moveEvent => {
                const bounds = waveformStage.getBoundingClientRect();
                const seconds = Math.max(0, Math.min(duration, (moveEvent.clientX - bounds.left) / bounds.width * duration));
                const position = state.segments.findIndex(item => item.index === segment.index);
                const previousEnd = position > 0 ? parseTimeToSeconds(state.segments[position - 1].end) : 0;
                const nextStart = position < state.segments.length - 1 ? parseTimeToSeconds(state.segments[position + 1].start) : duration;
                if (edge === 'start') {
                    segment.start = formatSecondsToTime(Math.max(previousEnd, Math.min(seconds, parseTimeToSeconds(segment.end) - 0.1)));
                } else {
                    segment.end = formatSecondsToTime(Math.min(nextStart, Math.max(seconds, parseTimeToSeconds(segment.start) + 0.1)));
                }
                const cueStart = parseTimeToSeconds(segment.start);
                const cueEnd = parseTimeToSeconds(segment.end);
                cue.style.left = `${cueStart / duration * 100}%`;
                cue.style.width = `${Math.max(0.25, (cueEnd - cueStart) / duration * 100)}%`;
                renderEditorQuality();
            };
            const finish = () => {
                handle.removeEventListener('pointermove', move);
                handle.removeEventListener('pointerup', finish);
                if (before !== snapshotEditor()) pushUndoSnapshot(before);
                renderEditor();
                scheduleEditorAutosave();
            };
            handle.addEventListener('pointermove', move);
            handle.addEventListener('pointerup', finish);
        });
    }

    async function loadEditorWaveform() {
        if (!state.videoPath) return;
        try {
            const response = await fetch(`/api/video/waveform?path=${encodeURIComponent(state.videoPath)}&bins=900`);
            if (!response.ok) throw new Error('Waveform unavailable');
            state.waveformPeaks = (await response.json()).peaks || [];
        } catch (error) {
            state.waveformPeaks = [];
        }
        drawWaveform();
    }

    async function loadEditorDraft() {
        if (!state.videoHash) return;
        try {
            const response = await fetch(`/api/editor/draft?video_hash=${encodeURIComponent(state.videoHash)}&target_language=${encodeURIComponent(state.targetLanguage || 'source')}`);
            if (!response.ok) return;
            const data = await response.json();
            if (data.draft?.segments?.length) state.segments = data.draft.segments;
            if (data.draft?.review) {
                state.reviewIssues = data.draft.review.issues || [];
                state.reviewApproval = data.draft.review.approval || null;
                state.reviewFieldState = data.draft.review.field_state || {};
            }
        } catch (error) {
            console.warn('Could not load subtitle draft', error);
        }
    }

    function scheduleEditorAutosave() {
        editorSaveState.textContent = 'Unsaved';
        state.reviewApproval = null;
        clearTimeout(editorAutosaveTimer);
        editorAutosaveTimer = setTimeout(saveEditorDraft, 700);
    }

    async function saveEditorDraft() {
        if (!state.videoHash) return;
        editorSaveState.textContent = 'Saving...';
        try {
            const driveContext = state.driveReviewContext;
            const response = await fetch(driveContext ? '/api/drive/batch/item/save' : '/api/editor/draft', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(driveContext ? {
                    batch_id: driveContext.batchId,
                    item_index: driveContext.itemIndex,
                    segments: state.segments,
                    translation_confirmed: true,
                } : {
                    video_hash: state.videoHash,
                    target_language: state.targetLanguage || 'source',
                    segments: state.segments,
                }),
            });
            if (!response.ok) throw new Error('Draft save failed');
            const data = await response.json();
            if (data.review) {
                state.reviewIssues = data.review.issues || [];
                state.reviewApproval = data.review.approval || null;
                state.reviewFieldState = data.review.field_state || {};
            }
            editorSaveState.textContent = 'Saved';
        } catch (error) {
            editorSaveState.textContent = 'Save failed';
        }
    }

    async function initializeEditor() {
        normalizeEditorSegments();
        state.editorUndo = [];
        state.editorRedo = [];
        await loadEditorDraft();
        normalizeEditorSegments();
        renderEditor();
        loadEditorWaveform();
    }

    function renderEditor() {
        renderEditorQuality();
        renderSegments();
        renderTimeline();
        drawWaveform();
        updateEditorHistoryButtons();
    }

    function highlightActiveSegment(index) {
        state.activeSegmentIndex = index;
        const cards = segmentsContainer.querySelectorAll('.segment-card');
        
        cards.forEach(card => {
            const cardIdx = parseInt(card.getAttribute('data-index'));
            if (cardIdx === index) {
                card.classList.add('active');
                card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                
                // Update overlay text in video player
                const seg = state.segments.find(s => s.index === index);
                if (seg) {
                    editorSubtitleOverlayText.textContent = seg.translation || seg.text;
                    editorSubtitleOverlay.style.display = 'flex';
                }
            } else {
                card.classList.remove('active');
            }
        });

        if (index === -1) {
            editorSubtitleOverlay.style.display = 'none';
        }
        renderTimeline();
    }

    // Sync editor video with segment cards scrolling & highlights
    editorVideo.addEventListener('timeupdate', () => {
        const time = editorVideo.currentTime;
        const duration = editorDuration();
        timelineCurrentTime.textContent = formatSecondsToTime(time);
        timelinePlayhead.style.left = duration ? `${time / duration * 100}%` : '0%';
        let activeIndex = -1;

        for (let i = 0; i < state.segments.length; i++) {
            const seg = state.segments[i];
            const start = parseTimeToSeconds(seg.start);
            const end = parseTimeToSeconds(seg.end);
            
            if (time >= start && time <= end) {
                activeIndex = seg.index;
                break;
            }
        }

        if (activeIndex !== -1 && activeIndex !== state.activeSegmentIndex) {
            highlightActiveSegment(activeIndex);
        } else if (activeIndex === -1 && state.activeSegmentIndex !== -1) {
            highlightActiveSegment(-1);
        }
    });

    // Search filter
    editorSearch.addEventListener('input', (e) => {
        renderSegments(e.target.value);
    });
    editorIssueFilter?.addEventListener('change', renderEditorQuality);

    function applySessionCapabilities(session) {
        state.session = {
            isLocal: !!session.is_local,
            lanEnabled: !!session.lan_enabled,
            clientPlatform: session.client_platform || 'other',
            clientBrowser: session.client_browser || 'other'
        };
        document.documentElement.dataset.clientPlatform = state.session.clientPlatform;
        document.documentElement.dataset.clientBrowser = state.session.clientBrowser;
        document.body.classList.toggle('mobile-session', !state.session.isLocal);
        btnMobileAccess.hidden = !state.session.isLocal || !state.session.lanEnabled;
        if (!state.session.isLocal) {
            if (localPathDivider) localPathDivider.hidden = true;
            if (localPathGroup) localPathGroup.hidden = true;
            if (outputDirGroup) outputDirGroup.hidden = true;
            if (navConfig) navConfig.hidden = true;
            if (setupBanner) setupBanner.hidden = true;
            if (document.getElementById('panel-config').classList.contains('active')) {
                switchTab('upload');
            }
        }
    }

    function selectedMobileAccessIndex() {
        const value = Number.parseInt(mobileNetworkSelect.value || '0', 10);
        return Number.isFinite(value) ? value : 0;
    }

    const mobilePlatformGuidance = {
        ios: 'Scan with Camera, or open the address in Safari or Chrome. If iOS displays a Local Network prompt, tap Allow. A browser that has never requested access may be absent from the Local Network settings list.',
        android: 'Scan with Camera, or open the address in your preferred browser. If the browser or Android asks for Local network or Nearby devices access, tap Allow. If access was denied, review that browser\'s site permissions for the SubGen address.'
    };

    function mobilePlatformLabel(platform) {
        if (platform === 'ios') return 'iPhone / iPad';
        if (platform === 'android') return 'Android';
        return 'mobile device';
    }

    function mobileBrowserLabel(browser) {
        const labels = { chrome: 'Chrome', safari: 'Safari', firefox: 'Firefox', edge: 'Edge', opera: 'Opera', samsung: 'Samsung Internet' };
        return labels[browser] || 'a mobile browser';
    }

    function selectMobileHelpPlatform(platform, persist = true) {
        const selected = platform === 'android' ? 'android' : 'ios';
        mobilePlatformOptions.forEach(option => {
            const active = option.dataset.mobilePlatform === selected;
            option.classList.toggle('active', active);
            option.setAttribute('aria-pressed', String(active));
        });
        mobilePlatformHelp.textContent = mobilePlatformGuidance[selected];
        if (persist) {
            try { localStorage.setItem('subgen-mobile-platform', selected); } catch (error) { /* optional preference */ }
        }
    }

    function renderLastMobileDevice(device) {
        if (!device || !['ios', 'android'].includes(device.platform)) {
            mobileDeviceStatus.hidden = true;
            return;
        }
        selectMobileHelpPlatform(device.platform, false);
        mobileDeviceStatus.textContent = `Last paired: ${mobilePlatformLabel(device.platform)} using ${mobileBrowserLabel(device.browser)}.`;
        mobileDeviceStatus.hidden = false;
    }

    function renderMobileAccessSelection() {
        const index = selectedMobileAccessIndex();
        const entry = state.mobileAccessUrls[index];
        if (!entry) return;
        mobileAccessUrl.value = entry.pairing_url;
        mobileAccessQr.src = `/api/mobile/qr?index=${index}&v=${Date.now()}`;
    }

    function renderMobileDiagnostics(diagnostics = {}) {
        const issues = diagnostics.issues || [];
        const blockers = issues.filter(issue => issue.severity === 'blocker');
        const warnings = issues.filter(issue => issue.severity === 'warning');
        mobileAccessIssues.replaceChildren();
        issues.forEach(issue => {
            const item = document.createElement('div');
            item.className = `mobile-access-issue ${issue.severity || ''}`;
            const title = document.createElement('strong');
            title.textContent = issue.title;
            const message = document.createElement('span');
            message.textContent = issue.message;
            item.append(title, message);
            mobileAccessIssues.appendChild(item);
        });
        mobileAccessIssues.hidden = issues.length === 0;
        btnRepairMobileAccess.hidden = !issues.some(issue => issue.repairable);
        mobileAccessQrFrame.classList.toggle('blocked', blockers.length > 0);
        mobileAccessIndicator.className = `status-indicator ${blockers.length ? 'blocked' : warnings.length ? 'warning' : 'online'}`;
        mobileAccessState.textContent = blockers.length
            ? 'Action required'
            : warnings.length
                ? 'Ready with VPN check'
                : 'Ready to pair';
    }

    async function loadMobileAccess() {
        mobileAccessState.textContent = 'Checking local network...';
        mobileAccessUnavailable.hidden = true;
        const response = await fetch('/api/mobile/access', { cache: 'no-store' });
        if (!response.ok) throw new Error('Mobile access settings are unavailable.');
        const data = await response.json();
        state.mobileAccessUrls = data.urls || [];
        renderLastMobileDevice(data.last_device);
        mobileNetworkSelect.innerHTML = '';
        state.mobileAccessUrls.forEach((entry, index) => {
            const option = document.createElement('option');
            option.value = String(index);
            option.textContent = `${entry.kind || 'Network'} (${entry.interface || entry.address}) - ${entry.base_url}`;
            mobileNetworkSelect.appendChild(option);
        });
        const multipleNetworks = state.mobileAccessUrls.length > 1;
        mobileNetworkSelect.hidden = !multipleNetworks;
        mobileNetworkLabel.hidden = !multipleNetworks;
        mobileAccessUnavailable.hidden = state.mobileAccessUrls.length > 0;
        mobileAccessQr.hidden = state.mobileAccessUrls.length === 0;
        renderMobileDiagnostics(data.diagnostics || {});
        if (!state.mobileAccessUrls.length) mobileAccessState.textContent = 'Network unavailable';
        if (state.mobileAccessUrls.length) renderMobileAccessSelection();
    }

    btnMobileAccess.addEventListener('click', async () => {
        mobileAccessModal.hidden = false;
        safeReplaceLucide();
        try {
            await loadMobileAccess();
        } catch (error) {
            mobileAccessState.textContent = error.message;
            mobileAccessUnavailable.hidden = false;
        }
    });
    let savedMobilePlatform = 'ios';
    try { savedMobilePlatform = localStorage.getItem('subgen-mobile-platform') || 'ios'; } catch (error) { /* optional preference */ }
    selectMobileHelpPlatform(savedMobilePlatform, false);
    mobilePlatformOptions.forEach(option => {
        option.addEventListener('click', () => selectMobileHelpPlatform(option.dataset.mobilePlatform));
    });
    mobileAccessModal.querySelectorAll('[data-mobile-access-close]').forEach(element => {
        element.addEventListener('click', () => { mobileAccessModal.hidden = true; });
    });
    mobileNetworkSelect.addEventListener('change', renderMobileAccessSelection);
    btnCopyMobileUrl.addEventListener('click', async () => {
        if (!mobileAccessUrl.value) return;
        await navigator.clipboard.writeText(mobileAccessUrl.value);
        mobileAccessState.textContent = 'Pairing link copied';
    });
    btnRefreshMobileAccess.addEventListener('click', async () => {
        if (!confirm('Create a new access token? Previously paired phones will be disconnected.')) return;
        const response = await fetch('/api/mobile/rotate', { method: 'POST' });
        if (!response.ok) throw new Error('Could not rotate the mobile access token.');
        await loadMobileAccess();
    });
    btnRepairMobileAccess.addEventListener('click', async () => {
        const accepted = confirm('Allow SubGen on this trusted private network? Windows will request administrator permission.');
        if (!accepted) return;
        const response = await fetch('/api/mobile/repair', { method: 'POST' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Could not start Windows network repair.');
        mobileAccessState.textContent = 'Waiting for Windows permission...';
        for (let attempt = 0; attempt < 60; attempt += 1) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            await loadMobileAccess();
            const repairStillNeeded = !btnRepairMobileAccess.hidden;
            if (!repairStillNeeded) break;
        }
    });

    const sessionReady = fetch('/api/session', { cache: 'no-store' })
        .then(response => response.json())
        .then(applySessionCapabilities)
        .catch(error => {
            console.error('Could not load session capabilities:', error);
            applySessionCapabilities({ is_local: true, lan_enabled: false });
        });

    btnEditorUndo.addEventListener('click', () => {
        if (!state.editorUndo.length) return;
        state.editorRedo.push(snapshotEditor());
        restoreEditorSnapshot(state.editorUndo.pop());
    });
    btnEditorRedo.addEventListener('click', () => {
        if (!state.editorRedo.length) return;
        state.editorUndo.push(snapshotEditor());
        restoreEditorSnapshot(state.editorRedo.pop());
    });
    btnEditorInsert.addEventListener('click', insertEditorSegment);
    btnRetranslateSelected.addEventListener('click', async () => {
        const cueIds = Array.from(segmentsContainer.querySelectorAll('.segment-card'))
            .filter(card => card.querySelector('.segment-retranslate-select')?.checked)
            .map(card => {
                const index = Number(card.dataset.index);
                return String(state.segments.find(segment => segment.index === index)?.id || '');
            })
            .filter(Boolean);
        if (!cueIds.length) {
            addLog('[ERROR] Select at least one source cue to retranslate.', 'error');
            return;
        }
        editorSaveState.textContent = 'Retranslating...';
        const driveContext = state.driveReviewContext;
        const response = await fetch(
            driveContext ? '/api/drive/batch/item/retranslate' : '/api/editor/retranslate',
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(driveContext ? {
                    batch_id: driveContext.batchId,
                    item_index: driveContext.itemIndex,
                    cue_ids: cueIds,
                } : {
                    video_hash: state.videoHash,
                    target_language: state.targetLanguage || 'source',
                    cue_ids: cueIds,
                }),
            },
        );
        const data = await response.json();
        if (!response.ok) {
            editorSaveState.textContent = 'Retranslation failed';
            addLog(`[ERROR] ${data.error || 'Selected-cue retranslation failed'}`, 'error');
            return;
        }
        state.segments = data.segments || state.segments;
        state.reviewApproval = null;
        state.reviewIssues = data.review?.issues || [];
        state.reviewFieldState = data.review?.field_state || {};
        editorSaveState.textContent = 'Selected cues retranslated';
        renderEditor();
        addLog(`[SYSTEM] Retranslated ${data.retranslation.translated_cue_count} selected cue(s); prior translations remain in history.`, 'success');
    });
    window.addEventListener('resize', () => {
        if (state.segments.length) drawWaveform();
    });

    // ==========================================
    // BURN & COMPILE PROCESS
    // ==========================================
    btnApproveDraft.addEventListener('click', async () => {
        if (!state.segments.length) return;
        const { errors, warnings } = validateEditorSegments();
        if (errors.length) {
            renderEditor();
            addLog('[ERROR] Resolve subtitle timing errors before approval.', 'error');
            return;
        }
        const acceptWarnings = !warnings.length || window.confirm(
            `This draft has ${warnings.length} non-blocking timing warning(s). Accept them and approve this exact version?`
        );
        if (!acceptWarnings) return;
        const response = await fetch(
            state.driveReviewContext ? '/api/drive/batch/item/approve' : '/api/process/approve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state.driveReviewContext ? {
                batch_id: state.driveReviewContext.batchId,
                item_index: state.driveReviewContext.itemIndex,
                segments: state.segments,
                translation_confirmed: true,
                accept_warnings: acceptWarnings,
            } : {
                video_hash: state.videoHash,
                target_language: state.targetLanguage || 'source',
                segments: state.segments,
                translation_confirmed: true,
                accept_warnings: acceptWarnings,
            }),
        });
        const data = await response.json();
        if (!response.ok) {
            state.reviewIssues = data.issues || state.reviewIssues;
            renderEditor();
            addLog(`[ERROR] ${data.error || 'Approval failed'}`, 'error');
            return;
        }
        state.reviewApproval = data.review.approval;
        state.reviewIssues = data.review.issues || [];
        state.reviewFieldState = data.review.field_state || {};
        editorSaveState.textContent = `Approved ${data.approved_draft_hash.slice(0, 10)}`;
        renderEditorQuality();
        addLog(`[SYSTEM] Approved exact subtitle draft ${data.approved_draft_hash}.`, 'success');
    });

    btnBurnFinal.addEventListener('click', async () => {
        if (state.segments.length === 0) return;
        const validationResponse = await fetch('/api/editor/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ segments: state.segments }),
        });
        const validation = await validationResponse.json();
        if (!validation.accept) {
            state.editorIssues = validation;
            renderEditor();
            addLog('[ERROR] Resolve subtitle timing errors before burning.', 'error');
            return;
        }
        if (!state.reviewApproval) {
            addLog('[ERROR] Approve this exact draft before burning.', 'error');
            return;
        }
        if (state.driveReviewContext) {
            const response = await fetch('/api/drive/batch/item/burn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    batch_id: state.driveReviewContext.batchId,
                    item_index: state.driveReviewContext.itemIndex,
                }),
            });
            const data = await response.json();
            if (!response.ok) {
                addLog(`[ERROR] ${data.error || 'Drive burn request failed'}`, 'error');
                return;
            }
            addLog('[SYSTEM] Drive source re-download and approved burn queued.', 'success');
            switchTab('batch');
            await refreshDriveBatch();
            return;
        }
        
        // Switch to progress tab
        switchTab('progress');
        runningBadge.style.display = 'block';
        
        // Reset progress UI for burning stage
        updateProgress(0, 'Preparing for burning...');
        setStageState('stage-burning', 'running');
        
        addLog(`[SYSTEM] Submitting edited subtitle segments to server...`, 'system');
        
        const payload = {
            video_path: state.videoPath,
            target_language: state.targetLanguage,
            segments: state.segments,
            style_config: state.styleConfig,
            force_burn: state.cacheAction !== 'reuse_all'
        };
        
        fetch('/api/process/burn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(async res => {
            const body = await res.json();
            if (!res.ok) throw new Error(body.error || 'Burning trigger failed');
            return body;
        })
        .then(data => {
            addLog(`[PIPELINE] Burning process started on server. Hardcoding styles: Font=${state.styleConfig.font_name}, Size=${state.styleConfig.font_size}px`, 'info');
            startPollingStatus();
        })
        .catch(err => {
            addLog(`[ERROR] ${err.message}`, 'error');
            updateProgress(0, 'Error');
            runningBadge.style.display = 'none';
        });
    });

    // ==========================================
    // EXPORT SCREEN
    // ==========================================
    function setupExportScreen(videoUrl, srtUrl) {
        // Use the API endpoints to serve the local files safely
        const ts = new Date().getTime();
        const finalVideoUrl = `/api/video?path=${encodeURIComponent(videoUrl)}&t=${ts}`;
        const finalSrtUrl = `/api/srt?path=${encodeURIComponent(srtUrl)}&t=${ts}`;
        
        finalVideo.src = finalVideoUrl;
        btnDownloadVideo.href = finalVideoUrl;
        btnDownloadSRT.href = finalSrtUrl;
        
        addLog(`[SYSTEM] Finished! Download your output files below.`, 'success');
    }

    btnRestart.addEventListener('click', () => {
        endCurrentVideoSession();
        // Reset state
        state.videoFile = null;
        state.videoPath = null;
        state.uploadJobId = null;
        state.detectedLanguage = null;
        state.status = 'idle';
        state.progress = 0;
        state.logs = [];
        state.segments = [];
        state.activeSegmentIndex = -1;
        state.cacheAction = 'reuse_all';
        state.editorUndo = [];
        state.editorRedo = [];
        state.editorIssues = { errors: [], warnings: [] };
        state.waveformPeaks = [];
        
        // Reset UI
        videoInput.value = '';
        dropzone.style.display = 'block';
        fileDetails.style.display = 'none';
        uploadPreviewContainer.style.display = 'none';
        uploadVideoPreview.src = '';
        previewVideo.src = '';
        previewVideo.style.display = 'none';
        document.getElementById('preview-overlay-bg').style.display = 'block';
        editorVideo.src = '';
        
        navEditor.disabled = true;
        navExport.disabled = true;
        
        logConsole.innerHTML = '<div class="log-line system">[SYSTEM] Ready. Upload a video and start processing.</div>';
        
        switchTab('upload');
    });

    // ==========================================
    // API SETTINGS & KEYS CONFIGURATION
    // ==========================================
    const apiProvidersContainer = document.getElementById('api-providers-container');
    
    const formSaveApiKey = document.getElementById('form-save-api-key');
    const saveProviderSelect = document.getElementById('save-provider-select');
    const saveProfileName = document.getElementById('save-profile-name');
    const saveApiKeyVal = document.getElementById('save-api-key-val');

    const transcriptionProviderSelect = configForm.querySelector('[name="transcription_provider"]');
    const transcriptionModelSelect = configForm.querySelector('[name="transcription_model"]');
    const groupTranscriptionModel = document.getElementById('group-transcription-model');
    const groupModelSize = document.getElementById('group-model-size');
    const timingAnchorProviderSelect = configForm.querySelector('[name="timing_anchor_provider"]');
    const groupTimingAnchorProvider = document.getElementById('group-timing-anchor-provider');
    const groupTimingMode = document.getElementById('group-timing-mode');
    const translationProviderSelect = configForm.querySelector('[name="translation_provider"]');
    const translationModelSelect = configForm.querySelector('[name="translation_model"]');
    const groupTranslationModel = document.getElementById('group-translation-model');
    const pipelinePlanTitle = document.getElementById('pipeline-plan-title');
    const pipelinePlanStages = document.getElementById('pipeline-plan-stages');
    const pipelinePlanError = document.getElementById('pipeline-plan-error');
    const modelGuideBody = document.getElementById('model-guide-body');
    const modelGuideSearch = document.getElementById('model-guide-search');
    const modelGuideProvider = document.getElementById('model-guide-provider');
    const modelGuideEmpty = document.getElementById('model-guide-empty');
    const modelGuideKindButtons = Array.from(document.querySelectorAll('[data-model-kind]'));
    let pipelineCatalog = { providers: {}, models: [] };
    let pipelinePreviewTimer = null;
    let modelGuideKind = 'all';
    state.pipelinePlanValid = true;

    const providerNames = {
        'openai': 'OpenAI (Whisper / GPT)',
        'google': 'Google Gemini',
        'deepseek': 'DeepSeek',
        'anthropic': 'Anthropic Claude',
        'xai': 'xAI',
        'cohere': 'Cohere'
    };

    function createCapabilityCell(available, label, detail = '') {
        const cell = document.createElement('td');
        cell.className = 'model-capability-cell';
        const status = document.createElement('span');
        status.className = `model-capability-status ${available ? 'is-yes' : 'is-no'}`;
        status.title = detail || `${label}: ${available ? 'available' : 'not available'}`;
        status.setAttribute('aria-label', status.title);
        const icon = document.createElement('i');
        icon.setAttribute('data-lucide', available ? 'circle-check' : 'circle-x');
        status.appendChild(icon);
        cell.appendChild(status);
        return cell;
    }

    function createOfficialLink(url, label, iconName) {
        if (!url || !/^https:\/\//i.test(url)) return null;
        const link = document.createElement('a');
        link.className = 'model-guide-link';
        link.href = url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.title = `${label} (opens official site)`;
        const icon = document.createElement('i');
        icon.setAttribute('data-lucide', iconName);
        const textNode = document.createElement('span');
        textNode.textContent = label;
        link.append(icon, textNode);
        return link;
    }

    function modelGuideTiming(model) {
        return model.guide_timing_kind || model.adapter_timing_kind || 'none';
    }

    function renderModelCapabilityTable() {
        if (!modelGuideBody || !modelGuideEmpty) return;
        const query = (modelGuideSearch?.value || '').trim().toLowerCase();
        const provider = modelGuideProvider?.value || 'all';
        const rows = (pipelineCatalog.models || [])
            .filter(model => {
                if (provider !== 'all' && model.provider !== provider) return false;
                if (modelGuideKind === 'transcription' && !model.audio_transcription) return false;
                if (modelGuideKind === 'translation' && !model.translation) return false;
                if (!query) return true;
                return [
                    model.provider_display,
                    model.display_name,
                    model.model,
                    model.model_family,
                    model.timing_description,
                ].some(value => String(value || '').toLowerCase().includes(query));
            })
            .sort((left, right) => {
                const providerOrder = String(left.provider_display || left.provider)
                    .localeCompare(String(right.provider_display || right.provider));
                return providerOrder || String(left.display_name).localeCompare(String(right.display_name));
            });

        modelGuideBody.innerHTML = '';
        rows.forEach(model => {
            const row = document.createElement('tr');
            if (model.subgen_supported === false) row.classList.add('is-not-integrated');

            const identity = document.createElement('th');
            identity.scope = 'row';
            identity.className = 'model-identity-cell';
            const providerName = document.createElement('span');
            providerName.className = 'model-provider-name';
            providerName.textContent = model.provider_display || providerNames[model.provider] || model.provider;
            const modelName = document.createElement('strong');
            modelName.textContent = model.display_name || model.model;
            const family = document.createElement('span');
            family.className = 'model-family';
            family.textContent = model.model_family || model.model;
            const timing = document.createElement('span');
            timing.className = `model-timing-note timing-${modelGuideTiming(model)}`;
            timing.textContent = model.timing_description || 'Timing capability not declared';
            identity.append(providerName, modelName, family, timing);
            row.appendChild(identity);

            const timingKind = modelGuideTiming(model);
            row.appendChild(createCapabilityCell(
                !!model.semantic_ai,
                'Semantic AI',
                model.semantic_ai ? 'Semantic language-model processing' : 'Dedicated non-LLM model',
            ));
            row.appendChild(createCapabilityCell(
                !!model.audio_transcription,
                'Transcript',
                model.audio_transcription ? 'Produces source-language transcript text' : 'Does not transcribe audio',
            ));
            row.appendChild(createCapabilityCell(
                timingKind === 'native_word',
                'Word timing',
                timingKind === 'native_word' ? 'Native word timestamps' : 'No native word timestamps',
            ));
            row.appendChild(createCapabilityCell(
                ['native_word', 'native_segment', 'prompted_segment'].includes(timingKind),
                'Segment timing',
                timingKind === 'prompted_segment'
                    ? 'LLM-generated segment timestamps; verification required'
                    : timingKind === 'native_word' || timingKind === 'native_segment'
                        ? 'Native segment timing is available'
                        : 'No segment timestamps',
            ));
            row.appendChild(createCapabilityCell(
                !!model.mixed_language,
                'Mixed-language audio',
            ));
            row.appendChild(createCapabilityCell(
                !!model.diarization,
                'Speaker labels',
            ));
            row.appendChild(createCapabilityCell(
                !!model.translation,
                'Semantic translation',
            ));
            row.appendChild(createCapabilityCell(
                model.subgen_supported !== false,
                'SubGen integration',
                model.subgen_supported === false ? 'Documented model; SubGen adapter not integrated' : 'Available in this SubGen branch',
            ));

            const linksCell = document.createElement('td');
            linksCell.className = 'model-guide-links';
            const docsLink = createOfficialLink(model.docs_url, 'Docs', 'book-open');
            const apiLink = createOfficialLink(model.api_url, 'API', 'external-link');
            if (docsLink) linksCell.appendChild(docsLink);
            if (apiLink) linksCell.appendChild(apiLink);
            if (!docsLink && !apiLink) linksCell.textContent = 'Custom';
            row.appendChild(linksCell);
            modelGuideBody.appendChild(row);
        });
        modelGuideEmpty.hidden = rows.length > 0;
        safeReplaceLucide();
    }

    function initializeModelCapabilityGuide() {
        if (!modelGuideProvider) return;
        const selectedProvider = modelGuideProvider.value;
        const providers = new Map();
        (pipelineCatalog.models || []).forEach(model => {
            providers.set(model.provider, model.provider_display || providerNames[model.provider] || model.provider);
        });
        modelGuideProvider.innerHTML = '<option value="all">All providers</option>';
        Array.from(providers.entries())
            .sort((left, right) => left[1].localeCompare(right[1]))
            .forEach(([providerId, label]) => {
                const option = document.createElement('option');
                option.value = providerId;
                option.textContent = label;
                modelGuideProvider.appendChild(option);
            });
        if (providers.has(selectedProvider)) modelGuideProvider.value = selectedProvider;
        renderModelCapabilityTable();
    }

    modelGuideSearch?.addEventListener('input', renderModelCapabilityTable);
    modelGuideProvider?.addEventListener('change', renderModelCapabilityTable);
    modelGuideKindButtons.forEach(button => {
        button.addEventListener('click', () => {
            modelGuideKind = button.dataset.modelKind || 'all';
            modelGuideKindButtons.forEach(candidate => {
                const active = candidate === button;
                candidate.classList.toggle('active', active);
                candidate.setAttribute('aria-pressed', String(active));
            });
            renderModelCapabilityTable();
        });
    });

    function loadApiKeysAndProfiles() {
        if (!state.session.isLocal) return;
        fetch('/api/config/keys')
            .then(res => res.json())
            .then(data => {
                const hasConfiguredProvider = Object.values(data.api_profiles || {})
                    .flat()
                    .some(profile => profile.configured);
                setupBanner.hidden = hasConfiguredProvider;
                if (!hasConfiguredProvider && !state.setupChecked) {
                    state.setupChecked = true;
                    switchTab('config');
                }
                if (apiProvidersContainer) {
                    apiProvidersContainer.innerHTML = '';
                    
                    Object.entries(data.api_profiles).forEach(([provId, profiles]) => {
                        const providerName = providerNames[provId] || provId;
                        
                        const provCard = document.createElement('div');
                        provCard.className = 'api-provider-group mb-6';
                        
                        let html = `
                            <div class="section-title-sm mb-3" style="display: flex; justify-content: space-between; align-items: center;">
                                <span>${providerName}</span>
                            </div>
                            <div class="api-profiles-list">
                        `;
                        
                        if (profiles.length === 0) {
                            html += `
                                <div class="empty-state py-4 text-center" style="background: rgba(255,255,255,0.01); border-radius: 8px;">
                                    <span class="muted font-sm">No profiles configured</span>
                                </div>
                            `;
                        } else {
                            profiles.forEach(p => {
                                html += `
                                    <div class="openai-profile-item ${p.active ? 'active' : ''}">
                                        <div class="api-status-info">
                                            <span class="api-status-name" style="text-transform: capitalize;">${p.label}</span>
                                            <span class="api-status-env">${p.env_key}</span>
                                        </div>
                                        <div class="openai-profile-actions">
                                            <span class="api-status-badge ${p.configured ? 'configured' : 'missing'}">
                                                ${p.configured ? 'Configured' : 'Missing Key'}
                                            </span>
                                            ${p.active ? 
                                                `<span class="api-status-badge configured" style="background: rgba(139, 92, 246, 0.2); color: #a78bfa;">Default</span>` : 
                                                `<button class="btn btn-xs btn-secondary btn-set-active" data-provider="${provId}" data-profile="${p.id}" type="button">Set as Default</button>`
                                            }
                                        </div>
                                    </div>
                                `;
                            });
                        }
                        
                        html += `</div>`;
                        provCard.innerHTML = html;
                        apiProvidersContainer.appendChild(provCard);
                    });

                    // Bind Set as Default buttons
                    apiProvidersContainer.querySelectorAll('.btn-set-active').forEach(btn => {
                        btn.addEventListener('click', (e) => {
                            const providerId = e.target.getAttribute('data-provider');
                            const profileName = e.target.getAttribute('data-profile');
                            fetch('/api/config/set-active-profile', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ provider: providerId, profile: profileName })
                            })
                            .then(res => res.json())
                            .then(() => {
                                addLog(`[SYSTEM] Default key for ${providerNames[providerId] || providerId} set to: ${profileName}`, 'success');
                                loadApiKeysAndProfiles();
                            })
                            .catch(err => addLog(`[ERROR] Failed to set default profile: ${err.message}`, 'error'));
                        });
                    });
                }
            })
            .catch(err => console.error('Failed to load API keys config:', err));
    }

    // Save/Update API Key form submit
    if (formSaveApiKey) {
        formSaveApiKey.addEventListener('submit', (e) => {
            e.preventDefault();
            const provider = saveProviderSelect.value;
            const name = saveProfileName.value.trim().toLowerCase();
            const apiKey = saveApiKeyVal.value.trim();
            
            fetch('/api/config/save-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: provider, name: name, api_key: apiKey })
            })
            .then(res => res.json())
            .then(() => {
                addLog(`[SYSTEM] Successfully saved API key for provider: ${providerNames[provider] || provider} [Profile: ${name}]`, 'success');
                saveProfileName.value = 'default';
                saveApiKeyVal.value = '';
                loadApiKeysAndProfiles();
            })
            .catch(err => addLog(`[ERROR] Failed to save API key: ${err.message}`, 'error'));
        });
    }

    // ==========================================
    // DYNAMIC SERVER CONFIG LOADER & VISIBILITY
    // ==========================================

    function modelOptions(providerId, capability) {
        const options = [];
        const seen = new Set();
        const provider = pipelineCatalog.providers?.[providerId] || {};
        const defaultModel = provider[`${capability}_model`];
        if (defaultModel) {
            options.push({ model: defaultModel, display_name: defaultModel });
            seen.add(defaultModel);
        }
        (pipelineCatalog.models || []).forEach(model => {
            const supportsCapability = capability === 'transcription'
                ? model.audio_transcription
                : model.translation;
            if (
                model.provider === providerId
                && supportsCapability
                && model.subgen_supported !== false
                && model.model
                && !seen.has(model.model)
            ) {
                options.push(model);
                seen.add(model.model);
            }
        });
        return options;
    }

    function syncProviderOptions(select, capability = null, includeLocal = false) {
        if (!select) return;
        const selected = select.value;
        Object.entries(pipelineCatalog.providers || {}).forEach(([providerId, provider]) => {
            if (
                (capability && !provider[capability])
                || Array.from(select.options).some(option => option.value === providerId)
            ) {
                return;
            }
            const option = document.createElement('option');
            option.value = providerId;
            option.textContent = provider.name || providerId;
            select.appendChild(option);
        });
        if (selected && Array.from(select.options).some(option => option.value === selected)) {
            select.value = selected;
        } else if (!includeLocal && select.value === 'local') {
            select.selectedIndex = 0;
        }
    }

    function populateModelSelect(select, providerId, capability, preferredModel) {
        if (!select) return;
        const options = modelOptions(providerId, capability);
        if (preferredModel && !options.some(item => item.model === preferredModel)) {
            options.unshift({ model: preferredModel, display_name: preferredModel });
        }
        select.innerHTML = '';
        options.forEach(item => {
            const option = document.createElement('option');
            option.value = item.model;
            option.textContent = item.display_name || item.model;
            select.appendChild(option);
        });
        if (preferredModel && options.some(item => item.model === preferredModel)) {
            select.value = preferredModel;
        }
        select.disabled = options.length === 0;
    }

    function selectedTranscriptionDescriptor() {
        const provider = transcriptionProviderSelect?.value;
        const model = transcriptionModelSelect?.value;
        return (pipelineCatalog.models || []).find(item => item.provider === provider && item.model === model);
    }

    function pipelinePreviewPayload() {
        return {
            transcription_provider: transcriptionProviderSelect?.value,
            transcription_model: transcriptionModelSelect?.value || null,
            timing_anchor_provider: timingAnchorProviderSelect?.value,
            api_transcript_timing_mode: configForm.querySelector('[name="api_transcript_timing_mode"]')?.value,
            translation_provider: translationProviderSelect?.value,
            translation_model: translationModelSelect?.value || null,
            model_size: configForm.querySelector('[name="model_size"]')?.value,
            source_language: document.getElementById('source-lang')?.value || null,
            target_language: document.getElementById('target-lang')?.value || null,
            subtitle_mode: configForm.querySelector('[name="subtitle_mode"]:checked')?.value || 'auto',
        };
    }

    function renderPipelinePlan(plan) {
        if (!pipelinePlanTitle || !pipelinePlanStages || !pipelinePlanError) return;
        pipelinePlanError.hidden = true;
        state.pipelinePlanValid = true;
        pipelinePlanTitle.textContent = plan.title;
        pipelinePlanStages.innerHTML = '';
        (plan.stages || []).forEach(stage => {
            const item = document.createElement('li');
            item.textContent = stage.label;
            pipelinePlanStages.appendChild(item);
        });
        safeReplaceLucide();
    }

    function refreshPipelinePreview() {
        clearTimeout(pipelinePreviewTimer);
        pipelinePreviewTimer = setTimeout(() => {
            fetch('/api/pipeline/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pipelinePreviewPayload()),
            })
            .then(async response => {
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Could not select a pipeline.');
                if (data.error) throw new Error(data.error);
                return data;
            })
            .then(data => renderPipelinePlan(data.plan))
            .catch(error => {
                if (!pipelinePlanTitle || !pipelinePlanStages || !pipelinePlanError) return;
                pipelinePlanTitle.textContent = 'Pipeline configuration needs attention';
                state.pipelinePlanValid = false;
                pipelinePlanStages.innerHTML = '';
                pipelinePlanError.textContent = error.message;
                pipelinePlanError.hidden = false;
            });
        }, 120);
    }

    function updateModelSizeVisibility() {
        if (transcriptionProviderSelect) {
            const val = transcriptionProviderSelect.value;
            const label = groupModelSize.querySelector('label');
            if (val === 'local') {
                if (groupTranscriptionModel) groupTranscriptionModel.style.display = 'none';
                groupModelSize.style.display = 'block';
                if (groupTimingAnchorProvider) groupTimingAnchorProvider.style.display = 'none';
                if (groupTimingMode) groupTimingMode.style.display = 'none';
                if (label) label.textContent = 'Local Model Size';
            } else {
                if (groupTranscriptionModel) groupTranscriptionModel.style.display = 'block';
                const descriptor = selectedTranscriptionDescriptor();
                const hasDirectTimestamps = descriptor && descriptor.adapter_timing_kind !== 'none';
                if (groupTimingAnchorProvider) {
                    groupTimingAnchorProvider.style.display = hasDirectTimestamps ? 'none' : 'block';
                }
                
                const anchorVal = timingAnchorProviderSelect ? timingAnchorProviderSelect.value : 'openai';
                if (!hasDirectTimestamps && anchorVal === 'local') {
                    groupModelSize.style.display = 'block';
                    if (label) label.textContent = 'Timing Anchor Model Size';
                } else {
                    groupModelSize.style.display = 'none';
                }
                if (groupTimingMode) {
                    groupTimingMode.style.display = hasDirectTimestamps ? 'none' : 'block';
                }
            }
        }
        if (groupTranslationModel && translationProviderSelect) {
            groupTranslationModel.style.display = translationProviderSelect.value === 'local' ? 'none' : 'block';
        }
    }

    if (transcriptionProviderSelect) {
        transcriptionProviderSelect.addEventListener('change', () => {
            populateModelSelect(
                transcriptionModelSelect,
                transcriptionProviderSelect.value,
                'transcription',
                pipelineCatalog.selected_models?.[transcriptionProviderSelect.value]?.transcription,
            );
            updateModelSizeVisibility();
            refreshPipelinePreview();
        });
    }
    if (transcriptionModelSelect) {
        transcriptionModelSelect.addEventListener('change', () => {
            pipelineCatalog.selected_models ||= {};
            pipelineCatalog.selected_models[transcriptionProviderSelect.value] ||= {};
            pipelineCatalog.selected_models[transcriptionProviderSelect.value].transcription = transcriptionModelSelect.value;
            updateModelSizeVisibility();
            refreshPipelinePreview();
        });
    }
    if (timingAnchorProviderSelect) {
        timingAnchorProviderSelect.addEventListener('change', () => {
            updateModelSizeVisibility();
            refreshPipelinePreview();
        });
    }
    if (translationProviderSelect) {
        translationProviderSelect.addEventListener('change', () => {
            populateModelSelect(
                translationModelSelect,
                translationProviderSelect.value,
                'translation',
                pipelineCatalog.selected_models?.[translationProviderSelect.value]?.translation,
            );
            updateModelSizeVisibility();
            refreshPipelinePreview();
        });
    }
    translationModelSelect?.addEventListener('change', () => {
        pipelineCatalog.selected_models ||= {};
        pipelineCatalog.selected_models[translationProviderSelect.value] ||= {};
        pipelineCatalog.selected_models[translationProviderSelect.value].translation = translationModelSelect.value;
        refreshPipelinePreview();
    });
    configForm.querySelector('[name="api_transcript_timing_mode"]')?.addEventListener('change', refreshPipelinePreview);
    configForm.querySelector('[name="model_size"]')?.addEventListener('change', refreshPipelinePreview);
    updateModelSizeVisibility();

    const outputDirInput = document.getElementById('output-dir');
    if (outputDirInput) {
        outputDirInput.addEventListener('change', () => {
            const val = outputDirInput.value.trim();
            if (val) {
                fetch('/api/config/update-output-dir', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ output_dir: val })
                })
                .then(res => res.json())
                .then(() => addLog(`[SYSTEM] Output & cache directory updated to: ${val}`, 'success'))
                .catch(err => addLog(`[ERROR] Failed to update output directory: ${err.message}`, 'error'));
            }
        });
    }

    async function resolveReviewIssue(issueId, severity) {
        const status = severity === 'critical' ? 'corrected' : 'accepted';
        const reason = severity === 'critical'
            ? window.prompt('Describe the correction made for this issue:')
            : 'Explicitly accepted during review';
        if (severity === 'critical' && !reason) return;
        const driveContext = state.driveReviewContext;
        const response = await fetch(
            driveContext ? '/api/drive/batch/item/issue/resolve' : '/api/editor/issue/resolve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(driveContext ? {
                batch_id: driveContext.batchId,
                item_index: driveContext.itemIndex,
                issue_id: issueId,
                status,
                reason,
            } : {
                video_hash: state.videoHash,
                target_language: state.targetLanguage || 'source',
                issue_id: issueId,
                status,
                reason,
            }),
        });
        const data = await response.json();
        if (!response.ok) {
            addLog(`[ERROR] ${data.error || 'Issue resolution failed'}`, 'error');
            return;
        }
        state.reviewIssues = data.review.issues || [];
        state.reviewApproval = null;
        renderEditor();
    }

    // ==========================================
    // GOOGLE DRIVE FOLDER BATCH
    // ==========================================
    let driveBatchPollId = null;
    let driveAuthPollId = null;

    function formatBytes(value) {
        const bytes = Number(value || 0);
        if (!bytes) return 'Unknown size';
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
        return `${(bytes / (1024 ** index)).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
    }

    async function driveJson(path, options = {}) {
        const response = await fetch(path, options);
        let data = {};
        try { data = await response.json(); } catch (error) { /* handled below */ }
        if (!response.ok) throw new Error(data.error || `Drive request failed (${response.status}).`);
        return data;
    }

    function showDriveMessage(element, message, error = false) {
        if (!element) return;
        element.textContent = message || '';
        element.hidden = !message;
        element.classList.toggle('error', Boolean(error));
    }

    function currentDriveConfiguration() {
        const subtitleMode = configForm.querySelector('[name="subtitle_mode"]:checked')?.value || 'auto';
        return {
            ...pipelinePreviewPayload(),
            subtitle_mode: subtitleMode,
            tiktok_style: subtitleMode === 'tiktok',
            style_config: { ...state.styleConfig },
        };
    }

    function selectedLabel(select) {
        return select?.selectedOptions?.[0]?.textContent?.trim() || 'Not selected';
    }

    function renderDrivePipelineSummary() {
        if (!driveBatchPipelineSummary) return;
        const subtitleMode = configForm.querySelector('[name="subtitle_mode"]:checked')?.value || 'auto';
        const rows = [
            `Text: ${selectedLabel(transcriptionModelSelect)}`,
            `Timing: ${selectedLabel(timingAnchorProviderSelect)}`,
            `Alignment: ${selectedLabel(configForm.querySelector('[name="api_transcript_timing_mode"]'))}`,
            `Translation: ${selectedLabel(translationModelSelect)}`,
            `Style: ${state.styleConfig.font_name}, ${state.styleConfig.font_size}px`,
            `Format: ${subtitleMode.charAt(0).toUpperCase()}${subtitleMode.slice(1)}`,
        ];
        driveBatchPipelineSummary.replaceChildren(...rows.map(label => {
            const chip = document.createElement('span');
            chip.className = 'batch-pipeline-chip';
            chip.textContent = label;
            return chip;
        }));
    }

    function renderDriveQueue(items) {
        if (!driveBatchQueue) return;
        if (!items?.length) {
            driveBatchQueue.innerHTML = '<div class="batch-empty-state"><i data-lucide="folder-open"></i><span>No supported videos found.</span></div>';
            safeReplaceLucide();
            return;
        }
        driveBatchQueue.replaceChildren(...items.map((item, index) => {
            const row = document.createElement('div');
            row.className = `batch-item ${item.status || 'pending'}`;

            const number = document.createElement('span');
            number.className = 'batch-item-index';
            number.textContent = String(index + 1).padStart(2, '0');

            const identity = document.createElement('div');
            identity.className = 'batch-item-name';
            const name = document.createElement('strong');
            name.textContent = item.source_name || item.name;
            const detail = document.createElement('span');
            detail.textContent = item.error || item.relative_path || formatBytes(item.size);
            identity.append(name, detail);

            const progress = document.createElement('div');
            progress.className = 'batch-item-progress';
            const fill = document.createElement('span');
            fill.style.width = `${Math.max(0, Math.min(100, Number(item.progress || 0)))}%`;
            progress.appendChild(fill);

            const status = document.createElement('span');
            status.className = 'batch-item-status';
            status.textContent = item.stage || item.status || 'queued';

            row.append(number, identity, progress, status);
            if (['ready_for_review', 'needs_attention', 'approved', 'burn_failed'].includes(item.status)) {
                const reviewButton = document.createElement('button');
                reviewButton.type = 'button';
                reviewButton.className = 'btn btn-secondary batch-review-button';
                reviewButton.textContent = item.status === 'approved' ? 'Open approved' : 'Review';
                reviewButton.addEventListener('click', () => openDriveReview(index));
                row.appendChild(reviewButton);
            }
            return row;
        }));
    }

    async function openDriveReview(itemIndex) {
        const data = await driveJson(
            `/api/drive/batch/item?id=${encodeURIComponent(state.driveBatchId)}&index=${itemIndex}`,
            { cache: 'no-store' },
        );
        if (!data.review || !data.segments?.length) throw new Error('Drive review draft is unavailable.');
        state.driveReviewContext = { batchId: state.driveBatchId, itemIndex };
        state.videoHash = data.review.video_id;
        state.videoPath = null;
        state.targetLanguage = data.review.target_language || 'source';
        state.detectedLanguage = data.review.source_language || null;
        state.segments = data.segments;
        state.reviewIssues = data.review.issues || [];
        state.reviewApproval = data.review.approval || null;
        state.reviewFieldState = data.review.field_state || {};
        state.editorUndo = [];
        state.editorRedo = [];
        editorVideo.pause();
        editorVideo.removeAttribute('src');
        editorVideo.load();
        navEditor.disabled = false;
        normalizeEditorSegments();
        renderEditor();
        switchTab('editor');
        addLog(`[SYSTEM] Opened Drive item ${itemIndex + 1} review. Source media remains remote until burn.`, 'system');
    }

    function renderDriveBatch(batch, running = false) {
        if (!batch) return;
        state.driveBatchId = batch.id || state.driveBatchId;
        const items = batch.items || [];
        const completed = Number(batch.completed ?? items.filter(item => item.status === 'completed').length);
        const failed = Number(batch.failed ?? items.filter(item => item.status === 'failed').length);
        const total = Number(batch.total ?? items.length);
        const aggregate = total
            ? items.reduce((sum, item) => sum + Number(item.progress || 0), 0) / total
            : 0;
        driveBatchTitle.textContent = batch.source_folder_name || 'Drive batch';
        driveBatchCounts.textContent = `${completed}/${total} complete${failed ? `, ${failed} failed` : ''}`;
        driveBatchProgress.style.width = `${Math.max(0, Math.min(100, aggregate))}%`;
        renderDriveQueue(items);
        driveOutputLink.hidden = !batch.output_folder_url;
        if (batch.output_folder_url) driveOutputLink.href = batch.output_folder_url;
        btnDriveStop.hidden = !running;
        btnDriveStop.disabled = Boolean(batch.stop_requested);
        btnDriveResume.hidden = running || !items.some(item => item.status === 'failed' || item.status === 'pending');
        btnDriveStart.disabled = running || !state.driveInspection;
        if (running) startDriveBatchPolling();
        else stopDriveBatchPolling();
    }

    function stopDriveBatchPolling() {
        if (driveBatchPollId) clearInterval(driveBatchPollId);
        driveBatchPollId = null;
    }

    async function refreshDriveBatch() {
        const suffix = state.driveBatchId ? `?id=${encodeURIComponent(state.driveBatchId)}` : '';
        try {
            const data = await driveJson(`/api/drive/batch/status${suffix}`, { cache: 'no-store' });
            if (data.batch) renderDriveBatch(data.batch, data.running);
        } catch (error) {
            showDriveMessage(driveFolderMessage, error.message, true);
            stopDriveBatchPolling();
        }
    }

    function startDriveBatchPolling() {
        if (driveBatchPollId) return;
        driveBatchPollId = setInterval(refreshDriveBatch, 2000);
    }

    function renderDriveConnection(status) {
        const connected = Boolean(status.connected);
        state.driveConnected = connected;
        const connecting = status.status === 'connecting';
        driveAuthPanel.classList.toggle('connected', connected);
        driveConnectionState.querySelector('.status-indicator').className = `status-indicator ${connected ? 'online' : connecting ? 'warning' : ''}`;
        driveConnectionState.querySelector('span:last-child').textContent = connected ? 'Connected' : connecting ? 'Connecting...' : 'Not connected';
        driveClientPathGroup.hidden = !status.can_configure || connected;
        btnDriveConnect.hidden = connected || !status.can_configure;
        btnDriveConnect.disabled = connecting || !status.client_configured;
        btnDriveDisconnect.hidden = !connected || !status.can_configure;
        btnDriveInspect.disabled = !connected || !driveSourceFolder.value.trim();
        if (status.error) showDriveMessage(driveAuthMessage, status.error, true);
        else if (connecting) showDriveMessage(driveAuthMessage, 'Complete Google authorization in the browser window.', false);
        else showDriveMessage(driveAuthMessage, '', false);
        if (status.latest_batch) renderDriveBatch(status.latest_batch, status.batch_running);
    }

    async function refreshDriveStatus() {
        if (!driveConnectionState) return;
        try {
            const status = await driveJson('/api/drive/status', { cache: 'no-store' });
            renderDriveConnection(status);
            if (status.status === 'connecting') {
                if (!driveAuthPollId) driveAuthPollId = setInterval(refreshDriveStatus, 1200);
            } else if (driveAuthPollId) {
                clearInterval(driveAuthPollId);
                driveAuthPollId = null;
            }
        } catch (error) {
            showDriveMessage(driveAuthMessage, error.message, true);
        }
    }

    btnDriveConfigure?.addEventListener('click', async () => {
        showDriveMessage(driveAuthMessage, 'Validating OAuth client...', false);
        try {
            await driveJson('/api/drive/configure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ client_json_path: driveClientJsonPath.value.trim() }),
            });
            showDriveMessage(driveAuthMessage, 'OAuth client configured.', false);
            await refreshDriveStatus();
        } catch (error) {
            showDriveMessage(driveAuthMessage, error.message, true);
        }
    });

    btnDriveBrowseClient?.addEventListener('click', async () => {
        if (!state.isTauriDesktop) return;
        try {
            const selectedPath = await window.__TAURI__.dialog.open({
                multiple: false,
                directory: false,
                title: 'Select Google OAuth desktop client JSON',
                filters: [{ name: 'JSON files', extensions: ['json'] }],
            });
            if (typeof selectedPath === 'string' && selectedPath) driveClientJsonPath.value = selectedPath;
        } catch (error) {
            showDriveMessage(driveAuthMessage, 'The desktop file picker could not open.', true);
        }
    });

    btnDriveConnect?.addEventListener('click', async () => {
        try {
            await driveJson('/api/drive/connect', { method: 'POST' });
            await refreshDriveStatus();
        } catch (error) {
            showDriveMessage(driveAuthMessage, error.message, true);
        }
    });

    btnDriveDisconnect?.addEventListener('click', async () => {
        try {
            await driveJson('/api/drive/disconnect', { method: 'POST' });
            state.driveInspection = null;
            await refreshDriveStatus();
        } catch (error) {
            showDriveMessage(driveAuthMessage, error.message, true);
        }
    });

    driveSourceFolder?.addEventListener('input', () => {
        state.driveInspection = null;
        btnDriveStart.disabled = true;
        btnDriveInspect.disabled = !state.driveConnected || !driveSourceFolder.value.trim();
    });

    btnDriveInspect?.addEventListener('click', async () => {
        showDriveMessage(driveFolderMessage, 'Reading folder...', false);
        btnDriveInspect.disabled = true;
        try {
            const data = await driveJson('/api/drive/folder/inspect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder_url: driveSourceFolder.value.trim() }),
            });
            state.driveInspection = data.folder;
            driveBatchTitle.textContent = data.folder.name;
            driveBatchCounts.textContent = `${data.folder.video_count} videos, ${formatBytes(data.folder.total_bytes)}`;
            renderDriveQueue(data.folder.videos.map(video => ({ ...video, status: 'pending', stage: 'queued', progress: 0 })));
            showDriveMessage(driveFolderMessage, `${data.folder.video_count} supported videos found.`, false);
            btnDriveStart.disabled = data.folder.video_count === 0;
        } catch (error) {
            state.driveInspection = null;
            showDriveMessage(driveFolderMessage, error.message, true);
        } finally {
            btnDriveInspect.disabled = !state.driveConnected || !driveSourceFolder.value.trim();
        }
    });

    btnDriveStart?.addEventListener('click', async () => {
        renderDrivePipelineSummary();
        showDriveMessage(driveFolderMessage, 'Creating batch...', false);
        btnDriveStart.disabled = true;
        try {
            const data = await driveJson('/api/drive/batch/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source_folder_url: driveSourceFolder.value.trim(),
                    destination_folder_url: driveDestinationFolder.value.trim(),
                    target_language: driveBatchTargetLanguage.value || null,
                    configuration: currentDriveConfiguration(),
                }),
            });
            state.driveBatchId = data.batch.id;
            renderDriveBatch(data.batch, true);
            showDriveMessage(driveFolderMessage, 'Batch started.', false);
        } catch (error) {
            showDriveMessage(driveFolderMessage, error.message, true);
            btnDriveStart.disabled = false;
        }
    });

    btnDriveStop?.addEventListener('click', async () => {
        try {
            btnDriveStop.disabled = true;
            const data = await driveJson('/api/drive/batch/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: state.driveBatchId }),
            });
            if (data.batch) renderDriveBatch(data.batch, true);
            btnDriveStop.disabled = true;
            showDriveMessage(driveFolderMessage, 'Stop requested. The current video will finish first.', false);
        } catch (error) {
            btnDriveStop.disabled = false;
            showDriveMessage(driveFolderMessage, error.message, true);
        }
    });

    btnDriveResume?.addEventListener('click', async () => {
        try {
            const data = await driveJson('/api/drive/batch/resume', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: state.driveBatchId }),
            });
            renderDriveBatch(data.batch, true);
        } catch (error) {
            showDriveMessage(driveFolderMessage, error.message, true);
        }
    });

    [transcriptionProviderSelect, transcriptionModelSelect, timingAnchorProviderSelect,
        translationProviderSelect, translationModelSelect, styleFont, styleSize,
        ...configForm.querySelectorAll('[name="subtitle_mode"]')]
        .filter(Boolean)
        .forEach(element => element.addEventListener('change', renderDrivePipelineSummary));

    // Populate languages dynamically
    const sourceLangSelect = document.getElementById('source-lang');
    const targetLangSelect = document.getElementById('target-lang');
    sourceLangSelect?.addEventListener('change', refreshPipelinePreview);
    targetLangSelect?.addEventListener('change', refreshPipelinePreview);
    const langNames = {
        'en': 'English',
        'he': 'Hebrew (עברית)',
        'fa': 'Persian (فارסי)',
        'ar': 'Arabic (العربية)',
        'es': 'Spanish (Español)',
        'fr': 'French (Français)',
        'de': 'German (Deutsch)',
        'it': 'Italian (Italiano)',
        'pt': 'Portuguese (Português)',
        'ru': 'Russian (Русский)',
        'zh': 'Chinese (中文)',
        'ja': 'Japanese (日本語)'
    };

    sessionReady.then(() => fetch('/api/languages'))
        .then(res => res.json())
        .then(langs => {
            if (sourceLangSelect && targetLangSelect) {
                // Clear and rebuild
                sourceLangSelect.innerHTML = '<option value="">Auto Detect (Recommended)</option>';
                targetLangSelect.innerHTML = '<option value="">Keep Original (No Translation)</option>';
                driveBatchTargetLanguage.innerHTML = '<option value="">Keep Original (No Translation)</option>';

                Object.entries(langs).forEach(([code, name]) => {
                    const displayName = langNames[code] || name;

                    const srcOpt = document.createElement('option');
                    srcOpt.value = code;
                    srcOpt.textContent = displayName;
                    sourceLangSelect.appendChild(srcOpt);

                    const tgtOpt = document.createElement('option');
                    tgtOpt.value = code;
                    tgtOpt.textContent = displayName;
                    targetLangSelect.appendChild(tgtOpt);

                    const batchOpt = tgtOpt.cloneNode(true);
                    driveBatchTargetLanguage.appendChild(batchOpt);
                });
                driveBatchTargetLanguage.value = targetLangSelect.value;
            }
            
            // Load config from server after languages are populated
            return fetch('/api/config');
        })
        .then(res => res.json())
        .then(async config => {
            try {
                const catalogResponse = await fetch('/api/pipeline/catalog');
                pipelineCatalog = await catalogResponse.json();
                syncProviderOptions(transcriptionProviderSelect, 'transcription', true);
                syncProviderOptions(translationProviderSelect, 'translation', true);
                syncProviderOptions(saveProviderSelect, null, false);
                initializeModelCapabilityGuide();
            } catch (error) {
                console.error('Failed to load pipeline catalog:', error);
            }
            if (config.source_language !== undefined) {
                const el = configForm.querySelector('[name="source_language"]');
                if (el) el.value = config.source_language || "";
            }
            if (config.target_language !== undefined) {
                const el = configForm.querySelector('[name="target_language"]');
                if (el) el.value = config.target_language || "";
            }
            if (config.transcription_provider) {
                const el = configForm.querySelector('[name="transcription_provider"]');
                if (el) el.value = config.transcription_provider;
            }
            populateModelSelect(
                transcriptionModelSelect,
                transcriptionProviderSelect?.value || 'local',
                'transcription',
                config.transcription_model
                    || pipelineCatalog.selected_models?.[transcriptionProviderSelect?.value]?.transcription,
            );
            if (config.model_size) {
                const el = configForm.querySelector('[name="model_size"]');
                if (el) el.value = config.model_size;
            }
            if (config.api_transcript_timing_mode) {
                const el = configForm.querySelector('[name="api_transcript_timing_mode"]');
                if (el) el.value = config.api_transcript_timing_mode;
            }
            if (config.timing_anchor_provider) {
                const el = configForm.querySelector('[name="timing_anchor_provider"]');
                if (el) el.value = config.timing_anchor_provider;
            }
            if (config.translation_provider) {
                const el = configForm.querySelector('[name="translation_provider"]');
                if (el) el.value = config.translation_provider;
            }
            populateModelSelect(
                translationModelSelect,
                translationProviderSelect?.value || 'local',
                'translation',
                config.translation_model
                    || pipelineCatalog.selected_models?.[translationProviderSelect?.value]?.translation,
            );
            const subtitleMode = ['auto', 'normal', 'tiktok'].includes(config.subtitle_mode)
                ? config.subtitle_mode
                : (config.tiktok_style ? 'tiktok' : 'normal');
            const subtitleModeInput = configForm.querySelector(
                `[name="subtitle_mode"][value="${subtitleMode}"]`
            );
            if (subtitleModeInput) subtitleModeInput.checked = true;
            if (config.last_output_dir) {
                const el = document.getElementById('output-dir');
                if (el) el.value = config.last_output_dir;
            }
            updateModelSizeVisibility();
            refreshPipelinePreview();
            loadApiKeysAndProfiles();
            renderDrivePipelineSummary();
            refreshDriveStatus();
        })
        .catch(err => console.error('Failed to load startup data:', err));
});
