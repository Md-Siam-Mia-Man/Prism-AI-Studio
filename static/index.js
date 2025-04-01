// Simple UUID function
function generateUUID() { return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.random() * 16 | 0; return (c == 'x' ? r : (r & 0x3 | 0x8)).toString(16); }); }

document.addEventListener('DOMContentLoaded', () => {
     // --- Globals & Config ---
     const CLIENT_ID = generateUUID();
     console.log("Client ID:", CLIENT_ID);
     let config = {};
     let uploadedFiles = {}; // { taskId: { originalName, size, status, outputUrl, inputElement, outputElement, blobUrl } }
     let processingQueue = new Set();
     let websocket = null;
     let allowedExtensions = [];
     const GFPGAN_MODEL_NAME = 'GFPGANv1.3.pth';

     // Globally defined MODEL_INFO - needed for frontend logic before config loads sometimes
     const MODEL_INFO = {
          'RealESRGAN_x4plus': {}, 'RealESRNet_x4plus': {}, 'RealESRGAN_x4plus_anime_6B': {},
          'RealESRGAN_x2plus': {}, 'realesr-animevideov3': {}, 'realesr-general-x4v3': {}
     };


     // --- DOM Element References ---
     // Sidebar Elements
     const themeSelectNative = document.getElementById('theme-select');
     const gpuInfoName = document.getElementById('gpu-name');
     const precisionInfo = document.getElementById('precision-info');
     const modelSelectNative = document.getElementById('model-select');
     const modelStatusText = document.getElementById('model-status');
     const downloadModelBtn = document.getElementById('download-model-btn');
     const upscaleFactorButtonsContainer = document.querySelector('.upscale-factor-buttons');
     const upscaleFactorNativeInput = document.getElementById('upscale-factor-native');
     const outputFormatButtonsContainer = document.querySelector('.output-format-buttons');
     const outputFormatNativeInput = document.getElementById('output-format-native');
     const faceEnhanceButtonsContainer = document.querySelector('.face-enhance-buttons');
     const faceEnhanceNativeInput = document.getElementById('face-enhance-native');
     const gfpganStatusText = document.getElementById('gfpgan-status');
     const downloadGfpganBtn = document.getElementById('download-gfpgan-btn');
     const denoiseControl = document.getElementById('denoise-strength-control');
     const denoiseSliderNative = document.getElementById('denoise-slider');
     const denoiseValueDisplay = document.getElementById('denoise-value');
     const enhanceButton = document.getElementById('enhance-button');
     const clearButton = document.getElementById('clear-button');
     const downloadAllButton = document.getElementById('download-all-button');
     const toggleAdvancedBtn = document.getElementById('toggle-advanced');
     const advancedOptionsDiv = document.getElementById('advanced-options');
     const tileSliderNative = document.getElementById('tile-slider');
     const tileValueDisplay = document.getElementById('tile-value');
     const tilePadSliderNative = document.getElementById('tile-pad-slider');
     const tilePadValueDisplay = document.getElementById('tile-pad-value');
     const prePadSliderNative = document.getElementById('pre-pad-slider');
     const prePadValueDisplay = document.getElementById('pre-pad-value');
     const fp32Control = document.getElementById('fp32-control');
     const fp32Checkbox = document.getElementById('fp32-checkbox');
     const cpuForceControl = document.getElementById('cpu-force-control');
     const cpuForceCheckbox = document.getElementById('cpu-force-checkbox');

     // Main Content Elements
     const dropArea = document.getElementById('drop-area');
     const fileInput = document.getElementById('fileElem');
     const inputPreviewsContainer = document.getElementById('input-previews');
     const outputPreviewsContainer = document.getElementById('output-previews');

     // Other Global Elements
     const notificationArea = document.getElementById('notification-area');

     // Custom Control References
     const customThemeSelect = document.getElementById('theme-select-custom');
     const customModelSelect = document.getElementById('model-select-custom');
     const customSliders = document.querySelectorAll('.custom-slider');


     // --- Initialization ---
     function initialize() {
          initCustomSelects();
          loadTheme(); // Load theme from localStorage or default
          fetchConfigAndConnect();
          setupEventListeners();
          initCustomSliders();
          updateButtonStates();
     }

     // --- Theme Handling ---
     function applyTheme(theme) {
          // Use 'default' for the new vibrant theme
          const themeValue = theme === 'default' ? '' : theme;
          // Set attribute, or remove if 'default'
          if (themeValue) {
               document.body.setAttribute('data-theme', themeValue);
          } else {
               document.body.removeAttribute('data-theme'); // Default vibrant needs no attribute
          }
          // Ensure the custom select display is updated correctly even for 'default'
          if (customThemeSelect) {
               updateCustomSelectDisplay(customThemeSelect, theme);
          } else {
               console.error("Could not find custom theme select element to update display.");
          }
     }

     function loadTheme() {
          const savedTheme = localStorage.getItem('theme') || 'default'; // Default to 'default' (vibrant)
          if (!themeSelectNative) {
               console.error("Theme select native element not found.");
               applyTheme('default'); // Apply default theme visually
               return;
          }
          const availableThemes = Array.from(themeSelectNative.options).map(opt => opt.value);
          // Ensure 'default' is considered valid
          const validTheme = availableThemes.includes(savedTheme) ? savedTheme : 'default';
          themeSelectNative.value = validTheme; // Set native select
          applyTheme(validTheme); // Apply the theme visuals
          localStorage.setItem('theme', validTheme); // Save the potentially corrected theme
     }


     // --- Config & WebSocket ---
     function fetchConfigAndConnect() {
          fetch('/get_config')
               .then(response => {
                    if (!response.ok) {
                         // Try to get error text, otherwise use status
                         return response.text().then(text => {
                              throw new Error(`HTTP error! status: ${response.status}, message: ${text || 'No details'}`);
                         });
                    }
                    return response.json();
               })
               .then(data => {
                    console.log("Config received:", data);
                    config = data;
                    allowedExtensions = data.allowed_extensions || [];
                    updateFileInputAccept();
                    populateModelSelect(data.available_models, data.default_model);
                    setupControls(data); // Setup controls based on received config
                    updateGPUInfo(data.has_gpu, data.gpu_name);
                    updatePrecisionInfo(); // Update precision based on potentially new GPU info
                    checkSelectedModelStatus(); // Check status after config is loaded
                    connectWebSocket(); // Connect WS after getting config
               })
               .catch(error => {
                    console.error('Error fetching config:', error);
                    showNotification(`Error fetching config: ${error.message}. Please reload.`, 'error', null, 'fas fa-plug-circle-exclamation');
               });
     }
     function connectWebSocket() {
          const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
          const wsUrl = `${wsProtocol}//${window.location.host}/ws/${CLIENT_ID}`;
          console.log("Connecting WebSocket:", wsUrl);
          if (websocket && websocket.readyState !== WebSocket.CLOSED) {
               console.log("Closing existing WebSocket connection.");
               websocket.close();
          }
          websocket = new WebSocket(wsUrl);
          websocket.onopen = (event) => {
               console.log("WebSocket connection opened");
               showNotification('Connected to server', 'success', 2000, 'fas fa-check-circle');
               checkSelectedModelStatus(); // Re-check status on connect/reconnect
               updateButtonStates(); // Update buttons which might depend on connection
          };
          websocket.onmessage = (event) => {
               try {
                    const data = JSON.parse(event.data);
                    // console.log("WS msg received:", data); // Reduce console noise
                    handleWebSocketMessage(data);
               } catch (e) {
                    console.error("Failed to parse WS msg:", event.data, e);
               }
          };
          websocket.onerror = (event) => {
               console.error("WebSocket error:", event);
               // Only show reconnecting message if WS was trying to connect or was open
               if (!websocket || websocket.readyState === WebSocket.CONNECTING || websocket.readyState === WebSocket.OPEN) {
                    showNotification('WebSocket connection error. Trying to reconnect...', 'error', 5000, 'fas fa-plug-circle-exclamation');
               }
               // Consider calling connectWebSocket() here immediately or rely on onclose
          };
          websocket.onclose = (event) => {
               console.log(`WS closed: Code=${event.code}, Reason='${event.reason}', Clean=${event.wasClean}`);
               websocket = null; // Clear the reference
               updateButtonStates(); // Update UI to reflect offline state
               if (!event.wasClean) {
                    showNotification(`Connection lost (${event.code}). Reconnecting...`, 'error', null, 'fas fa-plug-circle-xmark');
                    setTimeout(connectWebSocket, 5000); // Attempt reconnect after 5s
               } else {
                    showNotification('Connection closed.', 'info', 3000, 'fas fa-info-circle');
               }
          };
     }
     function sendWebSocketMessage(type, payload) {
          if (websocket && websocket.readyState === WebSocket.OPEN) {
               const message = JSON.stringify({ type, payload });
               // console.log("Sending WS msg:", message); // Reduce console noise
               websocket.send(message);
          } else {
               console.error("WS not connected, cannot send:", type, payload);
               showNotification("Cannot communicate with server (Offline).", "error", 4000, 'fas fa-plug-circle-xmark');
               // Attempt reconnect if closed or null
               if (!websocket || websocket.readyState === WebSocket.CLOSED) {
                    console.log("Attempting immediate reconnect due to send failure.");
                    connectWebSocket();
               }
          }
     }
     function handleWebSocketMessage(data) {
          const { type, ...payload } = data;
          switch (type) {
               case 'gpu_info': handleGpuInfoUpdate(payload); break;
               case 'status_update': handleStatusUpdate(payload); break;
               case 'model_download_progress': // Combined progress/status for models
               case 'model_download_status': handleModelDownloadUpdate(payload); break;
               default: console.warn("Unknown WS msg type:", type, payload);
          }
     }
     function handleGpuInfoUpdate(payload) {
          console.log('GPU Info Updated:', payload);
          config.has_gpu = payload.has_gpu;
          config.gpu_name = payload.gpu_name;
          updateGPUInfo(payload.has_gpu, payload.gpu_name);
          // Safely show/hide controls
          if (fp32Control) fp32Control.style.display = payload.has_gpu ? 'flex' : 'none';
          if (cpuForceControl) cpuForceControl.style.display = payload.has_gpu ? 'flex' : 'none';
          handleCpuForceChange(); // Re-evaluate precision settings
     }
     function handleStatusUpdate(payload) {
          // console.log('Task Status Update:', payload); // Reduce noise
          updatePreviewStatus(payload.task_id, payload.status, payload.message, payload.progress, payload.output_url);
          const fileInfo = uploadedFiles[payload.task_id];
          const fileName = fileInfo ? fileInfo.originalName : 'image'; // Safely get name
          if (payload.status === 'error') {
               showNotification(`Error processing ${fileName}: ${payload.message || 'Unknown error'}`, 'error', 6000, 'fas fa-circle-exclamation');
          } else if (payload.status === 'complete') {
               showNotification(`Finished ${fileName}!`, 'success', 3000, 'fas fa-check-circle');
          } else if (payload.status === 'warning') {
               showNotification(`Warning for ${fileName}: ${payload.message || 'Unknown warning'}`, 'warning', 5000, 'fas fa-triangle-exclamation');
          }
     }
     function handleModelDownloadUpdate(data) {
          const modelName = data.model_name;
          if (!modelName) {
               console.warn("Received model download update without model_name:", data);
               return;
          }
          // Sanitize ID: Replace dots and maybe other chars, ensure uniqueness
          const notifId = `modeldl-${modelName.replace(/[^a-zA-Z0-9-_]/g, '_')}`;
          let message = data.message || `Model ${modelName}: ${data.status}`;
          let type = 'info';
          let duration = null; // Default to persistent for ongoing downloads
          let progress = data.progress;
          let icon = 'fas fa-info-circle';

          switch (data.status) {
               case 'started':
               case 'downloading':
                    type = 'download'; // Use download type for styling/progress bar
                    message = data.message || `Downloading ${modelName}...`;
                    icon = 'fas fa-download';
                    break;
               case 'complete':
               case 'exists':
                    type = 'success';
                    message = data.message || `${modelName} is ready!`;
                    duration = 4000; // Auto-dismiss success
                    progress = 100;
                    icon = 'fas fa-check-circle';
                    checkSelectedModelStatus(); // Re-check status now that model is ready
                    break;
               case 'error':
                    type = 'error';
                    message = data.message || `Error downloading ${modelName}.`;
                    duration = 6000; // Keep error longer
                    icon = 'fas fa-circle-exclamation';
                    checkSelectedModelStatus(); // Re-check status (button might need re-enabling)
                    break;
               default: type = 'info'; // Fallback
          }
          showNotification(message, type, duration, icon, progress, notifId);
     }

     function populateModelSelect(models, defaultModel) {
          if (!customModelSelect || !modelSelectNative) {
               console.error("Model select elements not found.");
               return;
          }
          const optionsContainer = customModelSelect.querySelector('.select-options');
          if (!optionsContainer) {
               console.error("Model select options container not found.");
               return;
          }
          modelSelectNative.innerHTML = '';
          optionsContainer.innerHTML = '';

          if (!models || models.length === 0) {
               console.warn("No available models received from config.");
               const defaultOptionText = "No models available";
               modelSelectNative.innerHTML = `<option disabled selected>${defaultOptionText}</option>`;
               updateCustomSelectDisplay(customModelSelect, defaultOptionText, true); // Update display to show message
               return;
          }

          models.forEach(modelName => {
               const nativeOption = document.createElement('option');
               nativeOption.value = modelName;
               nativeOption.textContent = modelName;
               if (modelName === defaultModel) {
                    nativeOption.selected = true;
               }
               modelSelectNative.appendChild(nativeOption);

               const customOption = document.createElement('div');
               customOption.classList.add('select-option');
               customOption.setAttribute('data-value', modelName);
               customOption.textContent = modelName;
               if (modelName === defaultModel) {
                    customOption.classList.add('selected');
               }
               optionsContainer.appendChild(customOption);
          });

          // Ensure defaultModel is valid, otherwise fallback to the first model
          const finalDefault = models.includes(defaultModel) ? defaultModel : models[0];
          modelSelectNative.value = finalDefault; // Set native select value
          updateCustomSelectDisplay(customModelSelect, finalDefault); // Update custom select visual
          modelSelectNative.dispatchEvent(new Event('change')); // Trigger change to load status etc.
     }

     function setupControls(data) {
          // Safely access config data with defaults
          const defaultScale = data.default_outscale || 4;
          const defaultFormat = 'png'; // Standard default
          const defaultFace = (data.default_face_enhance === true).toString(); // Ensure boolean -> string
          const defaultTile = data.default_tile || 0;
          const gfpganModuleAvailable = data.gfpgan_available === true;

          if (upscaleFactorNativeInput) upscaleFactorNativeInput.value = defaultScale;
          if (outputFormatNativeInput) outputFormatNativeInput.value = defaultFormat;
          if (faceEnhanceNativeInput) faceEnhanceNativeInput.value = defaultFace;
          if (tileSliderNative) tileSliderNative.value = defaultTile;
          if (tilePadSliderNative) tilePadSliderNative.value = 10; // Default padding
          if (prePadSliderNative) prePadSliderNative.value = 0; // Default padding

          // Update button groups visually
          if (upscaleFactorButtonsContainer) selectButton(upscaleFactorButtonsContainer, defaultScale.toString());
          if (outputFormatButtonsContainer) selectButton(outputFormatButtonsContainer, defaultFormat);
          if (faceEnhanceButtonsContainer) {
               selectButton(faceEnhanceButtonsContainer, defaultFace);
               // Disable/Enable based on backend availability
               faceEnhanceButtonsContainer.classList.toggle('disabled', !gfpganModuleAvailable);
               faceEnhanceButtonsContainer.querySelectorAll('button').forEach(btn => btn.disabled = !gfpganModuleAvailable);
               if (!gfpganModuleAvailable && faceEnhanceNativeInput.value === 'true') {
                    // If default is true but module missing, force selection to false
                    faceEnhanceNativeInput.value = 'false';
                    selectButton(faceEnhanceButtonsContainer, 'false');
               }
          }

          // Setup precision checkboxes based on GPU availability
          const hasGpuDetected = data.has_gpu === true;
          if (fp32Control) fp32Control.style.display = hasGpuDetected ? 'flex' : 'none';
          if (cpuForceControl) cpuForceControl.style.display = hasGpuDetected ? 'flex' : 'none';
          if (fp32Checkbox) fp32Checkbox.checked = !hasGpuDetected; // Default FP32 for CPU, FP16 for GPU
          if (cpuForceCheckbox) cpuForceCheckbox.checked = !hasGpuDetected; // Don't force CPU if GPU exists

          handleCpuForceChange(); // Apply initial logic based on checks
          initCustomSliders(); // Update all sliders' visuals after setting values

          // Update custom select displays (ensure elements exist)
          if (customThemeSelect && themeSelectNative) updateCustomSelectDisplay(customThemeSelect, themeSelectNative.value);
          if (customModelSelect && modelSelectNative) updateCustomSelectDisplay(customModelSelect, modelSelectNative.value);
     }

     function updateGPUInfo(hasGpu, gpuName) {
          if (gpuInfoName) {
               gpuInfoName.textContent = gpuName || (hasGpu ? 'GPU Detected' : 'CPU');
          }
     }
     function updatePrecisionInfo() {
          if (!precisionInfo || !cpuForceCheckbox || !fp32Checkbox) return; // Guard

          const forceCpu = cpuForceCheckbox.checked;
          const useFp32 = fp32Checkbox.checked;
          let precisionText = 'Unknown';
          if (config.has_gpu) {
               if (forceCpu) {
                    precisionText = 'FP32 (CPU Forced)';
               } else {
                    precisionText = useFp32 ? 'FP32 (GPU)' : 'FP16 (GPU)';
               }
          } else {
               precisionText = 'FP32 (CPU)';
          }
          precisionInfo.textContent = precisionText;
     }
     function updateFileInputAccept() {
          if (!fileInput) return;
          if (allowedExtensions.length > 0) {
               const acceptString = allowedExtensions.map(ext => `.${ext.toLowerCase()}`).join(',');
               fileInput.accept = `image/*,${acceptString}`;
          } else {
               fileInput.accept = 'image/*';
          }
          console.log("File input accept set to:", fileInput.accept);
     }

     // --- Model Status Checking ---
     function checkSelectedModelStatus() {
          // Guard: Ensure necessary elements exist and a valid model is selected
          if (!modelSelectNative || !modelSelectNative.value || modelSelectNative.value === 'Loading...' || modelSelectNative.value === 'No models available') {
               console.log("Skipping model status check: No valid model selected or elements missing.");
               if (modelStatusText) {
                    modelStatusText.textContent = 'Select Model';
                    modelStatusText.className = 'model-status-text error';
               }
               if (gfpganStatusText) {
                    gfpganStatusText.textContent = ''; // Clear GFPGAN status
                    gfpganStatusText.className = 'model-status-text';
               }
               if (downloadModelBtn) downloadModelBtn.style.display = 'none';
               if (downloadGfpganBtn) downloadGfpganBtn.style.display = 'none';
               updateButtonStates(); // Update buttons based on invalid state
               return;
          }
          // Guard: Ensure config is loaded before proceeding, especially for gfpgan_available
          if (!config || typeof config.gfpgan_available === 'undefined') {
               console.warn("Skipping model status check: Config not yet loaded.");
               // Optionally show a temporary status like "Waiting for config..."
               return;
          }


          const selectedModel = modelSelectNative.value;
          const faceEnhanceEnabled = faceEnhanceNativeInput ? faceEnhanceNativeInput.value === 'true' : false;

          // Reset UI elements safely
          if (modelStatusText) {
               modelStatusText.textContent = 'Checking...';
               modelStatusText.className = 'model-status-text checking';
          }
          if (gfpganStatusText) {
               gfpganStatusText.textContent = 'Checking...';
               gfpganStatusText.className = 'model-status-text checking';
          }
          if (downloadModelBtn) downloadModelBtn.style.display = 'none';
          if (downloadGfpganBtn) downloadGfpganBtn.style.display = 'none';


          const gfpganModuleAvailable = config.gfpgan_available === true;
          // Ensure face enhance buttons reflect availability correctly
          if (faceEnhanceButtonsContainer) {
               faceEnhanceButtonsContainer.classList.toggle('disabled', !gfpganModuleAvailable);
               faceEnhanceButtonsContainer.querySelectorAll('button').forEach(btn => btn.disabled = !gfpganModuleAvailable);
               if (!gfpganModuleAvailable) {
                    if (gfpganStatusText) {
                         gfpganStatusText.textContent = 'Module Unavailable';
                         gfpganStatusText.className = 'model-status-text error';
                    }
                    if (faceEnhanceNativeInput && faceEnhanceNativeInput.value === 'true') {
                         faceEnhanceNativeInput.value = 'false'; // Force off if module missing
                         selectButton(faceEnhanceButtonsContainer, 'false');
                    }
               }
          }


          if (websocket && websocket.readyState === WebSocket.OPEN) {
               // Construct URL safely
               const params = new URLSearchParams({
                    model_name: selectedModel,
                    face_enhance: faceEnhanceEnabled.toString() // Ensure boolean as string
               });
               const checkUrl = `/check_model_status?${params.toString()}`;
               console.log("Fetching model status:", checkUrl); // Log the URL being fetched

               fetch(checkUrl)
                    .then(response => {
                         if (!response.ok) {
                              // --- MORE ROBUST ERROR HANDLING ---
                              console.warn(`Model status check failed with status: ${response.status}`);
                              // Try to parse JSON first, then text, then use status text
                              return response.json()
                                   .catch(() => response.text()) // If JSON fails, try text
                                   .then(err => {
                                        // Extract detail if object with detail, otherwise use the text/status
                                        const detail = err?.detail || (typeof err === 'string' && err ? err : `Server error ${response.status} ${response.statusText}`);
                                        // Throw a new error with the extracted/constructed message
                                        throw new Error(detail);
                                   });
                              // --- END ROBUST ERROR HANDLING ---
                         }
                         // If response IS ok, parse JSON
                         return response.json();
                    })
                    .then(data => {
                         // --- ADD GUARD FOR RESPONSE DATA ---
                         if (!data || typeof data.models_status === 'undefined' || typeof data.all_exist === 'undefined') {
                              console.error("Invalid response structure received from /check_model_status:", data);
                              throw new Error("Received invalid status data from server.");
                         }
                         // --- END GUARD ---
                         console.log("Model Status Response:", data);
                         // Update UI only if data is valid
                         updateModelStatusUI(selectedModel, data.models_status, gfpganModuleAvailable);
                    })
                    .catch(error => {
                         // This catches network errors AND errors thrown from .then blocks
                         console.error('Error checking model status:', error);
                         if (modelStatusText) {
                              modelStatusText.textContent = `Check Failed`; // More specific error
                              modelStatusText.className = 'model-status-text error';
                         }
                         if (gfpganModuleAvailable && gfpganStatusText) {
                              gfpganStatusText.textContent = `Check Failed`;
                              gfpganStatusText.className = 'model-status-text error';
                         }
                         // Show specific error message from backend if available
                         showNotification(`Error checking model status: ${error.message}`, 'error', 6000, 'fas fa-plug-circle-exclamation');
                         updateButtonStates(); // Ensure buttons are updated on error
                    });
          } else {
               console.warn("WebSocket not connected, skipping model status check.");
               if (modelStatusText) {
                    modelStatusText.textContent = 'Offline';
                    modelStatusText.className = 'model-status-text error';
               }
               if (gfpganStatusText) {
                    gfpganStatusText.textContent = gfpganModuleAvailable ? 'Offline' : 'Module Unavailable';
                    gfpganStatusText.className = 'model-status-text error';
               }
               updateButtonStates();
          }
     }
     function updateModelStatusUI(modelName, modelsStatus, gfpganModuleAvailable) {
          // Guard: Ensure required elements exist
          if (!modelStatusText || !gfpganStatusText || !downloadModelBtn || !downloadGfpganBtn || !faceEnhanceButtonsContainer) {
               console.error("Cannot update model status UI: One or more required elements are missing.");
               return;
          }
          // Guard: Ensure modelsStatus is a valid object
          if (typeof modelsStatus !== 'object' || modelsStatus === null) {
               console.error("Cannot update model status UI: Invalid modelsStatus data received:", modelsStatus);
               modelStatusText.textContent = 'Invalid Status Data';
               modelStatusText.className = 'model-status-text error';
               gfpganStatusText.textContent = 'Invalid Status Data';
               gfpganStatusText.className = 'model-status-text error';
               updateButtonStates();
               return;
          }


          let mainModelFiles = [];
          // Determine expected files based on backend logic (realesrgan_logic.py)
          // Use the globally defined MODEL_INFO for safety if config not fully loaded? No, rely on checkSelectedModelStatus guard.
          if (modelName === 'realesr-general-x4v3') {
               mainModelFiles.push('realesr-general-wdn-x4v3.pth'); // WDN model filename
               mainModelFiles.push('realesr-general-x4v3.pth'); // Main model filename
          } else if (modelName && MODEL_INFO[modelName]) { // Check if model is in known list
               // Derive the expected filename based on convention (e.g., RealESRGAN_x4plus.pth)
               mainModelFiles.push(`${modelName}.pth`);
          } else {
               // This case should ideally be caught before calling this function, but handle defensively
               console.warn(`updateModelStatusUI called with unknown or empty modelName: ${modelName}`);
               modelStatusText.textContent = 'Select Valid Model';
               modelStatusText.className = 'model-status-text error';
               gfpganStatusText.textContent = '';
               gfpganStatusText.className = 'model-status-text';
               updateButtonStates();
               return;
          }

          // Check existence using the received modelsStatus data
          const mainModelsExist = mainModelFiles.every(file => modelsStatus[file] === true);

          // --- Rest of the function remains the same ---
          if (mainModelsExist) {
               modelStatusText.textContent = 'Model Available';
               modelStatusText.className = 'model-status-text available';
               downloadModelBtn.style.display = 'none';
          } else {
               modelStatusText.textContent = 'Model Missing';
               modelStatusText.className = 'model-status-text unavailable';
               downloadModelBtn.style.display = 'inline-flex';
               downloadModelBtn.disabled = false;
               const downloadName = modelName === 'realesr-general-x4v3' ? 'General Models' : modelName;
               const btnSpan = downloadModelBtn.querySelector('span');
               if (btnSpan) btnSpan.textContent = `Download ${downloadName}`;
          }

          if (gfpganModuleAvailable) {
               // Use GFPGAN_MODEL_NAME constant to check status
               const gfpganModelExists = modelsStatus[GFPGAN_MODEL_NAME] === true;
               if (gfpganModelExists) {
                    gfpganStatusText.textContent = 'Model Available';
                    gfpganStatusText.className = 'model-status-text available';
                    downloadGfpganBtn.style.display = 'none';
               } else {
                    gfpganStatusText.textContent = 'Model Missing';
                    gfpganStatusText.className = 'model-status-text unavailable';
                    downloadGfpganBtn.style.display = 'inline-flex';
                    downloadGfpganBtn.disabled = false;
                    const btnSpan = downloadGfpganBtn.querySelector('span');
                    if (btnSpan) btnSpan.textContent = `Download GFPGAN`;
               }
               // Ensure buttons are enabled if module is available
               faceEnhanceButtonsContainer.classList.remove('disabled');
               faceEnhanceButtonsContainer.querySelectorAll('button').forEach(btn => btn.disabled = false);
          } else {
               // Module unavailable case
               gfpganStatusText.textContent = 'Module Unavailable';
               gfpganStatusText.className = 'model-status-text error';
               downloadGfpganBtn.style.display = 'none';
               faceEnhanceButtonsContainer.classList.add('disabled');
               faceEnhanceButtonsContainer.querySelectorAll('button').forEach(btn => btn.disabled = true);
          }

          updateButtonStates(); // Update buttons based on the final determined status
     }


     // --- Custom Control Initializers & Handlers ---
     function initCustomSelects() {
          document.querySelectorAll('.custom-select').forEach(select => {
               const selectedDisplay = select.querySelector('.select-selected');
               const optionsContainer = select.querySelector('.select-options');
               const targetSelectId = select.dataset.targetSelect;
               const nativeSelect = document.getElementById(targetSelectId);
               if (!nativeSelect || !selectedDisplay || !optionsContainer) {
                    console.error("Custom select structure invalid for:", select.id);
                    return;
               }
               updateCustomSelectDisplay(select, nativeSelect.value); // Sync initial display

               selectedDisplay.addEventListener('click', (e) => {
                    if (select.classList.contains('disabled')) return;
                    e.stopPropagation();
                    closeAllSelects(select);
                    select.classList.toggle('open');
               });

               optionsContainer.addEventListener('click', (e) => {
                    if (select.classList.contains('disabled')) return;
                    const option = e.target.closest('.select-option'); // Handle clicks on text inside option too
                    if (option && option.dataset.value !== undefined) { // Ensure option has value
                         const value = option.dataset.value;
                         nativeSelect.value = value; // Set native select
                         updateCustomSelectDisplay(select, value); // Update custom display
                         select.classList.remove('open');
                         nativeSelect.dispatchEvent(new Event('change', { bubbles: true })); // Trigger change event
                    }
               });
          });
          // Close dropdowns if clicking outside
          document.addEventListener('click', (e) => {
               if (!e.target.closest('.custom-select')) {
                    closeAllSelects(null);
               }
          });
     }
     function closeAllSelects(exceptThisOne) {
          document.querySelectorAll('.custom-select').forEach(select => {
               if (select !== exceptThisOne) {
                    select.classList.remove('open');
               }
          });
     }
     function updateCustomSelectDisplay(customSelectElement, newValue, forceText = false) {
          const selectedDisplay = customSelectElement.querySelector('.select-selected');
          const optionsContainer = customSelectElement.querySelector('.select-options');
          if (!selectedDisplay || !optionsContainer) {
               console.error("Cannot update custom select display: elements missing for", customSelectElement.id);
               return;
          }

          let newText = '';
          // Find the custom option element corresponding to the new value
          const selectedOption = optionsContainer.querySelector(`.select-option[data-value="${newValue}"]`);

          if (selectedOption) {
               newText = selectedOption.textContent;
               // Update visual selection state in custom dropdown
               optionsContainer.querySelectorAll('.select-option').forEach(opt => opt.classList.remove('selected'));
               selectedOption.classList.add('selected');
          } else if (forceText) {
               // If forceText is true, display the newValue directly (e.g., for "No models available")
               newText = newValue;
               optionsContainer.querySelectorAll('.select-option').forEach(opt => opt.classList.remove('selected'));
          } else {
               // Fallback: If custom option not found (e.g., initial load before population?), use native select's text or the value itself
               const nativeSelect = document.getElementById(customSelectElement.dataset.targetSelect);
               const nativeOption = nativeSelect?.querySelector(`option[value="${newValue}"]`);
               newText = nativeOption?.textContent || newValue || 'Select...'; // Use native text or value or default
               console.warn(`Option value "${newValue}" not found in custom select: ${customSelectElement.id}. Using fallback text: "${newText}".`);
               // Clear visual selection if value is not in custom options
               optionsContainer.querySelectorAll('.select-option').forEach(opt => opt.classList.remove('selected'));
          }
          selectedDisplay.textContent = newText;
          selectedDisplay.title = newText; // Add tooltip for potentially truncated text
     }

     function initCustomSliders() {
          customSliders.forEach(sliderWrapper => {
               const targetSliderId = sliderWrapper.dataset.targetSlider;
               if (!targetSliderId) {
                    console.error("Custom slider wrapper missing data-target-slider attribute:", sliderWrapper);
                    return;
               }
               const nativeSlider = document.getElementById(targetSliderId);
               if (!nativeSlider) {
                    console.error(`Native slider #${targetSliderId} not found for wrapper:`, sliderWrapper);
                    return;
               }
               updateSliderVisuals(nativeSlider); // Initial positioning and value display
          });
     }
     function updateSliderVisuals(nativeSlider) {
          if (!nativeSlider) return; // Guard
          const sliderWrapper = document.querySelector(`.custom-slider[data-target-slider="${nativeSlider.id}"]`);
          if (!sliderWrapper) {
               // console.warn(`Custom slider wrapper not found for native slider: #${nativeSlider.id}`);
               return; // Might happen if elements are dynamically removed
          }
          const thumb = sliderWrapper.querySelector('.slider-thumb');
          const fill = sliderWrapper.querySelector('.slider-fill');
          const track = sliderWrapper.querySelector('.slider-track');
          if (!thumb || !fill || !track) {
               console.error(`Slider visual elements (thumb, fill, track) missing for #${nativeSlider.id}`);
               return;
          }

          const min = parseFloat(nativeSlider.min);
          const max = parseFloat(nativeSlider.max);
          const value = parseFloat(nativeSlider.value);
          const percentage = max === min ? 0 : ((value - min) / (max - min)) * 100;

          // Position thumb based on percentage, centered on the value
          // The thumb's `left` style should be the percentage of the track width
          thumb.style.left = `${percentage}%`;
          fill.style.width = `${percentage}%`;

          updateSliderValueDisplay(nativeSlider); // Update associated value display
     }
     function updateSliderValueDisplay(nativeSlider) {
          if (!nativeSlider) return; // Guard
          const displaySpanId = nativeSlider.id.replace('-slider', '-value');
          const displaySpan = document.getElementById(displaySpanId);
          if (displaySpan) {
               let displayValue = nativeSlider.value;
               // Format denoise value specifically
               if (nativeSlider.id === 'denoise-slider') {
                    displayValue = parseFloat(nativeSlider.value).toFixed(2);
               }

               // Clear existing text content before adding new value/suffix
               displaySpan.textContent = ''; // Clear first
               displaySpan.appendChild(document.createTextNode(displayValue)); // Add value as text node

               // Add suffixes or extra text
               if (nativeSlider.id === 'tile-pad-slider' || nativeSlider.id === 'pre-pad-slider') {
                    displaySpan.insertAdjacentText('beforeend', 'px');
               } else if (nativeSlider.id === 'tile-slider' && displayValue === '0') {
                    displaySpan.insertAdjacentText('beforeend', ' (auto)');
               }
          }
     }


     // --- Button Group Handling ---
     function selectButton(buttonGroupContainer, value) {
          if (!buttonGroupContainer) return; // Guard
          const nativeInput = buttonGroupContainer.previousElementSibling;
          // Ensure nativeInput is the correct hidden input
          if (nativeInput && nativeInput.tagName === 'INPUT' && nativeInput.type === 'hidden') {
               nativeInput.value = value;
          } else {
               console.warn("Could not find associated native hidden input for button group:", buttonGroupContainer);
          }
          // Update button visual state
          buttonGroupContainer.querySelectorAll('button').forEach(btn => {
               btn.classList.toggle('selected', btn.dataset.value === value);
          });
          // console.log(`Button group ${nativeInput?.id || 'unknown'} set to:`, value); // Reduce noise
     }

     function handleButtonGroupClick(event) {
          const button = event.target.closest('button');
          // Check button exists, has value, is not disabled, and its parent group is not disabled
          if (button && button.dataset.value !== undefined && !button.disabled) {
               const container = button.closest('.button-group');
               if (container && !container.classList.contains('disabled')) {
                    selectButton(container, button.dataset.value);
                    const nativeInput = container.previousElementSibling;
                    // Dispatch change event on the hidden input
                    if (nativeInput && nativeInput.tagName === 'INPUT' && nativeInput.type === 'hidden') {
                         nativeInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }
               }
          }
     }


     // --- Event Listeners Setup ---
     function setupEventListeners() {
          // Theme Select (Native Change Handler)
          if (themeSelectNative) {
               themeSelectNative.addEventListener('change', (e) => {
                    applyTheme(e.target.value);
                    localStorage.setItem('theme', e.target.value);
               });
          }
          if (modelSelectNative) {
               modelSelectNative.addEventListener('change', handleModelChange);
          }

          // Button Groups (Event Delegation on a suitable parent)
          const controlsCont = document.querySelector('.controls-container');
          if (controlsCont) {
               controlsCont.addEventListener('click', (event) => {
                    // Delegate click for button groups
                    if (event.target.closest('.button-group button')) {
                         handleButtonGroupClick(event);
                    }
                    // Delegate click for small download buttons
                    else if (event.target.closest('#download-model-btn')) {
                         if (modelSelectNative) handleDownloadModelClick(modelSelectNative.value);
                    }
                    else if (event.target.closest('#download-gfpgan-btn')) {
                         handleDownloadModelClick(GFPGAN_MODEL_NAME);
                    }
               });
          }
          // Need explicit listener on hidden input change if other actions depend on it
          if (faceEnhanceNativeInput) faceEnhanceNativeInput.addEventListener('change', checkSelectedModelStatus);


          // Custom Sliders (Event delegation on a common parent or individual listeners)
          document.querySelectorAll('.custom-slider').forEach(sliderWrapper => {
               const targetSliderId = sliderWrapper.dataset.targetSlider;
               if (!targetSliderId) return;
               const nativeSlider = document.getElementById(targetSliderId);
               if (!nativeSlider) return;

               let isDragging = false;

               const handleInteraction = (event) => {
                    // Prevent default only for touchmove to allow scrolling elsewhere
                    if (event.type === 'touchmove') event.preventDefault();

                    const track = sliderWrapper.querySelector('.slider-track');
                    if (!track) return; // Need track for calculations
                    const rect = track.getBoundingClientRect();
                    const clientX = event.clientX ?? event.touches?.[0]?.clientX;
                    if (clientX === undefined) return;

                    const offsetX = Math.max(0, Math.min(rect.width, clientX - rect.left));
                    const trackWidth = track.offsetWidth;
                    if (trackWidth <= 0) return; // Avoid division by zero

                    let percentage = (offsetX / trackWidth) * 100;

                    const min = parseFloat(nativeSlider.min);
                    const max = parseFloat(nativeSlider.max);
                    const step = parseFloat(nativeSlider.step) || 1;

                    let value = min + (percentage / 100) * (max - min);
                    // Round to the nearest step
                    value = Math.round(value / step) * step;
                    // Ensure value stays within min/max bounds
                    value = Math.max(min, Math.min(max, value));

                    // Update only if value changed to avoid unnecessary events/updates
                    if (nativeSlider.value != value) {
                         nativeSlider.value = value;
                         updateSliderVisuals(nativeSlider); // Update visuals immediately
                         // Trigger 'input' event while dragging for live updates elsewhere if needed
                         nativeSlider.dispatchEvent(new Event('input', { bubbles: true }));
                    }
               };

               const startDragging = (event) => {
                    // Ignore right clicks
                    if (event.button === 2) return;
                    // Prevent text selection during drag
                    event.preventDefault();

                    isDragging = true;
                    sliderWrapper.classList.add('dragging');
                    // Add listeners to document to handle movement outside the element
                    document.addEventListener('mousemove', handleMove);
                    document.addEventListener('touchmove', handleMove, { passive: false }); // Need preventDefault
                    document.addEventListener('mouseup', stopDragging);
                    document.addEventListener('touchend', stopDragging);
                    document.addEventListener('mouseleave', stopDragging); // Handle mouse leaving window

                    handleInteraction(event); // Update on initial click/touch
               };

               const handleMove = (event) => {
                    if (!isDragging) return;
                    handleInteraction(event);
               };


               const stopDragging = () => {
                    if (!isDragging) return;
                    isDragging = false;
                    sliderWrapper.classList.remove('dragging');
                    // Remove document listeners
                    document.removeEventListener('mousemove', handleMove);
                    document.removeEventListener('touchmove', handleMove);
                    document.removeEventListener('mouseup', stopDragging);
                    document.removeEventListener('touchend', stopDragging);
                    document.removeEventListener('mouseleave', stopDragging);

                    // Trigger final 'change' event when dragging stops
                    nativeSlider.dispatchEvent(new Event('change', { bubbles: true }));
               };

               sliderWrapper.addEventListener('mousedown', startDragging);
               sliderWrapper.addEventListener('touchstart', startDragging, { passive: true }); // Let touchstart be passive
          });


          // Other Native Element Listeners
          if (fp32Checkbox) fp32Checkbox.addEventListener('change', updatePrecisionInfo);
          if (cpuForceCheckbox) cpuForceCheckbox.addEventListener('change', handleCpuForceChange);
          if (toggleAdvancedBtn) toggleAdvancedBtn.addEventListener('click', handleToggleAdvanced);

          // File Drop/Select
          if (dropArea && fileInput) {
               ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evName => dropArea.addEventListener(evName, preventDefaults, false));
               ['dragenter', 'dragover'].forEach(evName => dropArea.addEventListener(evName, () => dropArea.classList.add('dragover'), false));
               ['dragleave', 'drop'].forEach(evName => dropArea.addEventListener(evName, () => dropArea.classList.remove('dragover'), false));
               dropArea.addEventListener('drop', handleDrop, false);
               fileInput.addEventListener('change', handleFileSelect, false);
               dropArea.addEventListener('click', (e) => {
                    // Trigger file input only if clicking the drop area itself or its direct children (p, i)
                    if (e.target === dropArea || e.target.closest('.fa-upload') || e.target.closest('p')) {
                         fileInput.click();
                    }
               });
          }

          // Action Buttons (Sidebar) - Ensure they exist
          if (enhanceButton) enhanceButton.addEventListener('click', handleEnhanceClick);
          if (clearButton) clearButton.addEventListener('click', handleClearClick);
          if (downloadAllButton) downloadAllButton.addEventListener('click', handleDownloadAllClick);
          // Download model buttons handled via delegation above

          // Dynamic Listeners for Preview Buttons (using event delegation)
          if (inputPreviewsContainer) inputPreviewsContainer.addEventListener('click', handleInputPreviewAction);
          if (outputPreviewsContainer) outputPreviewsContainer.addEventListener('click', handleOutputPreviewAction);
     }

     function handleToggleAdvanced() {
          if (!advancedOptionsDiv || !toggleAdvancedBtn) return;
          const isHidden = advancedOptionsDiv.classList.toggle('hidden');
          const icon = toggleAdvancedBtn.querySelector('i');
          // Safely update text content of the first child node (the text)
          if (toggleAdvancedBtn.firstChild) {
               toggleAdvancedBtn.firstChild.textContent = isHidden ? 'Show Advanced ' : 'Hide Advanced ';
          }
          if (icon) {
               icon.classList.toggle('fa-chevron-down', isHidden);
               icon.classList.toggle('fa-chevron-up', !isHidden);
          }
          // Recalculate slider visuals if opening, might have been hidden initially
          if (!isHidden) {
               setTimeout(() => {
                    advancedOptionsDiv.querySelectorAll('.custom-slider').forEach(sliderWrapper => {
                         const nativeSlider = document.getElementById(sliderWrapper.dataset.targetSlider);
                         if (nativeSlider) {
                              updateSliderVisuals(nativeSlider);
                         }
                    });
               }, 50); // Small delay for transition
          }
     }
     function handleCpuForceChange() {
          if (!cpuForceCheckbox || !fp32Checkbox || !config) return;

          const forceCpu = cpuForceCheckbox.checked;
          if (forceCpu && config.has_gpu) {
               fp32Checkbox.checked = true; // Force FP32 when CPU is forced
               fp32Checkbox.disabled = true; // Disable user choice
          } else if (config.has_gpu) {
               fp32Checkbox.disabled = false; // Allow user choice on GPU
               // fp32Checkbox.checked = false; // Optionally default to FP16 when GPU enabled? Current logic keeps user choice unless forced.
          } else {
               // No GPU available
               fp32Checkbox.checked = true; // Must use FP32 on CPU
               fp32Checkbox.disabled = true; // Disable user choice
          }
          updatePrecisionInfo(); // Update display text
     }
     function handleModelChange() {
          if (!modelSelectNative || !denoiseControl || !denoiseSliderNative) return;
          const selectedModel = modelSelectNative.value;
          const showDenoise = selectedModel === 'realesr-general-x4v3';
          denoiseControl.style.display = showDenoise ? 'block' : 'none';
          if (showDenoise) {
               updateSliderVisuals(denoiseSliderNative); // Ensure denoise slider visual is correct
          }
          checkSelectedModelStatus(); // Check availability of the new model
     }
     function preventDefaults(e) {
          e.preventDefault();
          e.stopPropagation();
     }

     // --- File Handling & Preview Logic ---
     function handleDrop(e) {
          handleFiles(e.dataTransfer.files);
     }
     function handleFileSelect(e) {
          if (!e.target.files) return;
          handleFiles(e.target.files);
          e.target.value = null; // Reset input to allow selecting the same file again
     }
     function handleFiles(files) {
          if (!files || files.length === 0) return;

          const filesToUpload = [];
          let skippedCount = 0;
          // Use Set for efficient lookup
          const currentExtSet = new Set((allowedExtensions || []).map(ext => ext.toLowerCase()));

          for (let i = 0; i < files.length; i++) {
               const file = files[i];
               // Basic file validation
               if (!file || !file.name || !file.type) {
                    console.warn("Skipping invalid file object:", file);
                    skippedCount++;
                    continue;
               }
               const fileExt = file.name.split('.').pop()?.toLowerCase();
               const mimeMatch = file.type.startsWith('image/');
               // Check extension only if a list is provided, otherwise allow all image/* types
               const extMatch = currentExtSet.size > 0 ? (fileExt && currentExtSet.has(fileExt)) : true;

               if (mimeMatch && extMatch) {
                    filesToUpload.push(file);
               } else {
                    console.warn(`Skipping file: ${file.name}. Type: ${file.type}, Ext: ${fileExt}. Allowed: ${allowedExtensions.join(', ')}`);
                    showNotification(`Skipped invalid file type/extension: ${file.name}`, 'warning', 4000, 'fas fa-file-circle-exclamation');
                    skippedCount++;
               }
          }

          if (skippedCount > 0) console.warn(`Skipped ${skippedCount} invalid files.`);
          if (filesToUpload.length > 0) {
               uploadFiles(filesToUpload);
          } else if (skippedCount === 0 && files.length > 0) {
               // Only show if files were dropped/selected but none were valid
               showNotification('No valid image files selected/dropped.', 'warning', 4000, 'fas fa-folder-open');
          }
          updateButtonStates(); // Update buttons after processing files
     }
     function uploadFiles(files) {
          const formData = new FormData();
          formData.append('client_id', CLIENT_ID);
          files.forEach(file => formData.append('files', file, file.name)); // Include filename

          const notifId = showNotification(`Uploading ${files.length} file(s)...`, 'info', null, 'fas fa-upload');

          fetch('/upload', { method: 'POST', body: formData })
               .then(response => {
                    if (!response.ok) {
                         // Try to parse error detail from JSON response, fallback to status text
                         return response.json()
                              .catch(() => response.text()) // If json fails, get text
                              .then(err => {
                                   const detail = err?.detail || (typeof err === 'string' ? err : `Server error ${response.status}`);
                                   throw new Error(detail);
                              });
                    }
                    return response.json();
               })
               .then(data => {
                    console.log("Upload response:", data);
                    dismissNotification(notifId); // Dismiss 'Uploading...' notification
                    let successCount = 0;
                    if (data.uploaded_files && Array.isArray(data.uploaded_files)) {
                         data.uploaded_files.forEach(fileInfo => {
                              // Find the original File object based on name (needed for blob URL)
                              const originalFile = files.find(f => f.name === fileInfo.original_name);
                              addPreview(fileInfo, originalFile);
                              successCount++;
                         });
                    }
                    if (data.errors && Array.isArray(data.errors)) {
                         data.errors.forEach(err => showNotification(`Upload Error (${err.original_name}): ${err.error || 'Unknown reason'}`, 'error', 6000, 'fas fa-circle-exclamation'));
                    }

                    if (successCount > 0) {
                         showNotification(`Uploaded ${successCount} file(s). Ready to enhance.`, 'success', 3000, 'fas fa-check-circle');
                    } else if (!data.errors || data.errors.length === 0) {
                         // Only show warning if no files were processed and no specific errors were reported
                         if (files.length > 0 && data.message) {
                              showNotification(data.message || 'Upload processed, but no new files were added.', 'warning', 4000, 'fas fa-triangle-exclamation');
                         }
                    }
               })
               .catch(error => {
                    console.error('Error uploading files:', error);
                    dismissNotification(notifId);
                    showNotification(`Upload failed: ${error.message || 'Network error or server issue'}`, 'error', 6000, 'fas fa-circle-xmark');
               })
               .finally(() => {
                    updateButtonStates(); // Update buttons regardless of outcome
               });
     }

     function addPreview(fileInfo, originalFile) {
          if (!fileInfo || !fileInfo.task_id) {
               console.error("Invalid fileInfo received for preview:", fileInfo);
               return;
          }
          const taskId = fileInfo.task_id;
          if (uploadedFiles[taskId]) {
               console.warn(`Preview for task ${taskId} (${fileInfo.original_name}) already exists. Skipping.`);
               return;
          }
          if (!inputPreviewsContainer) {
               console.error("Input previews container not found.");
               return;
          }

          const inputThumb = document.createElement('div');
          inputThumb.classList.add('preview-thumbnail', 'pending'); // Start as pending
          inputThumb.setAttribute('data-task-id', taskId);
          inputThumb.title = `${fileInfo.original_name}\nSize: ${(fileInfo.size / 1024 / 1024).toFixed(2)} MB`;

          const img = document.createElement('img');
          img.alt = fileInfo.original_name;

          const statusOverlay = document.createElement('div');
          statusOverlay.classList.add('status-overlay');
          statusOverlay.innerHTML = `
          <i class="fa-solid fa-spinner"></i>      <!-- processing -->
          <i class="fa-solid fa-clock"></i>        <!-- pending/queued -->
          <i class="fa-solid fa-exclamation-triangle"></i> <!-- error -->
          <i class="fa-solid fa-check"></i>        <!-- complete (briefly) -->
          <span class="status-overlay-text">Pending</span>
      `;
          inputThumb.appendChild(statusOverlay);

          let blobUrl = null;
          if (originalFile instanceof File) { // Check if it's a valid File object
               try {
                    blobUrl = URL.createObjectURL(originalFile);
                    img.src = blobUrl;
                    img.onerror = () => {
                         console.error(`Error loading blob URL for ${fileInfo.originalName}`);
                         img.alt += " (Preview Load Error)";
                         // Maybe show a placeholder/icon if blob loading fails
                         inputThumb.classList.add('load-error'); // Add class for styling placeholder
                    };
               } catch (e) {
                    console.error(`Error creating object URL for ${fileInfo.originalName}:`, e);
                    img.alt += " (Preview Gen Error)";
                    inputThumb.classList.add('load-error');
               }
          } else {
               console.warn(`Original file object missing or invalid for ${fileInfo.originalName}. Preview may not load.`);
               img.alt += " (Preview unavailable)";
               inputThumb.classList.add('load-error'); // Style placeholder
          }
          inputThumb.appendChild(img);

          const removeBtn = document.createElement('button');
          removeBtn.classList.add('thumbnail-action-btn', 'thumbnail-remove-btn');
          removeBtn.title = 'Remove this image';
          removeBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
          removeBtn.setAttribute('data-action', 'remove');
          removeBtn.setAttribute('aria-label', `Remove ${fileInfo.original_name}`);
          inputThumb.appendChild(removeBtn);

          inputPreviewsContainer.appendChild(inputThumb);

          uploadedFiles[taskId] = {
               originalName: fileInfo.original_name,
               size: fileInfo.size,
               status: 'pending', // Initial status
               outputUrl: null,
               inputElement: inputThumb,
               outputElement: null, // Will be created on completion
               blobUrl: blobUrl // Store for potential revocation
          };
          updateButtonStates();
     }

     function handleInputPreviewAction(event) {
          const button = event.target.closest('.thumbnail-action-btn');
          if (!button) return;

          const action = button.dataset.action;
          const thumbnail = button.closest('.preview-thumbnail');
          const taskId = thumbnail?.dataset.taskId;

          if (!taskId || !uploadedFiles[taskId]) {
               console.warn("Action triggered on preview with invalid taskId or missing fileInfo:", taskId);
               return;
          }

          if (action === 'remove') {
               const fileInfo = uploadedFiles[taskId];
               if ((fileInfo.status === 'processing' || fileInfo.status === 'queued') && !confirm(`Image "${fileInfo.originalName}" might be processing. Remove anyway?`)) {
                    return; // User cancelled removal of active task
               }
               removeImage(taskId);
          }
     }

     function handleOutputPreviewAction(event) {
          const button = event.target.closest('.thumbnail-action-btn');
          if (!button) return;

          const action = button.dataset.action;
          const thumbnail = button.closest('.preview-thumbnail');
          const taskId = thumbnail?.dataset.taskId;

          if (!taskId || !uploadedFiles[taskId]) {
               console.warn("Action triggered on output preview with invalid taskId or missing fileInfo:", taskId);
               return;
          }

          if (action === 'download') {
               const fileInfo = uploadedFiles[taskId];
               if (fileInfo.outputUrl) {
                    const a = document.createElement('a');
                    a.href = fileInfo.outputUrl;
                    // Generate download filename based on original name and stored params/defaults
                    const nameParts = fileInfo.originalName.split('.');
                    // Use extension from the last successful enhancement, or fallback to current setting/default
                    const outputExt = config?.lastUsedParams?.ext || outputFormatNativeInput?.value || 'png';
                    const baseName = nameParts.slice(0, -1).join('.') || fileInfo.originalName; // Handle names with/without extension
                    a.download = `${baseName}_enhanced.${outputExt}`;
                    document.body.appendChild(a); // Required for FF
                    a.click();
                    document.body.removeChild(a); // Clean up
                    showNotification(`Downloading ${a.download}...`, 'success', 3000, 'fas fa-download');
               } else {
                    console.error(`Output URL missing for completed task ${taskId} (${fileInfo.originalName}).`);
                    showNotification(`Could not download: Output URL missing for ${fileInfo.originalName}.`, 'error', 4000, 'fas fa-circle-exclamation');
               }
          }
     }

     function removeImage(taskId) {
          const fileInfo = uploadedFiles[taskId];
          if (!fileInfo) return;

          console.log(`Removing image for task: ${taskId} (${fileInfo.originalName})`);
          const wasActive = (fileInfo.status === 'processing' || fileInfo.status === 'queued');

          // Remove DOM elements safely
          fileInfo.inputElement?.remove();
          fileInfo.outputElement?.remove();

          // Revoke blob URL if it exists
          if (fileInfo.blobUrl) {
               try { URL.revokeObjectURL(fileInfo.blobUrl); } catch (e) { console.warn("Error revoking blob URL:", e); }
          }

          // Remove from local state
          delete uploadedFiles[taskId];
          processingQueue.delete(taskId);

          // --- Send requests to backend ---
          // 1. Tell backend to delete the task's data (input/output files)
          fetch('/delete_task', { // Assuming endpoint exists - ADD THIS TO PYTHON IF MISSING
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({ client_id: CLIENT_ID, task_id: taskId })
          })
               .then(response => response.json().catch(() => ({ message: "Delete request sent (non-JSON response)." }))) // Handle non-json response
               .then(data => console.log(data.message || `Delete request for ${taskId} processed.`))
               .catch(error => console.error(`Error sending delete request for ${taskId}:`, error));

          // 2. If it was actively processing/queued, send a cancellation request via WebSocket
          if (wasActive) {
               console.log(`Sending cancel request for active task ${taskId}`);
               sendWebSocketMessage('cancel_task', { task_id: taskId }); // ADD THIS HANDLER TO PYTHON WS IF MISSING
          }

          updateButtonStates(); // Update UI
          showNotification(`Removed ${fileInfo.originalName}.`, 'info', 2500, 'fas fa-trash-can');
     }


     // --- Enhancement Process ---
     function handleEnhanceClick() {
          // Filter tasks that are currently in 'pending' state
          const taskIdsToProcess = Object.keys(uploadedFiles).filter(taskId => uploadedFiles[taskId]?.status === 'pending');

          if (taskIdsToProcess.length === 0) {
               showNotification('No pending images to enhance.', 'warning', 4000, 'fas fa-images');
               return;
          }

          if (!modelSelectNative || !modelSelectNative.value || modelSelectNative.value === 'Loading...' || modelSelectNative.value === 'No models available') {
               showNotification('Please select a valid upscaling model first.', 'warning', 4000, 'fas fa-hand-pointer');
               return;
          }
          const selectedModel = modelSelectNative.value;
          const faceEnhanceEnabled = faceEnhanceNativeInput ? faceEnhanceNativeInput.value === 'true' : false;

          // Pre-check model status before sending enhance request
          const params = new URLSearchParams({
               model_name: selectedModel,
               face_enhance: faceEnhanceEnabled.toString()
          });
          fetch(`/check_model_status?${params.toString()}`)
               .then(response => { if (!response.ok) return response.json().then(err => { throw new Error(err.detail || `Server error ${response.status}`) }); return response.json(); })
               .then(data => {
                    // Check if all *required* models exist based on current selection
                    let allRequiredExist = true;
                    let missingModels = [];

                    // Determine required files based on selection
                    let requiredFiles = [];
                    if (selectedModel === 'realesr-general-x4v3') {
                         requiredFiles.push('realesr-general-wdn-x4v3.pth', 'realesr-general-x4v3.pth');
                    } else if (selectedModel && MODEL_INFO[selectedModel]) {
                         requiredFiles.push(`${selectedModel}.pth`);
                    }

                    if (faceEnhanceEnabled && config.gfpgan_available === true) {
                         requiredFiles.push(GFPGAN_MODEL_NAME);
                    }

                    // Check against the status returned from the server
                    requiredFiles.forEach(file => {
                         if (data.models_status[file] !== true) {
                              allRequiredExist = false;
                              missingModels.push(file);
                         }
                    });


                    if (!allRequiredExist) {
                         const missingList = missingModels.join(', ');
                         showNotification(`Missing required model(s): ${missingList}. Check status & download.`, 'error', 5000, 'fas fa-puzzle-piece');
                         // Update UI to potentially show missing download buttons again
                         updateModelStatusUI(selectedModel, data.models_status, config.gfpgan_available);
                    } else {
                         // All required models exist, proceed with enhancement
                         startEnhancement(taskIdsToProcess);
                    }
               })
               .catch(error => {
                    console.error('Error pre-checking models before enhancement:', error);
                    showNotification(`Cannot start: Failed to check model status (${error.message})`, 'error', 5000, 'fas fa-circle-xmark');
               });
     }
     function startEnhancement(taskIdsToProcess) {
          if (!modelSelectNative || !upscaleFactorNativeInput || !denoiseSliderNative || !tileSliderNative || !tilePadSliderNative || !prePadSliderNative || !faceEnhanceNativeInput || !fp32Checkbox || !outputFormatNativeInput || !cpuForceCheckbox || !config) {
               console.error("Cannot start enhancement: One or more control elements are missing.");
               showNotification("Internal error: Cannot read enhancement settings.", "error", 5000);
               return;
          }
          const params = {
               model_name: modelSelectNative.value,
               outscale: parseInt(upscaleFactorNativeInput.value, 10) || 4,
               denoise_strength: modelSelectNative.value === 'realesr-general-x4v3' ? parseFloat(denoiseSliderNative.value) : null,
               tile: parseInt(tileSliderNative.value, 10) || 0,
               tile_pad: parseInt(tilePadSliderNative.value, 10) || 10,
               pre_pad: parseInt(prePadSliderNative.value, 10) || 0,
               face_enhance: faceEnhanceNativeInput.value === 'true',
               fp32: fp32Checkbox.checked,
               ext: outputFormatNativeInput.value || 'png',
               // Determine use_gpu based on config and checkbox
               use_gpu: config.has_gpu === true && !cpuForceCheckbox.checked,
          };
          config.lastUsedParams = params; // Store for potential later use (e.g., download names)

          // Update status for all processing tasks to 'queued' immediately
          taskIdsToProcess.forEach(taskId => {
               processingQueue.add(taskId); // Add to tracking set
               updatePreviewStatus(taskId, 'queued', 'Queued...'); // Update UI
          });
          updateButtonStates(); // Disable enhance button etc.

          showNotification(`Starting enhancement for ${taskIdsToProcess.length} image(s)...`, 'info', 3000, 'fas fa-rocket');

          // Send the enhancement request to the backend
          fetch('/enhance', {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({ client_id: CLIENT_ID, task_ids: taskIdsToProcess, params: params }),
          })
               .then(response => {
                    if (!response.ok) {
                         // Try to parse error detail from JSON response
                         return response.json().catch(() => response.text()).then(err => {
                              const detail = err?.detail || (typeof err === 'string' ? err : `Server error ${response.status}`);
                              throw new Error(detail);
                         });
                    }
                    return response.json();
               })
               .then(data => {
                    console.log('Enhancement request acknowledged:', data.message);
                    // Task status updates will come via WebSocket
               })
               .catch(error => {
                    console.error('Error starting enhancement:', error);
                    showNotification(`Error starting enhancement: ${error.message}`, 'error', 5000, 'fas fa-circle-xmark');
                    // Reset status of queued items to error on failure to start
                    taskIdsToProcess.forEach(taskId => {
                         if (processingQueue.has(taskId)) { // Check if it was actually queued
                              updatePreviewStatus(taskId, 'error', `Start failed: ${error.message}`);
                         }
                    });
                    updateButtonStates(); // Re-enable enhance button? Or require clear?
               });
     }
     function handleDownloadModelClick(modelName) {
          if (!modelName || modelName === 'Loading...' || modelName === 'No models available') {
               showNotification('Please select a valid model first.', 'warning', 4000, 'fas fa-hand-pointer');
               return;
          }
          const sanitizedId = `modeldl-${modelName.replace(/[^a-zA-Z0-9-_]/g, '_')}`;
          showNotification(`Requesting download: ${modelName}...`, 'download', null, 'fas fa-download', 0, sanitizedId);

          // Disable the correct button temporarily
          const buttonToDisable = (modelName === GFPGAN_MODEL_NAME) ? downloadGfpganBtn : downloadModelBtn;
          if (buttonToDisable) buttonToDisable.disabled = true;

          sendWebSocketMessage('request_model_download', { model_name: modelName });
     }

     // --- Cleanup and Download ---
     function handleClearClick() {
          const hasFiles = Object.keys(uploadedFiles).length > 0;
          const isProcessing = processingQueue.size > 0;

          if (!hasFiles && !isProcessing) {
               showNotification("Nothing to clear.", "info", 2000);
               return;
          }

          let confirmMsg = "Clear all images from this session?";
          if (isProcessing) {
               confirmMsg += " This will attempt to cancel any ongoing processing.";
          }
          confirmMsg += " (Cannot be undone)";


          if (confirm(confirmMsg)) {
               console.log("Clearing session...");

               // Clear UI immediately
               if (inputPreviewsContainer) inputPreviewsContainer.innerHTML = '';
               if (outputPreviewsContainer) outputPreviewsContainer.innerHTML = '';

               // Revoke blobs and clear local state
               Object.values(uploadedFiles).forEach(fileInfo => {
                    if (fileInfo.blobUrl) {
                         try { URL.revokeObjectURL(fileInfo.blobUrl); } catch (e) { }
                    }
               });
               const tasksToCancel = Array.from(processingQueue); // Get tasks that might be processing
               uploadedFiles = {};
               processingQueue.clear();
               updateButtonStates(); // Update UI buttons

               // Send clear request to backend
               fetch('/clear', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ client_id: CLIENT_ID, cancel_tasks: tasksToCancel }) // Send tasks to potentially cancel on backend
               })
                    .then(response => response.json().then(data => ({ ok: response.ok, data })).catch(err => ({ ok: false, data: { detail: "Failed to parse clear response." } })))
                    .then(({ ok, data }) => {
                         if (ok) {
                              showNotification(data.message || 'Session cleared.', 'success', 3000, 'fas fa-check-circle');
                         } else {
                              console.error("Server clear error response:", data);
                              showNotification(data.detail || 'Failed to clear server files.', 'error', 5000, 'fas fa-circle-xmark');
                         }
                         if (data.errors?.length > 0) {
                              console.warn("Server cleanup warnings:", data.errors);
                              showNotification(`Server cleanup warnings: ${data.errors.join(', ')}`, 'warning', 5000, 'fas fa-triangle-exclamation');
                         }
                    })
                    .catch(error => {
                         console.error('Error sending clear request:', error);
                         showNotification(`Failed to send clear request: ${error.message}`, 'error', 5000, 'fas fa-circle-xmark');
                    });
          }
     }
     function handleDownloadAllClick() {
          const completedTaskIds = Object.keys(uploadedFiles).filter(taskId => {
               const fileInfo = uploadedFiles[taskId];
               // Ensure status is complete and output URL exists
               return fileInfo?.status === 'complete' && fileInfo?.outputUrl;
          });

          if (completedTaskIds.length === 0) {
               showNotification('No completed images available to download.', 'warning', 4000, 'fas fa-box-open');
               return;
          }

          console.log("Requesting download for tasks:", completedTaskIds);
          const notifId = showNotification('Preparing download...', 'info', null, 'fas fa-file-zipper');

          fetch('/download_all', {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({ client_id: CLIENT_ID, task_ids: completedTaskIds })
          })
               .then(response => {
                    if (response.ok) {
                         const disposition = response.headers.get('Content-Disposition');
                         let filename = 'PrismAI_enhanced_images.zip'; // Default
                         if (disposition && disposition.includes('attachment')) {
                              // More robust filename extraction
                              const filenameMatch = disposition.match(/filename\*?=(?:UTF-8'')?([^;]+)/i);
                              if (filenameMatch && filenameMatch[1]) {
                                   try {
                                        // Trim quotes and decode
                                        let decoded = decodeURIComponent(filenameMatch[1].replace(/['"]/g, '').trim());
                                        filename = decoded;
                                   } catch (e) {
                                        console.error("Error decoding filename:", e, filenameMatch[1]);
                                        // Fallback to raw value if decoding fails
                                        filename = filenameMatch[1].replace(/['"]/g, '').trim();
                                   }
                              }
                         }
                         // Return blob and determined filename
                         return response.blob().then(blob => ({ blob, filename }));
                    } else {
                         // Handle error response from server
                         return response.json().catch(() => response.text()).then(err => {
                              const detail = err?.detail || (typeof err === 'string' ? err : `Download failed: Server error ${response.status}`);
                              throw new Error(detail);
                         });
                    }
               })
               .then(({ blob, filename }) => {
                    dismissNotification(notifId); // Dismiss "Preparing..."
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = filename; // Use the filename from header or default
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url); // Clean up blob URL
                    a.remove(); // Clean up link element
                    showNotification('Download started.', 'success', 3000, 'fas fa-download');
               })
               .catch(error => {
                    dismissNotification(notifId);
                    console.error('Error downloading files:', error);
                    showNotification(`Download failed: ${error.message}`, 'error', 6000, 'fas fa-circle-xmark');
               });
     }


     // --- UI Updates ---
     function updateButtonStates() {
          // Ensure elements exist before checking properties
          const modelValue = modelSelectNative?.value;
          const modelReady = modelStatusText?.classList.contains('available');
          const gfpganReady = !faceEnhanceNativeInput || faceEnhanceNativeInput.value !== 'true' || !config.gfpgan_available || gfpganStatusText?.classList.contains('available');

          // Determine button states based on conditions
          const modelSelectLoaded = modelValue && modelValue !== 'Loading...' && modelValue !== 'No models available';
          const hasFiles = Object.keys(uploadedFiles).length > 0;
          const hasPending = Object.values(uploadedFiles).some(f => f?.status === 'pending');
          const hasCompleted = Object.values(uploadedFiles).some(f => f?.status === 'complete' && f?.outputUrl);
          const isProcessingOrQueued = processingQueue.size > 0;
          const isOffline = !websocket || websocket.readyState !== WebSocket.OPEN;
          const modelsAvailable = modelSelectLoaded && modelReady && gfpganReady;

          // Safely enable/disable buttons
          if (enhanceButton) enhanceButton.disabled = isOffline || !hasFiles || !hasPending || isProcessingOrQueued || !modelsAvailable;
          if (clearButton) clearButton.disabled = !hasFiles && !isProcessingOrQueued; // Allow clear even if offline unless actively processing
          if (downloadAllButton) downloadAllButton.disabled = isOffline || !hasCompleted || isProcessingOrQueued;

          // Update enhance button tooltip
          if (enhanceButton) {
               let enhanceTitle = "Enhance pending images";
               if (enhanceButton.disabled) {
                    enhanceTitle = "Cannot enhance: ";
                    if (isOffline) enhanceTitle += "Offline. ";
                    else if (!hasFiles) enhanceTitle += "No images added. ";
                    else if (!hasPending) enhanceTitle += "No pending images to process. ";
                    else if (isProcessingOrQueued) enhanceTitle += "Processing/Queued in progress. ";
                    else if (!modelSelectLoaded) enhanceTitle += "Select a model. ";
                    else if (!modelsAvailable) enhanceTitle += "Missing required model(s). ";
                    else enhanceTitle += "Check status. "; // Generic fallback
               }
               enhanceButton.title = enhanceTitle.trim();
          }

          // Update other button tooltips
          if (downloadAllButton) downloadAllButton.title = downloadAllButton.disabled ? "No completed images to download or currently processing/offline." : "Download all completed images as ZIP";
          if (clearButton) clearButton.title = clearButton.disabled ? "Nothing to clear." : "Remove all images and clear session";
     }

     function updatePreviewStatus(taskId, status, message, progress = null, outputUrl = null) {
          const fileInfo = uploadedFiles[taskId];
          // Guard against missing fileInfo or inputElement
          if (!fileInfo || !fileInfo.inputElement) {
               console.warn(`Attempted to update status for unknown/removed task ID or missing input element: ${taskId}`);
               return;
          }

          const inputThumb = fileInfo.inputElement;
          const statusOverlay = inputThumb.querySelector('.status-overlay');
          const statusOverlayText = statusOverlay?.querySelector('.status-overlay-text');

          // Remove all previous status classes cleanly
          inputThumb.classList.remove('pending', 'queued', 'processing', 'error', 'complete');
          // Add the new status class (except for 'complete' on input, handled below)
          if (status !== 'complete') {
               inputThumb.classList.add(status);
          }

          // Update overlay text safely
          if (statusOverlayText) {
               statusOverlayText.textContent = message || status.charAt(0).toUpperCase() + status.slice(1);
          }

          // Update local fileInfo state
          fileInfo.status = status;
          if (outputUrl) fileInfo.outputUrl = outputUrl; // Store output URL if provided

          // Update processing queue tracking
          if (['queued', 'processing'].includes(status)) { // Removed 'downloading_model' as it's not a task status
               processingQueue.add(taskId);
          } else {
               processingQueue.delete(taskId); // Remove from queue if finished or error
          }

          // Handle different statuses specifically
          if (status === 'complete' && outputUrl) {
               if (statusOverlayText) statusOverlayText.textContent = 'Done';
               inputThumb.classList.add('complete'); // Add 'complete' class temporarily for input checkmark

               // Hide the 'complete' overlay on input after a delay
               setTimeout(() => {
                    if (fileInfo.inputElement) { // Check if element still exists
                         fileInfo.inputElement.classList.remove('complete'); // Remove class to hide overlay again
                    }
               }, 1500); // Hide after 1.5 seconds


               // Create or update output preview element
               if (outputPreviewsContainer) { // Ensure container exists
                    if (!fileInfo.outputElement) { // Create if doesn't exist
                         const outputThumb = document.createElement('div');
                         outputThumb.classList.add('preview-thumbnail');
                         outputThumb.setAttribute('data-task-id', taskId);
                         outputThumb.title = `${fileInfo.originalName} (Enhanced)`;

                         const outputImg = document.createElement('img');
                         outputImg.alt = `Enhanced: ${fileInfo.originalName}`;
                         // Add cache-busting query param to ensure fresh image load
                         outputImg.src = outputUrl + '?t=' + Date.now();
                         outputThumb.appendChild(outputImg);

                         const downloadBtn = document.createElement('button');
                         downloadBtn.classList.add('thumbnail-action-btn', 'thumbnail-download-btn');
                         downloadBtn.title = 'Download this image';
                         downloadBtn.innerHTML = '<i class="fa-solid fa-download"></i>';
                         downloadBtn.setAttribute('data-action', 'download');
                         downloadBtn.setAttribute('aria-label', `Download enhanced ${fileInfo.originalName}`);
                         outputThumb.appendChild(downloadBtn);

                         outputPreviewsContainer.appendChild(outputThumb);
                         fileInfo.outputElement = outputThumb; // Store reference
                    } else {
                         // If output element exists (e.g., re-processing), update image source
                         const existingImg = fileInfo.outputElement.querySelector('img');
                         if (existingImg) existingImg.src = outputUrl + '?t=' + Date.now();
                         fileInfo.outputElement.style.display = ''; // Ensure it's visible
                    }
               } else {
                    console.error("Output previews container not found.");
               }

          } else if (status === 'error') {
               if (statusOverlayText) statusOverlayText.textContent = 'Error';
               inputThumb.classList.add('error'); // Ensure error class is set on input
               // Remove or hide the corresponding output preview if it exists
               if (fileInfo.outputElement) {
                    fileInfo.outputElement.remove(); // Remove errored output preview
                    fileInfo.outputElement = null;
               }
          } else if (status === 'pending' || status === 'queued' || status === 'processing') {
               // Hide any existing output preview if reprocessing starts or task is active
               if (fileInfo.outputElement) {
                    fileInfo.outputElement.style.display = 'none';
               }
          }

          updateButtonStates(); // Update global button states based on new status
     }


     // --- Notifications ---
     let notificationIdCounter = 0;
     const activeNotifications = {}; // Use object for easy ID lookup/update

     function showNotification(message, type = 'info', duration = 5000, iconClass = null, progress = null, notificationId = null) {
          // Generate or sanitize ID
          const baseId = notificationId || `nid-${notificationIdCounter++}`;
          const currentId = baseId.replace(/[^a-zA-Z0-9-_]/g, '_'); // Sanitize

          let notificationEntry = activeNotifications[currentId];
          let notificationElement;

          // Determine icon if not provided
          if (!iconClass) {
               switch (type) {
                    case 'success': iconClass = 'fas fa-check-circle'; break;
                    case 'error': iconClass = 'fas fa-circle-exclamation'; break;
                    case 'warning': iconClass = 'fas fa-triangle-exclamation'; break;
                    case 'download': iconClass = 'fas fa-download'; break;
                    case 'processing': iconClass = 'fas fa-spinner fa-spin'; break; // Add processing icon
                    case 'info': default: iconClass = 'fas fa-info-circle'; break;
               }
          }

          // Create notification element if it doesn't exist for this ID
          if (!notificationEntry) {
               notificationElement = document.createElement('div');
               notificationElement.classList.add('notification', type);
               notificationElement.setAttribute('data-id', currentId);

               // Add Icon
               if (iconClass) {
                    const iconElement = document.createElement('i');
                    iconClass.split(' ').forEach(cls => iconElement.classList.add(cls));
                    notificationElement.appendChild(iconElement);
               }

               // Add Content Wrapper
               const contentWrapper = document.createElement('div');
               contentWrapper.classList.add('notification-content');

               // Add Message Span
               const messageSpan = document.createElement('span');
               messageSpan.classList.add('notification-message');
               messageSpan.textContent = message;
               contentWrapper.appendChild(messageSpan);

               // Add Progress Bar container (conditionally)
               if (progress !== null || type === 'download' || type === 'processing') {
                    const progressBar = document.createElement('div');
                    progressBar.classList.add('progress-bar');
                    const progressInner = document.createElement('div');
                    progressInner.classList.add('progress');
                    progressBar.appendChild(progressInner);
                    contentWrapper.appendChild(progressBar); // Append to content
               }

               notificationElement.appendChild(contentWrapper); // Append content

               // Add Close Button
               const closeBtn = document.createElement('button');
               closeBtn.innerHTML = '&times;'; // Multiplication sign as close symbol
               closeBtn.classList.add('notification-close-btn');
               closeBtn.setAttribute('aria-label', 'Close notification');
               closeBtn.onclick = () => dismissNotification(currentId);
               notificationElement.appendChild(closeBtn); // Append close button last

               // Add to DOM and active list
               if (notificationArea) notificationArea.appendChild(notificationElement);
               notificationEntry = { element: notificationElement, dismissTimer: null };
               activeNotifications[currentId] = notificationEntry;

               // Trigger entrance animation
               requestAnimationFrame(() => {
                    if (notificationElement) notificationElement.classList.add('show');
               });

          } else {
               // Update existing notification
               notificationElement = notificationEntry.element;
               notificationElement.className = `notification show ${type}`; // Update type class

               // Update Icon
               let iconElement = notificationElement.querySelector('i:first-child');
               if (iconClass) {
                    if (!iconElement) { // Add icon if missing
                         iconElement = document.createElement('i');
                         notificationElement.insertBefore(iconElement, notificationElement.firstChild);
                    }
                    iconElement.className = ''; // Reset classes
                    iconClass.split(' ').forEach(cls => iconElement.classList.add(cls));
               } else if (iconElement) {
                    iconElement.remove(); // Remove icon if not needed
               }

               // Update Message
               const messageSpan = notificationElement.querySelector('.notification-message');
               if (messageSpan) messageSpan.textContent = message;

               // Update Progress Bar (Add/Remove/Update)
               let contentWrapper = notificationElement.querySelector('.notification-content');
               let progressBar = contentWrapper?.querySelector('.progress-bar');
               const needsProgressBar = progress !== null || type === 'download' || type === 'processing';

               if (needsProgressBar && !progressBar) { // Add progress bar if needed and missing
                    progressBar = document.createElement('div');
                    progressBar.classList.add('progress-bar');
                    const progressInner = document.createElement('div');
                    progressInner.classList.add('progress');
                    progressBar.appendChild(progressInner);
                    contentWrapper?.appendChild(progressBar);
               } else if (!needsProgressBar && progressBar) { // Remove if not needed and present
                    progressBar.remove();
               }
          }

          // Update progress bar width if it exists
          const progressInner = notificationElement.querySelector('.progress-bar .progress');
          if (progressInner) {
               if (progress !== null) {
                    progressInner.style.width = `${Math.max(0, Math.min(100, progress))}%`;
                    progressInner.style.backgroundImage = 'none'; // Remove potential indeterminate style
                    progressInner.style.animation = 'none';
               } else if (type === 'download' || type === 'processing') {
                    // Indeterminate state - use CSS animation added earlier
                    progressInner.style.width = '100%'; // Full width for indeterminate visual
                    progressInner.style.removeProperty('background-image'); // Rely on CSS class animation
                    progressInner.style.removeProperty('animation');
               } else {
                    progressInner.style.width = '0%'; // Reset if no progress
                    progressInner.style.backgroundImage = 'none';
                    progressInner.style.animation = 'none';
               }
          }


          // Reset dismiss timer
          if (notificationEntry.dismissTimer) {
               clearTimeout(notificationEntry.dismissTimer);
          }
          if (duration !== null && duration > 0) { // Only set timer if duration is positive
               notificationEntry.dismissTimer = setTimeout(() => dismissNotification(currentId), duration);
          } else {
               notificationEntry.dismissTimer = null; // Persistent notification
          }
          return currentId; // Return the ID used
     }

     function dismissNotification(id) {
          const notificationEntry = activeNotifications[id];
          if (notificationEntry && notificationEntry.element) {
               const notificationElement = notificationEntry.element;
               notificationElement.classList.remove('show'); // Start fade/slide out animation

               // Use transitionend event for cleanup, with a fallback timeout
               const handleTransitionEnd = (event) => {
                    if (event.target === notificationElement && (event.propertyName === 'opacity' || event.propertyName === 'transform')) {
                         cleanupNotification(id);
                    }
               };

               notificationElement.addEventListener('transitionend', handleTransitionEnd);

               // Fallback cleanup in case transitionend doesn't fire reliably
               setTimeout(() => {
                    if (activeNotifications[id]) { // Check if still active
                         console.warn(`Notification ${id} removal fallback timer triggered.`);
                         cleanupNotification(id, handleTransitionEnd); // Pass listener to remove
                    }
               }, 500); // Should be slightly longer than CSS transition

               // Clear any existing dismiss timer
               if (notificationEntry.dismissTimer) {
                    clearTimeout(notificationEntry.dismissTimer);
                    notificationEntry.dismissTimer = null;
               }
          }
     }
     // Helper to actually remove element and state
     function cleanupNotification(id, listenerToRemove = null) {
          const notificationEntry = activeNotifications[id];
          if (notificationEntry && notificationEntry.element) {
               notificationEntry.element.remove(); // Remove from DOM
               if (listenerToRemove) {
                    notificationEntry.element.removeEventListener('transitionend', listenerToRemove);
               }
          }
          delete activeNotifications[id]; // Remove from active list
     }


     // --- Start the application ---
     initialize();
});