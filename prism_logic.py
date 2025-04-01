import cv2
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
# from basicsr.utils.download_util import load_file_from_url # Replaced with custom version
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import time # For progress simulation during download
import math # For progress calculation
import requests # For slightly better download progress
import tqdm # For download progress feedback
import traceback # Import traceback for better error logging
import logging # Use logging module

# Setup logger for this module
logger = logging.getLogger(__name__)

# Optional: GFPGAN for face enhancement
try:
    from gfpgan import GFPGANer
    gfpgan_available = True
except ImportError:
    gfpgan_available = False
    logger.warning("GFPGAN module not found. Face enhancement feature will be disabled. Install with: pip install gfpgan")
except Exception as e:
    gfpgan_available = False
    logger.error(f"Error importing GFPGAN: {e}. Face enhancement disabled.", exc_info=True)

# --- Model Definitions ---
MODEL_INFO = {
    'RealESRGAN_x4plus': {
        'class': RRDBNet,
        'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32, 'scale': 4},
        'netscale': 4,
        'urls': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    },
    'RealESRNet_x4plus': {
        'class': RRDBNet,
        'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32, 'scale': 4},
        'netscale': 4,
        'urls': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    },
    'RealESRGAN_x4plus_anime_6B': {
        'class': RRDBNet,
        'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 6, 'num_grow_ch': 32, 'scale': 4},
        'netscale': 4,
        'urls': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    },
    'RealESRGAN_x2plus': {
        'class': RRDBNet,
        'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32, 'scale': 2},
        'netscale': 2,
        'urls': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    },
    'realesr-animevideov3': {
        'class': SRVGGNetCompact,
        'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_conv': 16, 'upscale': 4, 'act_type': 'prelu'},
        'netscale': 4,
        'urls': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    },
    'realesr-general-x4v3': {
        'class': SRVGGNetCompact,
        'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_conv': 32, 'upscale': 4, 'act_type': 'prelu'},
        'netscale': 4,
        'urls': [ # Order matters: wdn model first if denoise < 1
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    }
}

GFPGAN_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
GFPGAN_MODEL_NAME = 'GFPGANv1.3.pth' # Consistent filename

# --- Helper Functions ---

def _send_message_passthrough(payload: dict):
    """Placeholder callback if none provided."""
    # This should ideally not be called if used correctly with the main app
    logger.warning(f"Message callback not provided. Message: {payload}")


def _make_progress_message(status, model_name, progress=None, message=None, task_id=None):
    """Helper to create progress message payload. Uses specific type."""
    payload = {
        'type': 'model_download_status', # Consistent type for WS handler
        'status': status,
        'model_name': model_name,
    }
    if progress is not None:
        payload['progress'] = max(0, min(100, int(progress))) # Clamp progress 0-100
    if message:
        payload['message'] = message
    # task_id is handled by the create_enhancement_callback wrapper now
    # if task_id: payload['task_id'] = task_id
    return payload

# Modified load_file_from_url to include progress reporting via callback
def load_file_from_url_with_progress(url, model_dir, file_name, model_name_for_msg, send_message_callback, task_id=None):
    """Downloads a file from a URL, reports progress via callback."""
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, file_name)
    logger.info(f"Attempting to download {model_name_for_msg} from {url} to {save_path}")

    try:
        response = requests.get(url, stream=True, timeout=60, allow_redirects=True) # Increased timeout, allow redirects
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 8 # 8 Kibibytes
        downloaded_size = 0

        # Send initial download message
        send_message_callback(
            _make_progress_message('downloading', model_name_for_msg, 0, f"Starting download: {file_name}", task_id)
        )

        # Use tqdm for progress bar logic
        progress_bar = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {file_name}", leave=False)

        last_reported_progress_bucket = -1
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:  # filter out keep-alive new chunks
                    chunk_size = len(chunk)
                    progress_bar.update(chunk_size)
                    f.write(chunk)
                    downloaded_size += chunk_size

                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        # Report progress only on significant changes (e.g., every 5% bucket)
                        current_progress_bucket = math.floor(progress / 5)
                        if current_progress_bucket > last_reported_progress_bucket:
                             send_message_callback(
                                 _make_progress_message('downloading', model_name_for_msg, progress, f"Downloading: {downloaded_size / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB", task_id)
                             )
                             last_reported_progress_bucket = current_progress_bucket

        progress_bar.close()

        # Final check on file size
        if total_size != 0 and downloaded_size != total_size:
             raise IOError(f"Download incomplete: Received {downloaded_size} of {total_size} bytes.")
        elif total_size == 0 and downloaded_size == 0:
             # Check if the file actually exists and has content if server didn't provide size
             if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
                  raise IOError("Download failed: Received 0 bytes and no content-length header.")
             else:
                  logger.warning(f"Downloaded {file_name} but server did not provide content-length. File size: {os.path.getsize(save_path)}")

        # Send final completion message
        logger.info(f"Successfully downloaded {model_name_for_msg} to {save_path}")
        send_message_callback(
            _make_progress_message('complete', model_name_for_msg, 100, f"{file_name} downloaded successfully.", task_id)
        )
        return save_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading {file_name} from {url}: {e}", exc_info=True)
        # Clean up potentially incomplete file
        if os.path.exists(save_path):
            try: os.remove(save_path)
            except OSError: pass
        send_message_callback(
            _make_progress_message('error', model_name_for_msg, None, f"Network error downloading {file_name}: {e}", task_id)
        )
        raise # Re-raise the exception to signal failure
    except Exception as e:
        logger.error(f"Error during download process for {file_name}: {e}", exc_info=True)
        if os.path.exists(save_path):
             try: os.remove(save_path)
             except OSError: pass
        send_message_callback(
            _make_progress_message('error', model_name_for_msg, None, f"Error downloading {file_name}: {e}", task_id)
        )
        raise


def get_model_path(model_name, model_dir='weights', send_message_callback=_send_message_passthrough, task_id=None):
    """Checks if model file exists, downloads if not, uses callback for progress. Returns list of paths if multiple files."""
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model name: {model_name}")

    info = MODEL_INFO[model_name]
    model_paths = []
    all_files_exist_locally = True # Track if *all* needed files already exist

    for i, url in enumerate(info.get('urls', [])): # Safely iterate URLs
        filename = os.path.basename(url)
        # Determine expected filename based on convention
        expected_filename = filename # Default
        model_name_for_msg = filename # Default message name
        if model_name == 'realesr-general-x4v3':
             if 'wdn' in filename:
                 expected_filename = 'realesr-general-wdn-x4v3.pth'
                 model_name_for_msg = expected_filename
             else:
                 expected_filename = 'realesr-general-x4v3.pth'
                 model_name_for_msg = expected_filename
        elif len(info.get('urls', [])) == 1: # Single file model
             expected_filename = f"{model_name}.pth"
             model_name_for_msg = expected_filename

        path = os.path.join(model_dir, expected_filename)

        if not os.path.isfile(path):
            all_files_exist_locally = False # At least one needs downloading
            logger.info(f"Model file {expected_filename} not found locally. Downloading from {url}...")
            try:
                # Download using the determined filename and message name
                downloaded_path = load_file_from_url_with_progress(
                    url=url,
                    model_dir=model_dir,
                    file_name=expected_filename,
                    model_name_for_msg=model_name_for_msg,
                    send_message_callback=send_message_callback,
                    task_id=task_id # Pass task_id if relevant
                )
                model_paths.append(downloaded_path)
            except Exception as e:
                 logger.error(f"Download failed for {expected_filename}. Aborting model setup for {model_name}.")
                 # Error message already sent by load_file_from_url_with_progress
                 raise # Stop processing if download fails
        else:
             logger.info(f"Model file {expected_filename} found locally at {path}.")
             model_paths.append(path)

    # Send a single "exists" message only if *all* required files were found locally at the start
    if all_files_exist_locally:
         # Use the base model name for the "exists" message for clarity
         send_message_callback(
             _make_progress_message('exists', model_name, 100, f"Model '{model_name}' already exists.", task_id)
         )

    # Return paths in the correct order if it's the general model
    if model_name == 'realesr-general-x4v3':
        # Ensure paths were actually added (download might have failed but loop completed)
        if len(model_paths) != 2:
            logger.error(f"Error: Expected 2 paths for {model_name} but found {len(model_paths)} after check/download.")
            raise RuntimeError(f"Required files missing or download failed for model {model_name}")

        wdn_path = next((p for p in model_paths if 'wdn' in p), None)
        main_path = next((p for p in model_paths if 'wdn' not in p), None)
        if wdn_path and main_path:
            # Return in the order expected by RealESRGANer: [wdn_path, main_path]
            return [wdn_path, main_path]
        else:
             logger.error(f"Error: Could not find both wdn and main paths for {model_name} in results: {model_paths}")
             raise RuntimeError(f"Required files structure incorrect for model {model_name}")
    elif len(model_paths) == 1:
        return model_paths[0] # Return single path string
    else:
        # Should not happen for other defined models, but handle defensively
        logger.warning(f"Model {model_name} resulted in unexpected number of paths: {model_paths}")
        if model_paths:
            return model_paths # Return list if multiple found somehow
        else:
            raise RuntimeError(f"No model paths found or downloaded for {model_name}")



def get_gfpgan_path(model_dir='weights', send_message_callback=_send_message_passthrough, task_id=None):
    """Checks if GFPGAN model file exists, downloads if not, uses callback."""
    if not gfpgan_available:
        logger.warning("GFPGAN not available, cannot get path.")
        return None

    path = os.path.join(model_dir, GFPGAN_MODEL_NAME)
    if not os.path.isfile(path):
        logger.info(f"Model {GFPGAN_MODEL_NAME} not found locally. Downloading...")
        try:
            path = load_file_from_url_with_progress(
                url=GFPGAN_URL,
                model_dir=model_dir,
                file_name=GFPGAN_MODEL_NAME,
                model_name_for_msg=GFPGAN_MODEL_NAME,
                send_message_callback=send_message_callback,
                task_id=task_id
            )
        except Exception as e:
            # Error message sent by helper
            logger.error(f"Download failed for {GFPGAN_MODEL_NAME}. GFPGAN cannot be used.", exc_info=False)
            return None # Return None if download fails
    else:
         logger.info(f"Model {GFPGAN_MODEL_NAME} found locally at {path}.")
         # Send exists message only if found locally, download sends 'complete'
         send_message_callback(
             _make_progress_message('exists', GFPGAN_MODEL_NAME, 100, f"{GFPGAN_MODEL_NAME} already exists.", task_id)
         )
    return path


# --- Main Processing Function ---
def enhance_image(args: dict, input_path: str, output_path_base: str, task_id: str, send_message_callback: callable):
    """
    Processes a single image using Real-ESRGAN. Uses callback for status updates.
    Now includes cancellation check.
    """
    current_status = 'starting' # Track local status

    def _emit_status(status, message=None, progress=None, output_url=None, type='status_update'):
        """Helper to format and send status updates via callback. Updates local status."""
        nonlocal current_status
        current_status = status # Update local status tracker
        payload = {'task_id': task_id, 'type': type, 'status': status}
        if message: payload['message'] = message
        if progress is not None: payload['progress'] = max(0, min(100, int(progress)))
        if output_url: payload['output_url'] = output_url
        try:
             send_message_callback(payload) # Callback now handles checking cancel flag
             # logger.debug(f"[Task {task_id}] Sent status: {status}") # Reduce noise
        except Exception as e:
             logger.warning(f"[Task {task_id}] Warning: Failed to send status update '{status}': {e}")

    # --- Cancellation Check Helper ---
    def check_cancel():
        # Simple placeholder - relies on callback no longer sending messages
        # For true interruption, the enhancement loop in RealESRGANer/GFPGANer would need modification
        return False

    _emit_status('processing', message='Starting enhancement...', progress=0)

    # --- Local Model/Upsampler Variables ---
    model = None
    upsampler = None
    face_enhancer = None
    device = None # Define device here for cleanup block
    model_weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'weights'))
    os.makedirs(model_weights_dir, exist_ok=True)

    try:
        # --- Parameter Validation and Setup ---
        model_name = args.get('model_name', 'RealESRGAN_x4plus')
        outscale = args.get('outscale', 4.0)
        denoise_strength = args.get('denoise_strength', 0.5)
        tile_size = args.get('tile', 0)
        tile_pad = args.get('tile_pad', 10)
        pre_pad = args.get('pre_pad', 0)
        face_enhance = args.get('face_enhance', False) and gfpgan_available # Only if requested AND available
        use_fp32 = args.get('fp32', False)
        gpu_id = args.get('gpu_id', None) # None lets RealESRGANer decide based on torch default
        alpha_upsampler = args.get('alpha_upsampler', 'realesrgan')
        ext = args.get('ext', 'png') # Default to png

        if model_name not in MODEL_INFO:
             raise ValueError(f"Invalid model name: {model_name}")

        model_info = MODEL_INFO[model_name]
        netscale = model_info['netscale']
        ModelClass = model_info['class']
        model_params = model_info['params']

        # --- Determine Device ---
        if gpu_id == -1: # Force CPU
            device = torch.device('cpu')
            use_fp32 = True # CPU requires fp32
            logger.info(f"[Task {task_id}] Forcing CPU usage.")
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            selected_gpu = gpu_id if gpu_id is not None and 0 <= gpu_id < torch.cuda.device_count() else 0
            try:
                 device = torch.device(f'cuda:{selected_gpu}')
                 # Set the default device for this thread/process (important!)
                 torch.cuda.set_device(device)
                 logger.info(f"[Task {task_id}] Using GPU ID: {selected_gpu} ({torch.cuda.get_device_name(selected_gpu)})")
            except Exception as e:
                 logger.warning(f"[Task {task_id}] Error selecting GPU {selected_gpu}: {e}. Falling back to CPU.", exc_info=True)
                 device = torch.device('cpu')
                 use_fp32 = True
                 gpu_id = -1 # Reflect CPU usage
        else:
            device = torch.device('cpu')
            use_fp32 = True # CPU requires fp32
            gpu_id = -1 # Reflect CPU usage
            logger.info(f"[Task {task_id}] CUDA not available or no devices found, using CPU.")

        # Half precision only on CUDA and if fp32 is not forced
        use_half_precision = (not use_fp32) and (device.type == 'cuda')
        logger.info(f"[Task {task_id}] Device selected: {device}, Half precision (fp16): {use_half_precision}")

        if check_cancel(): return # Check before loading models

        _emit_status('processing', message='Checking/Loading models...', progress=5)

        # --- Model Loading ---
        # Wrap the callback for model download progress within this task context
        def model_download_callback_wrapper(payload):
            # Forward the payload using the main task's callback
             _emit_status(payload['status'], payload.get('message'), payload.get('progress'), type='model_download_status') # Use correct type


        # Download Real-ESRGAN model if needed
        model_path_or_paths = get_model_path(model_name, model_weights_dir, model_download_callback_wrapper, task_id)
        if not model_path_or_paths: # Handle download failure
            raise RuntimeError(f"Failed to load or download required model file(s) for {model_name}")

        if check_cancel(): return

        _emit_status('processing', message='Initializing upsampler...', progress=15)

        # Instantiate the model architecture
        logger.info(f"[Task {task_id}] Instantiating model architecture: {ModelClass.__name__}")
        try:
             # Move model to device *before* potentially converting to half
             model = ModelClass(**model_params).to(device)
             model.eval()
             if use_half_precision:
                 model = model.half() # Convert to half precision if enabled AFTER moving to CUDA
                 logger.info(f"[Task {task_id}] Model converted to half precision (fp16).")
             logger.info(f"[Task {task_id}] Model architecture instantiated on {device}.")
        except Exception as model_inst_error:
             logger.error(f"[Task {task_id}] Error instantiating model {ModelClass.__name__}: {model_inst_error}", exc_info=True)
             raise RuntimeError(f"Failed to instantiate model") from model_inst_error

        # Handle multi-file models and denoise strength
        dni_weight = None
        actual_model_path = model_path_or_paths # Path(s) for loading weights
        if model_name == 'realesr-general-x4v3':
            if isinstance(model_path_or_paths, list) and len(model_path_or_paths) == 2:
                if 0 < denoise_strength < 1:
                    dni_weight = [denoise_strength, 1 - denoise_strength]
                    logger.info(f"[Task {task_id}] Using denoise strength blend: {dni_weight}")
                else: # Use only the main model if denoise is 0 or 1
                    actual_model_path = model_path_or_paths[1] # Use only main model path (index 1)
                    logger.info(f"[Task {task_id}] Using standard enhancement (denoise_strength={denoise_strength}).")
            else:
                 logger.error(f"[Task {task_id}] Error - Expected 2 paths for {model_name}, got {model_path_or_paths}.")
                 raise RuntimeError(f"Could not find both required model files for {model_name}")
        elif isinstance(model_path_or_paths, list):
             actual_model_path = model_path_or_paths[0] # Use first path if list returned unexpectedly


        # Initialize RealESRGANer (loads weights into the provided model instance)
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=actual_model_path, # Path(s) for loading weights
            dni_weight=dni_weight,
            model=model,               # Pass the instantiated model object
            tile=tile_size,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=use_half_precision,
            gpu_id=gpu_id if device.type == 'cuda' else None # Pass gpu_id only if using CUDA
        )
        logger.info(f"[Task {task_id}] Upsampler initialized.")

        # --- Face Enhancer Initialization (Optional) ---
        if face_enhance:
            if check_cancel(): return
            _emit_status('processing', message='Checking/Loading face enhancer...', progress=20)
            gfpgan_model_path = get_gfpgan_path(model_weights_dir, model_download_callback_wrapper, task_id)
            if gfpgan_model_path and gfpgan_available:
                try:
                    face_enhancer = GFPGANer(
                        model_path=gfpgan_model_path,
                        upscale=outscale,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=upsampler, # Use RealESRGAN instance as background upsampler
                        device=device # Use the same device
                        )
                    logger.info(f"[Task {task_id}] GFPGAN face enhancer loaded on device {device}.")
                    _emit_status('processing', message='Face enhancer loaded.', progress=25)
                except Exception as e:
                    logger.error(f"[Task {task_id}] Failed to load GFPGAN: {e}. Disabling face enhance.", exc_info=True)
                    _emit_status('warning', message=f"GFPGAN Load Error: {e}. Face enhancement disabled.")
                    face_enhancer = None # Disable if loading fails
            else:
                msg = "GFPGAN disabled: " + ("module not installed." if not gfpgan_available else "model file missing or download failed.")
                logger.warning(f"[Task {task_id}] {msg}")
                _emit_status('warning', message=msg)
                face_enhancer = None
        else:
             logger.info(f"[Task {task_id}] Face enhancement not requested or not available.")


        # --- Image Loading ---
        if check_cancel(): return
        _emit_status('processing', message='Loading image...', progress=30)
        try:
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"cv2.imread failed to load image (returned None)")
        except Exception as load_err:
            raise ValueError(f"Failed to load image: {input_path} - Error: {load_err}")

        img_mode = 'BGR'
        h, w = img.shape[0:2]
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
            logger.info(f"[Task {task_id}] Input image ({w}x{h}) is RGBA.")
        else:
             logger.info(f"[Task {task_id}] Input image ({w}x{h}) is {img_mode}.")

        # --- Enhancement ---
        if check_cancel(): return
        _emit_status('processing', message='Enhancing image...', progress=50)
        start_time = time.time()
        try:
            # Apply face enhancement first if enabled and loaded
            if face_enhance and face_enhancer:
                logger.info(f"[Task {task_id}] Applying GFPGAN face enhancement...")
                # Note: GFPGANer with bg_upsampler handles the full process
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                # Use only RealESRGAN upsampler
                logger.info(f"[Task {task_id}] Applying RealESRGAN enhancement...")
                output, _ = upsampler.enhance(img, outscale=outscale, alpha_upsampler=alpha_upsampler)

        except torch.cuda.OutOfMemoryError as oom_error:
            # Clear cache and report specific error
            if device.type == 'cuda': torch.cuda.empty_cache()
            error_message = 'CUDA Out of Memory. Try reducing Tile Size or Outscale Factor.'
            logger.error(f"[Task {task_id}] {error_message} - {oom_error}")
            _emit_status('error', message=error_message)
            return # Stop processing this image
        except RuntimeError as error:
            # Handle potential runtime OOM errors too
            if "CUDA out of memory" in str(error):
                 if device.type == 'cuda': torch.cuda.empty_cache()
                 error_message = 'Runtime CUDA Out of Memory. Try reducing Tile Size or Outscale Factor.'
                 logger.error(f"[Task {task_id}] {error_message} - {error}")
                 _emit_status('error', message=error_message)
                 return
            else: # Handle other runtime errors
                 error_message = f'Runtime Error during enhancement: {error}'
                 logger.error(f"[Task {task_id}] {error_message}", exc_info=True)
                 _emit_status('error', message=f'Runtime Error: {error}')
                 return
        except Exception as e:
            # Catch any other unexpected errors during enhancement
            error_message = f'Unexpected Error during enhancement: {e}'
            logger.error(f"[Task {task_id}] {error_message}", exc_info=True)
            _emit_status('error', message=f'Enhancement Error: {type(e).__name__}')
            return

        end_time = time.time()
        logger.info(f"[Task {task_id}] Enhancement took {end_time - start_time:.2f} seconds.")

        if check_cancel(): return # Check again before saving

        _emit_status('processing', message='Saving image...', progress=90)

        # --- Saving ---
        try:
            # Determine output extension
            _, input_ext = os.path.splitext(os.path.basename(input_path))
            save_extension = ext.lower()
            supported_write_ext = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'}

            if save_extension == 'auto':
                # Preserve original extension if supported, default to png
                save_extension = input_ext[1:].lower() if input_ext else 'png'
                if save_extension not in supported_write_ext:
                    logger.warning(f"[Task {task_id}] Input extension '{save_extension}' not directly supported for output, using PNG.")
                    save_extension = 'png'
            elif save_extension not in supported_write_ext:
                 logger.warning(f"[Task {task_id}] Unsupported output extension '{save_extension}', defaulting to PNG.")
                 save_extension = 'png'

            # Force PNG for RGBA images if requested format doesn't support alpha
            if img_mode == 'RGBA' and save_extension not in ['png', 'webp', 'tif', 'tiff']:
                logger.info(f"[Task {task_id}] Input is RGBA, forcing output to PNG (requested '{save_extension}' doesn't support alpha).")
                save_extension = 'png'

            # Construct final output path
            final_output_path = f"{output_path_base}.{save_extension}"

            # Set JPEG/WEBP quality if applicable (example: 95)
            write_params = []
            if save_extension in ['jpg', 'jpeg']:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            elif save_extension == 'webp':
                write_params = [cv2.IMWRITE_WEBP_QUALITY, 95]
            # Add params for other formats if needed (e.g., TIFF compression)

            success = cv2.imwrite(final_output_path, output, write_params)
            if not success:
                 raise IOError(f"cv2.imwrite failed to save {final_output_path}")

            logger.info(f"[Task {task_id}] Enhanced image saved to: {final_output_path}")

            # Generate a relative URL for the frontend
            relative_output_url = f"/outputs/{os.path.basename(final_output_path)}"

            _emit_status('complete', message='Enhancement complete!', progress=100, output_url=relative_output_url)

        except Exception as e:
            logger.error(f"[Task {task_id}] Error saving image to {output_path_base}.*: {e}", exc_info=True)
            _emit_status('error', message=f'Error saving image: {e}')

    except Exception as e:
        # Catch errors during setup (model loading, etc.)
        error_message = f'Setup or critical error in task {task_id}: {e}'
        logger.error(error_message, exc_info=True)
        # Send error status only if not already set (e.g., by OOM handler)
        if current_status != 'error':
             _emit_status('error', message=f'Server Error: {type(e).__name__}')

    finally:
        # Clean up GPU memory - ESSENTIAL
        # Use 'del' and 'torch.cuda.empty_cache()'
        # Important: Ensure variables exist before deleting
        if 'model' in locals() and model is not None: del model
        if 'upsampler' in locals() and upsampler is not None: del upsampler
        if 'face_enhancer' in locals() and face_enhancer is not None: del face_enhancer

        if device and device.type == 'cuda':
             try:
                 # Brief pause before clearing cache might sometimes help release resources
                 time.sleep(0.05)
                 torch.cuda.empty_cache()
                 logger.info(f"[Task {task_id}] CUDA cache cleared for device {device}.")
             except Exception as cache_err:
                  logger.warning(f"[Task {task_id}] Error clearing CUDA cache: {cache_err}")


# --- Main execution block (for testing import or standalone info) ---
if __name__ == '__main__':
    print("--- Real-ESRGAN Logic Module ---")
    print(f"GFPGAN Available: {gfpgan_available}")
    print("Available Models:")
    for name, info in MODEL_INFO.items():
        print(f"- {name} (Scale: {info['netscale']}x)")
    print("\nThis script is intended to be imported as a module by the FastAPI app.")