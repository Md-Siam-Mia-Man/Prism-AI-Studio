import time
import os
import shutil
import uuid
import zipfile
import threading
import torch
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging # Added logging
import traceback # For detailed error logging

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request,
    HTTPException, BackgroundTasks, Query
)
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError # Import ValidationError for explicit catch

# Import the enhancement function and model info from our modified realesrgan script
try:
    from prism_logic import (
        enhance_image, MODEL_INFO, get_model_path, get_gfpgan_path,
        gfpgan_available, GFPGAN_MODEL_NAME
    )
    print("Successfully imported from prism_logic.py")
    print(f"GFPGAN Available: {gfpgan_available}")
    print(f"Models Available: {list(MODEL_INFO.keys())}")
except ImportError as e:
    print(f"Error importing from prism_logic.py: {e}")
    print(traceback.format_exc()) # Print full traceback
    print("Please ensure prism_logic.py is in the same directory and all its dependencies (torch, basicsr, gfpgan, etc.) are installed.")
    exit(1)
except Exception as e: # Catch other potential init errors in prism_logic
    print(f"An unexpected error occurred during import or initialization in prism_logic.py: {e}")
    print(traceback.format_exc())
    exit(1)


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'outputs'
WEIGHTS_FOLDER = BASE_DIR / 'weights'
STATIC_FOLDER = BASE_DIR / 'static' # Assuming index.html is here
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'}

# --- Directory Setup ---
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
WEIGHTS_FOLDER.mkdir(parents=True, exist_ok=True)
STATIC_FOLDER.mkdir(parents=True, exist_ok=True) # Ensure static exists

# --- FastAPI App Initialization ---
app = FastAPI(title="prism_logic AI Studio")

# --- Global State (WARNING: Not multiprocess-safe! Use workers=1) ---
# tasks: Stores status of individual enhancement tasks
#   Key: task_id (str)
#   Value: {'status': str, 'output_url': Optional[str], 'client_id': str, 'original_name': str, 'input_path': Path, 'cancel_requested': bool}
tasks: Dict[str, Dict[str, Any]] = {}

# active_connections: Maps client_id to their active WebSocket connection
#   Key: client_id (str)
#   Value: WebSocket
active_connections: Dict[str, WebSocket] = {}

# Lock for thread-safe access to the shared tasks dictionary (basic protection)
tasks_lock = threading.Lock()

# --- GPU/CPU Detection (Run once at startup) ---
has_gpu = False
gpu_name = "CPU"
default_gpu_id = -1
default_fp32 = True
num_gpus = 0

try:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            gpu_name = torch.cuda.get_device_name(0)
            default_gpu_id = 0 # Default to the first GPU
            default_fp32 = False # Default to FP16 on GPU
            has_gpu = True
            logger.info(f"NVIDIA GPU detected: {gpu_name} (CUDA available, {num_gpus} device(s))")
        else:
            logger.info("CUDA available but no compatible devices found. Using CPU.")
    else:
        logger.info("No NVIDIA GPU detected or CUDA not properly installed, using CPU.")
except Exception as e:
    logger.warning(f"Error during GPU detection: {e}. Defaulting to CPU.", exc_info=True)
    has_gpu = False
    gpu_name = "CPU"
    default_gpu_id = -1
    default_fp32 = True
    num_gpus = 0


logger.info(f"Effective Device: {gpu_name}, Default GPU ID: {default_gpu_id}, Default FP32: {default_fp32}")

# --- Helper Functions ---
def allowed_file(filename: str) -> bool:
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_unique_task_id() -> str:
    """Generates a unique task identifier."""
    return str(uuid.uuid4())

# --- WebSocket Message Sending ---
async def send_ws_message(client_id: str, message: dict):
    """Sends a JSON message to a specific client via WebSocket if connected."""
    websocket = active_connections.get(client_id)
    if websocket:
        try:
            await websocket.send_json(message)
            # logger.debug(f"WS Sent to {client_id}: {message['type']}") # Reduce verbosity
        except WebSocketDisconnect:
             logger.warning(f"WebSocket disconnected for client {client_id} while trying to send message (type: {message.get('type')}).")
             # Connection removal is handled in the main websocket endpoint's finally block
        except Exception as e:
            logger.error(f"Error sending WebSocket message to client {client_id} (type: {message.get('type')}): {e}", exc_info=False) # Log less detail usually
    # else:
    #     logger.warning(f"Client {client_id} not connected, cannot send message type: {message.get('type')}")

# --- Background Task Runner ---
def create_enhancement_callback(loop: asyncio.AbstractEventLoop, client_id: str, task_id: str):
    """Creates a callback function that sends messages via WebSocket in the correct event loop
       and checks for cancellation flags."""
    def callback(payload: dict):
        # 1. Check cancellation flag before sending update
        with tasks_lock:
            task_info = tasks.get(task_id)
            if task_info and task_info.get('cancel_requested'):
                 logger.info(f"[Task {task_id}] Cancellation requested, skipping callback message: {payload.get('type')}")
                 # Optionally send a 'cancelled' status message once? Handled elsewhere usually.
                 return # Stop sending updates if cancelled

        # 2. Schedule the async send function to run in the main event loop
        if loop and loop.is_running():
             # Ensure task_id is always in the payload sent back
             payload_with_taskid = {'task_id': task_id, **payload}
             asyncio.run_coroutine_threadsafe(send_ws_message(client_id, payload_with_taskid), loop)
        else:
            logger.warning(f"[Task {task_id}] Event loop not running for callback. Cannot send: {payload.get('type')}")

    return callback

# --- Background Enhancement Thread ---
def run_enhancement_thread(loop: asyncio.AbstractEventLoop, args_dict: Dict, input_path: Path, output_base_path: Path, task_id: str, client_id: str, original_name: str):
    """Target function for the enhancement thread. Handles processing and status updates."""
    logger.info(f"Thread started for task {task_id} (Client: {client_id}, File: {original_name})")
    task_status = 'error' # Default status unless successfully completed
    error_message = 'Unknown processing error'

    if not loop:
         logger.error(f"[Task {task_id}] Event loop not provided to thread. Cannot send WS updates.")
         error_message = 'Server setup error (missing event loop)'
         # Update status directly (less safe but necessary here)
         with tasks_lock:
            if task_id in tasks: tasks[task_id]['status'] = 'error'
         # Cannot send WS message here
         return

    # Create the callback using the correct loop and task_id
    message_callback = create_enhancement_callback(loop, client_id, task_id)

    try:
        # Update task status to 'processing' (using lock for safety)
        with tasks_lock:
            if task_id in tasks:
                # Check if cancelled *before* starting processing
                if tasks[task_id].get('cancel_requested'):
                    logger.info(f"[Task {task_id}] Task was cancelled before processing thread started.")
                    tasks[task_id]['status'] = 'cancelled' # Set final status
                    # Send cancelled status update
                    asyncio.run_coroutine_threadsafe(send_ws_message(client_id, {'type':'status_update', 'task_id':task_id, 'status':'cancelled', 'message':'Task cancelled'}), loop)
                    return # Exit thread early
                tasks[task_id]['status'] = 'processing'
            else:
                 logger.warning(f"Task ID {task_id} not found in global dict when starting thread.")
                 # Should not happen if initialized correctly in /enhance
                 # If it does, we can't reliably process or update status.
                 return

        # Call the potentially long-running, blocking enhancement function
        # This function uses the callback internally for progress updates
        enhance_image(
            args=args_dict,
            input_path=str(input_path), # Ensure path is string
            output_path_base=str(output_path_base), # Ensure path is string
            task_id=task_id,
            send_message_callback=message_callback # Pass the wrapper callback
        )

        # If enhance_image finishes without raising an exception, assume it sent 'complete' or 'error' itself.
        # We need to check the final status set by the callback or the function itself.
        with tasks_lock:
            final_status = tasks.get(task_id, {}).get('status', 'unknown')
        logger.info(f"[Task {task_id}] Enhancement function finished with reported status: {final_status}")
        # We don't set status here, rely on the callback mechanism within enhance_image

    except Exception as e:
        error_message = f'Error in enhancement thread: {type(e).__name__}'
        logger.error(f"[Task {task_id}] {error_message}: {e}", exc_info=True) # Log full traceback
        # Update task status and send error message via callback
        with tasks_lock:
            if task_id in tasks: tasks[task_id]['status'] = 'error'
        error_payload = {
            'type': 'status_update',
            'task_id': task_id,
            'status': 'error',
            'message': error_message # Send concise error message
        }
        # Use the callback to send the error message
        message_callback(error_payload)

    finally:
         logger.info(f"Thread finished for task {task_id}")
# --- Pydantic Models ---
class EnhanceParams(BaseModel):
    model_name: str = Field(default='RealESRGAN_x4plus', description="Name of the RealESRGAN model to use.")
    outscale: float = Field(default=4.0, gt=0, description="Final upscaling factor.")
    denoise_strength: Optional[float] = Field(default=0.5, ge=0, le=1, description="Denoise strength (0-1) for specific models like realesr-general-x4v3.")
    tile: int = Field(default=0, ge=0, description="Tile size for processing (0 for auto/disabled). Reduce if OOM.")
    tile_pad: int = Field(default=10, ge=0, description="Padding for tiles.")
    pre_pad: int = Field(default=0, ge=0, description="Padding added before processing.")
    face_enhance: bool = Field(default=True, description="Enable GFPGAN face enhancement (if available).") # Default changed
    fp32: Optional[bool] = Field(default=None, description="Use 32-bit floating point precision (slower on GPU, required for CPU). If None, defaults based on device.")
    alpha_upsampler: str = Field(default='realesrgan', description="Upsampler for alpha channel ('realesrgan' or 'bicubic').")
    ext: str = Field(default='png', description="Output file extension ('auto', 'png', 'jpg', 'webp'). Defaulting to png.") # Default changed
    use_gpu: Optional[bool] = Field(default=None, description="Suggest whether to use GPU (True/False). If None, defaults based on detection.")

    # Optional: Suppress Pydantic warning if needed
    # model_config = {'protected_namespaces': ()}


class EnhanceRequest(BaseModel):
    client_id: str = Field(description="Unique identifier for the client session.")
    task_ids: List[str] = Field(description="List of task IDs corresponding to uploaded files to enhance.")
    params: EnhanceParams = Field(description="Enhancement parameters.")

class DownloadRequest(BaseModel):
    client_id: str = Field(description="Unique identifier for the client session.")
    task_ids: List[str] = Field(description="List of task IDs corresponding to completed files to download.")

class ClearRequest(BaseModel):
    client_id: str = Field(description="Unique identifier for the client session whose data should be cleared.")
    cancel_tasks: Optional[List[str]] = Field(default=[], description="List of task IDs to attempt cancellation for.") # Added for explicit cancellation

class TaskActionRequest(BaseModel): # New model for single task actions
    client_id: str
    task_id: str

class ModelDownloadRequest(BaseModel): # For WebSocket message payload validation
    model_name: str
    task_id: Optional[str] = None


# --- WebSocket Endpoint ---
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    if client_id in active_connections:
        logger.warning(f"Client {client_id} connected again? Closing previous connection.")
        old_ws = active_connections.get(client_id)
        if old_ws:
            try:
                await old_ws.close(code=1008, reason="New connection established")
            except Exception as e:
                 logger.warning(f"Error closing old WebSocket for {client_id}: {e}")
    active_connections[client_id] = websocket
    logger.info(f"WebSocket connection established for client: {client_id}")

    # Send initial GPU info
    await send_ws_message(client_id, {
        'type': 'gpu_info',
        'has_gpu': has_gpu,
        'gpu_name': gpu_name,
        'num_gpus': num_gpus
    })

    try:
        while True:
            # Listen for messages from the client
            data_text = await websocket.receive_text()
            # logger.debug(f"WS Received from {client_id}: {data_text}") # Reduce noise
            try:
                data = json.loads(data_text)
                message_type = data.get("type")
                payload = data.get("payload", {})

                if message_type == "request_model_download":
                    # Validate payload using Pydantic
                    try:
                        model_req = ModelDownloadRequest(**payload)
                        await handle_request_model_download(model_req, client_id)
                    except ValidationError as validation_error: # Catch Pydantic validation errors explicitly
                         logger.warning(f"Invalid model download request payload from {client_id}: {validation_error}")
                         await send_ws_message(client_id, {'type':'error', 'message': f'Invalid request format: {validation_error}'})

                elif message_type == "cancel_task":
                     task_id_to_cancel = payload.get('task_id')
                     if task_id_to_cancel:
                          logger.info(f"[Client {client_id}] Received cancellation request for task: {task_id_to_cancel}")
                          handle_cancel_task(task_id_to_cancel, client_id) # Call cancellation handler
                     else:
                          logger.warning(f"[Client {client_id}] Received cancel_task request without task_id.")

                else:
                     logger.warning(f"Received unknown WebSocket message type from {client_id}: {message_type}")

            except json.JSONDecodeError:
                 logger.error(f"Invalid JSON received from client {client_id}: {data_text}")
            except Exception as e:
                 logger.error(f"Error processing WebSocket message from client {client_id}: {e}", exc_info=True)


    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected for client: {client_id}, code: {e.code}, reason: {e.reason}")
        # Clean up connection handled in finally block

    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler for client {client_id}: {e}", exc_info=True)
        # Ensure connection is removed on unexpected error

    finally:
        # Clean up connection state when client disconnects or error occurs
        if client_id in active_connections and active_connections.get(client_id) == websocket:
             del active_connections[client_id]
             logger.info(f"Removed WebSocket connection for client: {client_id}")
        # Note: We are *not* clearing client tasks/uploads on simple disconnect.
        # This allows potential resume, but requires manual cleanup or a timeout mechanism later.


def handle_cancel_task(task_id: str, client_id: str):
    """Sets the cancellation flag for a given task."""
    with tasks_lock:
        task_info = tasks.get(task_id)
        if task_info and task_info.get('client_id') == client_id:
            if task_info['status'] in ['queued', 'processing']:
                 tasks[task_id]['cancel_requested'] = True
                 logger.info(f"Cancellation flag set for task {task_id} (Client: {client_id})")
                 # Note: Actual stopping depends on the enhancement logic checking this flag.
                 # We can also update status immediately if queued.
                 if tasks[task_id]['status'] == 'queued':
                      tasks[task_id]['status'] = 'cancelled'
                      # Send immediate feedback if it was just queued
                      try:
                           loop = asyncio.get_running_loop() # Get loop if possible
                           if loop and loop.is_running():
                                asyncio.run_coroutine_threadsafe(send_ws_message(client_id, {'type':'status_update', 'task_id':task_id, 'status':'cancelled', 'message':'Task cancelled'}), loop)
                           else:
                                logger.warning(f"[Task {task_id}] Cannot send immediate cancel status: No running loop.")
                      except RuntimeError:
                           logger.warning(f"[Task {task_id}] Cannot send immediate cancel status: No running loop.")


            else:
                 logger.info(f"Task {task_id} not in cancellable state (Status: {task_info['status']}). Ignoring cancel request.")
        else:
             logger.warning(f"Cannot cancel task {task_id}: Not found or client mismatch (Requested by {client_id}).")


async def handle_request_model_download(payload: ModelDownloadRequest, client_id: str):
    """Handles model download requests received via WebSocket."""
    model_name = payload.model_name
    task_id = payload.task_id # Optional task_id from payload
    logger.info(f"[Client {client_id}] Requested download for model: {model_name} (Task ID: {task_id})")

    # Basic check if model name is known
    if model_name != GFPGAN_MODEL_NAME and model_name not in MODEL_INFO:
        await send_ws_message(client_id, {
            'type': 'model_download_status',
            'model_name': model_name,
            'status': 'error',
            'message': 'Unknown model name provided.'
        })
        return

    # Get the running event loop to schedule the callback
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
         logger.error(f"Could not get running event loop for model download {model_name} (Client {client_id}). Cannot send WS updates.")
         await send_ws_message(client_id, {
            'type': 'model_download_status',
            'model_name': model_name,
            'status': 'error',
            'message': 'Server error: Cannot get event loop.'
         })
         return

    # Create the callback using the loop from the main thread (passing task_id if present)
    # Use a unique ID for the callback context, even if no task_id provided
    callback_context_id = task_id if task_id else f"dl_{model_name}_{uuid.uuid4().hex[:4]}"
    download_callback = create_enhancement_callback(loop, client_id, callback_context_id)

    # Determine which download function to call
    if model_name == GFPGAN_MODEL_NAME:
         target_func = get_gfpgan_path
         # Pass model_dir, callback, optional task_id
         target_args = (WEIGHTS_FOLDER, download_callback, task_id)
    elif model_name in MODEL_INFO:
         target_func = get_model_path
         # Pass model_name, model_dir, callback, optional task_id
         target_args = (model_name, WEIGHTS_FOLDER, download_callback, task_id)
    else:
         # This case should have been caught earlier, but handle defensively
         logger.error(f"Unexpected model name '{model_name}' in handle_request_model_download")
         await send_ws_message(client_id, {
            'type': 'model_download_status',
            'model_name': model_name,
            'status': 'error',
            'message': f"Internal Server Error: Unexpected model name '{model_name}'."
         })
         return

    try:
        # Notify client download is starting (send immediately via websocket)
        await send_ws_message(client_id, {
            'type': 'model_download_status', # Use a distinct type
            'model_name': model_name,
            'status': 'started',
            'message': f'Download initiated for {model_name}.'
            # We don't associate with task_id here, callback will handle it
        })

        # Run blocking download function in a separate thread using asyncio executor
        # This integrates better with the event loop than raw threading.Thread
        await loop.run_in_executor(
            None, # Use default thread pool executor
            target_func,
            *target_args # Unpack arguments
        )
        # Note: Status updates (downloading, complete, error) are sent via the callback *within* target_func

    except Exception as e:
        logger.error(f"Error starting or running model download task for {model_name} (Client {client_id}): {e}", exc_info=True)
        # Send error status back to client if starting the task failed
        await send_ws_message(client_id, {
            'type': 'model_download_status',
            'model_name': model_name,
            'status': 'error',
            'message': f'Error starting download: {e}'
        })


# --- HTTP Endpoints ---

# Serve Frontend (index.html)
@app.get("/", response_class=HTMLResponse, include_in_schema=False) # Hide from docs
async def read_root(request: Request):
    index_path = STATIC_FOLDER / "index.html"
    if not index_path.is_file():
         logger.error(f"{index_path} not found.")
         raise HTTPException(status_code=404, detail="Frontend application not found.")
    # logger.info(f"Serving frontend: {index_path}")
    return FileResponse(index_path, media_type="text/html")


# Serve Output Images
@app.get("/outputs/{filename}", include_in_schema=False) # Hide from docs
async def get_output_file(filename: str):
    # Basic path traversal check
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
         logger.warning(f"Attempted path traversal: {filename}")
         raise HTTPException(status_code=400, detail="Invalid filename format")

    file_path = OUTPUT_FOLDER / filename
    if not file_path.is_file():
        logger.warning(f"Output file not found: {file_path}")
        # Check if the task failed, maybe provide more info?
        # For now, just 404
        raise HTTPException(status_code=404, detail="Output file not found or task failed.")
    # logger.debug(f"Serving output file: {file_path}")
    return FileResponse(file_path)

# Get Initial Config
@app.get("/get_config")
async def get_config_endpoint():
    """Sends initial configuration to the frontend."""
    available_models = list(MODEL_INFO.keys())
    config_data = {
        'has_gpu': has_gpu,
        'gpu_name': gpu_name,
        'num_gpus': num_gpus,
        'available_models': available_models,
        'default_model': 'RealESRGAN_x4plus', # Or choose a different default
        'default_outscale': 4.0,
        'default_tile': 0,
        'gfpgan_available': gfpgan_available,
        'default_face_enhance': True, # CHANGED DEFAULT
        'allowed_extensions': list(ALLOWED_EXTENSIONS),
    }
    logger.info(f"Sending config: { {k:v for k,v in config_data.items() if k != 'available_models'} }") # Log without huge model list
    return JSONResponse(content=config_data)

# Check Model Status
@app.get("/check_model_status")
async def check_model_status_endpoint(model_name: str = Query(...), face_enhance: bool = Query(False)):
    """Checks if required models (RealESRGAN + optional GFPGAN) exist locally."""
    logger.info(f"Checking model status for model_name='{model_name}', face_enhance={face_enhance}") # Log input

    # Validate model_name early
    if not model_name or (model_name not in MODEL_INFO and model_name != GFPGAN_MODEL_NAME): # Allow checking GFPGAN directly too
         logger.warning(f"Invalid model name requested for status check: '{model_name}'")
         raise HTTPException(status_code=400, detail=f"Invalid model name specified: {model_name}")

    models_status = {} # Store status like {'filename.pth': True/False}
    all_required_exist = True # Tracks if models *needed* for this specific request exist

    try:
        required_esrgan_files = []
        # Check Real-ESRGAN model(s)
        if model_name in MODEL_INFO:
            info = MODEL_INFO[model_name]
            logger.debug(f"Model Info found for '{model_name}'. URLs: {info.get('urls', [])}")
            for i, url in enumerate(info.get('urls', [])): # Safely get URLs
                filename = os.path.basename(url)
                # Determine expected filename based on convention
                expected_filename = filename
                if model_name == 'realesr-general-x4v3':
                     if 'wdn' in filename: expected_filename = 'realesr-general-wdn-x4v3.pth'
                     else: expected_filename = 'realesr-general-x4v3.pth'
                elif len(info.get('urls', [])) == 1: # Single file model
                     expected_filename = f"{model_name}.pth"

                required_esrgan_files.append(expected_filename) # Add to list of files needed
                path = WEIGHTS_FOLDER / expected_filename
                exists = path.is_file()
                models_status[expected_filename] = exists
                logger.debug(f"Checking file '{expected_filename}': Exists={exists} at Path='{path}'")
                if not exists: all_required_exist = False # If any ESRGAN file missing, requirement fails
        else:
             logger.debug(f"'{model_name}' not in MODEL_INFO (likely checking GFPGAN directly).")


        # Check GFPGAN model status
        gfpgan_needed_for_request = face_enhance and gfpgan_available
        gfpgan_filename = GFPGAN_MODEL_NAME
        logger.debug(f"Checking GFPGAN '{gfpgan_filename}'. Needed for request: {gfpgan_needed_for_request}, Module available: {gfpgan_available}")

        if gfpgan_available:
             gfpgan_path = WEIGHTS_FOLDER / gfpgan_filename
             gfpgan_exists = gfpgan_path.is_file()
             models_status[gfpgan_filename] = gfpgan_exists
             logger.debug(f"GFPGAN file check: Exists={gfpgan_exists} at Path='{gfpgan_path}'")
             if gfpgan_needed_for_request and not gfpgan_exists:
                 all_required_exist = False # Mark as missing *if needed* but not present
                 logger.debug("GFPGAN needed but does not exist, setting all_required_exist=False.")
        else:
            models_status[gfpgan_filename] = False # Report as false if module unavailable
            if face_enhance: # If user requested it without module
                all_required_exist = False
                logger.debug("GFPGAN requested but module unavailable, setting all_required_exist=False.")


    except Exception as e:
        logger.error(f"Error checking model status for '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error checking model status: {e}")

    logger.info(f"Model status check result for '{model_name}' (face={face_enhance}): AllRequiredExist={all_required_exist}, StatusDict={models_status}") # Log result
    return JSONResponse({'models_status': models_status, 'all_exist': all_required_exist})


# Upload Files
@app.post("/upload")
async def upload_files_endpoint(
    client_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Handles file uploads, associating them with a client_id and creating task entries."""
    if not client_id:
         raise HTTPException(status_code=400, detail="Client ID is required.")

    uploaded_files_info = []
    errors = []

    if not files:
         logger.warning(f"Upload request received for client {client_id} with no files attached.")
         # Return immediately if no files
         return JSONResponse({
             'message': 'No files provided in the upload request.',
             'uploaded_files': [],
             'errors': []
         }, status_code=400)

    logger.info(f"[Client {client_id}] Received upload request with {len(files)} file(s).")

    for file in files:
        original_filename = file.filename
        if not original_filename:
            logger.warning("Received upload with empty filename.")
            errors.append({'original_name': '(empty)', 'error': 'File has no name.'})
            continue

        if allowed_file(original_filename):
            ext = original_filename.rsplit('.', 1)[1].lower()
            task_id = get_unique_task_id() # Unique ID for this image's task lifecycle
            # Use task ID for server filename to guarantee uniqueness and link to task
            server_filename = f"{task_id}.{ext}"
            server_path = UPLOAD_FOLDER / server_filename


            try:
                start_time = time.time()
                # Use shutil.copyfileobj for potentially better memory efficiency with large files
                with open(server_path, 'wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_size = server_path.stat().st_size # Get size after saving
                logger.info(f"[Task {task_id}] Saved '{original_filename}' as {server_filename} for client {client_id} ({file_size / (1024*1024):.2f} MB) in {(time.time() - start_time)*1000:.1f} ms")

                # Store task info immediately using lock
                with tasks_lock:
                    tasks[task_id] = {
                        'task_id': task_id, # Redundant but useful sometimes
                        'client_id': client_id,
                        'original_name': original_filename,
                        'input_path': server_path, # Store Path object
                        'status': 'pending', # Initial status after upload
                        'output_url': None,
                        'size': file_size,
                        'cancel_requested': False
                    }

                # Prepare info for response (don't send server_path)
                file_info_response = {
                    'original_name': original_filename,
                    'task_id': task_id,
                    'upload_success': True,
                    'size': file_size
                }
                uploaded_files_info.append(file_info_response)

            except Exception as e:
                 logger.error(f"Error saving uploaded file '{original_filename}' for client {client_id} (Task {task_id}): {e}", exc_info=True)
                 errors.append({'original_name': original_filename, 'error': f'Server save error: {e}'})
                 # Clean up partial file if it exists
                 if server_path.exists():
                      try: os.remove(server_path)
                      except OSError: logger.error(f"Failed to cleanup partial upload: {server_path}")
            finally:
                 await file.close() # Ensure file handle is closed

        else:
             # File type not allowed
             logger.warning(f"[Client {client_id}] Rejected file upload: Invalid type '{original_filename}'")
             errors.append({'original_name': original_filename, 'error': 'File type not allowed'})
             await file.close()

    return JSONResponse({
         'message': f'Processed {len(files)} files.',
         'uploaded_files': uploaded_files_info, # Send only successfully saved files' info
         'errors': errors
    })


# Enhance Files
@app.post("/enhance")
async def enhance_files_endpoint(enhance_req: EnhanceRequest):
    """Starts the enhancement process for specified files for a given client."""
    client_id = enhance_req.client_id
    task_ids_to_process = enhance_req.task_ids
    params = enhance_req.params

    logger.info(f"Received enhancement request for client {client_id}, tasks: {task_ids_to_process}")
    # logger.debug(f"Enhancement parameters: {params.model_dump()}")

    if not task_ids_to_process:
         raise HTTPException(status_code=400, detail="No task IDs provided for enhancement")

    # Check WebSocket connection (optional, but recommended for status updates)
    if client_id not in active_connections:
         logger.warning(f"WebSocket not connected for client {client_id} during /enhance request. Status updates will be unavailable.")
         # Proceeding anyway, but warn the user or raise error if updates are critical
         # raise HTTPException(status_code=400, detail="Client is not connected via WebSocket. Cannot guarantee status updates.")

    # --- Prepare arguments for enhance_image ---
    effective_gpu_id = default_gpu_id
    effective_fp32 = default_fp32

    if params.use_gpu is False: # User explicitly requested CPU
        effective_gpu_id = -1
        effective_fp32 = True
    elif params.use_gpu is True: # User explicitly requested GPU
         if has_gpu:
              effective_gpu_id = default_gpu_id # Use detected GPU
              effective_fp32 = params.fp32 if params.fp32 is not None else default_fp32
         else:
              logger.warning(f"[Client {client_id}] GPU requested but not available. Falling back to CPU.")
              effective_gpu_id = -1
              effective_fp32 = True
    elif has_gpu: # User didn't specify, use default GPU behavior if available
        effective_gpu_id = default_gpu_id
        effective_fp32 = params.fp32 if params.fp32 is not None else default_fp32
    else: # No GPU available and not explicitly requested
        effective_gpu_id = -1
        effective_fp32 = True

    # Final check: Ensure FP32 is true if CPU is used
    if effective_gpu_id == -1:
        effective_fp32 = True

    # Build args dict for the enhancement function
    args = {
        'model_name': params.model_name,
        'outscale': params.outscale,
        'denoise_strength': params.denoise_strength if params.model_name == 'realesr-general-x4v3' else None,
        'tile': params.tile,
        'tile_pad': params.tile_pad,
        'pre_pad': params.pre_pad,
        'face_enhance': params.face_enhance and gfpgan_available, # Only True if requested AND available
        'fp32': effective_fp32,
        'alpha_upsampler': params.alpha_upsampler,
        'ext': params.ext,
        'gpu_id': effective_gpu_id
    }

    logger.info(f"[Client {client_id}] Effective enhancement args: {args}")

    # --- Trigger Background Tasks ---
    files_queued_count = 0
    loop = asyncio.get_running_loop() # Get loop once for all threads started here

    with tasks_lock: # Lock when accessing/modifying tasks dict
        for task_id in task_ids_to_process:
            task_info = tasks.get(task_id)

            # Check if task exists, belongs to client, and is in 'pending' state
            if task_info and task_info.get('client_id') == client_id and task_info.get('status') == 'pending':
                input_path = task_info.get('input_path')
                original_name = task_info.get('original_name', 'Unknown')

                if input_path and isinstance(input_path, Path) and input_path.exists():
                    # Generate output base path (using task_id for uniqueness)
                    output_filename_base = f"{task_id}_out" # Simple base name using task_id
                    output_base_path = OUTPUT_FOLDER / output_filename_base

                    # Update task status to 'queued'
                    tasks[task_id]['status'] = 'queued'
                    tasks[task_id]['cancel_requested'] = False # Reset cancel flag

                    # Send queued status update via WebSocket (fire and forget)
                    asyncio.run_coroutine_threadsafe(send_ws_message(client_id, {
                        'type': 'status_update', 'task_id': task_id, 'status': 'queued', 'message': 'Queued...'
                        }), loop)

                    # Start the enhancement in a separate thread using the executor
                    logger.info(f"Queuing enhancement thread for task {task_id} ({original_name})")
                    loop.run_in_executor(
                        None, # Use default executor
                        run_enhancement_thread, # Target function
                        loop, # Pass the loop
                        args.copy(), # Pass a copy of args
                        input_path,
                        output_base_path,
                        task_id,
                        client_id,
                        original_name
                    )
                    files_queued_count += 1
                else:
                    logger.warning(f"Skipping task {task_id}: Input file missing or invalid path ({input_path}) for client {client_id}.")
                    tasks[task_id]['status'] = 'error' # Update status
                    # Send error status via WebSocket
                    asyncio.run_coroutine_threadsafe(send_ws_message(client_id, {
                        'type': 'status_update', 'task_id': task_id, 'status': 'error', 'message': 'Input file not found on server.'
                        }), loop)
            else:
                # Log why task wasn't processed (not found, wrong client, wrong status)
                status_reason = "Not found" if not task_info else f"Client mismatch ({task_info.get('client_id')})" if task_info.get('client_id') != client_id else f"Not pending (Status: {task_info.get('status')})"
                logger.warning(f"Skipping task {task_id}: {status_reason} for client {client_id}.")
                # Optionally send 'skipped' or 'error' status via WS? For now, just skip.


    if files_queued_count == 0 and task_ids_to_process:
         logger.warning(f"Enhancement request for client {client_id} had tasks, but none were in a valid state to be queued.")
         # Raise error if *no* files could be queued
         raise HTTPException(status_code=400, detail="None of the requested files were found or ready for enhancement.")


    return JSONResponse({'message': f'Enhancement process initiated for {files_queued_count} files.'})

# Delete a specific task's data
@app.post("/delete_task")
async def delete_task_endpoint(req: TaskActionRequest):
    """Deletes the input and output files associated with a specific task."""
    client_id = req.client_id
    task_id = req.task_id
    logger.info(f"[Client {client_id}] Request to delete task: {task_id}")

    deleted_input = False
    deleted_output = False
    error = None

    with tasks_lock:
        task_info = tasks.get(task_id)
        if task_info and task_info.get('client_id') == client_id:
            # Remove input file
            input_path = task_info.get('input_path')
            if input_path and isinstance(input_path, Path) and input_path.exists():
                try:
                    os.remove(input_path)
                    logger.info(f"Deleted input file for task {task_id}: {input_path}")
                    deleted_input = True
                except OSError as e:
                     error = f"Error deleting input file: {e}"
                     logger.error(f"[Task {task_id}] {error}")

            # Remove output file if URL exists
            output_url = task_info.get('output_url')
            if output_url and isinstance(output_url, str) and output_url.startswith('/outputs/'):
                 try:
                     relative_path = output_url.split('/', 2)[2]
                     if ".." not in relative_path and not relative_path.startswith("/"):
                         full_output_path = OUTPUT_FOLDER / relative_path
                         if full_output_path.exists():
                             os.remove(full_output_path)
                             logger.info(f"Deleted output file for task {task_id}: {full_output_path}")
                             deleted_output = True
                 except OSError as e:
                     err_msg = f"Error deleting output file: {e}"
                     if error: error += f"; {err_msg}" # Append errors
                     else: error = err_msg
                     logger.error(f"[Task {task_id}] {err_msg}")

            # Remove task from dictionary
            del tasks[task_id]
            logger.info(f"Removed task entry {task_id}")
        else:
             error = "Task not found or client ID mismatch."
             logger.warning(f"Delete task failed for {task_id} (Client {client_id}): {error}")
             raise HTTPException(status_code=404, detail=error)

    if error:
         # Return success but include error message about file deletion issues
         return JSONResponse({'message': f'Task entry {task_id} removed, but file deletion encountered issues.', 'error': error}, status_code=207) # Multi-status
    else:
         return JSONResponse({'message': f'Successfully deleted task {task_id} and associated files.'})

# Clear Session Files (Modified to use task dict)
@app.post("/clear")
async def clear_session_endpoint(clear_req: ClearRequest):
    """Clears all tasks and associated files for the specified client."""
    client_id = clear_req.client_id
    tasks_to_cancel = clear_req.cancel_tasks or [] # Get list of tasks to try cancelling
    cleared_input_count = 0
    cleared_output_count = 0
    cleared_task_count = 0
    cancelled_task_count = 0
    errors = []

    logger.info(f"Attempting to clear session for client: {client_id}, cancelling tasks: {tasks_to_cancel}")

    with tasks_lock:
        # Identify all tasks belonging to this client
        all_task_ids = list(tasks.keys())
        client_task_ids = [tid for tid in all_task_ids if tasks.get(tid, {}).get('client_id') == client_id]
        logger.info(f"Found {len(client_task_ids)} task entries for client {client_id}.")

        for task_id in client_task_ids:
            task_info = tasks.get(task_id) # Should exist as we just filtered
            if not task_info: continue # Should not happen, but safety check

            # 1. Attempt to cancel if requested and applicable
            if task_id in tasks_to_cancel and task_info['status'] in ['queued', 'processing']:
                tasks[task_id]['cancel_requested'] = True
                cancelled_task_count += 1
                logger.info(f"Set cancel flag for task {task_id} during clear.")
                # Also set status immediately if queued
                if tasks[task_id]['status'] == 'queued':
                    tasks[task_id]['status'] = 'cancelled'

            # 2. Delete associated files
            # Remove input file
            input_path = task_info.get('input_path')
            if input_path and isinstance(input_path, Path) and input_path.exists():
                try:
                    os.remove(input_path)
                    cleared_input_count += 1
                except OSError as e:
                     err_msg = f"Error removing input {input_path.name}: {e}"
                     errors.append(err_msg)
                     logger.error(f"[Task {task_id}] {err_msg}")

            # Remove output file
            output_url = task_info.get('output_url')
            if output_url and isinstance(output_url, str) and output_url.startswith('/outputs/'):
                try:
                    relative_path = output_url.split('/', 2)[2]
                    if ".." not in relative_path and not relative_path.startswith("/"):
                        full_output_path = OUTPUT_FOLDER / relative_path
                        if full_output_path.exists():
                            os.remove(full_output_path)
                            cleared_output_count += 1
                except OSError as e:
                     err_msg = f"Error removing output {output_url}: {e}"
                     errors.append(err_msg)
                     logger.error(f"[Task {task_id}] {err_msg}")

            # 3. Remove task entry from the global dictionary
            del tasks[task_id]
            cleared_task_count += 1


    logger.info(f"Cleanup summary for client {client_id}: {cleared_input_count} inputs, {cleared_output_count} outputs, {cleared_task_count} tasks removed. {cancelled_task_count} cancellation(s) attempted.")

    if cleared_task_count == 0:
         message = 'No active session data found to clear.'
    else:
         message = f'Cleared {cleared_task_count} task(s) and associated files.'
         if cancelled_task_count > 0:
             message += f" Attempted to cancel {cancelled_task_count} active task(s)."

    return JSONResponse({'message': message, 'errors': errors})


# Download All Enhanced Files for specified tasks
@app.post("/download_all")
async def download_all_endpoint(download_req: DownloadRequest):
    """Zips and sends all successfully generated output files for the client's specified tasks."""
    client_id = download_req.client_id
    task_ids_to_download = download_req.task_ids

    if not task_ids_to_download:
        raise HTTPException(status_code=400, detail="No task IDs provided for download")

    output_files_to_zip = []
    processed_original_names_in_zip = set()

    logger.info(f"Download request for client {client_id}, task IDs: {task_ids_to_download}")

    with tasks_lock: # Read task data safely
        for task_id in task_ids_to_download:
            task_data = tasks.get(task_id)
            # Check task exists, belongs to the client, and is complete with an output URL
            if (task_data and
                task_data.get('client_id') == client_id and
                task_data.get('status') == 'complete' and
                task_data.get('output_url')):

                output_url = task_data['output_url']
                original_name_from_task = task_data.get('original_name', 'unknown_original') # Get original name stored in task

                try:
                     # Expecting URL like '/outputs/filename.ext'
                     if output_url.startswith('/outputs/'):
                         relative_path = output_url.split('/', 2)[2]
                         # Basic path traversal check
                         if ".." in relative_path or relative_path.startswith("/"):
                              logger.warning(f"Skipping task {task_id}: Invalid path derived from URL '{output_url}'")
                              continue

                         full_output_path = OUTPUT_FOLDER / relative_path

                         if full_output_path.exists() and full_output_path.is_file():
                             # Determine a user-friendly name for the file inside the zip archive
                             # Use original name's stem + _enhanced + actual output extension
                             output_filename_base = Path(original_name_from_task).stem
                             output_ext = full_output_path.suffix # Get the actual extension
                             zip_arcname = f"{output_filename_base}_enhanced{output_ext}"

                             # Ensure unique name within the zip archive (handle potential duplicates)
                             counter = 1
                             while zip_arcname in processed_original_names_in_zip:
                                 zip_arcname = f"{output_filename_base}_enhanced_{counter}{output_ext}"
                                 counter += 1
                             processed_original_names_in_zip.add(zip_arcname)

                             output_files_to_zip.append({'path': full_output_path, 'arcname': zip_arcname})
                             # logger.debug(f"Adding to zip: {full_output_path} as {zip_arcname}")
                         else:
                             logger.warning(f"Skipping task {task_id}: Output file path not found or not a file: {full_output_path}")
                     else:
                         logger.warning(f"Skipping task {task_id}: Cannot determine output file path from URL format: {output_url}")
                except Exception as e:
                     logger.error(f"Error processing output URL {output_url} for task {task_id}: {e}", exc_info=True)
            # else: Task doesn't exist, belong to client, isn't complete, or lacks URL - skip silently

    if not output_files_to_zip:
        logger.warning(f"No completed output files found to zip for client {client_id} and tasks {task_ids_to_download}")
        raise HTTPException(status_code=404, detail="No completed output files found for the specified tasks.")

    # Create zip file (using a temporary file for safety)
    # Use a unique temporary zip filename within the output folder
    zip_filename_stem = f"enhanced_{client_id[:8]}_{uuid.uuid4().hex[:8]}"
    zip_filepath = OUTPUT_FOLDER / f"{zip_filename_stem}.zip"
    user_download_filename = "prism_logic_ai_enhanced_images.zip" # Consistent name for user download

    try:
        logger.info(f"Creating zip file: {zip_filepath} with {len(output_files_to_zip)} files.")
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_info in output_files_to_zip:
                zf.write(file_info['path'], arcname=file_info['arcname'])
        logger.info(f"Zip file created successfully: {zip_filepath}")

        # Prepare BackgroundTask for cleanup *after* response is sent
        cleanup_task = BackgroundTask(os.remove, zip_filepath)

        # Use FileResponse to send the zip file
        return FileResponse(
            path=zip_filepath,
            filename=user_download_filename, # User-friendly download name
            media_type='application/zip',
            background=cleanup_task # Register cleanup task
        )
    except Exception as e:
        logger.error(f"Error creating or sending zip file {zip_filepath}: {e}", exc_info=True)
        # Clean up temp file immediately if created before error during send
        if zip_filepath.exists():
             try: os.remove(zip_filepath)
             except OSError as rm_err: logger.error(f"Error cleaning up failed zip {zip_filepath}: {rm_err}")
        raise HTTPException(status_code=500, detail=f"Failed to create or send zip file: {e}")


# --- Mount Static Files ---
# Serve files from the 'static' directory at the root URL path AFTER API routes
# Any path not matched by API routes above will try to be served from 'static'
# Ensure index.html is directly inside STATIC_FOLDER
app.mount("/", StaticFiles(directory=STATIC_FOLDER, html=True), name="static")


# --- Main Execution (using Uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    # Note: reload=True is for development. Set to False for production.
    # workers=1 is CRUCIAL because global state is NOT multiprocess-safe.
    uvicorn.run(
        "prism:app",          # Correct format: filename:fastapi_instance_name
        host="127.0.0.1",   # Listen only on localhost
        port=3020,
        reload=True,        # Auto-reload on code changes (development)
        workers=1           # IMPORTANT: Keep workers=1 for in-memory state
        # log_level="debug" # Optional: Increase log level for more details
    )