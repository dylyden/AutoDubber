import os
import sys
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

# PyInstaller Subprocess Proxy
# When AutoDubber.exe is called with -c or -m, it must act like python.exe
if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    exec(sys.argv[2])
    sys.exit(0)
elif len(sys.argv) >= 3 and sys.argv[1] == "-m":
    import runpy
    module_name = sys.argv[2]
    sys.argv = [module_name] + sys.argv[3:]
    runpy.run_module(module_name, run_name="__main__")
    sys.exit(0)

# --- Dynamic Patch & Plugin Search Priority ---
app_root_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
for p_dir in [os.path.join(app_root_dir, "plugins"), os.path.join(app_root_dir, "patches"), app_root_dir]:
    if os.path.isdir(p_dir) and p_dir not in sys.path:
        sys.path.insert(0, p_dir)

def get_base_path():
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        if os.path.exists(os.path.join(app_dir, "index.html")):
            return app_dir
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def get_subtitles_dir():
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        target = os.path.join(exe_dir, "subtitles")
        internal_target = os.path.join(exe_dir, "_internal", "subtitles")
        os.makedirs(target, exist_ok=True)
        if not os.path.exists(internal_target):
            try:
                import subprocess
                subprocess.run(["cmd", "/c", "mklink", "/J", internal_target, target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        return target
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "subtitles"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "AutoDubber", "subtitles"),
        os.path.join(os.getcwd(), "subtitles")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    target = candidates[0]
    os.makedirs(target, exist_ok=True)
    return target

import json
import uuid
import asyncio
import logging
import subprocess
import warnings

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from contextlib import asynccontextmanager
import tkinter as tk
from tkinter import filedialog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from concurrent.futures import ThreadPoolExecutor

# Import the pipeline and key format utilities
from dubber import Dubber, identify_key_format, get_keys_breakdown, api_key_pool, get_ffmpeg_cmd, setup_ffmpeg_path
setup_ffmpeg_path()

import re
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoDubberServer")

main_loop = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop
    main_loop = asyncio.get_running_loop()


    # Ensure subtitles directory exists
    try:
        subtitles_dir = get_subtitles_dir()
        os.makedirs(subtitles_dir, exist_ok=True)
    except Exception as init_err:
        logger.error(f"Failed to initialize subtitles dir: {init_err}")

    send_log("DenZin Auto-Dubber Engine Initialized. Ready for actions.", "success")
    yield

    # Cleanup on application exit/shutdown: only reset the queue.
    # Segment audio previews are wiped on the NEXT startup (fresh start).
    # This allows the user to close and reopen the tool without losing previews mid-session.
    try:
        queue_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queue.json")
        with open(queue_file, "w") as f:
            json.dump([], f)
        logger.info("Queue reset on shutdown.")
    except Exception as shutdown_clean_err:
        logger.error(f"Shutdown cleanup error: {shutdown_clean_err}")

app = FastAPI(title="DenZin Auto-Dubber Server", lifespan=lifespan)

# Global State
active_websockets: List[WebSocket] = []
processing_task: Any = None
is_processing: bool = False
active_task_ids: set = set()
active_tasks: dict = {}
thread_pool = ThreadPoolExecutor(max_workers=4)

SETTINGS_FILE = os.path.join(get_base_path(), "settings.json")
QUEUE_FILE = os.path.join(get_base_path(), "queue.json")

queue: List[Dict[str, Any]] = []
if os.path.exists(QUEUE_FILE):
    try:
        with open(QUEUE_FILE, "r") as f:
            queue = json.load(f)
    except Exception as e:
        logger.error(f"Error loading queue: {e}")

def save_queue_to_disk():
    try:
        with open(QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving queue: {e}")

default_settings = {
    "api_type": "free",
    "output_type": "auto",
    "custom_output_path": "",
    "model": "gemini-3.5-flash",
    "vocals_volume": 100,
    "target_lang": "km-KH",
    "voice_gender": "auto",
    "tts_engine": "edge-tts",
    "hf_voxcpm_space": "dylyden/VoxCPM-Demo",
    "hf_token": "",
    "custom_male_voice_path": "",
    "custom_female_voice_path": "",
    "custom_narrator_male_voice_path": "",
    "custom_narrator_female_voice_path": "",
    "gpu_mode": "auto",
    "concurrent_threads": 1,
    "audio_sync_offset_ms": 0,
    "auto_shutdown_pc": False,
    "delete_original_file": False,
    "max_speech_speed": 1.6,
    "color_correction": False,
    "movie_recap_mode": False,
    "video_speed": 1.0,
    "voice_clone_mode": "custom_preset",
    "youtube_multi_audio": False,
    "youtube_extra_languages": [],
    "mixer": {
        "drums": True,
        "drums_vol": 100,
        "bass": True,
        "bass_vol": 100,
        "melody": True,
        "melody_vol": 100
    },
    "github_update_repo": "dylyden/AutoDubber",
    "auto_check_updates": True
}

settings = default_settings.copy()
api_keys = []

import base64

def obscure_settings(data: str) -> str:
    key = "DenzinSettingsKey2026"
    encrypted = ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(data, key * (len(data) // len(key) + 1)))
    return base64.b64encode(encrypted.encode('utf-8')).decode('utf-8')

def deobscure_settings(data: str) -> str:
    key = "DenzinSettingsKey2026"
    decrypted_bytes = base64.b64decode(data.encode('utf-8')).decode('utf-8')
    return ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(decrypted_bytes, key * (len(decrypted_bytes) // len(key) + 1)))

# Load Settings
if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
            content = f.read().strip()
            
        is_plain = content.startswith("{")
        if is_plain:
            data = json.loads(content)
        else:
            data = json.loads(deobscure_settings(content))
            
        settings.update(data.get("settings", {}))
        api_keys = data.get("keys", [])
        api_key_pool.set_keys(api_keys)
        
        # Auto-encrypt if it was plain text
        if is_plain:
            try:
                raw_json = json.dumps({"settings": settings, "keys": api_keys}, indent=4)
                with open(SETTINGS_FILE, "w") as f:
                    f.write(obscure_settings(raw_json))
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Error loading settings: {e}")

def save_settings_to_disk():
    try:
        raw_json = json.dumps({"settings": settings, "keys": api_keys}, indent=4)
        with open(SETTINGS_FILE, "w") as f:
            f.write(obscure_settings(raw_json))
    except Exception as e:
        logger.error(f"Error saving settings: {e}")

# Broadcast via WebSockets
async def broadcast(message: Dict[str, Any]):
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)

def broadcast_sync(message: Dict[str, Any]):
    """Sync helper to broadcast from other threads using the running loop."""
    try:
        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_thread_loop(broadcast(message), loop)
    except RuntimeError:
        # Fallback if no loop is running
        pass

def schedule_task(coro):
    try:
        loop = asyncio.get_running_loop()
        if loop is main_loop:
            loop.create_task(coro)
            return
    except RuntimeError:
        pass
    try:
        asyncio.run_coroutine_threadsafe(coro, main_loop)
    except Exception as e:
        logger.error(f"Failed to schedule task: {e}")

def send_log(msg: str, level: str = "info"):
    logger.info(f"[{level}] {msg}")
    # Run async broadcast inside event loop
    coro = broadcast({"type": "log", "message": msg, "level": level})
    schedule_task(coro)

# WebSockets Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    # Send current state
    await websocket.send_json({"type": "status_update", "is_processing": is_processing})
    await websocket.send_json({"type": "queue_update", "queue": queue})
    await websocket.send_json({"type": "api_status", "keys_count": len(api_keys), "status": "Loaded"})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

# HTML static routing
@app.get("/")
async def get_index():
    return FileResponse(os.path.join(get_base_path(), "index.html"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/style.css")
async def get_css():
    return FileResponse(os.path.join(get_base_path(), "style.css"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/app.js")
async def get_js():
    return FileResponse(os.path.join(get_base_path(), "app.js"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/favicon.ico")
async def get_favicon():
    fav_path = os.path.join(get_base_path(), "favicon.ico")
    if os.path.exists(fav_path):
        return FileResponse(fav_path)
    raise HTTPException(status_code=404, detail="Favicon not found")

# API: Queue management
@app.get("/api/queue")
async def get_queue():
    return {"queue": queue}

class DroppedFile(BaseModel):
    name: str
    size: int
    path: Optional[str] = ""

class AddDroppedRequest(BaseModel):
    files: List[DroppedFile]

@app.post("/api/queue/add-dropped")
async def add_dropped(req: AddDroppedRequest):
    # Try to resolve path locally on server side (Desktop, Downloads, or Workspace)
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    workspace = os.getcwd()
    
    sorted_files = sorted(req.files, key=lambda f: natural_sort_key(f.name))
    for f in sorted_files:
        resolved_path = ""
        
        if getattr(f, 'path', "") and os.path.exists(f.path):
            resolved_path = f.path
        else:
            # 1. Check workspace
            p_work = os.path.join(workspace, f.name)
            # 2. Check desktop
            p_desk = os.path.join(desktop, f.name)
            # 3. Check downloads
            p_down = os.path.join(downloads, f.name)
            
            if os.path.exists(p_work):
                resolved_path = p_work
            elif os.path.exists(p_desk):
                resolved_path = p_desk
            elif os.path.exists(p_down):
                resolved_path = p_down
            
        if resolved_path:
            queue.append({
                "id": str(uuid.uuid4()),
                "name": f.name,
                "path": resolved_path,
                "progress": 0,
                "status": "Queued"
            })
            send_log(f"Resolved dropped file path: {resolved_path}", "success")
        else:
            send_log(f"[WARNING] Could not resolve full path for dropped file '{f.name}'. Please add it using '+ Add Files' instead.", "warning")
            
    await broadcast({"type": "queue_update", "queue": queue})
    save_queue_to_disk()
    return {"queue": queue}

@app.delete("/api/queue/delete/{video_id}")
async def delete_queue_item(video_id: str):
    global queue
    try:
        import shutil
        subtitles_dir = get_subtitles_dir()
        subtitles_json_path = os.path.join(subtitles_dir, f"{video_id}.json")
        seg_preview_dir = os.path.join(subtitles_dir, str(video_id))
        if os.path.exists(subtitles_json_path):
            os.remove(subtitles_json_path)
        if os.path.exists(seg_preview_dir):
            shutil.rmtree(seg_preview_dir, ignore_errors=True)
    except Exception as cleanup_err:
        logger.error(f"Failed to cleanup files for video_id {video_id}: {cleanup_err}")

    queue = [item for item in queue if item["id"] != video_id]
    await broadcast({"type": "queue_update", "queue": queue})
    save_queue_to_disk()
    return {"queue": queue}

@app.post("/api/queue/reset-cancelled")
async def reset_cancelled():
    for item in queue:
        if item["status"] == "Cancelled":
            item["status"] = "Queued"
            item["progress"] = 0
    await broadcast({"type": "queue_update", "queue": queue})
    save_queue_to_disk()
    return {"queue": queue}

@app.post("/api/queue/retry-errors")
async def retry_errors():
    for item in queue:
        if item["status"] not in ["Queued", "Completed", "Cancelled"]:
            if item["id"] not in active_task_ids:
                item["status"] = "Queued"
                item["progress"] = 0
    await broadcast({"type": "queue_update", "queue": queue})
    save_queue_to_disk()
    return {"queue": queue}

@app.post("/api/queue/remove-completed")
async def remove_completed():
    global queue
    completed_items = [item for item in queue if item["status"] == "Completed"]
    try:
        import shutil
        subtitles_dir = get_subtitles_dir()
        for item in completed_items:
            video_id = item["id"]
            subtitles_json_path = os.path.join(subtitles_dir, f"{video_id}.json")
            seg_preview_dir = os.path.join(subtitles_dir, str(video_id))
            if os.path.exists(subtitles_json_path):
                os.remove(subtitles_json_path)
            if os.path.exists(seg_preview_dir):
                shutil.rmtree(seg_preview_dir, ignore_errors=True)
    except Exception as cleanup_err:
        logger.error(f"Failed to cleanup completed files: {cleanup_err}")

    queue = [item for item in queue if item["status"] != "Completed"]
    await broadcast({"type": "queue_update", "queue": queue})
    save_queue_to_disk()
    return {"queue": queue}

@app.post("/api/queue/clear-all")
async def clear_all():
    global queue
    try:
        from dubber import clear_locked_speaker_roles
        clear_locked_speaker_roles()
        import shutil
        subtitles_dir = get_subtitles_dir()
        for item in queue:
            video_id = item["id"]
            subtitles_json_path = os.path.join(subtitles_dir, f"{video_id}.json")
            seg_preview_dir = os.path.join(subtitles_dir, str(video_id))
            if os.path.exists(subtitles_json_path):
                os.remove(subtitles_json_path)
            if os.path.exists(seg_preview_dir):
                shutil.rmtree(seg_preview_dir, ignore_errors=True)
    except Exception as cleanup_err:
        logger.error(f"Failed to cleanup files in clear_all: {cleanup_err}")

    queue = []
    await broadcast({"type": "queue_update", "queue": queue})
    save_queue_to_disk()
    return {"queue": queue}

# File Explorer Browsing
def run_file_dialog(browse_type: str):
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    result = None
    try:
        if browse_type == "files":
            res = filedialog.askopenfilenames(
                title="Select Video Files",
                filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.flv *.ts *.webm *.3gp")]
            )
            result = list(res) if res else None
        elif browse_type == "folder":
            res = filedialog.askdirectory(title="Select Folder containing Videos")
            result = res if res else None
        elif browse_type == "output":
            res = filedialog.askdirectory(title="Select Output Folder")
            result = res if res else None
        elif browse_type == "male_voice":
            res = filedialog.askopenfilename(
                title="Select Male Reference Voice Audio",
                filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac")]
            )
            result = res if res else None
        elif browse_type == "female_voice":
            res = filedialog.askopenfilename(
                title="Select Female Reference Voice Audio",
                filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac")]
            )
            result = res if res else None
        elif browse_type == "narrator_male_voice":
            res = filedialog.askopenfilename(
                title="Select Narrator Male Reference Voice Audio",
                filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac")]
            )
            result = res if res else None
        elif browse_type == "narrator_female_voice":
            res = filedialog.askopenfilename(
                title="Select Narrator Female Reference Voice Audio",
                filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac")]
            )
            result = res if res else None
        elif browse_type in ["young_male_voice", "young_female_voice", "old_male_voice", "old_female_voice"]:
            res = filedialog.askopenfilename(
                title=f"Select {browse_type.replace('_', ' ').title()} Audio",
                filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac")]
            )
            result = res if res else None
    except Exception as e:
        logger.error(f"Failed to run file dialog: {e}")
    finally:
        root.destroy()
        
    return result

@app.get("/api/browse")
async def browse_path(type: str):
    loop = asyncio.get_event_loop()
    selected = await loop.run_in_executor(thread_pool, run_file_dialog, type)
    
    if not selected:
        return {"cancelled": True}
        
    if type in ["output", "male_voice", "female_voice", "narrator_male_voice", "narrator_female_voice", 
                "young_male_voice", "young_female_voice", "old_male_voice", "old_female_voice"]:
        return {"cancelled": False, "path": selected}
        
    added_count = 0
    if type == "files":
        selected_sorted = sorted(selected, key=lambda x: natural_sort_key(os.path.basename(x)))
        for filepath in selected_sorted:
            if filepath not in [item["path"] for item in queue]:
                queue.append({
                    "id": str(uuid.uuid4()),
                    "name": os.path.basename(filepath),
                    "path": filepath,
                    "progress": 0,
                    "status": "Queued"
                })
                added_count += 1
    elif type == "folder":
        for root, dirs, files in os.walk(selected):
            files_sorted = sorted(files, key=natural_sort_key)
            for file in files_sorted:
                if file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.flv', '.ts', '.webm')):
                    filepath = os.path.join(root, file)
                    if filepath not in [item["path"] for item in queue]:
                        queue.append({
                            "id": str(uuid.uuid4()),
                            "name": file,
                            "path": filepath,
                            "progress": 0,
                            "status": "Queued"
                        })
                        added_count += 1
                        
    await broadcast({"type": "queue_update", "queue": queue})
    return {"cancelled": False, "added_count": added_count, "queue": queue}

# Key Format Identification API Endpoint
@app.post("/api/keys/identify")
async def identify_keys(req: Dict[str, Any]):
    raw_keys = req.get("keys", [])
    if isinstance(raw_keys, str):
        raw_keys = raw_keys.split("\n")
    breakdown = get_keys_breakdown(raw_keys)
    return {"status": "success", "breakdown": breakdown}

@app.post("/api/verify")
async def verify_api(req: Dict[str, Any]):
    api_type = req.get("api_type") or settings.get("api_type")
    if api_type == "vertex":
        # Search for GCP JSON credentials in the workspace root
        creds_file = None
        project_id = None
        for filename in os.listdir("."):
            if filename.endswith(".json") and filename not in ["settings.json", "queue.json"]:
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "project_id" in data and "private_key" in data:
                        creds_file = filename
                        project_id = data["project_id"]
                        break
                except Exception:
                    pass
        
        if not creds_file:
            return {
                "valid": False,
                "keys_count": 0,
                "message": "Vertex AI mode active, but no GCP Service Account JSON credential file was found in the project root directory. Please download and place your key JSON file here.",
                "breakdown": get_keys_breakdown([])
            }
            
        try:
            from google import genai as google_genai
            from google.oauth2 import service_account
            
            credentials = service_account.Credentials.from_service_account_file(
                creds_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            client = google_genai.Client(
                vertexai=True,
                project=project_id,
                location="global",
                credentials=credentials
            )
            
            # Simple synchronous test request to verify API authentication
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents="hello"
            )
            if response.text:
                return {
                    "valid": True,
                    "keys_count": 1,
                    "message": f"Vertex AI verification passed! Successfully connected to GCP Project '{project_id}' using credentials file '{creds_file}'.",
                    "breakdown": {
                        "summary": f"GCP Service Account: {creds_file}",
                        "total": 1,
                        "google_oauth": 1
                    }
                }
        except Exception as e:
            return {
                "valid": False,
                "keys_count": 1,
                "message": f"Vertex AI verification failed for Project '{project_id}' with credentials '{creds_file}': {str(e)}",
                "breakdown": get_keys_breakdown([])
            }
        return {"valid": False, "keys_count": 1, "message": "Vertex AI verification failed", "breakdown": get_keys_breakdown([])}

    if api_type == "groq":
        raw_keys = req.get("keys") or api_keys
        if isinstance(raw_keys, str):
            raw_keys = [k.strip() for k in raw_keys.split("\n") if k.strip()]
        elif isinstance(raw_keys, list):
            raw_keys = [k.strip() for k in raw_keys if isinstance(k, str) and k.strip()]
        else:
            raw_keys = []
            
        groq_keys = [k for k in raw_keys if k.startswith("gsk_")]
        if not groq_keys:
            env_k = os.environ.get("GROQ_API_KEY", "").strip()
            if env_k.startswith("gsk_"):
                groq_keys.append(env_k)
                
        if not groq_keys:
            return {
                "valid": False,
                "keys_count": 0,
                "message": "Groq mode active, but no Groq API Key (starts with 'gsk_') was found in Manage Keys pool. Please paste your key there.",
                "breakdown": get_keys_breakdown(raw_keys)
            }
            
        test_key = groq_keys[0]
        masked = f"{test_key[:6]}...{test_key[-4:]}" if len(test_key) > 10 else "***"
        try:
            import httpx
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {test_key}", "Content-Type": "application/json"},
                    json={
                        "model": "openai/gpt-oss-120b",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5
                    }
                )
            if resp.status_code == 200:
                return {
                    "valid": True,
                    "keys_count": len(groq_keys),
                    "message": f"Groq verification passed! Key {masked} active on Groq LPU cluster. Pool has {len(groq_keys)} Groq key{'s' if len(groq_keys) > 1 else ''}.",
                    "breakdown": get_keys_breakdown(raw_keys)
                }
            else:
                err_detail = ""
                try:
                    err_detail = resp.json().get("error", {}).get("message", resp.text)
                except Exception:
                    err_detail = resp.text
                return {
                    "valid": False,
                    "keys_count": len(groq_keys),
                    "message": f"Groq verification failed ({resp.status_code}): {err_detail}",
                    "breakdown": get_keys_breakdown(raw_keys)
                }
        except Exception as groq_err:
            return {
                "valid": False,
                "keys_count": len(groq_keys),
                "message": f"Groq connection error: {str(groq_err)}",
                "breakdown": get_keys_breakdown(raw_keys)
            }

    # Legacy AI Studio / Free Pool path
    test_keys = req.get("keys") or api_keys
    if isinstance(test_keys, str):
        test_keys = [k.strip() for k in test_keys.split("\n") if k.strip()]
    elif isinstance(test_keys, list):
        test_keys = [k.strip() for k in test_keys if isinstance(k, str) and k.strip()]
    else:
        test_keys = []

    if not test_keys:
        return {"valid": False, "keys_count": 0, "message": "No keys loaded in pool.", "breakdown": get_keys_breakdown([])}
    
    breakdown = get_keys_breakdown(test_keys)
    
    import google.generativeai as genai
    candidate_models = []
    sel_model = settings.get("model", "gemini-2.5-flash")
    if sel_model:
        candidate_models.append(sel_model)
    for fb in ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.5-flash"]:
        if fb not in candidate_models:
            candidate_models.append(fb)
            
    last_error = None
    keys_to_try = test_keys[:min(3, len(test_keys))]
    for test_key in keys_to_try:
        key_info = identify_key_format(test_key)
        actual_key = key_info["key"] if key_info["key"] else test_key
        try:
            genai.configure(api_key=actual_key)
            for mod_name in candidate_models:
                try:
                    model = genai.GenerativeModel(mod_name)
                    response = model.generate_content("hello")
                    if response and response.text:
                        return {
                            "valid": True,
                            "keys_count": len(test_keys),
                            "message": f"Verification passed on {key_info['badge']} ({mod_name})! Pool contains {breakdown['summary']}.",
                            "breakdown": breakdown
                        }
                except Exception as mod_err:
                    last_error = f"{key_info['badge']} with {mod_name}: {str(mod_err)}"
                    if "404" in str(mod_err) or "not found" in str(mod_err).lower():
                        continue
                    break
        except Exception as e:
            last_error = f"{key_info['badge']}: {str(e)}"
            continue

    return {
        "valid": False,
        "keys_count": len(test_keys),
        "message": f"Verification failed for all tested keys: {last_error or 'No response'}",
        "breakdown": breakdown
    }

# Settings
@app.get("/api/settings")
async def get_settings():
    breakdown = get_keys_breakdown(api_keys)
    return {"settings": settings, "keys": api_keys, "breakdown": breakdown}

@app.post("/api/settings")
async def save_settings(req: Dict[str, Any]):
    global settings
    settings.update(req)
    save_settings_to_disk()
    return {"status": "success"}

@app.post("/api/settings/keys")
async def save_keys(req: Dict[str, List[str]]):
    global api_keys
    raw_keys = req.get("keys", [])
    cleaned_keys = []
    for k in raw_keys:
        info = identify_key_format(k)
        if info["key"]:
            cleaned_keys.append(info["key"])
    api_keys = cleaned_keys
    save_settings_to_disk()
    api_key_pool.set_keys(api_keys)
    breakdown = get_keys_breakdown(api_keys)
    return {"status": "success", "keys_count": len(api_keys), "breakdown": breakdown}

# Queue Processing Tasks
async def run_pipeline_for_video(video):
    active_task_ids.add(video["id"])
    # Callback helpers
    def log_cb(msg, lvl):
        send_log(f"[{video['name']}] {msg}", lvl)

    def prog_cb(progress, status):
        video["progress"] = progress
        video["status"] = status
        coro = broadcast({
            "type": "progress_update",
            "id": video["id"],
            "progress": progress,
            "status": status
        })
        schedule_task(coro)

    dubber = Dubber(settings, api_keys, log_cb, prog_cb)
    
    # Path guard
    if not video["path"] or not os.path.exists(video["path"]):
        log_cb(f"File path is empty or does not exist: '{video['name']}'. Please re-add using '+ Add Files' to browse.", "error")
        prog_cb(100, "Error")
        active_task_ids.discard(video["id"])
        return

    try:
        video["status"] = "Processing"
        if video.get("is_redub"):
            subtitles_dir = get_subtitles_dir()
            subtitles_json_path = os.path.join(subtitles_dir, f"{video['id']}.json")
            if os.path.exists(subtitles_json_path):
                with open(subtitles_json_path, "r", encoding="utf-8") as sf:
                    segments = json.load(sf)
                await dubber.re_dub_video(video["path"], video["id"], segments)
            else:
                log_cb("Subtitle database file not found. Falling back to full processing...", "warning")
                await dubber.process_video(video["path"], video["id"])
        else:
            await dubber.process_video(video["path"], video["id"])
    except Exception as e:
        video["status"] = "Error"
        logger.exception(f"Error processing video {video['name']}")
    finally:
        active_task_ids.discard(video["id"])
        active_tasks.pop(video["id"], None)
        # Save queue update
        save_queue_to_disk()
        coro = broadcast({"type": "queue_update", "queue": queue})
        schedule_task(coro)

async def queue_worker():
    global is_processing
    send_log("Dubbing queue processing started.", "success")
    
    while is_processing:
        # Dynamically check concurrent threads setting
        concurrency = int(settings.get("concurrent_threads", 2))
        
        # Find queued items and active tasks count
        queued_items = [item for item in queue if item["status"] == "Queued"]
        active_count = len(active_task_ids)
        
        if not queued_items and active_count == 0:
            # Everything done
            break
            
        # Spawn tasks up to concurrency limit
        slots_available = concurrency - active_count
        if slots_available > 0 and queued_items:
            for i in range(min(slots_available, len(queued_items))):
                item = queued_items[i]
                item["status"] = "Processing"
                # Start task
                task = asyncio.create_task(run_pipeline_for_video(item))
                active_tasks[item["id"]] = task
                
        await asyncio.sleep(1.0)
        
    is_processing = False
    send_log("All active tasks completed. Queue idle.", "success")
    await broadcast({"type": "status_update", "is_processing": False})
    
    if settings.get("auto_shutdown_pc"):
        send_log("[AUTO SHUTDOWN] All batch tasks completed! Windows will shut down in 30 seconds... (To cancel, type 'shutdown /a' in CMD)", "warning")
        try:
            subprocess.run(["shutdown", "/s", "/t", "30", "/c", "DenZin Auto-Dubber processing completed. Shutting down PC."])
        except Exception as se:
            send_log(f"[AUTO SHUTDOWN ERROR] Failed to trigger Windows shutdown: {se}", "error")

import hashlib
import requests
import subprocess

KEYAUTH_NAME = "Dylyden's Application"
KEYAUTH_OWNERID = "gpA63Cv1Br"
KEYAUTH_SECRET = "148550e533690353df9f41e49aa9514f1f75c2fe52f1a015a957140e56b6e838"
KEYAUTH_VERSION = "1.0"
KEYAUTH_API_URL = "https://keyauth.win/api/1.2/"

def get_hwid():
    try:
        if os.name == 'nt':
            hwid = subprocess.check_output('wmic csproduct get uuid').decode().split('\n')[1].strip()
            return hwid
        else:
            return "UNKNOWN_HWID"
    except:
        return "UNKNOWN_HWID"

def verify_keyauth(license_key):
    # License protection removed - Full Unrestricted Access
    return True, "Unrestricted", 9999

@app.post("/api/activate")
async def activate_license(req: Dict[str, str] = {}):
    # License protection removed - Full Unrestricted Access
    return {"status": "success", "message": "Full Access Enabled", "days_left": 9999}

@app.post("/api/process/start")
async def start_processing():
    global is_processing, processing_task

    if is_processing:
        return {"status": "error", "message": "Already processing."}
        
    is_processing = True
    # Reset completed/error items to queued if we start fresh
    for item in queue:
        if item["status"] in ["Error", "Cancelled"]:
            item["status"] = "Queued"
            item["progress"] = 0
            
    processing_task = asyncio.create_task(queue_worker())
    await broadcast({"type": "status_update", "is_processing": True})
    return {"status": "success"}

@app.post("/api/process/stop")
async def stop_processing():
    global is_processing, processing_task
    if not is_processing:
        return {"status": "error", "message": "Not processing."}
        
    is_processing = False
    # Set all active items to Cancelled and force kill running tasks
    for item in queue:
        if item["status"] in ["Processing", "Initializing..."]:
            item["status"] = "Cancelled"
            
        if item["id"] in active_tasks:
            active_tasks[item["id"]].cancel()
            
    if processing_task:
        processing_task.cancel()
        processing_task = None
        
    await broadcast({"type": "status_update", "is_processing": False})
    await broadcast({"type": "queue_update", "queue": queue})
    send_log("Processing cancelled by user.", "warning")
    return {"status": "success"}


def range_stream(file_path: str, start: int, end: int, chunk_size: int = 65536):
    """Generator that yields chunks of a file for range requests."""
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk

@app.get("/api/video/stream")
async def stream_video(path: str, dubbed: bool = False, request: Request = None):
    target_path = path
    if dubbed:
        name_without_ext = os.path.splitext(os.path.basename(path))[0]
        if settings.get("output_type") == "custom" and settings.get("custom_output_path"):
            out_dir = settings.get("custom_output_path")
        else:
            out_dir = os.path.join(os.path.dirname(path), "Dubbed_Output")
        dubbed_path = os.path.join(out_dir, f"[Dubbed]_{name_without_ext}.mp4")
        if os.path.exists(dubbed_path):
            target_path = dubbed_path
            
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Video file not found.")
    
    file_size = os.path.getsize(target_path)
    ext = os.path.splitext(target_path)[1].lower()
    media_type = "video/mp4" if ext == ".mp4" else "video/x-matroska" if ext == ".mkv" else "video/webm" if ext == ".webm" else "video/mp4"
    
    range_header = request.headers.get("Range") if request else None
    
    if range_header:
        try:
            range_val = range_header.strip().replace("bytes=", "")
            range_start, range_end = range_val.split("-")
            range_start = int(range_start)
            range_end = int(range_end) if range_end else file_size - 1
        except Exception:
            range_start = 0
            range_end = file_size - 1
        
        range_end = min(range_end, file_size - 1)
        content_length = range_end - range_start + 1
        
        return StreamingResponse(
            range_stream(target_path, range_start, range_end),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {range_start}-{range_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": "no-cache",
            }
        )
    
    return StreamingResponse(
        range_stream(target_path, 0, file_size - 1),
        status_code=200,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "no-cache",
        }
    )

import wave

def get_wav_duration(path):
    try:
        with wave.open(path, 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception:
        return None

def sanitize_segment_timestamps(segments):
    """Sanitizes segment timestamps so adjacent segments NEVER overlap in time."""
    if not segments or len(segments) <= 1:
        return segments
        
    segments.sort(key=lambda x: float(x.get("start", 0.0)))
    
    for i in range(len(segments) - 1):
        curr_start = float(segments[i].get("start", 0.0))
        curr_end = float(segments[i].get("end", 0.0))
        next_start = float(segments[i + 1].get("start", 0.0))
        
        if curr_end >= next_start:
            max_curr_end = round(next_start - 0.05, 2)
            curr_dur = max_curr_end - curr_start
            
            if curr_dur >= 0.6:
                segments[i]["end"] = max_curr_end
            else:
                segments[i]["end"] = round(curr_start + max(0.6, curr_dur), 2)
                segments[i + 1]["start"] = round(segments[i]["end"] + 0.05, 2)
                
                next_end = float(segments[i + 1].get("end", 0.0))
                if segments[i + 1]["start"] >= next_end:
                    segments[i + 1]["end"] = round(segments[i + 1]["start"] + 1.2, 2)
                    
    return segments

def get_all_subtitles_dirs():
    dirs = [get_subtitles_dir()]
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        dirs.extend([
            os.path.join(exe_dir, "subtitles"),
            os.path.join(exe_dir, "_internal", "subtitles"),
        ])
    app_dir = os.path.dirname(os.path.abspath(__file__))
    dirs.extend([
        os.path.join(app_dir, "subtitles"),
        os.path.join(app_dir, "dist", "AutoDubber", "subtitles"),
        os.path.join(os.getcwd(), "subtitles")
    ])
    seen = set()
    res = []
    for d in dirs:
        norm = os.path.normpath(d)
        if norm not in seen:
            seen.add(norm)
            res.append(norm)
    return res

def find_subtitles_json(video_id: str):
    dirs = get_all_subtitles_dirs()
    # 1. Exact video_id match
    for d in dirs:
        cand = os.path.join(d, f"{video_id}.json")
        if os.path.exists(cand):
            return cand, d

    # 2. Check if video matches any queue item
    v_item = next((item for item in queue if item.get("id") == video_id), None)
    if v_item:
        v_path = v_item.get("path", "")
        if v_path:
            v_dir = os.path.dirname(v_path)
            for out_cand in [os.path.join(v_dir, "Dubbed_Output"), os.path.join(v_dir, "YouTube_Subtitles")]:
                if os.path.isdir(out_cand):
                    for f in os.listdir(out_cand):
                        if f.endswith(".json"):
                            return os.path.join(out_cand, f), out_cand

    # 3. Fallback: If only 1 json file in any search dir, return it
    for d in dirs:
        if os.path.isdir(d):
            jsons = [f for f in os.listdir(d) if f.endswith(".json")]
            if len(jsons) == 1:
                return os.path.join(d, jsons[0]), d

    return None, dirs[0]

def find_segment_audio_file(video_id: str, seg_id: str):
    dirs = get_all_subtitles_dirs()
    # Direct video_id subfolder
    for d in dirs:
        cand = os.path.join(d, video_id, f"seg_{seg_id}.wav")
        if os.path.exists(cand):
            return cand

    # Any subfolder containing seg_{seg_id}.wav
    for d in dirs:
        if os.path.isdir(d):
            for sub in os.listdir(d):
                sub_dir = os.path.join(d, sub)
                if os.path.isdir(sub_dir):
                    cand = os.path.join(sub_dir, f"seg_{seg_id}.wav")
                    if os.path.exists(cand):
                        return cand
    return None

@app.get("/api/subtitles/{video_id}")
async def get_subtitles(video_id: str):
    subtitles_json_path, found_dir = find_subtitles_json(video_id)
    if not subtitles_json_path or not os.path.exists(subtitles_json_path):
        return {"subtitles": []}
    try:
        with open(subtitles_json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        segments = sanitize_segment_timestamps(segments)

        for idx, seg in enumerate(segments):
            seg_id = seg.get("id") or (idx + 1)
            seg["id"] = seg_id
            wav_path = find_segment_audio_file(video_id, str(seg_id))
            if wav_path and os.path.exists(wav_path):
                dur = get_wav_duration(wav_path)
                if dur:
                    seg["audio_duration"] = dur

        return {"subtitles": segments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read subtitles: {str(e)}")

class SubtitlesPayload(BaseModel):
    subtitles: List[Dict[str, Any]]

@app.post("/api/subtitles/{video_id}")
async def save_subtitles(video_id: str, req: SubtitlesPayload):
    subtitles_dir = get_subtitles_dir()
    os.makedirs(subtitles_dir, exist_ok=True)
    subtitles_json_path, _ = find_subtitles_json(video_id)
    if not subtitles_json_path:
        subtitles_json_path = os.path.join(subtitles_dir, f"{video_id}.json")
    try:
        sanitized_subtitles = sanitize_segment_timestamps(req.subtitles)
        with open(subtitles_json_path, "w", encoding="utf-8") as f:
            json.dump(sanitized_subtitles, f, ensure_ascii=False, indent=4)
        primary_path = os.path.join(subtitles_dir, f"{video_id}.json")
        if os.path.normpath(subtitles_json_path) != os.path.normpath(primary_path):
            with open(primary_path, "w", encoding="utf-8") as f:
                json.dump(sanitized_subtitles, f, ensure_ascii=False, indent=4)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save subtitles: {str(e)}")

@app.get("/api/subtitles/{video_id}/audio/{seg_id}")
async def get_segment_audio(video_id: str, seg_id: str, request: Request = None):
    seg_audio_path = find_segment_audio_file(video_id, str(seg_id))
    if not seg_audio_path or not os.path.exists(seg_audio_path):
        raise HTTPException(status_code=404, detail="Segment audio preview not found. Please dub/re-dub this segment first.")

    file_size = os.path.getsize(seg_audio_path)
    range_header = request.headers.get("Range") if request else None

    if range_header:
        try:
            range_val = range_header.strip().replace("bytes=", "")
            range_start, range_end = range_val.split("-")
            range_start = int(range_start)
            range_end = int(range_end) if range_end else file_size - 1
        except Exception:
            range_start = 0
            range_end = file_size - 1
        range_end = min(range_end, file_size - 1)
        content_length = range_end - range_start + 1
        return StreamingResponse(
            range_stream(seg_audio_path, range_start, range_end),
            status_code=206,
            media_type="audio/wav",
            headers={
                "Content-Range": f"bytes {range_start}-{range_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            }
        )

    return StreamingResponse(
        range_stream(seg_audio_path, 0, file_size - 1),
        status_code=200,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }
    )

class SingleSegmentReDubPayload(BaseModel):
    segments: List[Dict[str, Any]]

@app.post("/api/subtitles/{video_id}/re-dub/{seg_id}")
async def re_dub_single_segment_endpoint(video_id: str, seg_id: str, req: SingleSegmentReDubPayload):
    video_item = None
    for item in queue:
        if item["id"] == video_id:
            video_item = item
            break
            
    if not video_item:
        raise HTTPException(status_code=404, detail="Video item not found in queue.")
        
    try:
        subtitles_dir = get_subtitles_dir()
        subtitles_json_path = os.path.join(subtitles_dir, f"{video_id}.json")
        with open(subtitles_json_path, "w", encoding="utf-8") as f:
            json.dump(req.segments, f, ensure_ascii=False, indent=4)
            
        def log_cb(msg, lvl):
            send_log(f"[{video_item['name']}] [Seg {seg_id}] {msg}", lvl)

        def prog_cb(progress, status):
            pass
            
        dubber = Dubber(settings, api_keys, log_cb, prog_cb)
        await dubber.re_dub_single_segment(video_item["path"], video_id, seg_id, req.segments)
        
        # Get duration of newly synthesized segment
        wav_path = os.path.join(subtitles_dir, video_id, f"seg_{seg_id}.wav")
        new_dur = get_wav_duration(wav_path)
        
        if new_dur:
            try:
                for seg in req.segments:
                    if str(seg.get("id")) == str(seg_id):
                        seg["end"] = float(seg.get("start", 0)) + new_dur
                        break
                with open(subtitles_json_path, "w", encoding="utf-8") as f:
                    json.dump(req.segments, f, ensure_ascii=False, indent=4)
            except Exception as save_json_err:
                logger.error(f"Failed to update end timestamp on disk after re-dub: {save_json_err}")
                
        return {"status": "success", "audio_duration": new_dur}
    except Exception as e:
        logger.exception(f"Error re-dubbing segment {seg_id} for video {video_item['name']}")
        raise HTTPException(status_code=500, detail=f"Segment re-dub failed: {str(e)}")

@app.post("/api/process/re-dub/{video_id}")
async def re_dub_video_endpoint(video_id: str):
    global is_processing, processing_task
    # Find item in queue
    video_item = None
    for item in queue:
        if item["id"] == video_id:
            video_item = item
            break
            
    if not video_item:
        raise HTTPException(status_code=404, detail="Video item not found in queue.")
        
    video_item["status"] = "Queued"
    video_item["progress"] = 0
    video_item["is_redub"] = True
    
    # Trigger queue worker if not already running
    if not is_processing:
        is_processing = True
        # Reset completed/error items to queued if we start fresh
        for item in queue:
            if item["status"] in ["Error", "Cancelled"]:
                item["status"] = "Queued"
                item["progress"] = 0
        processing_task = asyncio.create_task(queue_worker())
        await broadcast({"type": "status_update", "is_processing": True})
        
    await broadcast({"type": "queue_update", "queue": queue})
    send_log(f"Re-dubbing queued for: {video_item['name']}", "info")
    return {"status": "success"}

# --- Universal Dual-Channel Auto-Updater Endpoints ---
APP_VERSION = "2.1.0"

@app.get("/api/system/version")
async def get_system_version():
    edge_ver = "unknown"
    try:
        import edge_tts
        edge_ver = getattr(edge_tts, "__version__", "unknown")
    except Exception:
        pass
    return {
        "app_version": APP_VERSION,
        "edge_tts_version": edge_ver,
        "python_version": sys.version.split()[0],
        "is_frozen": getattr(sys, 'frozen', False)
    }

@app.get("/api/system/check-updates")
async def check_updates_endpoint():
    result = {
        "current_app_version": APP_VERSION,
        "current_edge_tts": "unknown",
        "edge_tts": {
            "has_update": False,
            "latest_version": "",
            "download_url": "",
            "error": None
        },
        "github": {
            "repo": settings.get("github_update_repo", ""),
            "has_update": False,
            "latest_version": "",
            "release_notes": "",
            "download_url": "",
            "error": None
        }
    }
    
    # 1. Check Edge-TTS via official PyPI JSON API
    try:
        import edge_tts
        result["current_edge_tts"] = getattr(edge_tts, "__version__", "unknown")
    except Exception:
        pass

    try:
        import urllib.request, json
        req = urllib.request.Request("https://pypi.org/pypi/edge-tts/json", headers={"User-Agent": "AutoDubber-Updater/2.1"})
        with urllib.request.urlopen(req, timeout=7) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            latest_pypi_ver = data.get("info", {}).get("version", "")
            result["edge_tts"]["latest_version"] = latest_pypi_ver
            
            # Locate .whl download URL
            urls = data.get("urls", [])
            for u in urls:
                if u.get("packagetype") == "bdist_wheel":
                    result["edge_tts"]["download_url"] = u.get("url")
                    break
            if not result["edge_tts"]["download_url"] and urls:
                result["edge_tts"]["download_url"] = urls[0].get("url")
                
            # Compare versions
            if latest_pypi_ver and result["current_edge_tts"] != "unknown":
                from packaging import version as pkg_version
                result["edge_tts"]["has_update"] = pkg_version.parse(latest_pypi_ver) > pkg_version.parse(result["current_edge_tts"])
            elif latest_pypi_ver:
                result["edge_tts"]["has_update"] = latest_pypi_ver != result["current_edge_tts"]
    except Exception as e:
        result["edge_tts"]["error"] = str(e)

    # 2. Check GitHub Releases or Tags for core app updates
    repo = settings.get("github_update_repo", "").strip()
    if repo:
        try:
            import urllib.request, urllib.error, json
            gh_url = f"https://api.github.com/repos/{repo}/releases/latest"
            req = urllib.request.Request(gh_url, headers={"User-Agent": "AutoDubber-Updater/2.1", "Accept": "application/vnd.github.v3+json"})
            has_release = False
            try:
                with urllib.request.urlopen(req, timeout=7) as resp:
                    gh_data = json.loads(resp.read().decode("utf-8"))
                    tag_name = gh_data.get("tag_name", "").lstrip("vV")
                    result["github"]["latest_version"] = tag_name
                    result["github"]["release_notes"] = gh_data.get("body", "")
                    
                    # Check assets for zip
                    assets = gh_data.get("assets", [])
                    for a in assets:
                        if a.get("name", "").endswith(".zip"):
                            result["github"]["download_url"] = a.get("browser_download_url")
                            break
                    if not result["github"]["download_url"]:
                        result["github"]["download_url"] = gh_data.get("zipball_url")
                    has_release = True
            except urllib.error.HTTPError as he:
                if he.code == 404:
                    # Fallback to /tags if formal release not yet created
                    tags_url = f"https://api.github.com/repos/{repo}/tags"
                    treq = urllib.request.Request(tags_url, headers={"User-Agent": "AutoDubber-Updater/2.1", "Accept": "application/vnd.github.v3+json"})
                    with urllib.request.urlopen(treq, timeout=7) as tresp:
                        tags_data = json.loads(tresp.read().decode("utf-8"))
                        if tags_data:
                            latest_tag = tags_data[0]
                            tag_name = latest_tag.get("name", "").lstrip("vV")
                            result["github"]["latest_version"] = tag_name
                            result["github"]["download_url"] = latest_tag.get("zipball_url")
                            has_release = True
                else:
                    raise he

            if has_release and result["github"]["latest_version"]:
                from packaging import version as pkg_version
                result["github"]["has_update"] = pkg_version.parse(result["github"]["latest_version"]) > pkg_version.parse(APP_VERSION)
        except Exception as e:
            result["github"]["error"] = str(e)
            
    return result

@app.post("/api/system/update-edge-tts")
async def update_edge_tts_endpoint():
    """Downloads the latest wheel directly from PyPI and installs it into ./plugins/ without pip."""
    try:
        import urllib.request, json, zipfile, io, shutil
        
        # 1. Fetch metadata from PyPI
        pypi_url = "https://pypi.org/pypi/edge-tts/json"
        req = urllib.request.Request(pypi_url, headers={"User-Agent": "AutoDubber-Updater/2.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        latest_ver = data.get("info", {}).get("version", "")
        wheel_url = None
        for u in data.get("urls", []):
            if u.get("packagetype") == "bdist_wheel":
                wheel_url = u.get("url")
                break
        if not wheel_url:
            raise RuntimeError("No wheel found for edge-tts on PyPI.")
            
        send_log(f"Downloading edge-tts v{latest_ver} wheel from PyPI...", "info")
        
        # 2. Download wheel
        wheel_req = urllib.request.Request(wheel_url, headers={"User-Agent": "AutoDubber-Updater/2.1"})
        with urllib.request.urlopen(wheel_req, timeout=30) as resp:
            wheel_bytes = resp.read()
            
        # 3. Target plugins directory
        app_root = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.join(app_root, "plugins")
        os.makedirs(plugins_dir, exist_ok=True)
        
        # 4. Extract edge_tts package from wheel zip
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as zf:
            for item in zf.namelist():
                if item.startswith("edge_tts/") or item.startswith("edge_tts\\"):
                    zf.extract(item, plugins_dir)
                    
        # Ensure plugins_dir is at sys.path[0]
        if plugins_dir not in sys.path:
            sys.path.insert(0, plugins_dir)
            
        # Also sync into _internal/edge_tts if running frozen so both paths are hot-patched
        if getattr(sys, 'frozen', False):
            internal_edge = os.path.join(app_root, "_internal", "edge_tts")
            if os.path.isdir(os.path.join(plugins_dir, "edge_tts")):
                try:
                    shutil.copytree(os.path.join(plugins_dir, "edge_tts"), internal_edge, dirs_exist_ok=True)
                except Exception:
                    pass

        # 5. Flush sys.modules so imported instances reload
        for mod_name in list(sys.modules.keys()):
            if mod_name == "edge_tts" or mod_name.startswith("edge_tts."):
                del sys.modules[mod_name]
                
        import edge_tts
        new_active_ver = getattr(edge_tts, "__version__", latest_ver)
        send_log(f"Edge-TTS hot-patch installed successfully: v{new_active_ver}", "success")
        return {"status": "success", "new_version": new_active_ver}
    except Exception as e:
        logger.exception("Failed to update edge-tts from PyPI")
        send_log(f"Failed to update edge-tts: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))

class ApplyUpdateRequest(BaseModel):
    download_url: Optional[str] = None

@app.post("/api/system/apply-github-update")
async def apply_github_update_endpoint(req: ApplyUpdateRequest):
    """Downloads patch zip and extracts updated python/js files into app directory."""
    try:
        import urllib.request, zipfile, io, shutil
        url = req.download_url
        if not url:
            repo = settings.get("github_update_repo", "").strip()
            if not repo:
                raise HTTPException(status_code=400, detail="No GitHub update repository configured.")
            gh_url = f"https://api.github.com/repos/{repo}/releases/latest"
            gh_req = urllib.request.Request(gh_url, headers={"User-Agent": "AutoDubber-Updater/2.1"})
            with urllib.request.urlopen(gh_req, timeout=10) as resp:
                gh_data = json.loads(resp.read().decode("utf-8"))
                for a in gh_data.get("assets", []):
                    if a.get("name", "").endswith(".zip"):
                        url = a.get("browser_download_url")
                        break
                if not url:
                    url = gh_data.get("zipball_url")
                    
        send_log("Downloading patch archive from GitHub...", "info")
        patch_req = urllib.request.Request(url, headers={"User-Agent": "AutoDubber-Updater/2.1"})
        with urllib.request.urlopen(patch_req, timeout=60) as resp:
            patch_bytes = resp.read()
            
        app_root = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        
        # Safe extraction: only extract allowed runtime files
        allowed_extensions = (".py", ".js", ".html", ".css", ".json")
        allowed_files = ("dubber.py", "server.py", "app.js", "style.css", "index.html")
        
        extracted_count = 0
        with zipfile.ZipFile(io.BytesIO(patch_bytes)) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                basename = os.path.basename(member.filename)
                if basename in allowed_files or any(member.filename.endswith(ext) for ext in allowed_extensions):
                    target_file = os.path.join(app_root, basename)
                    with zf.open(member) as src, open(target_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted_count += 1
                    
        send_log(f"Applied patch with {extracted_count} updated components! Please refresh or restart.", "success")
        return {"status": "success", "extracted_files": extracted_count}
    except Exception as e:
        logger.exception("Failed to apply GitHub patch")
        send_log(f"GitHub patch failed: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))

# Lifespan handles initializations now

if __name__ == "__main__":
    import sys
    import time
    import webbrowser
    import threading
    import urllib.request
    import hashlib
    import json
    import os
    import tkinter as tk
    from tkinter import messagebox



    # Set working directory to executable directory for full portability
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        os.chdir(exe_dir)
    except Exception:
        pass

    # Auto-open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open("http://127.0.0.1:8500")
        except Exception as e:
            print(f"Failed to auto-open browser: {e}")
            
    threading.Thread(target=open_browser, daemon=True).start()

    # Verify that ffmpeg is accessible
    setup_ffmpeg_path()
    ffmpeg_cmd = get_ffmpeg_cmd()
    try:
        subprocess.run([ffmpeg_cmd, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("[CRITICAL ERROR] ffmpeg is not installed or not found!")
        print("Please place ffmpeg.exe next to AutoDubber.exe or install ffmpeg.")
        sys.exit(1)
        
    uvicorn.run(app, host="127.0.0.1", port=8500, log_level="info")
