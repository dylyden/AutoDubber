import os
import sys
import subprocess
import uuid
import json
import re
import asyncio
import shutil
import logging
import warnings
import threading

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_base_path():
    if getattr(sys, 'frozen', False):
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

def get_ffmpeg_cmd():
    search_paths = [get_base_path()]
    if getattr(sys, 'frozen', False):
        # Also check next to the .exe itself
        search_paths.append(os.path.dirname(sys.executable))
    search_paths.append(os.getcwd())
        
    for base in search_paths:
        local_ffmpeg_0 = os.path.join(base, "ffmpeg.exe")
        if os.path.exists(local_ffmpeg_0):
            return local_ffmpeg_0
            
        local_ffmpeg_1 = os.path.join(base, "bin", "ffmpeg.exe")
        if os.path.exists(local_ffmpeg_1):
            return local_ffmpeg_1
            
        local_ffmpeg_2 = os.path.join(base, "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe")
        if os.path.exists(local_ffmpeg_2):
            return local_ffmpeg_2
            
    return "ffmpeg"

def setup_ffmpeg_path():
    search_paths = [get_base_path()]
    if getattr(sys, 'frozen', False):
        search_paths.insert(0, os.path.dirname(sys.executable))
    search_paths.append(os.getcwd())
    for base in search_paths:
        for sub in ["", "bin", os.path.join("ffmpeg-master-latest-win64-gpl", "bin")]:
            target_dir = os.path.join(base, sub) if sub else base
            target_exe = os.path.join(target_dir, "ffmpeg.exe")
            if os.path.exists(target_exe):
                curr_path = os.environ.get("PATH", "")
                if target_dir not in curr_path:
                    os.environ["PATH"] = target_dir + os.pathsep + curr_path
                return target_exe
    return "ffmpeg"

setup_ffmpeg_path()

def resolve_voice_path(p):
    if not p:
        return None
    if os.path.exists(p):
        return p
    basename = os.path.basename(p)
    search_dirs = [get_base_path(), os.getcwd()]
    if getattr(sys, 'frozen', False):
        search_dirs.append(os.path.dirname(sys.executable))
    for sd in search_dirs:
        for sub in ["", "custom_voices", "voice_samples", os.path.join("voice_samples", "teen_male"), os.path.join("voice_samples", "strong_male")]:
            candidate = os.path.join(sd, sub, basename) if sub else os.path.join(sd, basename)
            if os.path.exists(candidate):
                return candidate
    return None

# --- CUDA DLL Path Injection (must happen BEFORE importing faster_whisper/ctranslate2) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform == "win32":
    try:
        import importlib
        _nvidia_modules = ["nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_nvrtc"]
        _nvidia_dll_paths = []
        for _mod_name in _nvidia_modules:
            try:
                _mod = importlib.import_module(_mod_name)
                _mod_file = getattr(_mod, "__file__", None)
                if _mod_file:
                    _nvidia_dll_paths.append(os.path.join(os.path.dirname(_mod_file), "bin"))
            except Exception:
                pass
        for _p in _nvidia_dll_paths:
            if os.path.isdir(_p):
                os.environ["PATH"] = _p + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(_p)
    except Exception:
        pass
# --- End CUDA DLL Path Injection ---

from pydub import AudioSegment  # type: ignore
import torch  # type: ignore
from faster_whisper import WhisperModel  # type: ignore
import google.generativeai as genai  # type: ignore
import google.generativeai.client as client_mod  # type: ignore

try:
    import googleapiclient.discovery  # type: ignore
    import googleapiclient.http  # type: ignore
    import httplib2  # type: ignore

    # Monkeypatch FileServiceClient._setup_discovery_api to support new AQ. format keys.
    def patched_setup_discovery_api(self, metadata=()):  # type: ignore
        api_key = getattr(self._client_options, "api_key", None)
        if api_key is None:
            raise ValueError("Invalid operation: Uploading to the File API requires an API key. Please provide a valid API key.")
        
        uri = f"{client_mod.GENAI_API_DISCOVERY_URL}?version=v1beta"
        request = googleapiclient.http.HttpRequest(
            http=httplib2.Http(),
            postproc=lambda resp, content: (resp, content),
            uri=uri,
            headers=dict(metadata),
        )
        response, content = request.execute()
        request.http.close()

        discovery_doc = content.decode("utf-8")
        self._local.discovery_api = googleapiclient.discovery.build_from_document(
            discovery_doc, developerKey=api_key
        )

    client_mod.FileServiceClient._setup_discovery_api = patched_setup_discovery_api
except Exception:
    pass

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DubberPipeline")

# Voice map for edge-tts
# Maps language code to (Male Voice, Female Voice)
VOICES = {
    # Primary
    "km-KH": ("km-KH-PisethNeural", "km-KH-SreymomNeural"),
    
    # User requested: Thai & Myanmar
    "th-TH": ("th-TH-NiwatNeural", "th-TH-PremwadeeNeural"),
    "my-MM": ("my-MM-ThihaNeural", "my-MM-NilarNeural"),
    
    # YouTube High RPM Tier 1 (US, UK, Germany, France, Netherlands, Spain, Italy, Brazil)
    "en-US": ("en-US-GuyNeural", "en-US-JennyNeural"),
    "en-GB": ("en-GB-RyanNeural", "en-GB-SoniaNeural"),
    "de-DE": ("de-DE-FlorianMultilingualNeural", "de-DE-SeraphinaMultilingualNeural"),
    "fr-FR": ("fr-FR-HenriNeural", "fr-FR-DeniseNeural"),
    "es-ES": ("es-ES-AlvaroNeural", "es-ES-ElviraNeural"),
    "nl-NL": ("nl-NL-MaartenNeural", "nl-NL-FennaNeural"),
    "it-IT": ("it-IT-DiegoNeural", "it-IT-ElsaNeural"),
    "pt-BR": ("pt-BR-AntonioNeural", "pt-BR-FranciscaNeural"),
    
    # East Asia High RPM
    "ja-JP": ("ja-JP-KeitaNeural", "ja-JP-NanamiNeural"),
    "ko-KR": ("ko-KR-InJoonNeural", "ko-KR-SunHiNeural"),
    "zh-CN": ("zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"),
    
    # Southeast Asia
    "vi-VN": ("vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"),
    "id-ID": ("id-ID-ArdiNeural", "id-ID-GadisNeural"),
    "fil-PH": ("fil-PH-AngeloNeural", "fil-PH-BlessicaNeural")
}

LANGUAGE_PROMPTS = {
    "km-KH": "Khmer (ភាសាខ្មែរ)",
    "th-TH": "Thai (ภาษาไทย)",
    "my-MM": "Myanmar / Burmese (မြန်မာဘာသာ)",
    "en-US": "English (US)",
    "en-GB": "English (UK)",
    "de-DE": "German (Deutsch)",
    "fr-FR": "French (Français)",
    "es-ES": "Spanish (Español)",
    "ja-JP": "Japanese (日本語)",
    "ko-KR": "Korean (한국어)",
    "nl-NL": "Dutch (Nederlands)",
    "it-IT": "Italian (Italiano)",
    "pt-BR": "Portuguese (Português do Brasil)",
    "zh-CN": "Chinese Mandarin (简体中文)",
    "vi-VN": "Vietnamese (Tiếng Việt)",
    "id-ID": "Indonesian (Bahasa Indonesia)",
    "fil-PH": "Filipino / Tagalog (Wikang Filipino)"
}

def identify_key_format(key: str) -> dict:
    """Identifies the format of a Gemini API key or credential string."""
    if not isinstance(key, str):
        key = str(key) if key is not None else ""
        
    key_clean = key.strip().strip('"\'')
    if key_clean.startswith("GEMINI_API_KEY="):
        key_clean = key_clean.replace("GEMINI_API_KEY=", "").strip()

    if not key_clean:
        return {
            "raw": key,
            "key": "",
            "masked": "",
            "type": "empty",
            "label": "Empty Key",
            "badge": "EMPTY",
            "badge_class": "badge-gray",
            "description": "No key text provided.",
            "length": 0
        }

    length = len(key_clean)
    masked = key_clean[:6] + "..." + key_clean[-4:] if length > 10 else "***"

    if key_clean.startswith("AQ.") or key_clean.startswith("AQ"):
        return {
            "raw": key,
            "key": key_clean,
            "masked": masked,
            "type": "gemini_new_aq",
            "label": "New Gemini API Key (AQ....)",
            "badge": "NEW AQ FORMAT",
            "badge_class": "badge-cyan",
            "description": "Modern high-security Gemini Developer / enterprise API key format.",
            "length": length
        }
    elif key_clean.startswith("AIza"):
        return {
            "raw": key,
            "key": key_clean,
            "masked": masked,
            "type": "gemini_legacy_aiza",
            "label": "Legacy Google AI Studio Key (AIza...)",
            "badge": "AIZA LEGACY",
            "badge_class": "badge-gold",
            "description": "Standard Google AI Studio developer API key format.",
            "length": length
        }
    elif key_clean.startswith("ya29."):
        return {
            "raw": key,
            "key": key_clean,
            "masked": masked,
            "type": "google_oauth",
            "label": "Google OAuth Access Token (ya29...)",
            "badge": "OAUTH TOKEN",
            "badge_class": "badge-purple",
            "description": "Short-lived Google Cloud / Vertex AI OAuth access token.",
            "length": length
        }
    elif key_clean.startswith("gsk_"):
        return {
            "raw": key,
            "key": key_clean,
            "masked": masked,
            "type": "groq_api_key",
            "label": "Groq Cloud API Key (gsk_...)",
            "badge": "GROQ CLOUD",
            "badge_class": "badge-orange",
            "description": "Ultra-fast Groq LPU API key for Llama 3.3 70B & 3.1 8B.",
            "length": length
        }
    else:
        return {
            "raw": key,
            "key": key_clean,
            "masked": masked,
            "type": "unknown",
            "label": "Custom / Unknown Format",
            "badge": "CUSTOM KEY",
            "badge_class": "badge-blue",
            "description": "Custom key format or proxy authentication credential.",
            "length": length
        }

def get_keys_breakdown(keys_list: list) -> dict:
    """Returns structured format breakdown statistics for a list of API keys."""
    details = [identify_key_format(k) for k in keys_list if k and k.strip()]
    counts = {
        "gemini_new_aq": 0,
        "gemini_legacy_aiza": 0,
        "google_oauth": 0,
        "groq_api_key": 0,
        "unknown": 0,
        "total": len(details)
    }
    for item in details:
        t = item.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
        
    summary_parts = []
    if counts["groq_api_key"] > 0:
        summary_parts.append(f"{counts['groq_api_key']} Groq Key{'s' if counts['groq_api_key'] > 1 else ''}")
    if counts["gemini_new_aq"] > 0:
        summary_parts.append(f"{counts['gemini_new_aq']} New AQ Key{'s' if counts['gemini_new_aq'] > 1 else ''}")
    if counts["gemini_legacy_aiza"] > 0:
        summary_parts.append(f"{counts['gemini_legacy_aiza']} Legacy AIza Key{'s' if counts['gemini_legacy_aiza'] > 1 else ''}")
    if counts["google_oauth"] > 0:
        summary_parts.append(f"{counts['google_oauth']} OAuth Token{'s' if counts['google_oauth'] > 1 else ''}")
    if counts["unknown"] > 0:
        summary_parts.append(f"{counts['unknown']} Custom Key{'s' if counts['unknown'] > 1 else ''}")
        
    summary_str = ", ".join(summary_parts) if summary_parts else "No valid keys"

    return {
        "total_count": len(details),
        "counts": counts,
        "summary": summary_str,
        "details": details
    }

def extract_json_payload(text: str) -> str:
    """Extracts the first valid JSON array or object substring from a text string."""
    if not text:
        return ""
    
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    if first_brace == -1 and first_bracket == -1:
        return text
    
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        last_brace = text.rfind('}')
        if last_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace+1]
    elif first_bracket != -1:
        last_bracket = text.rfind(']')
        if last_bracket != -1 and last_bracket > first_bracket:
            return text[first_bracket:last_bracket+1]
            
        # Truncation Recovery: If array was cut off mid-output, slice up to the last complete object '}' and close the array ']'
        last_brace = text.rfind('}')
        if last_brace != -1 and last_brace > first_bracket:
            import re
            repaired = text[first_bracket:last_brace+1]
            repaired = re.sub(r',\s*$', '', repaired)
            return repaired + "\n]"
            
    return text

import hashlib
import time
from dataclasses import dataclass, field

@dataclass
class KeyHealth:
    raw_key: str
    clean_key: str
    masked_key: str
    key_type: str
    cooldown_until: float = 0.0
    in_use: int = 0
    uploaded_files: dict = field(default_factory=dict)  # file_hash -> uploaded_file_obj

    def is_available(self) -> bool:
        return time.monotonic() >= self.cooldown_until


class ApiKeyPool:
    def __init__(self):
        self.keys = []
        self.leased_keys = {}  # key -> count of tasks currently using it
        self.health_map = {}   # clean_key -> KeyHealth
        self.lock = asyncio.Lock()

    def set_keys(self, keys):
        self.keys = list(keys) if keys else []
        active_clean = set()
        for k in self.keys:
            info = identify_key_format(k)
            clean = info["key"] if info["key"] else str(k).strip()
            if clean:
                active_clean.add(clean)
                if clean not in self.health_map:
                    self.health_map[clean] = KeyHealth(
                        raw_key=k,
                        clean_key=clean,
                        masked_key=info.get("masked", "***"),
                        key_type=info.get("type", "unknown")
                    )
                else:
                    self.health_map[clean].raw_key = k

        for key in list(self.leased_keys.keys()):
            if key not in self.keys:
                del self.leased_keys[key]
        for clean_k in list(self.health_map.keys()):
            if clean_k not in active_clean:
                del self.health_map[clean_k]

    @staticmethod
    def compute_file_hash(path: str) -> str:
        if not path or not os.path.exists(path):
            return ""
        st = os.stat(path)
        raw = f"{os.path.abspath(path)}:{st.st_size}:{st.st_mtime_ns}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    async def acquire_router_key(self, preferred_file_hash: str = "", failed_keys=None) -> KeyHealth:
        async with self.lock:
            if not self.keys:
                raise ValueError("No Gemini API keys provided. Please load keys in the API Manager.")
            if not self.health_map:
                self.set_keys(self.keys)

            gemini_candidates = [
                kh for kh in self.health_map.values()
                if kh.key_type in ("gemini_new_aq", "gemini_legacy_aiza", "unknown")
            ]
            if not gemini_candidates:
                gemini_candidates = list(self.health_map.values())

            now = time.monotonic()
            available = [kh for kh in gemini_candidates if kh.cooldown_until <= now]
            if not available:
                soonest = min(gemini_candidates, key=lambda x: x.cooldown_until)
                wait_s = max(0.5, min(15.0, soonest.cooldown_until - now))
                await asyncio.sleep(wait_s)
                available = [soonest]

            def router_priority(kh: KeyHealth):
                is_failed = 1 if (failed_keys and (kh.raw_key in failed_keys or kh.clean_key in failed_keys)) else 0
                has_cached = 0 if (preferred_file_hash and preferred_file_hash in kh.uploaded_files) else 1
                return (is_failed, has_cached, kh.in_use)

            available.sort(key=router_priority)
            chosen = available[0]
            chosen.in_use += 1
            self.leased_keys[chosen.raw_key] = chosen.in_use
            return chosen

    async def release_router_key(self, kh: KeyHealth, rate_limited: bool = False, cooldown_seconds: float = 60.0):
        async with self.lock:
            kh.in_use = max(0, kh.in_use - 1)
            if kh.raw_key in self.leased_keys:
                self.leased_keys[kh.raw_key] = kh.in_use
            if rate_limited:
                kh.cooldown_until = time.monotonic() + cooldown_seconds

    async def acquire_key(self, failed_keys=None):
        kh = await self.acquire_router_key(failed_keys=failed_keys)
        return kh.raw_key

    async def release_key(self, key, rate_limited: bool = False):
        async with self.lock:
            info = identify_key_format(key)
            clean = info["key"] if info["key"] else str(key).strip()
            if clean in self.health_map:
                kh = self.health_map[clean]
                kh.in_use = max(0, kh.in_use - 1)
                if rate_limited:
                    kh.cooldown_until = time.monotonic() + 60.0
            if key in self.leased_keys:
                self.leased_keys[key] = max(0, self.leased_keys[key] - 1)

api_key_pool = ApiKeyPool()

locked_speaker_roles = {}
def clear_locked_speaker_roles():
    locked_speaker_roles.clear()

whisper_gpu_lock = None
gemini_api_lock = None
global_whisper_model = None
def get_whisper_lock():
    global whisper_gpu_lock
    if whisper_gpu_lock is None:
        whisper_gpu_lock = asyncio.Semaphore(1)
    return whisper_gpu_lock

def get_gemini_lock():
    global gemini_api_lock
    if gemini_api_lock is None:
        gemini_api_lock = asyncio.Semaphore(1)
    return gemini_api_lock

voxcpm_gpu_lock = None
global_voxcpm_model = None
global_voxcpm_4gb_model = None
global_openvoice_converter = None

def get_voxcpm_lock():
    global voxcpm_gpu_lock
    if voxcpm_gpu_lock is None:
        voxcpm_gpu_lock = asyncio.Semaphore(1)
    return voxcpm_gpu_lock

def get_local_openvoice_converter(device: str = "auto"):
    global global_openvoice_converter
    if global_openvoice_converter is None:
        from voice_converter import ToneColorConverterEngine
        base_dir = get_base_path()
        model_dir = os.path.join(base_dir, "models", "OpenVoiceV2")
        if getattr(sys, "frozen", False):
            exe_model_dir = os.path.join(os.path.dirname(sys.executable), "models", "OpenVoiceV2")
            if os.path.exists(exe_model_dir):
                model_dir = exe_model_dir
        run_device = "cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu"
        global_openvoice_converter = ToneColorConverterEngine(model_dir=model_dir, device=run_device)
    return global_openvoice_converter

def get_local_voxcpm_4gb_model(device: str = "auto"):
    global global_voxcpm_4gb_model, global_whisper_model
    # Park Faster-Whisper into System RAM (0.7s PCIe copy, zero disk reload) before VoxCPM 4GB runs
    if global_whisper_model is not None:
        try:
            if hasattr(global_whisper_model, "model") and hasattr(global_whisper_model.model, "unload_model"):
                global_whisper_model.model.unload_model(to_cpu=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if global_voxcpm_4gb_model is None:
        import os, glob
        import torch  # type: ignore
        from voice_converter import VoxCPM4GBEngine

        base_dir = get_base_path()
        project_models = [
            os.path.join(base_dir, "models", "VoxCPM2"),
            os.path.join(os.path.dirname(sys.executable), "models", "VoxCPM2"),
        ]
        local_path = None
        for p in project_models:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "model.safetensors")):
                local_path = p
                break
        if not local_path:
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
            snapshots = glob.glob(os.path.join(hf_home, "models--openbmb--VoxCPM2", "snapshots", "*"))
            for snap in snapshots:
                if os.path.isdir(snap) and os.path.exists(os.path.join(snap, "model.safetensors")):
                    local_path = snap
                    break
        model_path = local_path if local_path else os.path.join(base_dir, "models", "VoxCPM2")
        run_device = "cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu"
        global_voxcpm_4gb_model = VoxCPM4GBEngine(model_dir=model_path, device=run_device)
    else:
        global_voxcpm_4gb_model.reload_to_gpu()
    return global_voxcpm_4gb_model

def get_local_voxcpm_model(device: str = "auto"):
    global global_voxcpm_model
    if global_voxcpm_model is None:
        import os, glob
        import torch  # type: ignore
        from voxcpm import VoxCPM  # type: ignore

        # 1. Check local project folder: models/VoxCPM2
        base_dir = get_base_path()
        project_models = [
            os.path.join(base_dir, "models", "VoxCPM2"),
            os.path.join(os.path.dirname(sys.executable), "models", "VoxCPM2")
        ]
        local_path = None
        for p in project_models:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "model.safetensors")):
                local_path = p
                break

        # 2. Fall back to user HuggingFace hub cache
        if not local_path:
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
            snapshots = glob.glob(os.path.join(hf_home, "models--openbmb--VoxCPM2", "snapshots", "*"))
            for snap in snapshots:
                if os.path.isdir(snap) and os.path.exists(os.path.join(snap, "model.safetensors")):
                    local_path = snap
                    break

        model_path = local_path if local_path else "openbmb/VoxCPM2"

        run_device = "cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu"
        try:
            global_voxcpm_model = VoxCPM(
                voxcpm_model_path=model_path,
                enable_denoiser=False,
                optimize=False,
                device=run_device
            )
        except Exception as e:
            if run_device == "cuda":
                logging.getLogger("AutoDubber").warning(f"Failed loading VoxCPM on CUDA ({e}), falling back to CPU...")
                global_voxcpm_model = VoxCPM(
                    voxcpm_model_path=model_path,
                    enable_denoiser=False,
                    optimize=False,
                    device="cpu"
                )
            else:
                raise e
    return global_voxcpm_model


def fast_overlay_segments(base_duration_ms, placements, frame_rate=44100, channels=2):
    """Overlays multiple (pos_ms, AudioSegment) pairs into a single timeline using one pre-allocated NumPy buffer."""
    import numpy as np
    total_frames = max(1, int(round(base_duration_ms * frame_rate / 1000.0)))
    acc = np.zeros((total_frames, channels), dtype=np.int32)
    for pos_ms, seg in placements:
        if seg is None or len(seg) == 0:
            continue
        s = seg
        if s.frame_rate != frame_rate:
            s = s.set_frame_rate(frame_rate)
        if s.channels != channels:
            s = s.set_channels(channels)
        if s.sample_width != 2:
            s = s.set_sample_width(2)
        arr = np.frombuffer(s.raw_data, dtype=np.int16)
        n_frames = len(arr) // channels
        if n_frames <= 0:
            continue
        arr = arr[:n_frames * channels].reshape((n_frames, channels))
        start_idx = max(0, int(round(pos_ms * frame_rate / 1000.0)))
        if start_idx >= total_frames:
            continue
        end_idx = min(total_frames, start_idx + n_frames)
        acc[start_idx:end_idx] += arr[:end_idx - start_idx]
    np.clip(acc, -32768, 32767, out=acc)
    out_bytes = acc.astype(np.int16).tobytes()
    return AudioSegment(
        data=out_bytes,
        sample_width=2,
        frame_rate=frame_rate,
        channels=channels,
    )


def fast_duck_audio_segment(bg_seg, merged_intervals, duck_db):
    """Applies dynamic volume ducking with smooth linear crossfade ramps in-place on a single NumPy array."""
    import numpy as np
    if not merged_intervals or bg_seg is None or len(bg_seg) == 0:
        return bg_seg
    s = bg_seg
    if s.sample_width != 2:
        s = s.set_sample_width(2)
    frame_rate = s.frame_rate
    channels = s.channels
    arr = np.frombuffer(s.raw_data, dtype=np.int16).copy()
    n_frames = len(arr) // channels
    if n_frames <= 0:
        return bg_seg
    arr_2d = arr[:n_frames * channels].reshape((n_frames, channels)).astype(np.float32)
    gain = float(10.0 ** (-float(duck_db) / 20.0))

    for s_ms, e_ms in merged_intervals:
        duration_ms = e_ms - s_ms
        if duration_ms < 100:
            continue
        s_idx = max(0, min(n_frames, int(round(s_ms * frame_rate / 1000.0))))
        e_idx = max(0, min(n_frames, int(round(e_ms * frame_rate / 1000.0))))
        if e_idx <= s_idx:
            continue
        fade_ms = min(200, int(duration_ms / 2))
        fade_frames = int(round(fade_ms * frame_rate / 1000.0))
        seg_frames = e_idx - s_idx
        if fade_ms > 10 and fade_frames * 2 <= seg_frames and fade_frames > 0:
            ramp_down = np.linspace(1.0, gain, fade_frames, dtype=np.float32)[:, None]
            ramp_up = np.linspace(gain, 1.0, fade_frames, dtype=np.float32)[:, None]
            arr_2d[s_idx:s_idx + fade_frames] *= ramp_down
            arr_2d[s_idx + fade_frames:e_idx - fade_frames] *= gain
            arr_2d[e_idx - fade_frames:e_idx] *= ramp_up
        else:
            arr_2d[s_idx:e_idx] *= gain

    np.clip(arr_2d, -32768.0, 32767.0, out=arr_2d)
    out_bytes = arr_2d.astype(np.int16).tobytes()
    return AudioSegment(
        data=out_bytes,
        sample_width=2,
        frame_rate=frame_rate,
        channels=channels,
    )


class Dubber:
    def __init__(self, settings, api_keys, log_callback=None, progress_callback=None):
        self.settings = settings
        self.api_keys = api_keys
        self.log_callback = log_callback or (lambda msg, lvl: print(f"[{lvl}] {msg}"))
        self.progress_callback = progress_callback or (lambda prog, status: None)
        self.current_key_index = 0
        
        # Populate global api_key_pool if it is empty (fallback for standalone calls)
        if api_keys and not api_key_pool.keys:
            api_key_pool.set_keys(api_keys)

        self.temp_dir = None
        self._hf_client = None
        self._hf_space = None
        self._audio_handles = {}

    def log(self, message, level="info"):
        self.log_callback(message, level)

    def update_progress(self, progress, status):
        self.progress_callback(progress, status)

    def get_api_key(self):
        if not self.api_keys:
            raise ValueError("No Gemini API keys provided. Please load keys in the API Manager.")
        raw_key = self.api_keys[self.current_key_index]
        key_info = identify_key_format(raw_key)
        return key_info["key"] if key_info["key"] else raw_key.strip()

    async def rotate_api_key(self):
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.log(f"Rotating to API key #{self.current_key_index + 1}...", "warning")
        else:
            self.log("Only 1 API key available. Retrying with same key in 10s...", "warning")
            await asyncio.sleep(10)

    def get_clean_env(self):
        env = os.environ.copy()
        # Remove PyInstaller variables that cause subprocesses to crash with 0xc0000142
        env.pop('_MEIPASS2', None)
        env.pop('LD_LIBRARY_PATH', None)
        return env

    def run_cmd(self, cmd, desc="Command"):
        self.log(f"Running: {' '.join(cmd)[:150]}...", "muted")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore', env=self.get_clean_env())
        if result.returncode != 0:
            logger.error(f"{desc} failed: {result.stderr}")
            raise RuntimeError(f"{desc} failed: {result.stderr}")
        return result.stdout

    async def async_run_cmd(self, cmd, desc="Command"):
        self.log(f"Running: {' '.join(cmd)[:150]}...", "muted")
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=self.get_clean_env()
        )
        try:
            stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            process.terminate()
            raise

        if process.returncode != 0:
            stderr_dec = stderr.decode('utf-8', errors='ignore') if stderr else ''
            logger.error(f"{desc} failed: {stderr_dec}")
            raise RuntimeError(f"{desc} failed: {stderr_dec}")
            
        return stdout.decode('utf-8', errors='ignore') if stdout else ''

    async def run_demucs(self, audio_path, output_dir):
        """Runs local Demucs to isolate vocals and background stems with low-PC optimizations."""
        mixer = self.settings.get("mixer", {"drums": True, "bass": True, "melody": True})
        def get_stem_vol(stem):
            try:
                if not mixer.get(stem, True):
                    return 0.0
                return float(mixer.get(f"{stem}_vol", 100))
            except (ValueError, TypeError):
                return 100.0 if mixer.get(stem, True) else 0.0

        d_vol = get_stem_vol("drums")
        b_vol = get_stem_vol("bass")
        m_vol = get_stem_vol("melody")
        # When all 3 stems are enabled at default 100% (or all muted), drums + bass + melody == no_vocals.wav.
        # Only pay for 4-stem disk I/O if the user selectively muted one stem or set custom unequal stem volumes.
        all_default_100 = (abs(d_vol - 100.0) < 1.0 and abs(b_vol - 100.0) < 1.0 and abs(m_vol - 100.0) < 1.0)
        all_muted = (d_vol <= 0.0 and b_vol <= 0.0 and m_vol <= 0.0)
        need_individual_stems = not (all_default_100 or all_muted)
        mode_label = "Full 4-Stem" if need_individual_stems else "High-Speed 2-Stem"
        self.log(f"Running Demucs vocal separation ({mode_label}, shifts=0, overlap=0.1 for max speed)...", "info")
        
        # Demucs python module invocation
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "-j", "2",
            "--segment", "7",
            "--shifts", "0",
            "--overlap", "0.1",
            "--out", output_dir
        ]
        
        if not need_individual_stems:
            cmd.extend(["--two-stems", "vocals"])
            
        cmd.append(audio_path)
        
        # If GPU mode is cpu, force cpu.
        if self.settings.get("gpu_mode") == "cpu":
            cmd.append("--cpu")

        # Run as a subprocess async to prevent blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.run_cmd, cmd, "Demucs Separation")

    async def _transcribe_reference_locally(self, audio_path):
        """Transcribes reference audio locally using global Whisper model to guide VoxCPM."""
        if not audio_path or not os.path.exists(audio_path):
            return ""
            
        txt_path = os.path.splitext(audio_path)[0] + ".txt"
        
        # Check cache
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    cached_text = f.read().strip()
                if cached_text:
                    self.log(f"Using cached transcript for {os.path.basename(audio_path)}", "info")
                    return cached_text
            except Exception as e:
                self.log(f"Error reading transcript cache: {e}", "warning")
                
        self.log(f"Transcribing reference audio: {os.path.basename(audio_path)} for Ultimate Cloning...", "info")
        try:
            loop = asyncio.get_event_loop()
            
            def run_whisper():
                global global_whisper_model
                if global_whisper_model is None:
                    model_base = get_base_path()
                    model_path = os.path.join(model_base, "models", "faster-whisper-large-v3-turbo")
                    global_whisper_model = WhisperModel(model_path, device="cpu", compute_type="int8")
                segs, _ = global_whisper_model.transcribe(audio_path, language="en")
                return " ".join([s.text for s in segs]).strip()
                    
            transcript = await loop.run_in_executor(None, run_whisper)
            
            if transcript:
                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(transcript)
                    self.log(f"Successfully transcribed reference audio and cached to {txt_path}.", "success")
                except Exception as e:
                    self.log(f"Failed to cache transcript: {e}", "warning")
                    
            return transcript
        except Exception as e:
            self.log(f"Local transcription failed: {e}", "error")
            return ""

    async def synthesize_tts(self, text, voice, output_path, speed_ratio=1.0, is_male=False, is_narrator=False, emotion="dramatic", retries=5, age_group="adult"):
        """Synthesizes text to speech using selected TTS engine."""
        tts_engine = self.settings.get("tts_engine", "edge-tts")
        if tts_engine == "edge-tts":
            target_lang = self.settings.get("target_lang", "km-KH")
            edge_voice = "en-US-GuyNeural" if is_male else "en-US-JennyNeural"
            if target_lang in VOICES:
                edge_voice = VOICES[target_lang][0] if is_male else VOICES[target_lang][1]
            
            rate_str = "+0%"
            if speed_ratio != 1.0:
                rate_diff = int((speed_ratio - 1.0) * 100)
                rate_str = f"+{rate_diff}%" if rate_diff >= 0 else f"{rate_diff}%"

            clean_text = text.strip() if text else ""
            clean_text = re.sub(r"[\*\_\#\`]", "", clean_text).strip()
            clean_text = re.sub(r"\s+", " ", clean_text)

            if not clean_text or clean_text in ["...", "…", "-", ""]:
                AudioSegment.silent(duration=300).export(output_path, format="mp3")
                return

            import edge_tts  # type: ignore
            max_edge_retries = 5
            for edge_attempt in range(1, max_edge_retries + 1):
                try:
                    communicate = edge_tts.Communicate(clean_text, edge_voice, rate=rate_str)
                    await communicate.save(output_path)
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                        self.log(f"Synthesized segment via native edge-tts ({edge_voice})", "info")
                        return
                    else:
                        raise RuntimeError("Edge-TTS produced an empty output file.")
                except Exception as e:
                    if edge_attempt < max_edge_retries:
                        backoff = 1.0 * (1.5 ** (edge_attempt - 1))
                        self.log(f"Edge-TTS attempt {edge_attempt} failed ({e}). Retrying in {backoff:.1f}s...", "warning")
                        await asyncio.sleep(backoff)
                        if edge_attempt >= 3 and target_lang in VOICES:
                            edge_voice = VOICES[target_lang][1] if is_male else VOICES[target_lang][0]
                            rate_str = "+0%"
                    else:
                        self.log(f"Native edge-tts failed after {max_edge_retries} attempts ({e}). Writing silent audio placeholder.", "error")
                        AudioSegment.silent(duration=500).export(output_path, format="mp3")
                        return
            return

        # Custom reference voice cloning audio
        custom_ref = None
        if age_group == "young" or "child" in age_group or "kid" in age_group:
            custom_ref = self.settings.get("custom_young_male_voice_path") if is_male else self.settings.get("custom_young_female_voice_path")
        elif age_group == "old" or "elder" in age_group or "senior" in age_group:
            custom_ref = self.settings.get("custom_old_male_voice_path") if is_male else self.settings.get("custom_old_female_voice_path")
            
        custom_ref = resolve_voice_path(custom_ref)
        if not custom_ref:
            if is_narrator:
                custom_ref = self.settings.get("custom_narrator_male_voice_path") if is_male else self.settings.get("custom_narrator_female_voice_path")
            else:
                custom_ref = self.settings.get("custom_male_voice_path") if is_male else self.settings.get("custom_female_voice_path")
            custom_ref = resolve_voice_path(custom_ref)

        # Fallback to default custom_voices/ sample if user did not explicitly browse one
        if not custom_ref:
            default_sample_name = "Male_1.wav" if is_male else "Female_1.wav"
            if age_group == "young" or "child" in str(age_group):
                default_sample_name = "ក្មេងប្រុស.wav" if is_male else "ក្មេងស្រី.mp3"
            elif age_group == "old" or "elder" in str(age_group):
                default_sample_name = "ចាស់ប្រុស.wav" if is_male else "ចាស់ស្រី.wav"
            custom_ref = resolve_voice_path(default_sample_name)

        # TEST 2: Hybrid Edge-TTS (Khmer Reader) + OpenVoiceV2 Tone-Color Converter (131 MB VRAM)
        if tts_engine == "edge-openvoice":
            target_lang = self.settings.get("target_lang", "km-KH")
            edge_voice = "en-US-GuyNeural" if is_male else "en-US-JennyNeural"
            if target_lang in VOICES:
                edge_voice = VOICES[target_lang][0] if is_male else VOICES[target_lang][1]

            rate_str = "+0%"
            if speed_ratio != 1.0:
                rate_diff = int((speed_ratio - 1.0) * 100)
                rate_str = f"+{rate_diff}%" if rate_diff >= 0 else f"{rate_diff}%"

            clean_text = re.sub(r"\[.*?\]", "", text or "")
            clean_text = re.sub(r"\(.*?\)", "", clean_text)
            clean_text = re.sub(r"[\*\_\#\`]", "", clean_text).strip()
            clean_text = re.sub(r"\s+", " ", clean_text)

            if not clean_text or clean_text in ["...", "…", "-", ""]:
                AudioSegment.silent(duration=300).export(output_path, format="mp3")
                return

            import edge_tts  # type: ignore
            import soundfile as sf  # type: ignore
            temp_edge_mp3 = output_path + f".edge_base_{uuid.uuid4().hex[:6]}.mp3"
            temp_conv_wav = output_path + f".ov2_conv_{uuid.uuid4().hex[:6]}.wav"
            try:
                for edge_attempt in range(1, 5):
                    try:
                        communicate = edge_tts.Communicate(clean_text, edge_voice, rate=rate_str)
                        await communicate.save(temp_edge_mp3)
                        if os.path.exists(temp_edge_mp3) and os.path.getsize(temp_edge_mp3) > 100:
                            break
                    except Exception as e:
                        if edge_attempt < 4:
                            await asyncio.sleep(1.0 * edge_attempt)
                        else:
                            raise e

                if not custom_ref or not os.path.exists(custom_ref):
                    shutil.copy(temp_edge_mp3, output_path)
                    self.log(f"Synthesized via Edge-TTS ({edge_voice}) [No reference .wav found for conversion]", "info")
                    return

                role_type = "Narrator" if is_narrator else "Actor"
                actor_label = f"Male {role_type}" if is_male else f"Female {role_type}"
                self.log(
                    f"Converting Edge-TTS ({edge_voice}) -> OpenVoiceV2 Clone [{actor_label}: {os.path.basename(custom_ref)}]...",
                    "info",
                )

                loop = asyncio.get_event_loop()
                gpu_setting = self.settings.get("gpu_mode", "auto")
                device_target = "cpu" if gpu_setting == "cpu" else "cuda"

                def run_ov2_conversion():
                    converter = get_local_openvoice_converter(device=device_target)
                    wav_np, sr = converter.convert_file(temp_edge_mp3, custom_ref, tau=0.3)
                    if output_path.lower().endswith(".wav"):
                        sf.write(output_path, wav_np, sr)
                    else:
                        sf.write(temp_conv_wav, wav_np, sr)
                        seg = AudioSegment.from_wav(temp_conv_wav)
                        seg.export(output_path, format="mp3", bitrate="192k")
                    return output_path

                async with get_voxcpm_lock():
                    await loop.run_in_executor(None, run_ov2_conversion)
                return
            except Exception as ov_err:
                self.log(f"OpenVoiceV2 conversion fallback to base Edge-TTS ({ov_err})", "warning")
                if os.path.exists(temp_edge_mp3) and os.path.getsize(temp_edge_mp3) > 100:
                    shutil.copy(temp_edge_mp3, output_path)
                else:
                    AudioSegment.silent(duration=500).export(output_path, format="mp3")
                return
            finally:
                for tmp_f in (temp_edge_mp3, temp_conv_wav):
                    try:
                        if os.path.exists(tmp_f):
                            os.remove(tmp_f)
                    except Exception:
                        pass

        # Ultimate clone mode logic (transcript-guided cloning)
        use_prompt = False
        prompt_txt = ""
        if self.settings.get("voxcpm_ultimate_clone", False) and custom_ref:
            prompt_txt = await self._transcribe_reference_locally(custom_ref)
            if prompt_txt:
                use_prompt = True

        attempt = 1
        while True:
            try:
                role_type = "Narrator" if is_narrator else "Actor"
                actor_label = f"Male {role_type}" if is_male else f"Female {role_type}"
                clone_str = f" [Voice Clone: {os.path.basename(custom_ref)}]" if custom_ref else " [Default Voice]"
                if use_prompt:
                    clone_str += " (Ultimate Clone Mode)"
                engine_label = "VoxCPM2 4GB INT8 Engine"
                self.log(f"Synthesizing segment via {engine_label} ({actor_label}{clone_str}) [{emotion}] (Attempt {attempt})...", "info")
                loop = asyncio.get_event_loop()
                
                def run_local_synthesis():
                    import re, os
                    import soundfile as sf  # type: ignore
                    from pydub import AudioSegment  # type: ignore

                    # Clean unintentional parentheses but keep valid non-verbal tags like [laughing], [sigh], [Uhm], etc.
                    text_no_parens = re.sub(r"[\(\)]", "", text)

                    def clean_brackets(match):
                        content = match.group(1)
                        if re.match(r"^[a-zA-Z\-]+$", content):
                            return f"[{content}]"
                        else:
                            return content

                    speech_clean = re.sub(r"\[([^\]]+)\]", clean_brackets, text_no_parens).strip()

                    emo = emotion.lower().strip()
                    emotion_prompt_map = {
                        "angry": "intense, aggressive, angry movie actor voice, high emotional power, furious delivery",
                        "furious": "furious, roaring, intense angry movie actor voice, powerful emotional delivery",
                        "dramatic": "highly dramatic, cinematic, intense emotion, expressive movie dubbing voice",
                        "sad": "sorrowful, emotional, sad, tearful, soft melancholic movie voice",
                        "crying": "tearful, crying, deeply emotional, sobbing melancholic voice",
                        "happy": "joyful, cheerful, energetic, vibrant, happy movie actor voice",
                        "excited": "excited, enthusiastic, high energy, dynamic expressiveness",
                        "fearful": "fearful, anxious, trembling, scared movie dubbing voice",
                        "whisper": "whispering, soft, quiet, secret intimate voice",
                        "romantic": "warm, romantic, affectionate, sweet gentle voice",
                        "calm": "calm, relaxed, clear articulate voice"
                    }
                    emo_desc = emotion_prompt_map.get(emo, f"{emo} mood, expressive cinematic voice")

                    if is_male:
                        base_prompt = "consistent friendly male narrator, warm articulate storytelling voice" if is_narrator else "consistent male lead movie actor, clear cinematic voice"
                    else:
                        base_prompt = "consistent female narrator, articulate gentle storytelling voice" if is_narrator else "consistent female lead movie actress, clear cinematic voice"

                    ctrl_prompt = f"{base_prompt}, {emo_desc}"
                    cfg_val = float(self.settings.get("voxcpm_cfg", 2.0))

                    target_text = f"({ctrl_prompt}){speech_clean}" if (ctrl_prompt and not use_prompt) else speech_clean

                    gpu_setting = self.settings.get("gpu_mode", "auto")
                    device_target = "cpu" if gpu_setting == "cpu" else "cuda"
                    local_model = get_local_voxcpm_4gb_model(device=device_target)

                    gen_kwargs = {
                        "text": target_text,
                        "cfg_value": cfg_val,
                        "inference_timesteps": 8
                    }
                    if custom_ref and os.path.exists(custom_ref):
                        if use_prompt and prompt_txt:
                            gen_kwargs["prompt_wav_path"] = custom_ref
                            gen_kwargs["prompt_text"] = prompt_txt
                            gen_kwargs["reference_wav_path"] = custom_ref
                        else:
                            gen_kwargs["reference_wav_path"] = custom_ref

                    wav = local_model.generate(**gen_kwargs)

                    if output_path.lower().endswith(".wav"):
                        sf.write(output_path, wav, 48000)
                    else:
                        temp_wav = output_path + f".tmp_{uuid.uuid4().hex[:6]}.wav"
                        sf.write(temp_wav, wav, 48000)
                        seg = AudioSegment.from_wav(temp_wav)
                        seg.export(output_path, format="mp3", bitrate="192k")
                        try:
                            if os.path.exists(temp_wav):
                                os.remove(temp_wav)
                        except Exception:
                            pass
                    return output_path

                async with get_voxcpm_lock():
                    result_path = await loop.run_in_executor(None, run_local_synthesis)

                if result_path and os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                    return
            except Exception as e:
                self.log(f"Local VoxCPM2 attempt {attempt} failed ({e}). Retrying in 2 seconds...", "warning")
                attempt += 1
                await asyncio.sleep(2)

    async def async_time_stretch_audio(self, input_path, output_path, speed):
        """Stretches audio speed using FFmpeg's high-quality atempo filter with chaining for extreme speeds."""
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            raise ValueError("Input audio file for time stretch is empty or missing.")
            
        if abs(speed - 1.0) < 0.05:
            # No stretch needed, convert directly to wav
            cmd = [get_ffmpeg_cmd(), "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_path]
            await self.async_run_cmd(cmd, "Audio Format Conversion")
            return

        # Build chained atempo filters (each filter supports 0.5 to 2.0)
        speed_clamped = max(0.1, min(10.0, speed))
        filters = []
        curr = speed_clamped
        while curr > 2.0:
            filters.append("atempo=2.0")
            curr /= 2.0
        while curr < 0.5:
            filters.append("atempo=0.5")
            curr /= 0.5
        filters.append(f"atempo={curr:.3f}")
        filter_str = ",".join(filters)

        cmd = [
            get_ffmpeg_cmd(), "-y",
            "-i", input_path,
            "-vn",
            "-filter:a", filter_str,
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            output_path
        ]
        await self.async_run_cmd(cmd, "Audio Time Stretch")

    def find_gcp_credentials(self):
        """Scans root directory for any JSON file containing GCP service account fields."""
        for filename in os.listdir("."):
            if filename.endswith(".json") and filename not in ["settings.json", "queue.json"]:
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "project_id" in data and "private_key" in data:
                        return filename, data["project_id"]
                except Exception:
                    pass
        return None, None

    async def call_groq_with_retry(self, prompt: str, audio_file_path: str = None) -> str:
        """Translates Whisper transcript using Groq Cloud API (Llama 3.3 70B / 3.1 8B) with key rotation."""
        import httpx

        # 1. Discover all Groq keys
        groq_keys = []
        pool_keys = api_key_pool.keys if api_key_pool.keys else self.api_keys
        if pool_keys:
            for k in pool_keys:
                k_clean = str(k).strip().strip('"\'')
                if k_clean.startswith("gsk_") and k_clean not in groq_keys:
                    groq_keys.append(k_clean)

        setting_key = self.settings.get("groq_api_key", "").strip()
        if setting_key and setting_key not in groq_keys:
            groq_keys.insert(0, setting_key)

        env_key = os.environ.get("GROQ_API_KEY", "").strip()
        if env_key and env_key not in groq_keys:
            groq_keys.append(env_key)

        if not groq_keys:
            raise ValueError(
                "Groq API selected, but no Groq API Key (starts with 'gsk_') was found in Manage Keys. "
                "Please get a free API key at console.groq.com and paste it in Manage Keys."
            )

        self.log(f"Groq Cloud Engine activated. Pool has {len(groq_keys)} Groq key{'s' if len(groq_keys) > 1 else ''}.", "info")

        # 2. Determine model sequence with dynamic live discovery and failover
        preferred_model = self.settings.get("model", "openai/gpt-oss-120b")
        if not preferred_model or "gemini" in str(preferred_model).lower() or "llama" in str(preferred_model).lower():
            preferred_model = "openai/gpt-oss-120b"

        discovered_models = []
        try:
            async with httpx.AsyncClient(timeout=8.0) as disc_client:
                m_resp = await disc_client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {groq_keys[0]}"}
                )
                if m_resp.status_code == 200:
                    raw_models = [m.get("id") for m in m_resp.json().get("data", []) if m.get("id")]
                    discovered_models = [
                        m for m in raw_models 
                        if not any(skip in m.lower() for skip in ["whisper", "guard", "orpheus", "prompt"])
                    ]
        except Exception:
            pass

        priority_ranked = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "allam-2-7b"]
        models_to_try = []
        
        if preferred_model and (not discovered_models or preferred_model in discovered_models):
            models_to_try.append(preferred_model)
            
        for pr in priority_ranked:
            if pr not in models_to_try and (not discovered_models or pr in discovered_models):
                models_to_try.append(pr)
                
        for dm in discovered_models:
            if dm not in models_to_try:
                models_to_try.append(dm)
                
        if not models_to_try:
            models_to_try = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]

        # 3. Attempt generation across models and keys
        last_error = None
        for active_model in models_to_try:
            for key_idx, current_key in enumerate(groq_keys):
                masked_key = f"{current_key[:6]}...{current_key[-4:]}" if len(current_key) > 10 else "***"
                self.log(f"Calling Groq API [{active_model}] using Key #{key_idx + 1} ({masked_key})...", "info")
                
                headers = {
                    "Authorization": f"Bearer {current_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": active_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert film and television dubbing dialogue director and translator. "
                                "Translate the provided dialogue segments accurately, expressively, and cinematically. "
                                "Strictly follow the JSON schema instructions provided in the user prompt. "
                                "Return ONLY valid raw JSON without markdown formatting."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 8192
                }

                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )

                    if response.status_code == 200:
                        data = response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if content:
                            self.log(f"Groq API generation successful! Model: {active_model} | Output: {len(content)} chars.", "success")
                            return content
                        else:
                            raise ValueError("Empty content returned from Groq API.")

                    elif response.status_code == 429:
                        err_json = {}
                        try:
                            err_json = response.json()
                        except Exception:
                            pass
                        err_msg = err_json.get("error", {}).get("message", response.text)
                        self.log(f"Groq rate limit (429) hit on Key #{key_idx + 1} ({masked_key}): {err_msg}. Rotating to next key/model...", "warning")
                        last_error = f"429 Rate Limit: {err_msg}"
                        await asyncio.sleep(1.5)
                        continue

                    else:
                        err_text = response.text
                        self.log(f"Groq API HTTP {response.status_code} error: {err_text}", "warning")
                        last_error = f"HTTP {response.status_code}: {err_text}"
                        continue

                except Exception as req_err:
                    self.log(f"Groq request exception on {active_model}: {req_err}", "warning")
                    last_error = str(req_err)
                    continue

        raise RuntimeError(f"All Groq API models and keys failed. Last error: {last_error}")

    def classify_router_task(self, audio_file_path, prompt: str) -> str:
        """
        Classifies task modality and tier for the Hybrid Gemini Router:
        - 'bulk_translate': Text-only translation (e.g., YouTube extra language tracks)
        - 'story_recap': Text-only narrative storytelling (Movie Recap Mode with Whisper transcript)
        - 'multimodal_dub': Acoustic voice pitch/gender/emotion analysis + dialogue translation
        """
        if not audio_file_path or not os.path.exists(str(audio_file_path)):
            return "bulk_translate"
        if "Input Segments:" in prompt and "Maintain the exact same segment structure" in prompt:
            return "bulk_translate"
        if self.settings.get("movie_recap_mode", False) and "Here is the raw dialogue spoken in the video:" in prompt:
            return "story_recap"
        return "multimodal_dub"

    async def cleanup_router_files(self):
        """Deletes any cached Gemini File API uploads across all keys in the pool at the end of a video."""
        loop = asyncio.get_event_loop()
        import google.generativeai.client as client_mod  # type: ignore
        from google.api_core.client_options import ClientOptions  # type: ignore

        for kh in list(api_key_pool.health_map.values()):
            if not kh.uploaded_files:
                continue
            try:
                client_options = ClientOptions(api_key=kh.clean_key)
                file_client = client_mod.FileServiceClient(client_options=client_options)
                for f_hash, f_obj in list(kh.uploaded_files.items()):
                    try:
                        await loop.run_in_executor(None, lambda name=f_obj.name: file_client.delete_file(name=name))
                    except Exception:
                        pass
                    kh.uploaded_files.pop(f_hash, None)
            except Exception:
                kh.uploaded_files.clear()

    async def call_gemini_with_retry(self, audio_file_path, prompt, reference_audios=None, expected_segments=None):
        """
        Hybrid Type-3 (Task/Modality) + Type-4 (Multi-Key/Provider Failover) Gemini Router.
        Routes requests dynamically across multimodal vs text-only tiers, reuses uploaded
        File API handles across batches on the same key, and enforces a 60s rate-limit circuit breaker.
        """
        task_mode = self.classify_router_task(audio_file_path, prompt)
        need_audio = (task_mode == "multimodal_dub" and audio_file_path and os.path.exists(str(audio_file_path)))

        # 1. Explicit Groq Cloud API path OR oportunistic Groq offload for bulk text translation
        pool_keys = api_key_pool.keys if api_key_pool.keys else self.api_keys
        has_groq_key = any(str(k).strip().strip('"\'').startswith("gsk_") for k in (pool_keys or [])) or bool(self.settings.get("groq_api_key", "").strip())
        if self.settings.get("api_type") == "groq" or (task_mode == "bulk_translate" and has_groq_key and self.settings.get("api_type") != "vertex"):
            try:
                return await self.call_groq_with_retry(prompt, audio_file_path)
            except Exception as groq_err:
                if self.settings.get("api_type") == "groq":
                    raise groq_err
                self.log(f"Router Groq offload failed ({groq_err}), routing to Gemini Flash-Lite...", "warning")

        # 2. Vertex AI path (Modality-Aware)
        if self.settings.get("api_type") == "vertex":
            creds_file, project_id = self.find_gcp_credentials()
            if not creds_file:
                raise ValueError("Vertex AI selected, but no GCP Service Account JSON credentials found in project root folder. Please download your service account key JSON and paste it in the folder.")
                
            from google.oauth2 import service_account  # type: ignore
            from google.api_core import exceptions as gcp_exceptions  # type: ignore
            
            try:
                from google import genai as google_genai  # type: ignore
                from google.genai import types  # type: ignore
            except ImportError:
                raise ImportError("Required library 'google-genai' is missing. Please run: pip install google-genai google-api-core google-cloud-storage")
                
            credentials = service_account.Credentials.from_service_account_file(
                creds_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            audio_bytes = None
            audio_mime_type = "audio/wav"
            if need_audio:
                target_audio_path = audio_file_path
                file_size = os.path.getsize(target_audio_path)
                if file_size > 10 * 1024 * 1024 and str(audio_file_path).lower().endswith(".wav"):
                    temp_mp3 = os.path.join(self.temp_dir, "vertex_inline_compressed.mp3")
                    if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
                        self.log(f"Compressing vocals ({file_size / (1024*1024):.1f} MB) to mono 96k MP3 for rapid Vertex AI upload...", "info")
                        cmd = [get_ffmpeg_cmd(), "-y", "-i", target_audio_path, "-ac", "1", "-b:a", "96k", temp_mp3]
                        await self.async_run_cmd(cmd, "Audio Compression for Vertex AI")
                    if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 0:
                        target_audio_path = temp_mp3
                        audio_mime_type = "audio/mp3"

                with open(target_audio_path, "rb") as f:
                    audio_bytes = f.read()
                
            if task_mode == "bulk_translate":
                model_priorities = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3.6-flash"]
            elif task_mode == "story_recap":
                model_priorities = ["gemini-2.5-pro", "gemini-3.6-pro", "gemini-2.5-flash", "gemini-3.6-flash"]
            else:
                model_priorities = [
                    "gemini-2.5-flash",
                    "gemini-3.6-flash",
                    "gemini-2.5-pro",
                    "gemini-3.6-pro",
                    "gemini-1.5-flash-002"
                ]
            server_regions = [
                "global",
                "us-central1",
                "europe-west1",
                "asia-east1"
            ]
            
            is_timeline = "JSON structure" in prompt or "Split the transcription" in prompt
            
            if is_timeline:
                response_schema = {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "start": {"type": "NUMBER"},
                            "end": {"type": "NUMBER"},
                            "text": {"type": "STRING"},
                            "translation": {"type": "STRING"},
                            "gender": {"type": "STRING"},
                            "speaker_role": {"type": "STRING"},
                            "age_group": {"type": "STRING"},
                            "emotion": {"type": "STRING"}
                        },
                        "required": ["id", "start", "end", "text", "translation", "gender", "speaker_role", "age_group", "emotion"]
                    }
                }
            else:
                response_schema = {
                    "type": "OBJECT",
                    "properties": {
                        "transcript": {"type": "STRING"},
                        "english_translation": {"type": "STRING"},
                        "gender": {"type": "STRING"},
                        "age_group": {"type": "STRING"}
                    },
                    "required": ["transcript", "english_translation", "gender", "age_group"]
                }
                
            for model in model_priorities:
                for region in server_regions:
                    self.log(f"Gemini Router [{task_mode}] -> Vertex AI {model} ({region})...", "info")
                    try:
                        client = google_genai.Client(
                            vertexai=True,
                            project=project_id,
                            location=region,
                            credentials=credentials
                        )
                        
                        loop = asyncio.get_event_loop()
                        def do_vertex_generate():
                            contents = []
                            if need_audio and audio_bytes:
                                if reference_audios:
                                    contents.append(f"I am providing {len(reference_audios)} reference audio clips for established character voices:")
                                    for role_key, path in reference_audios.items():
                                        if os.path.exists(path):
                                            with open(path, "rb") as f:
                                                ref_bytes = f.read()
                                            contents.append(f"Reference Audio for role: {role_key}")
                                            contents.append(types.Part.from_bytes(data=ref_bytes, mime_type="audio/wav"))
                                
                                contents.append("Here is the main audio you need to transcribe and translate:")
                                contents.append(types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime_type))
                            contents.append(prompt)
                            
                            return client.models.generate_content(
                                model=model,
                                contents=contents,
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=response_schema
                                )
                            )
                            
                        response = await asyncio.wait_for(
                            loop.run_in_executor(None, do_vertex_generate),
                            timeout=600.0
                        )
                        
                        text_response = response.text
                        self.log(f"Vertex AI generation successful using Model: {model} in Region: {region}", "success")
                        return text_response
                        
                    except asyncio.TimeoutError:
                        self.log(f"Vertex AI failover: timeout on {model} ({region}) after 600s", "warning")
                        await asyncio.sleep(2)
                    except (gcp_exceptions.ResourceExhausted, gcp_exceptions.InternalServerError) as limit_err:
                        self.log(f"Vertex AI failover: limit/server error on {model} ({region}): {limit_err}", "warning")
                        await asyncio.sleep(2)
                    except Exception as other_err:
                        self.log(f"Vertex AI failover: unexpected error on {model} ({region}): {other_err}", "warning")
                        await asyncio.sleep(2)
                        
            self.log("Vertex AI failover exhausted: All model/region combinations failed. Falling back to AI Studio Router...", "warning")

        # 3. Hybrid AI Studio Gemini Router Path (Sticky File Cache + Task Tiering + 60s Circuit Breaker)
        import google.generativeai.client as client_mod  # type: ignore
        import google.ai.generativelanguage as glm  # type: ignore
        from google.api_core.client_options import ClientOptions  # type: ignore
        from google.generativeai.types import file_types  # type: ignore
        import pathlib

        loop = asyncio.get_event_loop()
        if self.api_keys and not api_key_pool.keys:
            api_key_pool.set_keys(self.api_keys)

        # Prepare compressed upload target once if audio is required
        upload_target_path = None
        mime_type = "audio/wav"
        preferred_hash = ""
        if need_audio:
            upload_target_path = audio_file_path
            file_sz = os.path.getsize(upload_target_path)
            if file_sz > 2 * 1024 * 1024 and str(audio_file_path).lower().endswith(".wav"):
                compressed_mp3 = os.path.join(self.temp_dir, "gemini_upload_compressed.mp3")
                if not os.path.exists(compressed_mp3) or os.path.getsize(compressed_mp3) == 0:
                    self.log(f"Compressing vocals ({file_sz / (1024*1024):.1f} MB) to 16kHz 48k Mono MP3 for fast upload...", "info")
                    compress_cmd = [get_ffmpeg_cmd(), "-y", "-i", audio_file_path, "-ac", "1", "-ar", "16000", "-b:a", "48k", compressed_mp3]
                    await self.async_run_cmd(compress_cmd, "Audio Compression for Gemini Upload")
                if os.path.exists(compressed_mp3) and os.path.getsize(compressed_mp3) > 0:
                    upload_target_path = compressed_mp3
                    mime_type = "audio/mp3"
            preferred_hash = api_key_pool.compute_file_hash(upload_target_path)

        # Build model route tier based on task classification
        user_model = self.settings.get("model", "gemini-2.5-flash")
        if task_mode == "bulk_translate":
            model_candidates = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.5-flash", user_model]
        elif task_mode == "story_recap":
            model_candidates = ["gemini-2.5-pro", user_model, "gemini-2.5-flash", "gemini-3.5-flash"]
        else:
            model_candidates = [user_model, "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.6-flash", "gemini-3.5-flash"]
        # Deduplicate while preserving order
        seen_m = set()
        model_candidates = [m for m in model_candidates if m and not (m in seen_m or seen_m.add(m))]

        failed_keys = set()
        pool_size = max(len(api_key_pool.keys), 1)
        max_attempts = max(pool_size * 2 + 1, 5)

        for attempt in range(max_attempts):
            kh = None
            rate_limited = False
            try:
                kh = await api_key_pool.acquire_router_key(preferred_file_hash=preferred_hash, failed_keys=failed_keys)
                self.log(
                    f"Gemini Router [{task_mode}] leased key {kh.masked_key} (In-use: {kh.in_use}, Cached files: {len(kh.uploaded_files)})",
                    "info"
                )

                client_options = ClientOptions(api_key=kh.clean_key)
                generative_client = glm.GenerativeServiceClient(client_options=client_options)
                file_client = client_mod.FileServiceClient(client_options=client_options)

                async def get_or_upload_file(local_path: str, m_type: str):
                    f_hash = api_key_pool.compute_file_hash(local_path)
                    if f_hash and f_hash in kh.uploaded_files:
                        self.log(f"Router cache hit: Reusing uploaded {os.path.basename(local_path)} on key {kh.masked_key} (0 upload cost).", "info")
                        return kh.uploaded_files[f_hash]

                    p_obj = pathlib.Path(os.fspath(local_path))
                    self.log(f"Uploading {p_obj.name} ({os.path.getsize(local_path)/(1024*1024):.2f} MB) to Gemini File API...", "info")
                    def _create():
                        resp = file_client.create_file(
                            path=p_obj, mime_type=m_type, name=None, display_name=p_obj.name, resumable=True
                        )
                        return file_types.File(resp)
                    up_file = await loop.run_in_executor(None, _create)
                    while getattr(up_file, "state", None) and up_file.state.name == "PROCESSING":
                        await asyncio.sleep(1)
                        up_file = await loop.run_in_executor(None, lambda: file_client.get_file(name=up_file.name))
                    if getattr(up_file, "state", None) and up_file.state.name == "FAILED":
                        raise RuntimeError(f"Uploaded audio {p_obj.name} failed processing in Gemini storage.")
                    if f_hash:
                        kh.uploaded_files[f_hash] = up_file
                    return up_file

                uploaded_refs = []
                uploaded_file = None
                if need_audio and upload_target_path:
                    if reference_audios:
                        for role_key, ref_path in reference_audios.items():
                            if os.path.exists(ref_path):
                                try:
                                    ref_f = await get_or_upload_file(ref_path, "audio/wav")
                                    uploaded_refs.append((role_key, ref_f))
                                except Exception as ref_err:
                                    self.log(f"Failed to upload reference audio {ref_path}: {ref_err}", "warning")
                    uploaded_file = await get_or_upload_file(upload_target_path, mime_type)
                else:
                    self.log(f"Router [{task_mode}]: Skipping audio upload (text-only fast path).", "info")

                def do_generate():
                    generate_contents = []
                    if uploaded_file is not None:
                        if uploaded_refs:
                            generate_contents.append(f"I am providing {len(uploaded_refs)} reference audio clips for established character voices:")
                            for r_key, r_file in uploaded_refs:
                                generate_contents.append(f"Reference Audio for role: {r_key}")
                                generate_contents.append(r_file)
                        generate_contents.append("Here is the main audio you need to transcribe and translate:")
                        generate_contents.append(uploaded_file)
                    generate_contents.append(prompt)

                    last_gen_err = None
                    for m_candidate in model_candidates:
                        try:
                            m_inst = genai.GenerativeModel(m_candidate)
                            m_inst._client = generative_client
                            resp = m_inst.generate_content(
                                generate_contents,
                                generation_config={"response_mime_type": "application/json"}
                            )
                            text_out = resp.text or ""
                            if expected_segments and text_out.count('"id"') < int(expected_segments * 0.75):
                                raise ValueError(f"Model {m_candidate} returned incomplete segment count (<75% of {expected_segments}).")
                            return resp, m_candidate
                        except Exception as m_err:
                            last_gen_err = m_err
                            err_s = str(m_err).lower()
                            if "404" in err_s or "not found" in err_s or "incomplete segment count" in err_s:
                                continue
                            raise m_err
                    if last_gen_err:
                        raise last_gen_err

                response, used_model = await asyncio.wait_for(
                    loop.run_in_executor(None, do_generate),
                    timeout=120.0
                )

                text_response = response.text
                self.log(f"Router [{task_mode}] completed via {used_model} on key {kh.masked_key}.", "success")
                await api_key_pool.release_router_key(kh, rate_limited=False)
                return text_response

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "resourceexhausted" in err_str.lower():
                    rate_limited = True
                if kh is not None:
                    failed_keys.add(kh.raw_key)
                    await api_key_pool.release_router_key(kh, rate_limited=rate_limited, cooldown_seconds=60.0)
                    kh = None

                masked = identify_key_format(list(failed_keys)[-1])["masked"] if failed_keys else "Unknown"
                status_tag = "429 Rate-Limited (60s cooldown)" if rate_limited else "Error"
                self.log(f"Router failover on Key {masked} [{status_tag}]: {err_str}", "warning")

                if attempt < max_attempts - 1:
                    wait_time = 6 if len(api_key_pool.keys) <= 1 else 1.5
                    self.log(f"Router switching key/tier (attempt {attempt + 2}/{max_attempts}) in {wait_time}s...", "warning")
                    await asyncio.sleep(wait_time)

        raise RuntimeError("Failed to translate: All Gemini API keys and router tiers exhausted.")

    async def process_video(self, video_path, video_id):
        """Main processing pipeline for a single video."""
        self.update_progress(5, "Initializing...")
        
        # Paths setup
        filename = os.path.basename(video_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Temp dir inside workspace
        self.temp_dir = os.path.join(os.path.dirname(video_path), f"temp_dub_{video_id}")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        temp_audio = os.path.join(self.temp_dir, "original_audio.wav")
        vocals_wav = os.path.join(self.temp_dir, "vocals.wav")
        
        try:
            # 1. Extract Audio from Video
            self.log(f"Step 1A: Extracting audio from {filename}...", "info")
            self.update_progress(10, "Extracting audio...")
            extract_cmd = [
                get_ffmpeg_cmd(), "-y",
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                temp_audio
            ]
            await self.async_run_cmd(extract_cmd, "Audio Extraction")
            
            # Load original audio duration
            audio_segment = AudioSegment.from_wav(temp_audio)
            original_duration_ms = len(audio_segment)
            original_duration_s = original_duration_ms / 1000.0
            
            # 2. Audio Stem Separation
            demucs_output_dir = os.path.join(self.temp_dir, "demucs")
            vocals_path = None
            drums_path = None
            bass_path = None
            melody_path = None
            
            # Run separation if at least one stem mixer is selected OR we want clean vocals
            try:
                self.update_progress(15, "Demucs Isolating...")
                await self.run_demucs(temp_audio, demucs_output_dir)
                
                # Locate stems
                track_dir = os.path.join(demucs_output_dir, "htdemucs", "original_audio")
                vocals_path = os.path.join(track_dir, "vocals.wav")
                drums_path = os.path.join(track_dir, "drums.wav")
                bass_path = os.path.join(track_dir, "bass.wav")
                melody_path = os.path.join(track_dir, "other.wav")
                
                # Copy vocals out for easier access
                shutil.copy(vocals_path, vocals_wav)
                self.log("Vocal isolation completed successfully.", "success")
            except Exception as de_err:
                self.log(f"Demucs isolation failed, falling back to original audio: {de_err}", "warning")
                shutil.copy(temp_audio, vocals_wav)
                vocals_path = None
            
            # 3. Local ASR using Faster-Whisper
            self.update_progress(30, "Transcribing with Whisper...")
            async with get_whisper_lock():
                self.log("Step 1B: Running local Faster-Whisper to transcribe and timestamp speech...", "info")
                
                whisper_segments_list = []
                try:
                        device = "cuda" if torch.cuda.is_available() and self.settings.get("gpu_mode") != "cpu" else "cpu"
                        model_base = get_base_path()
                        if getattr(sys, 'frozen', False):
                            # Prioritize models folder next to the .exe
                            exe_models = os.path.join(os.path.dirname(sys.executable), "models")
                            if os.path.exists(exe_models):
                                model_base = os.path.dirname(sys.executable)
                                
                        model_path = os.path.join(model_base, "models", "faster-whisper-large-v3-turbo")
                    

                        loop = asyncio.get_event_loop()
                        def load_whisper():
                            global global_whisper_model, global_voxcpm_4gb_model
                            if global_voxcpm_4gb_model is not None:
                                try:
                                    global_voxcpm_4gb_model.offload_to_ram()
                                except Exception:
                                    pass
                            if global_whisper_model is not None:
                                try:
                                    if hasattr(global_whisper_model, "model") and hasattr(global_whisper_model.model, "load_model"):
                                        global_whisper_model.model.load_model()
                                except Exception:
                                    pass
                                return global_whisper_model

                            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                            os.environ['OMP_NUM_THREADS'] = '1'

                            if device == "cuda":
                                comp_types = ["float16", "int8_float16", "int8", "float32"]
                                try:
                                    major, minor = torch.cuda.get_device_capability(0)
                                    if major < 7:
                                        self.log(f"GPU Compute Capability {major}.{minor} detected (Pascal or older). Using int8 directly.", "info")
                                        comp_types = ["int8", "float32"]
                                except Exception as cap_err:
                                    self.log(f"Failed to query GPU Compute Capability: {cap_err}", "warning")

                                for comp_type in comp_types:
                                    try:
                                        self.log(f"Loading Faster-Whisper on CUDA with compute_type={comp_type}...", "info")
                                        wm = WhisperModel(model_path, device="cuda", compute_type=comp_type)
                                        self.log(f"Successfully loaded Faster-Whisper on CUDA with compute_type={comp_type}.", "success")
                                        global_whisper_model = wm
                                        return wm
                                    except Exception as e:
                                        self.log(f"Failed loading on CUDA with {comp_type}: {e}", "warning")

                            self.log("Loading Faster-Whisper on CPU with compute_type=int8...", "info")
                            wm = WhisperModel(model_path, device="cpu", compute_type="int8")
                            self.log("Successfully loaded Faster-Whisper on CPU.", "success")
                            global_whisper_model = wm
                            return wm

                        whisper_model = await loop.run_in_executor(None, load_whisper)
                    
                        def run_transcription():
                            segs_gen, info = whisper_model.transcribe(
                                vocals_wav,
                                beam_size=1,
                                best_of=1,
                                vad_filter=False,
                                condition_on_previous_text=False,
                                word_timestamps=False
                            )
                            return list(segs_gen)
                    
                        segs = await loop.run_in_executor(None, run_transcription)

                        # Acoustic onset/offset trimmer + hallucination filter on vocals_wav (replaces buggy word_timestamps DTW seek jump)
                        import numpy as np
                        voc_arr_16k = None
                        try:
                            voc_seg_16k = AudioSegment.from_wav(vocals_wav).set_channels(1).set_frame_rate(16000).set_sample_width(2)
                            voc_arr_16k = np.frombuffer(voc_seg_16k.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                        except Exception:
                            voc_arr_16k = None

                        hallucination_substrings = (
                            "优优独播剧场", "yoyo television", "ming pao", "amara.org",
                            "字幕志愿者", "感谢观看", "谢谢观看", "请不吝点赞", "订阅转发"
                        )
                    
                        prev_txt = ""
                        prev_end = 0.0
                        for s in segs:
                            txt = s.text.strip()
                            if txt:
                                txt_low = txt.lower()
                                if any(h in txt_low for h in hallucination_substrings):
                                    continue

                                w_start = round(s.start, 3)
                                w_end = round(min(original_duration_s, s.end), 3)
                                if w_end <= w_start:
                                    continue

                                if voc_arr_16k is not None:
                                    s_i = max(0, int(w_start * 16000))
                                    e_i = min(len(voc_arr_16k), int(w_end * 16000))
                                    if e_i - s_i >= 800:
                                        chunk_16k = voc_arr_16k[s_i:e_i]
                                        overall_db = 20.0 * float(np.log10(np.sqrt(np.mean(chunk_16k ** 2)) + 1e-9))
                                        # Drop pure silence hallucinations (< -50 dB)
                                        if overall_db < -50.0:
                                            continue
                                        # Trim leading silence if Whisper attached preceding pause (> 1.2s span) to segment start
                                        if (w_end - w_start) > 1.2:
                                            win = 400  # 25ms frame at 16kHz
                                            n_w = (e_i - s_i) // win
                                            if n_w > 2:
                                                frames = chunk_16k[:n_w * win].reshape(n_w, win)
                                                rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
                                                peak_db = float(np.max(rms_db))
                                                thresh_db = max(-36.0, min(-24.0, peak_db - 14.0))
                                                active_idx = np.where(rms_db > thresh_db)[0]
                                                if len(active_idx) > 0:
                                                    trimmed_start = round(max(w_start, w_start + (int(active_idx[0]) * 0.025) - 0.06), 3)
                                                    if w_end - trimmed_start >= 0.35:
                                                        w_start = trimmed_start

                                # Skip immediate consecutive identical text repeats within 1.0s
                                if txt == prev_txt and abs(w_start - prev_end) < 1.0:
                                    continue

                                whisper_segments_list.append({
                                    "id": len(whisper_segments_list) + 1,
                                    "start": w_start,
                                    "end": w_end,
                                    "text": txt
                                })
                                prev_txt = txt
                                prev_end = w_end
                    
                        self.log(f"Faster-Whisper transcription completed. Detected {len(whisper_segments_list)} speech segments.", "success")
                except Exception as whisper_err:
                    self.log(f"Faster-Whisper transcription failed: {whisper_err}. Falling back to default Gemini-only processing.", "error")
            
            # 4. Gemini Translation & Voice Analysis
            self.update_progress(40, "Translating voice...")
            self.log("Step 2: Sending audio to Gemini for translation & voice profiling...", "info")
            
            target_lang_code = self.settings.get("target_lang", "km-KH")
            target_lang_name = LANGUAGE_PROMPTS.get(target_lang_code, target_lang_code)
            conciseness = ""
            if "concise-20" in self.settings.get("translation_mode", ""):
                conciseness = "Make the translations concise (approximately 20% shorter in syllables than original speech) to ensure it fits the timing."
            elif "concise-10" in self.settings.get("translation_mode", ""):
                conciseness = "Make the translations slightly concise (approximately 10% shorter) to fit the timing."
            
            movie_recap_mode = self.settings.get("movie_recap_mode", False)
            if movie_recap_mode:
                # Target chunks based on total duration (average 12-15s per story beat)
                target_chunks = max(3, min(12, int(original_duration_s / 12.0)))
                words_per_chunk = max(15, int((original_duration_s / target_chunks) * 1.8))
                
                if whisper_segments_list:
                    full_transcript_text = "\n".join([f"- {s.get('text', '')}" for s in whisper_segments_list if s.get('text', '').strip()])
                    prompt = f"""
                    You are an expert movie recap storyteller and scriptwriter (សម្រាយរឿង / Movie Recap Narrator).
                    Your task is to write a captivating, continuous third-person story recap narrative for the audience in target language: '{target_lang_name}'.
                    
                    Here is the raw dialogue spoken in the video:
                    {full_transcript_text}
                    
                    CRITICAL RECAP RULES:
                    1. DO NOT translate character dialogue line-by-line.
                    2. DO NOT keep the original conversation or segment structure.
                    3. Tell the complete story from a third-person storyteller perspective (describing the characters, setup, rising conflict, key actions, and outcome).
                    4. Split your recap narrative into exactly {target_chunks} chronological story beats/paragraphs.
                    5. For each chunk:
                       - "translation": The exact storytelling sentence(s) to be SPOKEN ALOUD by the narrator in '{target_lang_name}'. Write natural, engaging narrative prose. DO NOT output English here if the target language is not English.
                       - "text": A brief 1-line note of the scene or context (e.g. "Protagonist discovers the mud pond").
                       - Budget approximately {words_per_chunk} words per chunk so it flows smoothly across the total video duration ({original_duration_s:.1f} seconds).
                    6. Assign sequential, non-overlapping "start" and "end" timestamps spread naturally across the video duration [0.0 - {original_duration_s:.1f}].
                    7. Set "speaker_role" to "narrator", "gender" to "male", and "emotion" to "dramatic" for all segments.
                    8. Return ONLY a valid JSON array of objects. Do not include markdown code block syntax.
                    
                    JSON structure:
                    [
                      {{
                        "id": 1,
                        "start": 1.0,
                        "end": 14.0,
                        "text": "Introduction of the story context",
                        "translation": "នៅក្នុងសាច់រឿងនេះ បុរសម្នាក់កំពុងតែមានការព្រួយបារម្ភយ៉ាងខ្លាំង...",
                        "gender": "male",
                        "speaker_role": "narrator",
                        "emotion": "dramatic"
                      }}
                    ]
                    """
                else:
                    self.log("Using fallback Gemini ASR + recap prompt...", "warning")
                    prompt = f"""
                    You are an expert movie recap storyteller and scriptwriter (សម្រាយរឿង / Movie Recap Narrator).
                    Analyze the uploaded audio (total duration: {original_duration_s:.2f} seconds).
                    Your task is to write a captivating, continuous third-person story recap narrative for the audience in target language: '{target_lang_name}'.
                    
                    CRITICAL RECAP RULES:
                    1. DO NOT transcribe or translate line-by-line character dialogue.
                    2. Listen to the audio, understand the full plot, and write a continuous third-person movie recap narrative in target language: '{target_lang_name}'.
                    3. Split your recap narrative into exactly {target_chunks} chronological story beats/paragraphs spread naturally across the total video duration ({original_duration_s:.2f} seconds).
                    4. For each chunk:
                       - "translation": The exact storytelling sentence(s) to be SPOKEN ALOUD by the narrator in '{target_lang_name}'. Write natural, engaging narrative prose. DO NOT output English here if the target language is not English.
                       - "text": A brief 1-line note of the scene or context.
                       - Budget approximately {words_per_chunk} words per chunk.
                    5. Assign sequential, non-overlapping "start" and "end" timestamps across [0.0 - {original_duration_s:.1f}].
                    6. Set "speaker_role" to "narrator", "gender" to "male", and "emotion" to "dramatic" for all segments.
                    7. Return ONLY a valid JSON array of objects.
                    
                    JSON structure:
                    [
                      {{
                        "id": 1,
                        "start": 1.0,
                        "end": 14.0,
                        "text": "Introduction of the story context",
                        "translation": "នៅក្នុងសាច់រឿងនេះ បុរសម្នាក់កំពុងតែមានការព្រួយបារម្ភយ៉ាងខ្លាំង...",
                        "gender": "male",
                        "speaker_role": "narrator",
                        "emotion": "dramatic"
                      }}
                    ]
                    """
            else:
                if whisper_segments_list:
                    whisper_json_str = json.dumps(whisper_segments_list, ensure_ascii=False, indent=2)
                    prompt = f"""
                    You are an AI video dubbing translator. Analyze the uploaded dialogue audio.
                    We have pre-transcribed the speech using Faster-Whisper to guarantee 100% correct timestamps and original wording.
                    Here is the exact transcript with timestamps:
                    {whisper_json_str}
                
                    Your task is to:
                    1. Listen to the uploaded dialogue audio file to understand the speaker's tone, pitch (for gender), and emotion of each segment.
                    2. For each segment in the provided JSON, translate the "text" into the target language: '{target_lang_name}'.
                    3. Detect the speaker's gender ('male' or 'female') for each segment.
                        CRITICAL GENDER DETECTION INSTRUCTIONS:
                        - Prioritize Audio Pitch over Transcribed Subtitles: You MUST trust the voice pitch and acoustic properties in the audio above all else. High-pitched, feminine voices are 'female'; deep, low-pitched, masculine voices are 'male'.
                        - Do NOT Confuse Speaker with Object/Subject of Dialogue: If a segment contains words like "他" (him), "先生" (Mister), "哥哥" (brother), or "少爷" (Master), it does NOT mean the speaker is male. A female character could be talking to or about a male. You MUST determine the gender of the person actually SPEAKING the words at that timestamp, not the gender of who they are talking to or about.
                        - Segment-Specific Listening: Listen strictly to the audio within the 'start' and 'end' timestamp boundaries of the segment to check the voice.
                        - Relational & Contextual Double-Check: Look for relational address terms in the dialogue (such as "哥", "姐", "先生", "女士", "老婆", "老公", "爸爸", "妈妈", "儿子", "女儿") to cross-reference and verify the speaker's actual gender.
                        - Enforce Speaker Consistency: A single speaker's gender, speaker_role, and age_group MUST remain strictly consistent throughout the entire video. Do not let a speaker's attributes flip back and forth between segments just because their emotion, pitch, or volume changes.
                    4. Detect the speaker role ('lead_actor' or 'narrator'). If a character is part of the story or on-screen, they are ALWAYS 'lead_actor' (do not use 'secondary'). Only use 'narrator' for off-screen disembodied voice-overs.
                    5. Detect the age_group ('young', 'adult', 'old'). Default to 'adult' unless the voice is unmistakably a child ('young') or an elderly person ('old'). This must also remain consistent for the same character.
                    6. Detect the speaker emotion/feeling (e.g. 'dramatic', 'angry', 'sad', 'happy', 'excited', 'fearful', 'whisper', 'romantic', 'crying', 'calm').
                    7. CRITICAL SEGMENT LOCKING: Maintain the exact same segment structure. You MUST output exactly {len(whisper_segments_list)} JSON objects, exactly one for each input segment from id=1 to id={len(whisper_segments_list)}. NEVER skip, omit, combine, or drop any segment, even if it is only a single word or short expression (e.g. '傻', '吃吧', '好'). Maintain the exact "id", "start", and "end" timestamps.
                    8. PROPORTIONAL DURATION & CONCISENESS (CRITICAL LIP-SYNC CONSTRAINTS):
                       - The spoken length of your translation MUST fit comfortably within the segment duration ('end' - 'start').
                       - For fast dialogue retorts (duration under 1.8 seconds), write extremely punchy, compact, and concise translations (maximum 3 to 5 words). NEVER inflate short 4-to-6 character rapid retorts into long compound sentences.
                       - For example:
                         * "你真往家带" (0.88s) -> "យកទៅផ្ទះមែន?" (NOT "ឯងពិតជានាំវាទៅផ្ទះមែន?")
                         * "不然让他冻死" (0.96s) -> "ទុកឱ្យវាកកស្លាប់?" (NOT "បើមិនចឹង ទុកឱ្យវាកកស្លាប់មែន?")
                         * "你家米缸都见底了" (1.36s) -> "អង្ករផ្ទះឯងអស់ហើយ" (NOT "អង្ករផ្ទះឯងជិតអស់ពីពាងហើយ")
                       - Keep translated dialogue natural, expressive, and compact so actors can finish speaking completely before the next sentence starts. {conciseness}
                    9. Return ONLY a valid JSON array of objects. Do not include markdown code block syntax.
                
                    JSON structure:
                    [
                      {{
                        "id": 1,
                        "start": 1.25,
                        "end": 4.5,
                        "text": "original text here",
                        "translation": "translated text here",
                        "gender": "male",
                        "speaker_role": "lead_actor",
                        "emotion": "dramatic"
                      }}
                    ]
                    """
                else:
                    self.log("Using fallback Gemini ASR + translation prompt...", "warning")
                    prompt = f"""
                    You are an AI video dubbing translator. Analyze the uploaded dialogue audio.
                    Perform transcription and translate it into target language: '{target_lang_name}'.
                
                    CRITICAL TIMING & DURATION CONSTRAINTS:
                    - The total duration of this audio file is EXACTLY {original_duration_s:.2f} seconds.
                      All segment 'start' and 'end' timestamps MUST be within the range [0.0, {original_duration_s:.2f}].
                      Do NOT output any segment with a timestamp greater than {original_duration_s:.2f} seconds.
                    - DO NOT STOP EARLY: There may be long silent gaps or music pauses (up to 15-20 seconds) in the audio.
                      You MUST continue analyzing the entire audio file until the very end ({original_duration_s:.2f} seconds)
                      and transcribe any spoken dialogue that appears after these silent gaps.
                    - MAXIMUM SEGMENT DURATION: A single segment duration ('end' - 'start') must NEVER exceed 8.0 seconds. 
                      If a character speaks for a long time, split it into multiple smaller segments.
                    - PROPORTIONALITY: The duration of a segment must be proportional to the length of the text spoken.
                    - SEQUENTIAL TIMING: Ensure timestamps are strictly chronological: start < end, and start of segment N >= end of segment N-1.
                
                    Requirements:
                    1. Split the transcription into logical timeline segments.
                    2. For each segment, output the start and end timestamp in seconds.
                    3. Detect the speaker's gender for each segment (output EXACTLY 'male' or 'female').
                       CRITICAL GENDER DETECTION INSTRUCTIONS:
                       - Prioritize Audio Pitch over Transcribed Subtitles: You MUST trust the voice pitch and acoustic properties in the audio above all else. High-pitched, feminine voices are 'female'; deep, low-pitched, masculine voices are 'male'.
                       - Do NOT Confuse Speaker with Object/Subject of Dialogue: If a segment contains words like "他" (him), "先生" (Mister), "哥哥" (brother), or "少爷" (Master), it does NOT mean the speaker is male. A female character could be talking to or about a male. You MUST determine the gender of the person actually SPEAKING the words at that timestamp, not the gender of who they are talking to or about.
                       - Segment-Specific Listening: Listen strictly to the audio within the 'start' and 'end' timestamp boundaries of the segment to check the voice.
                       - Relational & Contextual Double-Check: Look for relational address terms in the dialogue (such as "哥", "姐", "先生", "女士", "老婆", "老公", "爸爸", "妈妈", "儿子", "女儿") to cross-reference and verify the speaker's actual gender.
                       - Enforce Speaker Consistency: A single speaker's gender, speaker_role, and age_group MUST remain strictly consistent throughout the entire video. Do not let a speaker's attributes flip back and forth between segments just because their emotion, pitch, or volume changes.
                    4. Detect the speaker role for each segment (output 'lead_actor' or 'narrator'). If a character is part of the story or on-screen, they are ALWAYS 'lead_actor' (do not use 'secondary'). Only use 'narrator' for off-screen disembodied voice-overs.
                    5. Detect the speaker age group for each segment (output 'young', 'adult', or 'old'). Default to 'adult' unless the voice is unmistakably a child ('young') or an elderly person ('old').
                    6. Detect the speaker emotion/feeling for each segment accurately.
                    7. Provide natural, expressive, poetic, and cinematic translated dialogue for each segment in target language. {conciseness}
                    8. Return ONLY a valid JSON array of objects. Do not include markdown code block syntax.
                
                    JSON structure:
                    [
                      {{
                        "start": 1.25,
                        "end": 4.5,
                        "text": "Hello, welcome to this movie tutorial.",
                        "translation": "សួស្តី សូមស្វាគមន៍មកកាន់ការណែនាំភាពយន្តនេះ។",
                        "gender": "male",
                        "speaker_role": "lead_actor",
                        "emotion": "dramatic"
                      }}
                    ]
                    """
            
            recap_refs = None if movie_recap_mode else locked_speaker_roles
            raw_response = ""

            try:
                if whisper_segments_list and not movie_recap_mode and len(whisper_segments_list) > 45:
                    batch_size = 45
                    batches = [whisper_segments_list[i:i + batch_size] for i in range(0, len(whisper_segments_list), batch_size)]
                    self.log(f"Whisper transcript has {len(whisper_segments_list)} segments. Processing in {len(batches)} batches of ~{batch_size} via cached File API handle...", "info")
                    
                    batch_sem = asyncio.Semaphore(3)

                    async def _run_batch(b_idx, batch_segs):
                        self.log(f"Translating dialogue batch {b_idx}/{len(batches)} ({len(batch_segs)} segments)...", "info")
                        b_json_str = json.dumps(batch_segs, ensure_ascii=False, indent=2)
                        b_prompt = f"""
                        You are an AI video dubbing translator. Analyze the uploaded dialogue audio.
                        We have pre-transcribed the speech using Faster-Whisper to guarantee 100% correct timestamps and original wording.
                        Here is the exact transcript for this batch ({len(batch_segs)} segments):
                        {b_json_str}
                    
                        Your task is to:
                        1. Listen to the uploaded dialogue audio file to understand the speaker's tone, pitch (for gender), and emotion of each segment.
                        2. For each segment in the provided JSON, translate the "text" into the target language: '{target_lang_name}'.
                        3. Detect the speaker's gender ('male' or 'female') for each segment.
                        4. Detect the speaker role ('lead_actor' or 'narrator').
                        5. Detect the age_group ('young', 'adult', 'old'). Default to 'adult'.
                        6. Detect the speaker emotion/feeling (e.g. 'dramatic', 'angry', 'sad', 'happy', 'excited', 'fearful', 'whisper', 'romantic', 'crying', 'calm').
                        7. CRITICAL SEGMENT LOCKING: Maintain the exact same segment structure. You MUST output exactly {len(batch_segs)} JSON objects, exactly one for each input segment. Maintain the exact "id", "start", and "end" timestamps.
                        8. Keep translated dialogue natural, expressive, and compact so actors can finish speaking completely before the next sentence starts. {conciseness}
                        9. Return ONLY a valid JSON array of objects. Do not include markdown code block syntax.
                    
                        JSON structure:
                        [
                          {{
                            "id": {batch_segs[0].get('id', 1)},
                            "start": {batch_segs[0].get('start', 0.0)},
                            "end": {batch_segs[0].get('end', 1.0)},
                            "text": "{batch_segs[0].get('text', '')}",
                            "translation": "translated text here",
                            "gender": "male",
                            "speaker_role": "lead_actor",
                            "emotion": "dramatic"
                          }}
                        ]
                        """
                        if locked_speaker_roles and not movie_recap_mode:
                            b_prompt += f"\n\nCRITICAL SPEAKER LOCKING: Match character voices against reference audios."
                        
                        async with batch_sem:
                            async with get_gemini_lock():
                                b_raw = await self.call_gemini_with_retry(
                                    vocals_wav,
                                    b_prompt,
                                    reference_audios=recap_refs,
                                    expected_segments=len(batch_segs)
                                )
                        
                        b_clean = extract_json_payload(b_raw)
                        b_clean = re.sub(r',\s*([\]}])', r'\1', b_clean)
                        try:
                            b_parsed = json.loads(b_clean, strict=False)
                        except Exception:
                            last_brace = b_clean.rfind('}')
                            if last_brace != -1 and b_clean.find('[') != -1:
                                b_repaired = b_clean[b_clean.find('['):last_brace+1]
                                b_repaired = re.sub(r',\s*$', '', b_repaired) + "\n]"
                                b_parsed = json.loads(b_repaired, strict=False)
                            else:
                                b_parsed = []
                                
                        if isinstance(b_parsed, list) and len(b_parsed) > 0:
                            out_segs = b_parsed
                        else:
                            self.log(f"[WARNING] Batch {b_idx} JSON parsing incomplete. Falling back to original segments.", "warning")
                            out_segs = []
                            for seg in batch_segs:
                                out_segs.append({
                                    "id": seg.get("id"),
                                    "start": seg.get("start", 0.0),
                                    "end": seg.get("end", 1.0),
                                    "text": seg.get("text", ""),
                                    "translation": seg.get("text", ""),
                                    "gender": "male",
                                    "speaker_role": "lead_actor",
                                    "emotion": "calm"
                                })
                        return b_idx, b_raw, out_segs

                    # Warm up sticky File API cache with Batch 1 first, then fire remaining batches concurrently
                    first_res = await _run_batch(1, batches[0])
                    batch_results = [first_res]
                    if len(batches) > 1:
                        rem_tasks = [_run_batch(idx, b_segs) for idx, b_segs in enumerate(batches[1:], 2)]
                        rem_results = await asyncio.gather(*rem_tasks)
                        batch_results.extend(rem_results)

                    batch_results.sort(key=lambda x: x[0])
                    segments = []
                    for b_idx, b_raw, out_segs in batch_results:
                        raw_response += f"\n--- BATCH {b_idx} ---\n" + b_raw
                        segments.extend(out_segs)
                            
                    self.log(f"All {len(batches)} batches processed successfully. Received {len(segments)} segments total.", "success")
                else:
                    self.log("Waiting for Gemini API access (Concurrency Lock) to prevent Rate Limiting...", "muted")
                    
                    if locked_speaker_roles and not movie_recap_mode:
                        prompt += f"\n\nCRITICAL SPEAKER LOCKING: I have provided {len(locked_speaker_roles)} reference audio files representing established character voices in this video series. You MUST compare the speakers in the MAIN audio against these references. If a voice in the MAIN audio matches a reference voice, you MUST assign the EXACT SAME 'speaker_role' (e.g. 'lead_actor', 'narrator') as the reference. Do not create new roles for voices that match. IMPORTANT: DO NOT transcribe the reference audios! ONLY transcribe the MAIN audio."
                    
                    recap_refs = None if movie_recap_mode else locked_speaker_roles
                    expected_cnt = len(whisper_segments_list) if (whisper_segments_list and not movie_recap_mode) else None
                    async with get_gemini_lock():
                        raw_response = await self.call_gemini_with_retry(
                            vocals_wav,
                            prompt,
                            reference_audios=recap_refs,
                            expected_segments=expected_cnt
                        )
                    
                    # Clean JSON response from markdown blocks if present
                    clean_json = extract_json_payload(raw_response)
                    # Fix common JSON trailing commas
                    clean_json = re.sub(r',\s*([\]}])', r'\1', clean_json)
                    
                    try:
                        segments = json.loads(clean_json, strict=False)
                    except Exception as first_parse_err:
                        last_brace = clean_json.rfind('}')
                        if last_brace != -1 and clean_json.find('[') != -1:
                            repaired_json = clean_json[clean_json.find('['):last_brace+1]
                            repaired_json = re.sub(r',\s*$', '', repaired_json) + "\n]"
                            segments = json.loads(repaired_json, strict=False)
                            self.log(f"Auto-recovered truncated AI response: salvaged {len(segments)} valid segments.", "warning")
                        else:
                            raise first_parse_err
                    self.log(f"Successfully processed translation. Found {len(segments)} raw dialogue segments.", "success")
                
                # Field swap protection & ID initialization for movie recap mode
                if movie_recap_mode and isinstance(segments, list):
                    for idx, seg in enumerate(segments):
                        if not seg.get("id"):
                            seg["id"] = idx + 1
                        seg["speaker_role"] = "narrator"
                        tr = str(seg.get("translation", "")).strip()
                        tx = str(seg.get("text", "")).strip()
                        if target_lang_code == "km-KH":
                            if re.search(r'[\u1780-\u17FF]', tx) and not re.search(r'[\u1780-\u17FF]', tr):
                                seg["translation"] = tx
                                seg["text"] = tr
                        elif target_lang_code == "th-TH":
                            if re.search(r'[\u0E00-\u0E7F]', tx) and not re.search(r'[\u0E00-\u0E7F]', tr):
                                seg["translation"] = tx
                                seg["text"] = tr
                        elif target_lang_code == "my-MM":
                            if re.search(r'[\u1000-\u109F]', tx) and not re.search(r'[\u1000-\u109F]', tr):
                                seg["translation"] = tx
                                seg["text"] = tr
                
                # Enforce the exact original Whisper timestamps to prevent LLM modification/drift
                if whisper_segments_list and not self.settings.get("movie_recap_mode", False):
                    self.log("Anchoring translated segments to exact Faster-Whisper acoustic timestamps...", "info")
                    whisper_id_map = {int(w["id"]): w for w in whisper_segments_list if "id" in w}
                    used_whisper_ids = set()
                    
                    for idx, seg in enumerate(segments):
                        seg_id = seg.get("id")
                        matched_w = None
                        
                        # 1. Match by exact ID
                        if seg_id is not None:
                            try:
                                sid_int = int(seg_id)
                                if sid_int in whisper_id_map:
                                    matched_w = whisper_id_map[sid_int]
                                    used_whisper_ids.add(sid_int)
                            except (ValueError, TypeError):
                                pass
                                
                        # 2. Match by closest timestamp proximity
                        if matched_w is None:
                            try:
                                seg_s = float(seg.get("start", 0))
                                best_dist = float("inf")
                                best_cand = None
                                for w in whisper_segments_list:
                                    w_id = int(w.get("id", -1))
                                    if w_id not in used_whisper_ids:
                                        dist = abs(float(w["start"]) - seg_s)
                                        if dist < best_dist:
                                            best_dist = dist
                                            best_cand = w
                                if best_cand and best_dist < 4.0:
                                    matched_w = best_cand
                                    used_whisper_ids.add(int(best_cand.get("id", -1)))
                            except Exception:
                                pass
                                
                        # 3. Fallback to index if within bounds
                        if matched_w is None and idx < len(whisper_segments_list):
                            matched_w = whisper_segments_list[idx]
                            
                        if matched_w:
                            seg["start"] = float(matched_w["start"])
                            seg["end"] = float(matched_w["end"])
                            if "id" not in seg or not seg["id"]:
                                seg["id"] = matched_w.get("id", idx + 1)
                                
                    # Safety recovery for any acoustic segments omitted by LLM translation
                    missing_whisper = [w for w in whisper_segments_list if int(w.get("id", -1)) not in used_whisper_ids]
                    if missing_whisper:
                        self.log(f"Safety Net: Recovered {len(missing_whisper)} dialogue segment(s) omitted by LLM: {[w.get('text') for w in missing_whisper]}", "warning")
                        for mw in missing_whisper:
                            mw_text = mw.get("text", "").strip()
                            if mw_text:
                                segments.append({
                                    "id": mw.get("id"),
                                    "start": float(mw["start"]),
                                    "end": float(mw["end"]),
                                    "text": mw_text,
                                    "translation": mw_text,
                                    "gender": "female",
                                    "speaker_role": "lead_actor",
                                    "age_group": "adult",
                                    "emotion": "calm"
                                })
                elif not self.settings.get("movie_recap_mode", False):
                    self.log("No Whisper segments available, relying on LLM predicted timestamps.", "warning")

                # Validate parsed segments
                if not isinstance(segments, list):
                    raise ValueError("Parsed JSON is not a list of segments.")
                
                # Filter out segments starting after the video end, and clip segments exceeding video end
                if segments:
                    valid_segments = []
                    for seg in segments:
                        start = seg.get("start", 0.0)
                        end = seg.get("end", 0.0)
                        if start >= original_duration_s:
                            self.log(f"Discarding segment beyond video duration: [{start:.2f}s - {end:.2f}s] '{seg.get('text', '')}' -> '{seg.get('translation', '')}'", "info")
                            continue
                        if end > original_duration_s:
                            self.log(f"Clipping segment end from {end:.2f}s to {original_duration_s:.2f}s: '{seg.get('text', '')}' -> '{seg.get('translation', '')}'", "info")
                            end = original_duration_s
                        
                        if end > start + 0.1:  # Keep only if duration is at least 0.1s
                            seg["start"] = round(start, 3)
                            seg["end"] = round(end, 3)
                            valid_segments.append(seg)
                    segments = valid_segments

                # Filter out overlapping secondary/background voices to ensure only the dominant voice is dubbed
                filtered_segments = []
                for seg in segments:
                    role = str(seg.get("speaker_role", "")).lower().strip()
                    if "secondary" in role or "background" in role or "distant" in role:
                        has_heavy_overlap = False
                        for existing in filtered_segments:
                            overlap_start = max(seg["start"], existing["start"])
                            overlap_end = min(seg["end"], existing["end"])
                            if overlap_end > overlap_start:
                                overlap_dur = overlap_end - overlap_start
                                if overlap_dur > 0.6 and overlap_dur / max(0.1, seg["end"] - seg["start"]) > 0.6:
                                    has_heavy_overlap = True
                                    break
                        if has_heavy_overlap:
                            self.log(f"Filtered out heavily overlapping secondary/background voice segment: [{seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s] ({seg.get('text', '')})", "muted")
                            continue
                    filtered_segments.append(seg)
                
                # Sort segments strictly chronologically by start timestamp
                filtered_segments.sort(key=lambda x: float(x.get("start", 0)))
                segments = filtered_segments
                self.log(f"Retained {len(segments)} primary dominant voice segments for dubbing.", "info")
                
                # Cross-Video Voice Locking: Extract reference audios for new roles
                import subprocess
                for seg in segments:
                    s_role = str(seg.get("speaker_role", "")).lower().strip()
                    if s_role and s_role not in locked_speaker_roles:
                        # Extract this segment from vocals_wav
                        s_start = seg.get("start", 0.0)
                        s_end = seg.get("end", 0.0)
                        duration = s_end - s_start
                        if duration >= 1.0: # Only lock if it's long enough to be a good sample
                            locked_path = os.path.join(self.temp_dir, f"locked_ref_{s_role}.wav")
                            try:
                                subprocess.run([
                                    get_ffmpeg_cmd(), "-y", "-i", vocals_wav,
                                    "-ss", str(s_start), "-t", str(min(duration, 10.0)), # Keep up to 10s for better reference
                                    locked_path
                                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                                locked_speaker_roles[s_role] = locked_path
                                self.log(f"Locked reference voice for role '{s_role}' (Segment: {s_start:.2f}s - {s_start + min(duration, 10.0):.2f}s)", "success")
                            except Exception as e:
                                logger.warning(f"Failed to extract locked voice for {s_role}: {e}")
            except Exception as parse_err:
                self.log(f"Failed to parse Gemini response as JSON ({parse_err}). Raw response length: {len(raw_response)}", "error")
                logger.error(f"Raw Gemini response: {raw_response}")
                try:
                    err_txt_path = os.path.join(self.temp_dir, "error_response.txt")
                    with open(err_txt_path, "w", encoding="utf-8") as f:
                        f.write(raw_response)
                except Exception as e:
                    self.log(f"Failed to write error_response: {e}", "warning")
                raise RuntimeError("Failed to parse dialogue timeline from AI. (Check error_response.txt for raw output)")
            
            # 4. Text-To-Speech Synthesis
            self.update_progress(60, "Dubbing voice...")
            self.log("Step 3: Synthesizing dubbed voice lines...", "info")
            
            # Create voice tracks list
            dubbed_audio = AudioSegment.silent(duration=original_duration_ms)
            
            # Load vocals track if available to check vocal energy
            vocals_audio = None
            if vocals_wav and os.path.exists(vocals_wav):
                try:
                    vocals_audio = AudioSegment.from_wav(vocals_wav)
                except Exception as ve:
                    logger.warning(f"Failed to load vocals wav for energy analysis: {ve}")
            
            # Target voice mapping
            lang_code = self.settings.get("target_lang", "km-KH")
            male_voice, female_voice = VOICES.get(lang_code, VOICES["km-KH"])
            
            # CONCURRENT TTS SYNTHESIS BLOCK
            tts_eng = self.settings.get("tts_engine", "edge-tts")
            sem = asyncio.Semaphore(6 if tts_eng in ("edge-tts", "edge-openvoice") else 4)

            async def process_segment(idx, seg):
                seg_id = seg.get("id") or (idx + 1)
                seg["id"] = seg_id
                start = seg["start"]
                end = seg["end"]
                
                # Check isolated vocal energy during the segment's timeframe to filter true digital silence
                if vocals_audio is not None and not whisper_segments_list:
                    seg_start_ms = int(start * 1000.0)
                    seg_end_ms = int(end * 1000.0)
                    vocal_chunk = vocals_audio[max(0, seg_start_ms):min(len(vocals_audio), seg_end_ms)]
                    if len(vocal_chunk) > 100:
                        vocal_db = vocal_chunk.dBFS
                        if vocal_db < -55.0 and not str(seg.get("text", "")).strip():
                            self.log(f"Segment {seg_id}/{len(segments)} [{start:.2f}s - {end:.2f}s] filtered out as silent hallucination (Vocal energy: {vocal_db:.1f} dB).", "warning")
                            return None
                
                target_dur = end - start
                raw_text_to_speak = str(seg.get("translation", ""))
                
                # Strip bracketed non-verbal sound annotations like [Silence], [Music], [ការមិនមានបញ្ចេញសំឡេង]
                text_to_speak = re.sub(r"\[.*?\]", "", raw_text_to_speak)
                text_to_speak = re.sub(r"\(.*?\)", "", text_to_speak).strip()
                
                # Clean leading colon/punctuation and spaces
                text_to_speak = text_to_speak.lstrip(":-; ").strip()
                
                # Deduplicate repetitive phrase loops
                text_to_speak = re.sub(r'(.+?[\?\!\.\,\s]+)\1+', r'\1', text_to_speak).strip()
                
                # If stripping parens stripped the whole sentence, retain inner text
                if not text_to_speak and raw_text_to_speak:
                    fallback_text = re.sub(r"^[\[\(（\s]+|[\]\)）\s]+$", "", raw_text_to_speak).strip()
                    if re.search(r"[\u1780-\u17FF\w]", fallback_text):
                        text_to_speak = fallback_text

                if not text_to_speak or text_to_speak in ["...", "…", "-", ""]:
                    self.log(f"Segment {seg_id}/{len(segments)} is empty or non-verbal sound ({raw_text_to_speak}). Skipping voice synthesis.", "muted")
                    return None
                
                # Select voice gender and role
                gender_str = str(seg.get("gender", "female")).lower().strip()
                role_str = str(seg.get("speaker_role", "lead_actor")).lower().strip()
                is_female = "female" in gender_str or "woman" in gender_str or "girl" in gender_str
                is_male = ("male" in gender_str and not is_female) or "man" in gender_str or "boy" in gender_str
                is_narrator_segment = "narrator" in role_str or "third" in role_str or "secondary" in role_str
                if self.settings.get("movie_recap_mode", False):
                    is_narrator_segment = True
                    is_male = True
                    is_female = False
                
                if self.settings.get("voice_gender") == "male":
                    is_male_segment = True
                    selected_voice = male_voice
                elif self.settings.get("voice_gender") == "female":
                    is_male_segment = False
                    selected_voice = female_voice
                else:
                    is_male_segment = is_male if (is_male or is_female) else False
                    selected_voice = male_voice if is_male_segment else female_voice
                
                emotion = str(seg.get("emotion", "dramatic")).lower().strip()
                gender_label = "narrator male" if (is_male_segment and is_narrator_segment) else ("narrator female" if is_narrator_segment else ("male" if is_male_segment else "female"))
                self.log(f"Dubbing segment {seg_id}/{len(segments)} [{start:.2f}s - {end:.2f}s] ({gender_label}) [{emotion}]: {text_to_speak}", "muted")
                
                raw_ext = "mp3" if tts_eng == "edge-tts" else "wav"
                temp_tts_path = os.path.join(self.temp_dir, f"seg_{seg_id}_raw.{raw_ext}")
                temp_stretch_path = os.path.join(self.temp_dir, f"seg_{seg_id}_stretched.wav")

                # Calculate required duration & look ahead to next segment to allow natural pacing
                target_dur = max(0.1, end - start)
                next_start = original_duration_s
                if idx + 1 < len(segments):
                    next_start = float(segments[idx + 1].get("start", original_duration_s))
                max_allowed_dur = max(target_dur, next_start - start - 0.15)

                pre_rate = 1.0
                if not movie_recap_mode and tts_eng in ("edge-tts", "edge-openvoice"):
                    est_cps = 9.5 if lang_code in ("km-KH", "th-TH", "my-MM") else 13.5
                    est_dur = len(text_to_speak) / est_cps
                    if est_dur > max_allowed_dur * 1.08:
                        pre_rate = min(1.30, max(1.0, est_dur / max_allowed_dur))
                
                # SYNTHESIZE WITH SEMAPHORE TO PREVENT API RATE LIMITING
                async with sem:
                    await self.synthesize_tts(text_to_speak, selected_voice, temp_tts_path, speed_ratio=pre_rate, is_male=is_male_segment, is_narrator=is_narrator_segment, emotion=emotion, age_group=seg.get("age_group", "adult"))
                
                if not os.path.exists(temp_tts_path) or os.path.getsize(temp_tts_path) == 0:
                    self.log(f"TTS audio output empty for segment {seg_id}. Skipping.", "warning")
                    return None

                # Get actual TTS duration
                tts_segment = AudioSegment.from_wav(temp_tts_path) if raw_ext == "wav" else AudioSegment.from_file(temp_tts_path)

                if movie_recap_mode:
                    # In recap mode, allow narration to play naturally at 1.0x without chopping or artificial speedup
                    final_seg = tts_segment
                else:
                    max_user_speed = float(self.settings.get("max_speech_speed", 1.5))
                    # Dynamic auto-fit: If user hasn't set an aggressive speed limit (or has 1.0 baseline),
                    # allow dynamic auto-stretch up to 1.45x so character dialogue never gets brutally cut off
                    max_fit_speed = max(1.45, max_user_speed if max_user_speed > 1.05 else 1.45)
                    
                    tts_dur = len(tts_segment) / 1000.0
                    
                    # Calculate required speed
                    if tts_dur <= target_dur:
                        chosen_speed = 1.0
                    elif tts_dur <= max_allowed_dur:
                        # Fits comfortably in the available gap before next sentence
                        chosen_speed = 1.0
                    else:
                        ideal_speed = tts_dur / max_allowed_dur
                        chosen_speed = max(1.0, min(max_fit_speed, ideal_speed))
                    
                    # Speed stretch logic: skip FFmpeg stretch if chosen_speed is 1.0
                    if abs(chosen_speed - 1.0) < 0.05:
                        final_seg = tts_segment
                    else:
                        try:
                            await self.async_time_stretch_audio(temp_tts_path, temp_stretch_path, chosen_speed)
                            final_seg = AudioSegment.from_wav(temp_stretch_path)
                        except Exception as stretch_err:
                            self.log(f"Stretching failed for segment {seg_id}: {stretch_err}. Using raw voice.", "warning")
                            final_seg = tts_segment
                    
                    # Boost volume for whispered segments to make sure they are audible over background music
                    if emotion == "whisper" or "whisper" in emotion:
                        self.log(f"Boosting volume of whispered segment {seg_id} by +12 dB.", "info")
                        final_seg = final_seg + 12
                    
                    # Enforce boundary: dubbed audio must not collide heavily with next dialogue.
                    # Allow a soft conversational margin (up to 0.30s) so characters finish speaking naturally instead of being chopped in half mid-word.
                    max_duration_limit = max_allowed_dur + 0.30
                    final_dur_s = len(final_seg) / 1000.0
                    if final_dur_s > max_duration_limit:
                        cut_ms = int(max_duration_limit * 1000.0)
                        fade_ms = min(150, int(cut_ms * 0.1))
                        final_seg = final_seg[:cut_ms].fade_out(duration=fade_ms)
                
                # Save a permanent copy for preview
                try:
                    seg_preview_dir = os.path.join(get_subtitles_dir(), str(video_id))
                    os.makedirs(seg_preview_dir, exist_ok=True)
                    seg_preview_path = os.path.join(seg_preview_dir, f"seg_{seg_id}.wav")
                    final_seg.export(seg_preview_path, format="wav")
                    self.log(f"Segment {seg_id} audio preview saved for playback.", "muted")
                    
                    meta_path = os.path.join(seg_preview_dir, f"seg_{seg_id}.meta")
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump({
                            "translation": seg.get("translation", ""),
                            "gender": seg.get("gender", ""),
                            "speaker_role": seg.get("speaker_role", ""),
                            "emotion": seg.get("emotion", "")
                        }, mf, ensure_ascii=False, indent=2)
                except Exception as preview_save_err:
                    self.log(f"WARNING: Failed to save segment {seg_id} audio preview: {preview_save_err}", "warning")
                    logger.error(f"Failed to save segment preview audio: {preview_save_err}")
                
                # Return the start time and the final audio segment
                return (start, final_seg)
            
            # Execute all tasks concurrently
            tasks = [process_segment(idx, seg) for idx, seg in enumerate(segments)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            if movie_recap_mode:
                # Dynamic sequential timeline layout for movie recap mode
                self.log("Laying out movie recap narrative timeline with cinematic storytelling pauses...", "info")
                valid_items = []
                for idx, (seg, res) in enumerate(zip(segments, results)):
                    if res and not isinstance(res, Exception):
                        valid_items.append((seg, res[1]))
                
                if valid_items:
                    durs = [len(audio) / 1000.0 for _, audio in valid_items]
                    total_audio_s = sum(durs)
                    max_avail_s = max(5.0, original_duration_s - 1.0)
                    
                    global_stretch = 1.0
                    if total_audio_s > max_avail_s:
                        global_stretch = min(1.25, total_audio_s / max_avail_s)
                        self.log(f"Recap narration slightly exceeds timeline ({total_audio_s:.1f}s > {max_avail_s:.1f}s). Applying gentle auto-fit {global_stretch:.2f}x...", "info")
                    
                    effective_durs = [d / global_stretch for d in durs]
                    total_eff = sum(effective_durs)
                    remaining_silence = max(0.0, max_avail_s - total_eff)
                    gap = min(2.5, remaining_silence / (len(valid_items) + 1)) if len(valid_items) > 0 else 0.5
                    
                    curr_pos_s = max(0.5, gap)
                    for item_idx, (seg, raw_audio) in enumerate(valid_items):
                        eff_dur = effective_durs[item_idx]
                        if abs(global_stretch - 1.0) >= 0.04:
                            temp_recap_str = os.path.join(self.temp_dir, f"recap_str_{seg.get('id', item_idx+1)}.wav")
                            try:
                                seg_raw_path = os.path.join(self.temp_dir, f"seg_{seg.get('id', item_idx+1)}_raw.mp3")
                                if os.path.exists(seg_raw_path):
                                    await self.async_time_stretch_audio(seg_raw_path, temp_recap_str, global_stretch)
                                    placed_audio = AudioSegment.from_wav(temp_recap_str)
                                else:
                                    placed_audio = raw_audio
                            except Exception:
                                placed_audio = raw_audio
                        else:
                            placed_audio = raw_audio
                            
                        start_s = round(curr_pos_s, 2)
                        end_s = round(start_s + (len(placed_audio) / 1000.0), 2)
                        
                        seg["start"] = start_s
                        seg["end"] = end_s
                        
                        pos_ms = int(start_s * 1000.0)
                        if "recap_placements" not in locals():
                            recap_placements = []
                        recap_placements.append((pos_ms, placed_audio))
                        
                        try:
                            seg_preview_dir = os.path.join(get_subtitles_dir(), str(video_id))
                            seg_preview_path = os.path.join(seg_preview_dir, f"seg_{seg.get('id', item_idx+1)}.wav")
                            placed_audio.export(seg_preview_path, format="wav")
                        except Exception:
                            pass
                            
                        curr_pos_s = end_s + max(0.4, gap)
                    if "recap_placements" in locals() and recap_placements:
                        dubbed_audio = fast_overlay_segments(original_duration_ms, recap_placements)
            else:
                # Overlay results onto dubbed timeline in a single O(N) NumPy pass
                placements = []
                for result in results:
                    if isinstance(result, Exception):
                        self.log(f"Segment processing failed due to exception: {result}", "error")
                        continue
                    if result is None:
                        continue
                    start, final_seg = result
                    sync_offset_ms = float(self.settings.get("audio_sync_offset_ms", 0))
                    target_pos_ms = max(0, int((start * 1000.0) + sync_offset_ms))
                    placements.append((target_pos_ms, final_seg))
                dubbed_audio = fast_overlay_segments(original_duration_ms, placements)
            

            
            # Save dubbed voice track
            dubbed_voice_wav = os.path.join(self.temp_dir, "dubbed_voice.wav")
            dubbed_audio.export(dubbed_voice_wav, format="wav")
            self.log("Dubbed voice timeline completed.", "success")
            
            # 5. Audio Mixing
            self.update_progress(80, "Mixing audio stems...")
            self.log("Step 4: Mixing background music and dubbed vocals...", "info")
            
            mixer = self.settings.get("mixer", {"drums": True, "bass": True, "melody": True})
            
            def is_stem_active(stem):
                try:
                    return bool(mixer.get(stem) and float(mixer.get(f"{stem}_vol", 100)) > 0)
                except (ValueError, TypeError):
                    return bool(mixer.get(stem))
            
            any_bg_enabled = is_stem_active("drums") or is_stem_active("bass") or is_stem_active("melody")
            
            # Mix background stems if Demucs succeeded
            bg_mixed = AudioSegment.silent(duration=original_duration_ms)
            stems_mixed = False
            
            if not any_bg_enabled:
                self.log("All background sound options are turned off. Dubbed video will contain vocals only (pure speech).", "info")
            elif vocals_path and os.path.exists(vocals_path):
                # Mix stems with volume controls
                import math
                
                def apply_stem_volume(audio, vol_pct):
                    try:
                        vol_pct = float(vol_pct)
                    except (ValueError, TypeError):
                        vol_pct = 100.0
                    if vol_pct <= 0:
                        return None
                    if abs(vol_pct - 100.0) < 1.0:
                        return audio
                    gain_db = 20.0 * math.log10(vol_pct / 100.0)
                    return audio + gain_db

                if is_stem_active("drums") and drums_path and os.path.exists(drums_path):
                    vol = mixer.get("drums_vol", 100)
                    stem_audio = apply_stem_volume(AudioSegment.from_wav(drums_path), vol)
                    if stem_audio is not None:
                        bg_mixed = bg_mixed.overlay(stem_audio)
                        stems_mixed = True
                        
                if is_stem_active("bass") and bass_path and os.path.exists(bass_path):
                    vol = mixer.get("bass_vol", 100)
                    stem_audio = apply_stem_volume(AudioSegment.from_wav(bass_path), vol)
                    if stem_audio is not None:
                        bg_mixed = bg_mixed.overlay(stem_audio)
                        stems_mixed = True
                        
                if is_stem_active("melody") and melody_path and os.path.exists(melody_path):
                    vol = mixer.get("melody_vol", 100)
                    stem_audio = apply_stem_volume(AudioSegment.from_wav(melody_path), vol)
                    if stem_audio is not None:
                        bg_mixed = bg_mixed.overlay(stem_audio)
                        stems_mixed = True
                
                # Fallback to no_vocals.wav ONLY if the user actually enabled background music
                if not stems_mixed:
                    no_vocals_path = os.path.join(track_dir, "no_vocals.wav")
                    if os.path.exists(no_vocals_path):
                        bg_mixed = AudioSegment.from_wav(no_vocals_path)
                        stems_mixed = True
                        self.log("Loaded isolated instrumental background track (no_vocals.wav).", "info")
                    else:
                        self.log("No background stems selected. Dubbed video will contain vocals only.", "info")
            else:
                # Demucs failed, use original audio as background ONLY if background sound is enabled
                self.log("No stems available. Preparing original audio track for dynamic mixing...", "info")
                bg_mixed = audio_segment
            
            # Apply dynamic audio ducking to the background track during active dialogue segments (only if background sound is enabled)
            if segments and any_bg_enabled:
                # If stems were separated, duck background slightly (-8dB) so it remains rich.
                # If Demucs failed, duck background heavily (-35dB) to drown out the original voices.
                # For movie recap mode, duck -14dB so narrator voice cuts cleanly over background music.
                if movie_recap_mode:
                    duck_db = 14 if (vocals_path and os.path.exists(vocals_path)) else 35
                else:
                    duck_db = 8 if (vocals_path and os.path.exists(vocals_path)) else 35
                self.log(f"Applying dynamic audio ducking to background music (-{duck_db} dB)...", "info")
                
                # Merge overlapping/close segments
                sorted_segs = sorted(segments, key=lambda x: x["start"])
                merged_intervals = []
                for seg in sorted_segs:
                    s_ms = max(0, int(seg["start"] * 1000.0) - 300)
                    e_ms = min(original_duration_ms, int(seg["end"] * 1000.0) + 300)
                    if not merged_intervals:
                        merged_intervals.append([s_ms, e_ms])
                    else:
                        last = merged_intervals[-1]
                        if s_ms <= last[1]:
                            last[1] = max(last[1], e_ms)
                        else:
                            merged_intervals.append([s_ms, e_ms])

                # Process ducking with crossfades in a single O(N) NumPy pass
                bg_mixed = fast_duck_audio_segment(bg_mixed, merged_intervals, duck_db)

            # Final Audio Mix
            final_audio = bg_mixed.overlay(dubbed_audio)
            final_audio_path = os.path.join(self.temp_dir, "final_mixed_audio.wav")
            final_audio.export(final_audio_path, format="wav")
            
            # Save a copy of the clean background audio track for future re-dubs
            try:
                seg_preview_dir = os.path.join(get_subtitles_dir(), str(video_id))
                os.makedirs(seg_preview_dir, exist_ok=True)
                bg_cache_path = os.path.join(seg_preview_dir, "bg_mixed.wav")
                if any_bg_enabled and vocals_path and os.path.exists(vocals_path) and stems_mixed:
                    self.log("Saving clean background music track to cache for future re-dubs...", "info")
                    bg_mixed.export(bg_cache_path, format="wav")
                else:
                    if os.path.exists(bg_cache_path):
                        os.remove(bg_cache_path)
            except Exception as bg_save_err:
                logger.error(f"Failed to handle cached background audio track: {bg_save_err}")
            
            # 6. Re-remux with original video
            self.update_progress(90, "Re-muxing video...")
            self.log("Step 5: Remuxing new audio track into video...", "info")
            
            # Determine output directory & clean naming
            clean_name = name_without_ext
            for pfx in ["[Dubbed]_", "[Recap]_"]:
                if clean_name.startswith(pfx):
                    clean_name = clean_name[len(pfx):]
            
            tag = "[Recap]" if self.settings.get("movie_recap_mode", False) else "[Dubbed]"
            parent_dir = os.path.dirname(os.path.abspath(video_path))
            
            if self.settings.get("output_type") == "custom" and self.settings.get("custom_output_path"):
                out_dir = self.settings.get("custom_output_path")
            elif os.path.basename(parent_dir).lower() == "dubbed_output":
                out_dir = parent_dir
            else:
                out_dir = os.path.join(parent_dir, "Dubbed_Output")
            
            os.makedirs(out_dir, exist_ok=True)
            output_video_path = os.path.join(out_dir, f"{tag}_{clean_name}.mp4")
            
            remux_cmd = [
                get_ffmpeg_cmd(), "-y",
                "-i", video_path,
                "-i", final_audio_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-max_interleave_delta", "0",
                "-avoid_negative_ts", "make_zero",
                "-shortest",
                output_video_path
            ]
            await self.async_run_cmd(remux_cmd, "Video Remuxing")
            
            # Save subtitle segments to a permanent JSON file in subtitles/ folder
            try:
                subtitles_dir = get_subtitles_dir()
                os.makedirs(subtitles_dir, exist_ok=True)
                subtitles_json_path = os.path.join(subtitles_dir, f"{video_id}.json")
                with open(subtitles_json_path, "w", encoding="utf-8") as sf:
                    json.dump(segments, sf, ensure_ascii=False, indent=4)
                self.log("Saved subtitle segments to JSON database.", "info")
            except Exception as save_err:
                logger.error(f"Failed to save subtitles: {save_err}")

            # YouTube Multi-Audio Tracks Export (.m4a stems)
            try:
                await self.export_youtube_audio_tracks(
                    video_path=video_path,
                    video_id=video_id,
                    clean_name=clean_name,
                    out_dir=out_dir,
                    segments=segments,
                    bg_mixed=bg_mixed,
                    any_bg_enabled=any_bg_enabled,
                    original_duration_ms=original_duration_ms,
                    original_duration_s=original_duration_s,
                    vocals_wav=vocals_wav
                )
            except Exception as yt_err:
                self.log(f"Warning: YouTube Multi-Audio export encountered an issue: {yt_err}", "warning")
                logger.error(f"YouTube Multi-Audio export error: {yt_err}")
                
            self.update_progress(100, "Completed")
            self.log(f"[SUCCESS] Video processed successfully: {os.path.basename(output_video_path)}", "success")

            # Optional cleanup: Delete original source video if requested
            if self.settings.get("delete_original_file", False):
                try:
                    if (
                        output_video_path
                        and os.path.exists(output_video_path)
                        and os.path.getsize(output_video_path) > 1024
                        and os.path.abspath(video_path) != os.path.abspath(output_video_path)
                        and os.path.exists(video_path)
                    ):
                        os.remove(video_path)
                        self.log(f"[CLEANUP] Deleted original video: {os.path.basename(video_path)}", "warning")
                        logger.info(f"Deleted original file after dubbing: {video_path}")
                except Exception as del_err:
                    self.log(f"[CLEANUP ERROR] Failed to delete original video: {str(del_err)}", "error")
                    logger.error(f"Failed to delete original file {video_path}: {del_err}")
            
        except Exception as e:
            self.update_progress(100, "Error")
            self.log(f"[ERROR] Failed to process video {filename}: {str(e)}", "error")
            raise e
        finally:
            # Clean up cached Gemini File API uploads and local temp folder
            try:
                await self.cleanup_router_files()
            except Exception:
                pass
            try:
                if self.temp_dir and os.path.exists(self.temp_dir):
                    # We keep logs but delete heavy wav stems
                    shutil.rmtree(self.temp_dir)
            except Exception as clean_err:
                logger.warning(f"Failed to clean up temp files: {clean_err}")

    async def re_dub_video(self, video_path, video_id, segments):
        """Re-runs the TTS synthesis, mixing, and remuxing pipeline using modified segments, skipping Whisper/Gemini."""
        self.update_progress(5, "Initializing Re-Dubbing...")
        
        filename = os.path.basename(video_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Temp dir inside workspace
        self.temp_dir = os.path.join(os.path.dirname(video_path), f"temp_dub_re_{video_id}_{uuid.uuid4().hex}")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        temp_audio = os.path.join(self.temp_dir, "original_audio.wav")
        
        try:
            # 1. Extract Audio from Video
            self.log(f"Step 1: Extracting audio from {filename}...", "info")
            self.update_progress(15, "Extracting audio...")
            extract_cmd = [
                get_ffmpeg_cmd(), "-y",
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                temp_audio
            ]
            await self.async_run_cmd(extract_cmd, "Audio Extraction")
            
            # Load original audio duration
            audio_segment = AudioSegment.from_wav(temp_audio)
            original_duration_ms = len(audio_segment)
            original_duration_s = original_duration_ms / 1000.0
            
            # 2. Text-To-Speech Synthesis
            self.update_progress(40, "Synthesizing corrected voice lines...")
            self.log("Step 2: Synthesizing dubbed voice lines...", "info")
            
            dubbed_audio = AudioSegment.silent(duration=original_duration_ms)
            
            # Target voice mapping
            lang_code = self.settings.get("target_lang", "km-KH")
            male_voice, female_voice = VOICES.get(lang_code, VOICES["km-KH"])
            
            seg_preview_dir = os.path.join(get_subtitles_dir(), str(video_id))
            tts_eng = self.settings.get("tts_engine", "edge-tts")
            sem = asyncio.Semaphore(6 if tts_eng in ("edge-tts", "edge-openvoice") else 4)
            
            async def process_re_dub_segment(idx, seg):
                seg_id = seg.get("id") or (idx + 1)
                seg["id"] = seg_id
                start = float(seg["start"])
                end = float(seg["end"])
                
                # Check bounds
                if start >= original_duration_s:
                    return None
                if end > original_duration_s:
                    end = original_duration_s
                
                target_dur = max(0.1, end - start)
                
                seg_preview_path = os.path.join(seg_preview_dir, f"seg_{seg_id}.wav")
                meta_path = os.path.join(seg_preview_dir, f"seg_{seg_id}.meta")
                
                # Check if meta file matches current segment properties
                cache_valid = False
                if os.path.exists(seg_preview_path) and os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        if (meta.get("translation") == seg.get("translation") and
                            meta.get("gender") == seg.get("gender") and
                            meta.get("speaker_role") == seg.get("speaker_role") and
                            meta.get("emotion") == seg.get("emotion")):
                            cache_valid = True
                    except Exception:
                        cache_valid = False
                
                if cache_valid:
                    self.log(f"Reusing existing audio for segment {seg_id}...", "info")
                    try:
                        final_seg = AudioSegment.from_wav(seg_preview_path)
                        return (start, final_seg)
                    except Exception as load_err:
                        self.log(f"Failed to load cached WAV for segment {seg_id} ({load_err}). Re-synthesizing...", "warning")
                        final_seg = None
                else:
                    final_seg = None
                
                if final_seg is None:
                    raw_text_to_speak = str(seg.get("translation", ""))
                    
                    # Strip bracketed annotations
                    text_to_speak = re.sub(r"\[.*?\]", "", raw_text_to_speak)
                    text_to_speak = re.sub(r"\(.*?\)", "", text_to_speak).strip()
                    text_to_speak = text_to_speak.lstrip(":-; ").strip()
                    text_to_speak = re.sub(r'(.+?[\?\!\.\,\s]+)\1+', r'\1', text_to_speak).strip()
                    
                    # If stripping parens stripped the whole sentence, retain inner text
                    if not text_to_speak and raw_text_to_speak:
                        fallback_text = re.sub(r"^[\[\(（\s]+|[\]\)）\s]+$", "", raw_text_to_speak).strip()
                        if re.search(r"[\u1780-\u17FF\w]", fallback_text):
                            text_to_speak = fallback_text

                    if not text_to_speak or text_to_speak in ["...", "…", "-", ""]:
                        return None
                    
                    # Select voice gender and role
                    gender_str = str(seg.get("gender", "female")).lower().strip()
                    role_str = str(seg.get("speaker_role", "lead_actor")).lower().strip()
                    is_female = "female" in gender_str or "woman" in gender_str or "girl" in gender_str
                    is_male = ("male" in gender_str and not is_female) or "man" in gender_str or "boy" in gender_str
                    is_narrator_segment = "narrator" in role_str or "third" in role_str or "secondary" in role_str
                    if self.settings.get("movie_recap_mode", False):
                        is_narrator_segment = True
                        is_male = True
                        is_female = False
                    
                    if self.settings.get("voice_gender") == "male":
                        is_male_segment = True
                        selected_voice = male_voice
                    elif self.settings.get("voice_gender") == "female":
                        is_male_segment = False
                        selected_voice = female_voice
                    else:
                        is_male_segment = is_male if (is_male or is_female) else False
                        selected_voice = male_voice if is_male_segment else female_voice
                    
                    emotion = str(seg.get("emotion", "dramatic")).lower().strip()
                    self.log(f"Synthesizing segment {seg_id}/{len(segments)} [{start:.2f}s - {end:.2f}s]: {text_to_speak}", "muted")
                    
                    raw_ext = "mp3" if tts_eng == "edge-tts" else "wav"
                    temp_tts_path = os.path.join(self.temp_dir, f"seg_{seg_id}_raw.{raw_ext}")
                    temp_stretch_path = os.path.join(self.temp_dir, f"seg_{seg_id}_stretched.wav")

                    next_start = original_duration_s
                    if idx + 1 < len(segments):
                        next_start = float(segments[idx + 1].get("start", original_duration_s))
                    max_allowed_dur = max(target_dur, next_start - start - 0.15)

                    pre_rate = 1.0
                    if not self.settings.get("movie_recap_mode", False) and tts_eng in ("edge-tts", "edge-openvoice"):
                        est_cps = 9.5 if lang_code in ("km-KH", "th-TH", "my-MM") else 13.5
                        est_dur = len(text_to_speak) / est_cps
                        if est_dur > max_allowed_dur * 1.08:
                            pre_rate = min(1.30, max(1.0, est_dur / max_allowed_dur))
                    
                    # Synthesize
                    async with sem:
                        await self.synthesize_tts(text_to_speak, selected_voice, temp_tts_path, speed_ratio=pre_rate, is_male=is_male_segment, is_narrator=is_narrator_segment, emotion=emotion, age_group=seg.get("age_group", "adult"))
                    
                    if not os.path.exists(temp_tts_path) or os.path.getsize(temp_tts_path) == 0:
                        return None

                    # Get actual TTS duration
                    tts_segment = AudioSegment.from_wav(temp_tts_path) if raw_ext == "wav" else AudioSegment.from_file(temp_tts_path)
                    
                    if self.settings.get("movie_recap_mode", False):
                        # In recap mode, allow narration to play naturally at 1.0x without chopping or artificial speedup
                        final_seg = tts_segment
                    else:
                        tts_dur = len(tts_segment) / 1000.0
                        
                        # Speed stretch logic
                        if tts_dur <= target_dur:
                            chosen_speed = 1.0
                        elif tts_dur <= max_allowed_dur:
                            chosen_speed = 1.0
                        else:
                            ideal_speed = tts_dur / max_allowed_dur
                            max_user_speed = float(self.settings.get("max_speech_speed", 1.5))
                            max_fit_speed = max(1.45, max_user_speed if max_user_speed > 1.05 else 1.45)
                            chosen_speed = max(1.0, min(max_fit_speed, ideal_speed))
                        
                        if abs(chosen_speed - 1.0) < 0.05:
                            final_seg = tts_segment
                        else:
                            try:
                                await self.async_time_stretch_audio(temp_tts_path, temp_stretch_path, chosen_speed)
                                final_seg = AudioSegment.from_wav(temp_stretch_path)
                            except Exception as stretch_err:
                                self.log(f"Stretching failed: {stretch_err}. Using raw voice.", "warning")
                                final_seg = tts_segment
                        
                        if emotion == "whisper" or "whisper" in emotion:
                            final_seg = final_seg + 12
                        
                        # Enforce limit with soft conversational breathing margin
                        max_duration_limit = max_allowed_dur + 0.30
                        final_dur_s = len(final_seg) / 1000.0
                        if final_dur_s > max_duration_limit:
                            cut_ms = int(max_duration_limit * 1000.0)
                            fade_ms = min(150, int(cut_ms * 0.1))
                            final_seg = final_seg[:cut_ms].fade_out(duration=fade_ms)
                    
                    # Save a permanent copy for preview in subtitles/{video_id}/ directory
                    try:
                        os.makedirs(seg_preview_dir, exist_ok=True)
                        final_seg.export(seg_preview_path, format="wav")
                        
                        # Save metadata cache file
                        meta_path = os.path.join(seg_preview_dir, f"seg_{seg_id}.meta")
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            json.dump({
                                "translation": seg.get("translation", ""),
                                "gender": seg.get("gender", ""),
                                "speaker_role": seg.get("speaker_role", ""),
                                "emotion": seg.get("emotion", "")
                            }, mf, ensure_ascii=False, indent=2)
                    except Exception as preview_save_err:
                        logger.error(f"Failed to save segment preview audio: {preview_save_err}")
                    
                    return (start, final_seg)
                return (start, final_seg)

            re_dub_tasks = [process_re_dub_segment(idx, seg) for idx, seg in enumerate(segments)]
            re_dub_results = await asyncio.gather(*re_dub_tasks, return_exceptions=True)
            
            if self.settings.get("movie_recap_mode", False):
                self.log("Laying out movie recap narrative timeline with cinematic storytelling pauses...", "info")
                valid_items = []
                for idx, (seg, res) in enumerate(zip(segments, re_dub_results)):
                    if res and not isinstance(res, Exception):
                        valid_items.append((seg, res[1]))
                
                if valid_items:
                    durs = [len(audio) / 1000.0 for _, audio in valid_items]
                    total_audio_s = sum(durs)
                    max_avail_s = max(5.0, original_duration_s - 1.0)
                    
                    global_stretch = 1.0
                    if total_audio_s > max_avail_s:
                        global_stretch = min(1.25, total_audio_s / max_avail_s)
                        self.log(f"Recap narration slightly exceeds timeline ({total_audio_s:.1f}s > {max_avail_s:.1f}s). Applying gentle auto-fit {global_stretch:.2f}x...", "info")
                    
                    effective_durs = [d / global_stretch for d in durs]
                    total_eff = sum(effective_durs)
                    remaining_silence = max(0.0, max_avail_s - total_eff)
                    gap = min(2.5, remaining_silence / (len(valid_items) + 1)) if len(valid_items) > 0 else 0.5
                    
                    curr_pos_s = max(0.5, gap)
                    for item_idx, (seg, raw_audio) in enumerate(valid_items):
                        eff_dur = effective_durs[item_idx]
                        if abs(global_stretch - 1.0) >= 0.04:
                            temp_recap_str = os.path.join(self.temp_dir, f"recap_str_{seg.get('id', item_idx+1)}.wav")
                            try:
                                seg_raw_path = os.path.join(self.temp_dir, f"seg_{seg.get('id', item_idx+1)}_raw.mp3")
                                if os.path.exists(seg_raw_path):
                                    await self.async_time_stretch_audio(seg_raw_path, temp_recap_str, global_stretch)
                                    placed_audio = AudioSegment.from_wav(temp_recap_str)
                                else:
                                    placed_audio = raw_audio
                            except Exception:
                                placed_audio = raw_audio
                        else:
                            placed_audio = raw_audio
                            
                        start_s = round(curr_pos_s, 2)
                        end_s = round(start_s + (len(placed_audio) / 1000.0), 2)
                        
                        seg["start"] = start_s
                        seg["end"] = end_s
                        
                        pos_ms = int(start_s * 1000.0)
                        if "redub_recap_placements" not in locals():
                            redub_recap_placements = []
                        redub_recap_placements.append((pos_ms, placed_audio))
                        
                        try:
                            seg_preview_dir = os.path.join(get_subtitles_dir(), str(video_id))
                            seg_preview_path = os.path.join(seg_preview_dir, f"seg_{seg.get('id', item_idx+1)}.wav")
                            placed_audio.export(seg_preview_path, format="wav")
                        except Exception:
                            pass
                            
                        curr_pos_s = end_s + max(0.4, gap)
                    if "redub_recap_placements" in locals() and redub_recap_placements:
                        dubbed_audio = fast_overlay_segments(original_duration_ms, redub_recap_placements)
            else:
                redub_placements = []
                for res in re_dub_results:
                    if isinstance(res, Exception):
                        self.log(f"Segment re-dub failed: {res}", "error")
                        continue
                    if res is None:
                        continue
                    seg_start, final_seg = res
                    sync_offset_ms = float(self.settings.get("audio_sync_offset_ms", 0))
                    target_pos_ms = max(0, int((seg_start * 1000.0) + sync_offset_ms))
                    redub_placements.append((target_pos_ms, final_seg))
                dubbed_audio = fast_overlay_segments(original_duration_ms, redub_placements)
            


            
            # Save dubbed voice track
            dubbed_voice_wav = os.path.join(self.temp_dir, "dubbed_voice.wav")
            dubbed_audio.export(dubbed_voice_wav, format="wav")
            
            # 3. Audio Mixing with Dynamic Ducking
            self.update_progress(70, "Mixing audio stems...")
            self.log("Step 3: Mixing background music and dubbed vocals...", "info")
            
            mixer = self.settings.get("mixer", {"drums": True, "bass": True, "melody": True})
            def is_stem_active(stem):
                try:
                    return bool(mixer.get(stem) and float(mixer.get(f"{stem}_vol", 100)) > 0)
                except (ValueError, TypeError):
                    return bool(mixer.get(stem))
            any_bg_enabled = is_stem_active("drums") or is_stem_active("bass") or is_stem_active("melody")
            has_clean_bg = False
            
            if not any_bg_enabled:
                self.log("All background sound options are turned off. Dubbed video will contain vocals only (pure speech).", "info")
                bg_mixed = AudioSegment.silent(duration=original_duration_ms)
            else:
                # Check if we have a permanently saved clean background track
                bg_mixed_wav_path = os.path.join(get_subtitles_dir(), str(video_id), "bg_mixed.wav")
                if os.path.exists(bg_mixed_wav_path):
                    self.log("Loading clean background music track from cache...", "info")
                    try:
                        bg_mixed = AudioSegment.from_wav(bg_mixed_wav_path)
                        has_clean_bg = True
                    except Exception as load_bg_err:
                        self.log(f"Failed to load cached background track: {load_bg_err}. Falling back to original audio.", "warning")
                        bg_mixed = audio_segment
                        has_clean_bg = False
                else:
                    self.log("No cached background music track found. Using original audio with ducking.", "warning")
                    bg_mixed = audio_segment
                    has_clean_bg = False
            
            # If we don't have a clean background track, we duck the original audio heavily (-35 dB) to completely drown out the original vocals.
            # If we do have a clean background track, we duck it slightly (-8 dB) to make speech clearer.
            # In movie recap mode, duck -14 dB so narrator cuts cleanly over background music.
            if self.settings.get("movie_recap_mode", False):
                duck_db = 14 if has_clean_bg else 35
            else:
                duck_db = 8 if has_clean_bg else 35
            
            if segments and any_bg_enabled:
                self.log(f"Applying dynamic audio ducking to background music (-{duck_db} dB)...", "info")
                # Merge overlapping/close segments
                sorted_segs = sorted(segments, key=lambda x: float(x["start"]))
                merged_intervals = []
                for seg in sorted_segs:
                    s_ms = max(0, int(float(seg["start"]) * 1000.0) - 300)
                    e_ms = min(original_duration_ms, int(float(seg["end"]) * 1000.0) + 300)
                    if not merged_intervals:
                        merged_intervals.append([s_ms, e_ms])
                    else:
                        last = merged_intervals[-1]
                        if s_ms <= last[1]:
                            last[1] = max(last[1], e_ms)
                        else:
                            merged_intervals.append([s_ms, e_ms])

                bg_mixed = fast_duck_audio_segment(bg_mixed, merged_intervals, duck_db)
            
            final_audio = bg_mixed.overlay(dubbed_audio)
            final_audio_path = os.path.join(self.temp_dir, "final_mixed_audio.wav")
            final_audio.export(final_audio_path, format="wav")
            
            # 4. Re-remux with original video
            self.update_progress(85, "Re-muxing video...")
            self.log("Step 4: Remuxing new audio track into video...", "info")
            
            # Determine output directory & clean naming
            clean_name = name_without_ext
            for pfx in ["[Dubbed]_", "[Recap]_"]:
                if clean_name.startswith(pfx):
                    clean_name = clean_name[len(pfx):]
            
            tag = "[Recap]" if self.settings.get("movie_recap_mode", False) else "[Dubbed]"
            parent_dir = os.path.dirname(os.path.abspath(video_path))
            
            if self.settings.get("output_type") == "custom" and self.settings.get("custom_output_path"):
                out_dir = self.settings.get("custom_output_path")
            elif os.path.basename(parent_dir).lower() == "dubbed_output":
                out_dir = parent_dir
            else:
                out_dir = os.path.join(parent_dir, "Dubbed_Output")
            
            os.makedirs(out_dir, exist_ok=True)
            output_video_path = os.path.join(out_dir, f"{tag}_{clean_name}.mp4")
            
            remux_cmd = [
                get_ffmpeg_cmd(), "-y",
                "-i", video_path,
                "-i", final_audio_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-max_interleave_delta", "0",
                "-avoid_negative_ts", "make_zero",
                "-shortest",
                output_video_path
            ]
            await self.async_run_cmd(remux_cmd, "Video Remuxing")
            
            # Save updated subtitles JSON database permanently as well
            try:
                subtitles_dir = get_subtitles_dir()
                os.makedirs(subtitles_dir, exist_ok=True)
                subtitles_json_path = os.path.join(subtitles_dir, f"{video_id}.json")
                with open(subtitles_json_path, "w", encoding="utf-8") as sf:
                    json.dump(segments, sf, ensure_ascii=False, indent=4)
            except Exception as se:
                logger.error(f"Failed to update subtitles json in re-dub: {se}")

            # YouTube Multi-Audio Tracks Export (.m4a stems)
            try:
                await self.export_youtube_audio_tracks(
                    video_path=video_path,
                    video_id=video_id,
                    clean_name=clean_name,
                    out_dir=out_dir,
                    segments=segments,
                    bg_mixed=bg_mixed,
                    any_bg_enabled=any_bg_enabled,
                    original_duration_ms=original_duration_ms,
                    original_duration_s=original_duration_s,
                    vocals_wav=None
                )
            except Exception as yt_err:
                self.log(f"Warning: YouTube Multi-Audio export encountered an issue: {yt_err}", "warning")
                logger.error(f"YouTube Multi-Audio export error: {yt_err}")
                
            self.update_progress(100, "Completed")
            self.log(f"[SUCCESS] Video re-dubbed successfully: {os.path.basename(output_video_path)}", "success")

            # Optional cleanup: Delete original source video if requested
            if self.settings.get("delete_original_file", False):
                try:
                    if (
                        output_video_path
                        and os.path.exists(output_video_path)
                        and os.path.getsize(output_video_path) > 1024
                        and os.path.abspath(video_path) != os.path.abspath(output_video_path)
                        and os.path.exists(video_path)
                    ):
                        os.remove(video_path)
                        self.log(f"[CLEANUP] Deleted original video: {os.path.basename(video_path)}", "warning")
                        logger.info(f"Deleted original file after re-dub: {video_path}")
                except Exception as del_err:
                    self.log(f"[CLEANUP ERROR] Failed to delete original video: {str(del_err)}", "error")
                    logger.error(f"Failed to delete original file {video_path}: {del_err}")
            
        except Exception as e:
            self.update_progress(100, "Error")
            self.log(f"[ERROR] Failed to re-dub video {filename}: {str(e)}", "error")
            raise e
        finally:
            # Clean up temp folder
            try:
                if self.temp_dir and os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
            except Exception as clean_err:
                logger.warning(f"Failed to clean up re-dub temp files: {clean_err}")

    async def re_dub_single_segment(self, video_path, video_id, target_seg_id, segments):
        """Re-synthesizes a single segment's voice and rebuilds the final dubbed audio/video."""
        self.update_progress(5, f"Re-dubbing segment {target_seg_id}...")
        
        filename = os.path.basename(video_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        self.temp_dir = os.path.join(os.path.dirname(video_path), f"temp_dub_re_{video_id}_{uuid.uuid4().hex}")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        temp_audio = os.path.join(self.temp_dir, "original_audio.wav")
        subtitles_dir = get_subtitles_dir()
        seg_preview_dir = os.path.join(subtitles_dir, str(video_id))
        os.makedirs(seg_preview_dir, exist_ok=True)
        
        try:
            # 1. Extract Audio from Video
            extract_cmd = [
                get_ffmpeg_cmd(), "-y",
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                temp_audio
            ]
            await self.async_run_cmd(extract_cmd, "Audio Extraction")
            
            # Load original audio duration
            audio_segment = AudioSegment.from_wav(temp_audio)
            original_duration_ms = len(audio_segment)
            original_duration_s = original_duration_ms / 1000.0
            
            # 2. Text-To-Speech Synthesis for target segment only
            lang_code = self.settings.get("target_lang", "km-KH")
            male_voice, female_voice = VOICES.get(lang_code, VOICES["km-KH"])
            
            # Find the target segment dict
            target_seg = None
            for seg in segments:
                if int(seg["id"]) == int(target_seg_id):
                    target_seg = seg
                    break
                    
            if not target_seg:
                raise Exception(f"Segment ID {target_seg_id} not found in segments database.")
                
            start = float(target_seg["start"])
            end = float(target_seg["end"])
            target_dur = max(0.1, end - start)
            raw_text_to_speak = str(target_seg.get("translation", ""))
            
            # Strip annotations
            text_to_speak = re.sub(r"\[.*?\]", "", raw_text_to_speak)
            text_to_speak = re.sub(r"\(.*?\)", "", text_to_speak).strip()
            text_to_speak = text_to_speak.lstrip(":-; ").strip()
            text_to_speak = re.sub(r'(.+?[\?\!\.\,\s]+)\1+', r'\1', text_to_speak).strip()
            
            if not text_to_speak and raw_text_to_speak:
                fallback_text = re.sub(r"^[\[\(（\s]+|[\]\)）\s]+$", "", raw_text_to_speak).strip()
                if re.search(r"[\u1780-\u17FF\w]", fallback_text):
                    text_to_speak = fallback_text

            if text_to_speak and text_to_speak not in ["...", "…", "-", ""]:
                # Determine voice and settings
                gender_str = str(target_seg.get("gender", "female")).lower().strip()
                role_str = str(target_seg.get("speaker_role", "lead_actor")).lower().strip()
                is_female = "female" in gender_str or "woman" in gender_str or "girl" in gender_str
                is_male = ("male" in gender_str and not is_female) or "man" in gender_str or "boy" in gender_str
                is_narrator_segment = "narrator" in role_str or "third" in role_str or "secondary" in role_str
                if self.settings.get("movie_recap_mode", False):
                    is_narrator_segment = True
                    is_male = True
                    is_female = False
                
                if self.settings.get("voice_gender") == "male":
                    is_male_segment = True
                    selected_voice = male_voice
                elif self.settings.get("voice_gender") == "female":
                    is_male_segment = False
                    selected_voice = female_voice
                else:
                    is_male_segment = is_male if (is_male or is_female) else False
                    selected_voice = male_voice if is_male_segment else female_voice
                
                emotion = str(target_seg.get("emotion", "dramatic")).lower().strip()
                
                tts_eng = self.settings.get("tts_engine", "edge-tts")
                raw_ext = "mp3" if tts_eng == "edge-tts" else "wav"
                temp_tts_path = os.path.join(self.temp_dir, f"seg_{target_seg_id}_raw.{raw_ext}")
                temp_stretch_path = os.path.join(self.temp_dir, f"seg_{target_seg_id}_stretched.wav")
                
                await self.synthesize_tts(text_to_speak, selected_voice, temp_tts_path, is_male=is_male_segment, is_narrator=is_narrator_segment, emotion=emotion, age_group=target_seg.get("age_group", "adult"))
                
                if os.path.exists(temp_tts_path) and os.path.getsize(temp_tts_path) > 0:
                    tts_segment = AudioSegment.from_wav(temp_tts_path) if raw_ext == "wav" else AudioSegment.from_file(temp_tts_path)
                    tts_dur = len(tts_segment) / 1000.0
                    
                    # Calculate speed stretch
                    next_start = original_duration_s
                    for idx, s in enumerate(segments):
                        if int(s["id"]) == int(target_seg_id) and idx + 1 < len(segments):
                            next_start = float(segments[idx + 1].get("start", original_duration_s))
                            break
                            
                    max_allowed_dur = max(target_dur, next_start - start - 0.15)
                    max_user_speed = float(self.settings.get("max_speech_speed", 1.5))
                    max_fit_speed = max(1.45, max_user_speed if max_user_speed > 1.05 else 1.45)
                    chosen_speed = 1.0 if tts_dur <= target_dur else (1.0 if tts_dur <= max_allowed_dur else max(1.0, min(max_fit_speed, tts_dur / max_allowed_dur)))
                    
                    if abs(chosen_speed - 1.0) < 0.05:
                        final_seg = tts_segment
                    else:
                        try:
                            await self.async_time_stretch_audio(temp_tts_path, temp_stretch_path, chosen_speed)
                            final_seg = AudioSegment.from_wav(temp_stretch_path)
                        except Exception:
                            final_seg = tts_segment
                        
                    if emotion == "whisper" or "whisper" in emotion:
                        final_seg = final_seg + 12
                        
                    max_duration_limit = max_allowed_dur + 0.30
                    final_dur_s = len(final_seg) / 1000.0
                    if final_dur_s > max_duration_limit:
                        cut_ms = int(max_duration_limit * 1000.0)
                        fade_ms = min(150, int(cut_ms * 0.1))
                        final_seg = final_seg[:cut_ms].fade_out(duration=fade_ms)
                        
                    seg_preview_path = os.path.join(seg_preview_dir, f"seg_{target_seg_id}.wav")
                    final_seg.export(seg_preview_path, format="wav")
                    
                    # Save metadata cache file
                    meta_path = os.path.join(seg_preview_dir, f"seg_{target_seg_id}.meta")
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump({
                            "translation": target_seg.get("translation", ""),
                            "gender": target_seg.get("gender", ""),
                            "speaker_role": target_seg.get("speaker_role", ""),
                            "emotion": target_seg.get("emotion", "")
                        }, mf, ensure_ascii=False, indent=2)
                    return
                else:
                    seg_preview_path = os.path.join(seg_preview_dir, f"seg_{target_seg_id}.wav")
                    if os.path.exists(seg_preview_path):
                        os.remove(seg_preview_path)
                    return
            else:
                seg_preview_path = os.path.join(seg_preview_dir, f"seg_{target_seg_id}.wav")
                if os.path.exists(seg_preview_path):
                    os.remove(seg_preview_path)
                return
            
            # 3. Mix all segments onto dubbed timeline
            self.update_progress(50, "Re-mixing all audio segments...")
            dubbed_audio = AudioSegment.silent(duration=original_duration_ms)
            
            for seg in segments:
                s_id = int(seg["id"])
                s_start = float(seg["start"])
                s_end = float(seg["end"])
                
                if s_start >= original_duration_s:
                    continue
                
                seg_file = os.path.join(seg_preview_dir, f"seg_{s_id}.wav")
                meta_path = os.path.join(seg_preview_dir, f"seg_{s_id}.meta")
                
                cache_valid = False
                if os.path.exists(seg_file) and os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        if (meta.get("translation") == seg.get("translation") and
                            meta.get("gender") == seg.get("gender") and
                            meta.get("speaker_role") == seg.get("speaker_role") and
                            meta.get("emotion") == seg.get("emotion")):
                            cache_valid = True
                    except Exception:
                        cache_valid = False
                
                if not cache_valid:
                    raw_s_text = str(seg.get("translation", ""))
                    s_text = re.sub(r"\[.*?\]", "", raw_s_text)
                    s_text = re.sub(r"\(.*?\)", "", s_text).strip()
                    s_text = s_text.lstrip(":-; ").strip()
                    s_text = re.sub(r'(.+?[\?\!\.\,\s]+)\1+', r'\1', s_text).strip()
                    if not s_text and raw_s_text:
                        fallback_text = re.sub(r"^[\[\(（\s]+|[\]\)）\s]+$", "", raw_s_text).strip()
                        if re.search(r"[\u1780-\u17FF\w]", fallback_text):
                            s_text = fallback_text

                    if s_text and s_text not in ["...", "…", "-", ""]:
                        s_gender = str(seg.get("gender", "female")).lower().strip()
                        s_role = str(seg.get("speaker_role", "lead_actor")).lower().strip()
                        s_is_female = "female" in s_gender or "woman" in s_gender
                        s_is_male = ("male" in s_gender and not s_is_female)
                        s_is_narrator = "narrator" in s_role
                        
                        s_selected_voice = male_voice if s_is_male else female_voice
                        s_emotion = str(seg.get("emotion", "dramatic")).lower().strip()
                        
                        tts_eng = self.settings.get("tts_engine", "edge-tts")
                        raw_ext = "mp3" if tts_eng == "edge-tts" else "wav"
                        t_raw = os.path.join(self.temp_dir, f"seg_{s_id}_raw.{raw_ext}")
                        t_str = os.path.join(self.temp_dir, f"seg_{s_id}_stretched.wav")
                        
                        await self.synthesize_tts(s_text, s_selected_voice, t_raw, is_male=s_is_male, is_narrator=s_is_narrator, emotion=s_emotion, age_group=seg.get("age_group", "adult"))
                        if os.path.exists(t_raw) and os.path.getsize(t_raw) > 0:
                            s_seg = AudioSegment.from_wav(t_raw) if raw_ext == "wav" else AudioSegment.from_file(t_raw)
                            s_dur = len(s_seg) / 1000.0
                            s_target_dur = max(0.1, s_end - s_start)
                            max_user_speed = float(self.settings.get("max_speech_speed", 1.5))
                            if max_user_speed < 1.0:
                                max_user_speed = 1.5
                            s_speed = 1.0 if s_dur <= s_target_dur else max(1.0, min(max_user_speed, s_dur / s_target_dur))
                            try:
                                await self.async_time_stretch_audio(t_raw, t_str, s_speed)
                                final_s = AudioSegment.from_wav(t_str)
                            except Exception:
                                final_s = s_seg
                            if s_emotion == "whisper" or "whisper" in s_emotion:
                                final_s = final_s + 12
                            
                            # Export WAV
                            final_s.export(seg_file, format="wav")
                            
                            # Export Meta
                            with open(meta_path, "w", encoding="utf-8") as mf:
                                json.dump({
                                    "translation": seg.get("translation", ""),
                                    "gender": seg.get("gender", ""),
                                    "speaker_role": seg.get("speaker_role", ""),
                                    "emotion": seg.get("emotion", "")
                                }, mf, ensure_ascii=False, indent=2)
                
                if os.path.exists(seg_file):
                    seg_clip = AudioSegment.from_wav(seg_file)
                    sync_offset_ms = float(self.settings.get("audio_sync_offset_ms", 0))
                    pos_ms = max(0, int((s_start * 1000.0) + sync_offset_ms))
                    if "single_placements" not in locals():
                        single_placements = []
                    single_placements.append((pos_ms, seg_clip))
            if "single_placements" in locals() and single_placements:
                dubbed_audio = fast_overlay_segments(original_duration_ms, single_placements)
            
            # 4. Mixing background stems with dynamic ducking
            self.update_progress(75, "Remixing background track...")
            
            mixer = self.settings.get("mixer", {"drums": True, "bass": True, "melody": True})
            def is_stem_active(stem):
                try:
                    return bool(mixer.get(stem) and float(mixer.get(f"{stem}_vol", 100)) > 0)
                except (ValueError, TypeError):
                    return bool(mixer.get(stem))
            any_bg_enabled = is_stem_active("drums") or is_stem_active("bass") or is_stem_active("melody")
            
            if not any_bg_enabled:
                bg_mixed = AudioSegment.silent(duration=original_duration_ms)
            else:
                bg_mixed_wav_path = os.path.join(get_subtitles_dir(), str(video_id), "bg_mixed.wav")
                if os.path.exists(bg_mixed_wav_path):
                    try:
                        bg_mixed = AudioSegment.from_wav(bg_mixed_wav_path)
                    except Exception:
                        bg_mixed = audio_segment
                else:
                    bg_mixed = audio_segment
            duck_db = 15
            
            if segments and any_bg_enabled:
                sorted_segs = sorted(segments, key=lambda x: float(x["start"]))
                merged_intervals = []
                for seg in sorted_segs:
                    s_ms = max(0, int(float(seg["start"]) * 1000.0) - 300)
                    e_ms = min(original_duration_ms, int(float(seg["end"]) * 1000.0) + 300)
                    if not merged_intervals:
                        merged_intervals.append([s_ms, e_ms])
                    else:
                        last = merged_intervals[-1]
                        if s_ms <= last[1]:
                            last[1] = max(last[1], e_ms)
                        else:
                            merged_intervals.append([s_ms, e_ms])

                bg_mixed = fast_duck_audio_segment(bg_mixed, merged_intervals, duck_db)
            
            final_audio = bg_mixed.overlay(dubbed_audio)
            final_audio_path = os.path.join(self.temp_dir, "final_mixed_audio.wav")
            final_audio.export(final_audio_path, format="wav")
            
            # 5. Remux video
            self.update_progress(90, "Re-muxing final video...")
            if self.settings.get("output_type") == "custom" and self.settings.get("custom_output_path"):
                out_dir = self.settings.get("custom_output_path")
            else:
                out_dir = os.path.join(os.path.dirname(video_path), "Dubbed_Output")
            
            os.makedirs(out_dir, exist_ok=True)
            output_video_path = os.path.join(out_dir, f"[Dubbed]_{name_without_ext}.mp4")
            
            remux_cmd = [
                get_ffmpeg_cmd(), "-y",
                "-i", video_path,
                "-i", final_audio_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-max_interleave_delta", "0",
                "-avoid_negative_ts", "make_zero",
                "-shortest",
                output_video_path
            ]
            
            await self.async_run_cmd(remux_cmd, "Video Remuxing")
            
            # Ensure we update subtitles JSON permanently on disk as well
            try:
                subtitles_json_path = os.path.join(subtitles_dir, f"{video_id}.json")
                with open(subtitles_json_path, "w", encoding="utf-8") as sf:
                    json.dump(segments, sf, ensure_ascii=False, indent=4)
            except Exception as se:
                logger.error(f"Failed to update subtitles json in segment re-dub: {se}")
                
            self.update_progress(100, "Completed")
            self.log(f"[SUCCESS] Segment {target_seg_id} re-dubbed successfully.", "success")
            
        except Exception as e:
            self.update_progress(100, "Error")
            self.log(f"[ERROR] Failed to re-dub segment {target_seg_id}: {str(e)}", "error")
            raise e
        finally:
            try:
                if self.temp_dir and os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
            except Exception:
                pass

    async def synthesize_edge_for_lang(self, text, voice_name, output_path, speed_ratio=1.0):
        """Synthesizes text using native Edge-TTS with retry logic for arbitrary language voices."""
        clean_text = text.strip() if text else ""
        clean_text = re.sub(r"[\*\_\#\`]", "", clean_text).strip()
        clean_text = re.sub(r"\s+", " ", clean_text)
        if not clean_text or clean_text in ["...", "…", "-", ""]:
            AudioSegment.silent(duration=300).export(output_path, format="mp3")
            return
            
        rate_str = "+0%"
        if speed_ratio != 1.0:
            rate_diff = int((speed_ratio - 1.0) * 100)
            rate_str = f"+{rate_diff}%" if rate_diff >= 0 else f"{rate_diff}%"
            
        import edge_tts  # type: ignore
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                comm = edge_tts.Communicate(clean_text, voice_name, rate=rate_str)
                await comm.save(output_path)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                    return
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
                else:
                    self.log(f"Edge-TTS failed for {voice_name}: {e}", "warning")
                    AudioSegment.silent(duration=500).export(output_path, format="mp3")

    async def export_youtube_audio_tracks(self, video_path, video_id, clean_name, out_dir, segments, bg_mixed, any_bg_enabled, original_duration_ms, original_duration_s, vocals_wav=None):
        """Generates standalone broadcast-quality .m4a stems and .srt files for YouTube Studio."""
        if not self.settings.get("youtube_multi_audio", False):
            return
            
        extra_langs = self.settings.get("youtube_extra_languages", [])
        if not extra_langs or not isinstance(extra_langs, list):
            return
            
        primary_lang = self.settings.get("target_lang", "km-KH")
        target_langs = [lg for lg in extra_langs if lg != primary_lang and lg in VOICES]
        if not target_langs:
            self.log("YouTube Multi-Audio: No extra languages selected (or primary was skipped).", "info")
            return
            
        self.log(f"Starting YouTube Multi-Audio generation for {len(target_langs)} extra language(s): {', '.join(target_langs)}", "info")
        
        youtube_tracks_dir = os.path.join(out_dir, "YouTube_Audio_Tracks")
        youtube_subs_dir = os.path.join(out_dir, "YouTube_Subtitles")
        os.makedirs(youtube_tracks_dir, exist_ok=True)
        os.makedirs(youtube_subs_dir, exist_ok=True)
        
        def to_srt_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int(round((seconds - int(seconds)) * 1000))
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        # Export primary language SRT
        try:
            primary_srt_path = os.path.join(youtube_subs_dir, f"[Dubbed]_{clean_name}_{primary_lang}.srt")
            with open(primary_srt_path, "w", encoding="utf-8") as psrt:
                for idx, seg in enumerate(segments, 1):
                    s_t = to_srt_time(float(seg.get("start", 0)))
                    e_t = to_srt_time(float(seg.get("end", 0)))
                    txt = str(seg.get("translation", "")).strip()
                    psrt.write(f"{idx}\n{s_t} --> {e_t}\n{txt}\n\n")
        except Exception as psrt_err:
            logger.warning(f"Could not save primary SRT: {psrt_err}")

        total_extra = len(target_langs)
        lang_sem = asyncio.Semaphore(3)

        # Compact payload: only send {id, start, end, text} to cut input/output tokens by 50%
        compact_items = []
        for idx, s in enumerate(segments):
            orig_text = s.get("text") or s.get("translation", "")
            compact_items.append({
                "id": int(s.get("id", idx + 1)),
                "start": round(float(s.get("start", 0.0)), 2),
                "end": round(float(s.get("end", 0.0)), 2),
                "text": orig_text
            })

        batch_size = 50
        item_batches = [compact_items[i:i + batch_size] for i in range(0, len(compact_items), batch_size)]

        async def _process_youtube_language(lang_idx, lang_code):
            async with lang_sem:
                lang_label = LANGUAGE_PROMPTS.get(lang_code, lang_code)
                self.log(f"[YouTube Track {lang_idx}/{total_extra}] Translating dialogue to {lang_label} ({lang_code}) in {len(item_batches)} batch(es)...", "info")

                async def _translate_yt_batch(b_items):
                    trans_prompt = f"""
                    You are an expert film and television dubbing dialogue director and translator.
                    Translate the dialogue for each segment into target language: '{lang_label}'.
                    
                    CRITICAL CONSTRAINTS:
                    1. Maintain the exact same segment structure and IDs ({len(b_items)} items).
                    2. The spoken length of your translation MUST fit comfortably within the segment duration ('end' - 'start').
                    3. Keep translated dialogue natural, expressive, and concise so actors can finish speaking naturally.
                    4. Return ONLY a valid JSON array of objects with "id" and "translation". Do not include markdown code block syntax.
                    
                    Input Segments:
                    {json.dumps(b_items, ensure_ascii=False)}
                    
                    JSON structure:
                    [
                      {{
                        "id": {b_items[0]["id"]},
                        "translation": "translated text in {lang_label}"
                      }}
                    ]
                    """
                    raw_resp = await self.call_gemini_with_retry(
                        None,
                        trans_prompt,
                        expected_segments=len(b_items)
                    )
                    clean_json_str = extract_json_payload(raw_resp)
                    clean_json_str = re.sub(r',\s*([\]}])', r'\1', clean_json_str)
                    try:
                        parsed = json.loads(clean_json_str, strict=False)
                    except Exception:
                        last_brace = clean_json_str.rfind('}')
                        if last_brace != -1 and clean_json_str.find('[') != -1:
                            repaired = clean_json_str[clean_json_str.find('['):last_brace + 1]
                            repaired = re.sub(r',\s*$', '', repaired) + "\n]"
                            parsed = json.loads(repaired, strict=False)
                        else:
                            parsed = []
                    return parsed if isinstance(parsed, list) else []

                translated_segments = []
                try:
                    batch_tasks = [_translate_yt_batch(b) for b in item_batches]
                    batch_lists = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    for b_res in batch_lists:
                        if isinstance(b_res, list):
                            translated_segments.extend(b_res)
                except Exception as trans_err:
                    self.log(f"Translation failed for {lang_code}: {trans_err}. Falling back to original segments.", "error")
                    translated_segments = [dict(s) for s in segments]

                anchor_map = {int(s.get("id", i + 1)): s for i, s in enumerate(translated_segments) if isinstance(s, dict)}
                final_lang_segments = []
                for idx, orig_seg in enumerate(segments):
                    sid = int(orig_seg.get("id", idx + 1))
                    match = anchor_map.get(sid, {})
                    merged = dict(orig_seg)
                    merged["translation"] = match.get("translation") or orig_seg.get("translation", "")
                    final_lang_segments.append(merged)

                # Write matching .srt file
                lang_srt_path = os.path.join(youtube_subs_dir, f"[Dubbed]_{clean_name}_{lang_code}.srt")
                try:
                    with open(lang_srt_path, "w", encoding="utf-8") as sf:
                        for s_idx, seg in enumerate(final_lang_segments, 1):
                            s_t = to_srt_time(float(seg.get("start", 0)))
                            e_t = to_srt_time(float(seg.get("end", 0)))
                            txt = str(seg.get("translation", "")).strip()
                            sf.write(f"{s_idx}\n{s_t} --> {e_t}\n{txt}\n\n")
                except Exception as srt_err:
                    logger.warning(f"Failed to write SRT for {lang_code}: {srt_err}")

                # Step 2: Synthesize TTS
                self.log(f"[YouTube Track {lang_idx}/{total_extra}] Synthesizing {lang_label} voice acting...", "info")
                voices_pair = VOICES.get(lang_code, ("en-US-GuyNeural", "en-US-JennyNeural"))
                male_voice = voices_pair[0]
                female_voice = voices_pair[1]

                sem = asyncio.Semaphore(6)
                extra_temp_dir = os.path.join(self.temp_dir, f"yt_{lang_code}")
                os.makedirs(extra_temp_dir, exist_ok=True)

                max_user_speed = float(self.settings.get("max_speech_speed", 1.5))
                max_fit_speed = max(1.45, max_user_speed if max_user_speed > 1.05 else 1.45)

                async def process_extra_seg(idx, seg):
                    raw_text = str(seg.get("translation", ""))
                    text_to_speak = re.sub(r"\[.*?\]", "", raw_text)
                    text_to_speak = re.sub(r"\(.*?\)", "", text_to_speak).strip()
                    text_to_speak = text_to_speak.lstrip(":-; ").strip()
                    text_to_speak = re.sub(r'(.+?[\?\!\.\,\s]+)\1+', r'\1', text_to_speak).strip()
                    if not text_to_speak and raw_text:
                        fallback_text = re.sub(r"^[\[\(（\s]+|[\]\)）\s]+$", "", raw_text).strip()
                        if fallback_text:
                            text_to_speak = fallback_text
                    if not text_to_speak or text_to_speak in ["...", "…", "-", ""]:
                        return None

                    seg_id = seg.get("id", idx + 1)
                    start = float(seg.get("start", 0))
                    end = float(seg.get("end", 0))

                    gender_str = str(seg.get("gender", "male")).lower().strip()
                    is_female = "female" in gender_str or "woman" in gender_str or "girl" in gender_str
                    is_male = ("male" in gender_str and not is_female) or "man" in gender_str or "boy" in gender_str
                    if self.settings.get("voice_gender") == "male":
                        chosen_voice = male_voice
                    elif self.settings.get("voice_gender") == "female":
                        chosen_voice = female_voice
                    else:
                        chosen_voice = male_voice if (is_male and not is_female) else female_voice

                    target_dur = max(0.1, end - start)
                    next_start = original_duration_s
                    if idx + 1 < len(final_lang_segments):
                        next_start = float(final_lang_segments[idx + 1].get("start", original_duration_s))
                    max_allowed = max(target_dur, next_start - start - 0.15)

                    # Estimate speaking rate upfront for tight windows to minimize secondary FFmpeg stretches
                    est_chars_per_sec = 14.0 if lang_code in ("en-US", "en-GB", "de-DE", "fr-FR", "es-ES", "nl-NL", "it-IT", "pt-BR", "vi-VN", "id-ID", "fil-PH") else 8.5
                    est_dur = len(text_to_speak) / est_chars_per_sec
                    pre_rate = 1.0
                    if est_dur > max_allowed * 1.08:
                        pre_rate = min(1.30, max(1.0, est_dur / max_allowed))

                    temp_tts_file = os.path.join(extra_temp_dir, f"seg_{seg_id}.mp3")
                    temp_stretch_file = os.path.join(extra_temp_dir, f"seg_{seg_id}_str.wav")

                    async with sem:
                        await self.synthesize_edge_for_lang(text_to_speak, chosen_voice, temp_tts_file, speed_ratio=pre_rate)

                    if not os.path.exists(temp_tts_file) or os.path.getsize(temp_tts_file) == 0:
                        return None

                    tts_audio = AudioSegment.from_file(temp_tts_file)
                    tts_dur = len(tts_audio) / 1000.0
                    if tts_dur <= max_allowed:
                        speed = 1.0
                    else:
                        speed = min(max_fit_speed, tts_dur / max_allowed)

                    if abs(speed - 1.0) < 0.05:
                        final_audio_seg = tts_audio
                    else:
                        try:
                            await self.async_time_stretch_audio(temp_tts_file, temp_stretch_file, speed)
                            final_audio_seg = AudioSegment.from_wav(temp_stretch_file)
                        except Exception:
                            final_audio_seg = tts_audio

                    emotion = str(seg.get("emotion", "dramatic")).lower().strip()
                    if emotion == "whisper" or "whisper" in emotion:
                        final_audio_seg = final_audio_seg + 12

                    max_limit = max_allowed + 0.30
                    if (len(final_audio_seg) / 1000.0) > max_limit:
                        cut_ms = int(max_limit * 1000.0)
                        fade_ms = min(150, int(cut_ms * 0.1))
                        final_audio_seg = final_audio_seg[:cut_ms].fade_out(duration=fade_ms)

                    return (start, final_audio_seg)

                tts_tasks = [process_extra_seg(i, s) for i, s in enumerate(final_lang_segments)]
                tts_results = await asyncio.gather(*tts_tasks, return_exceptions=True)

                extra_placements = []
                for res in tts_results:
                    if res and not isinstance(res, Exception):
                        start_sec, seg_audio = res
                        sync_offset_ms = float(self.settings.get("audio_sync_offset_ms", 0))
                        target_ms = max(0, int((start_sec * 1000.0) + sync_offset_ms))
                        extra_placements.append((target_ms, seg_audio))
                extra_dubbed_audio = fast_overlay_segments(original_duration_ms, extra_placements)

                # Step 3: Mix with BGM stem and export .m4a directly (zero video re-encoding)
                extra_full_mix = bg_mixed.overlay(extra_dubbed_audio)
                temp_mix_wav = os.path.join(extra_temp_dir, f"mix_{lang_code}.wav")
                extra_full_mix.export(temp_mix_wav, format="wav")

                output_m4a_path = os.path.join(youtube_tracks_dir, f"[Dubbed]_{clean_name}_{lang_code}.m4a")
                m4a_cmd = [
                    get_ffmpeg_cmd(), "-y",
                    "-i", temp_mix_wav,
                    "-c:a", "aac",
                    "-b:a", "384k",
                    "-ar", "48000",
                    output_m4a_path
                ]
                await self.async_run_cmd(m4a_cmd, f"YouTube Audio Stem ({lang_code})")
                self.log(f"[SUCCESS] YouTube stem exported: YouTube_Audio_Tracks\\[Dubbed]_{clean_name}_{lang_code}.m4a", "success")

        lang_tasks = [_process_youtube_language(idx, l_code) for idx, l_code in enumerate(target_langs, 1)]
        await asyncio.gather(*lang_tasks, return_exceptions=True)

# Sync runtime version into frozen __main__ (AutoDubber.exe) so hot-patches update the UI version badge
try:
    _main_mod = sys.modules.get("__main__")
    if _main_mod is not None and hasattr(_main_mod, "APP_VERSION"):
        _main_mod.APP_VERSION = "2.2.1"
except Exception:
    pass

