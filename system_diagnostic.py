import sys
import os
import subprocess
import importlib.util
import socket

def print_status(component, status, details=""):
    color = "\033[92m" if status == "OK" else "\033[91m"
    reset = "\033[0m"
    print(f"[{color}{status:^6}{reset}] {component:<25} {details}")

def check_python_version():
    version = sys.version_info
    status = "OK" if version.major >= 3 and version.minor >= 10 else "WARN"
    print_status("Python Version", status, f"{version.major}.{version.minor}.{version.micro}")

def check_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print_status("FFmpeg", "OK", version_line)
        else:
            print_status("FFmpeg", "FAIL", "Command failed")
    except FileNotFoundError:
        # Check local path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_ffmpeg = os.path.join(base_dir, "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe")
        if os.path.exists(local_ffmpeg):
            print_status("FFmpeg", "OK", f"Found locally at {local_ffmpeg}")
        else:
            print_status("FFmpeg", "FAIL", "Not found in PATH or local directory")

def check_dependencies():
    deps = [
        "fastapi", "uvicorn", "pydub", "google.genai", 
        "multipart", "demucs", "faster_whisper"
    ]
    for dep in deps:
        if importlib.util.find_spec(dep) is not None:
            print_status(f"Package: {dep}", "OK")
        else:
            print_status(f"Package: {dep}", "FAIL", "Not installed")

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            print_status(f"Port {port}", "OK", "Available")
        except socket.error:
            print_status(f"Port {port}", "FAIL", "In use")

if __name__ == "__main__":
    print("=== AutoDubber System Diagnostic ===")
    check_python_version()
    check_ffmpeg()
    check_dependencies()
    check_port(8500)
    check_port(8080)
    print("====================================")
