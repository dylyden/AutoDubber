"""
Automated Patch Publisher for AutoDubber
Called by Publish_Update.bat
"""

import os
import sys
import json
import zipfile
import re
import time

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    version_json_path = os.path.join(root_dir, "version.json")
    server_py = os.path.join(root_dir, "server.py")
    dubber_py = os.path.join(root_dir, "dubber.py")

    # 1. Load or determine current version
    current_ver = "2.2.5"
    manifest_data = {
        "version": current_ver,
        "manifest_url": "https://raw.githubusercontent.com/dylyden/AutoDubber/main/version.json",
        "download_url": "https://github.com/dylyden/AutoDubber/archive/refs/heads/main.zip",
        "updated_at": "",
        "changelog": "Bug fixes and performance updates"
    }

    if os.path.exists(version_json_path):
        try:
            with open(version_json_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    manifest_data.update(saved)
                    current_ver = str(saved.get("version", current_ver)).lstrip("vV")
        except Exception:
            pass
    elif os.path.exists(server_py):
        try:
            with open(server_py, "r", encoding="utf-8") as f:
                m = re.search(r'APP_VERSION\s*=\s*["\']([^"\']+)["\']', f.read())
                if m:
                    current_ver = m.group(1).lstrip("vV")
        except Exception:
            pass

    # Read CLI args from Publish_Update.bat
    raw_ver = sys.argv[1].strip().lstrip("vV") if len(sys.argv) >= 2 else ""
    changelog = sys.argv[2].strip() if len(sys.argv) >= 3 else ""

    target_ver = raw_ver if raw_ver else current_ver
    changelog = changelog if changelog else manifest_data.get("changelog", "Bug fixes and performance updates")

    # Update version.json
    manifest_data["version"] = target_ver
    manifest_data["changelog"] = changelog
    manifest_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(version_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Updated version.json -> v{target_ver}")

    # 2. Update server.py and dubber.py APP_VERSION
    if os.path.exists(server_py):
        try:
            with open(server_py, "r", encoding="utf-8") as f:
                s_content = f.read()
            s_updated = re.sub(r'APP_VERSION\s*=\s*["\'].*?["\']', f'APP_VERSION = "{target_ver}"', s_content)
            with open(server_py, "w", encoding="utf-8") as f:
                f.write(s_updated)
        except Exception as e:
            print(f"Warning updating server.py: {e}")

    if os.path.exists(dubber_py):
        try:
            with open(dubber_py, "r", encoding="utf-8") as f:
                d_content = f.read()
            d_updated = re.sub(r'_main_mod\.APP_VERSION\s*=\s*["\'].*?["\']', f'_main_mod.APP_VERSION = "{target_ver}"', d_content)
            with open(dubber_py, "w", encoding="utf-8") as f:
                f.write(d_updated)
        except Exception as e:
            print(f"Warning updating dubber.py: {e}")

    # 3. Rebuild app.zip
    app_zip_path = os.path.join(root_dir, "app.zip")
    core_files = [
        "server.py",
        "dubber.py",
        "voice_converter.py",
        "dialog_picker.py",
        "run_voxcpm_studio.py",
        "vertex_batch_processor.py",
        "system_diagnostic.py",
        "app.js",
        "index.html",
        "style.css",
        "version.json"
    ]

    try:
        with zipfile.ZipFile(app_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in core_files:
                fpath = os.path.join(root_dir, fname)
                if os.path.exists(fpath):
                    zf.write(fpath, fname)
        print("Rebuilt app.zip successfully")
    except Exception as e:
        print(f"Warning rebuilding app.zip: {e}")

if __name__ == "__main__":
    main()
