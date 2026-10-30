"""
Automated Patch Publisher for AutoDubber
Usage:
    python publish_patch.py 2.1.1 "Fixed audio sync and edge-tts handshake"
"""

import os
import sys
import json
import zipfile
import subprocess
import re

def _get_current_and_next_version(server_py_path: str):
    curr_ver = "2.2.0"
    if os.path.exists(server_py_path):
        try:
            with open(server_py_path, "r", encoding="utf-8") as f:
                m = re.search(r'APP_VERSION\s*=\s*["\']([^"\']+)["\']', f.read())
                if m:
                    curr_ver = m.group(1).lstrip("vV")
        except Exception:
            pass
    parts = [int(x) if x.isdigit() else 0 for x in curr_ver.split(".")]
    while len(parts) < 3:
        parts.append(0)
    parts[-1] += 1
    return curr_ver, ".".join(str(x) for x in parts)


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    server_py = os.path.join(root_dir, "server.py")
    curr_ver, suggested_ver = _get_current_and_next_version(server_py)

    if len(sys.argv) < 2:
        print(f"Current Version : v{curr_ver}")
        print(f"Suggested Next  : v{suggested_ver}\n")
        try:
            u_ver = input(f"Enter new version [Press ENTER for {suggested_ver}]: ").strip().lstrip("vV")
            new_version = u_ver if u_ver else suggested_ver
            u_notes = input("Enter release notes [Press ENTER for default]: ").strip()
            notes = u_notes if u_notes else f"Release v{new_version}: Speed & Voice Engine Update"
        except EOFError:
            new_version = suggested_ver
            notes = f"Release v{new_version}: Speed & Voice Engine Update"
        print(f"\n[AUTO] Publishing v{new_version}...")
    else:
        new_version = sys.argv[1].lstrip("vV")
        notes = sys.argv[2] if len(sys.argv) >= 3 else f"Release v{new_version}"

    # 1. Bump APP_VERSION in server.py and dubber.py
    if os.path.exists(server_py):
        with open(server_py, "r", encoding="utf-8") as f:
            content = f.read()
        updated_content = re.sub(r'APP_VERSION\s*=\s*["\'].*?["\']', f'APP_VERSION = "{new_version}"', content)
        with open(server_py, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"[OK] Bumped APP_VERSION to {new_version} in server.py")

    dubber_py = os.path.join(root_dir, "dubber.py")
    if os.path.exists(dubber_py):
        with open(dubber_py, "r", encoding="utf-8") as f:
            d_content = f.read()
        d_updated = re.sub(r'_main_mod\.APP_VERSION\s*=\s*["\'].*?["\']', f'_main_mod.APP_VERSION = "{new_version}"', d_content)
        with open(dubber_py, "w", encoding="utf-8") as f:
            f.write(d_updated)
        print(f"[OK] Bumped _main_mod.APP_VERSION to {new_version} in dubber.py")

    # 2. Build lightweight patch.zip
    patch_zip_name = "patch.zip"
    patch_zip_path = os.path.join(root_dir, patch_zip_name)
    files_to_pack = ["dubber.py", "voice_converter.py", "server.py", "app.js", "index.html", "style.css"]
    
    with zipfile.ZipFile(patch_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in files_to_pack:
            fpath = os.path.join(root_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)
                print(f"[PACKED] {fname} ({os.path.getsize(fpath) / 1024:.1f} KB)")

    print(f"[OK] Generated {patch_zip_name} ({os.path.getsize(patch_zip_path) / 1024:.1f} KB total)")

    # 3. Git commit & tag
    tag_name = f"v{new_version}"
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", f"Release {tag_name}: {notes}"], check=False)
        subprocess.run(["git", "tag", "-a", tag_name, "-m", notes], check=False)
        print(f"[OK] Created Git tag {tag_name}")
        
        # Check if remote exists
        res = subprocess.run(["git", "remote"], capture_output=True, text=True)
        if "origin" in res.stdout:
            print(f"[PUSHING] Pushing commit and tag {tag_name} to GitHub...")
            subprocess.run(["git", "push", "origin", "main", "--tags"], check=False)
            subprocess.run(["git", "push", "origin", "master", "--tags"], check=False)
            print("[SUCCESS] Pushed to GitHub successfully!")
        else:
            print("[INFO] No remote 'origin' configured yet. Run 'git remote add origin <url>' to connect.")
    except Exception as e:
        print(f"[WARN] Git step encountered: {e}")

    print("\n=======================================================")
    print(f"PATCH READY: {patch_zip_path}")
    print(f"Attach {patch_zip_name} to your GitHub Release for {tag_name}.")
    print("=======================================================")

if __name__ == "__main__":
    main()
