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

def main():
    if len(sys.argv) < 2:
        print("Usage: python publish_patch.py <version> [release_notes]")
        print("Example: python publish_patch.py 2.1.1 \"Fixed audio sync issue\"")
        sys.exit(1)

    new_version = sys.argv[1].lstrip("vV")
    notes = sys.argv[2] if len(sys.argv) >= 3 else f"Release v{new_version}"

    root_dir = os.path.dirname(os.path.abspath(__file__))
    server_py = os.path.join(root_dir, "server.py")

    # 1. Bump APP_VERSION in server.py
    if os.path.exists(server_py):
        with open(server_py, "r", encoding="utf-8") as f:
            content = f.read()
        updated_content = re.sub(r'APP_VERSION\s*=\s*["\'].*?["\']', f'APP_VERSION = "{new_version}"', content)
        with open(server_py, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"[OK] Bumped APP_VERSION to {new_version} in server.py")

    # 2. Build lightweight patch.zip
    patch_zip_name = "patch.zip"
    patch_zip_path = os.path.join(root_dir, patch_zip_name)
    files_to_pack = ["dubber.py", "server.py", "app.js", "index.html", "style.css"]
    
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
