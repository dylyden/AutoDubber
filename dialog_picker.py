import sys
import json
import os

def pick(browse_type):
    # Try Tkinter first
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.focus_force()

        res = None
        if browse_type == "files":
            f = filedialog.askopenfilenames(
                title="Select Video Files",
                filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.flv *.ts *.webm *.3gp"), ("All Files", "*.*")]
            )
            res = list(f) if f else None
        elif browse_type in ["folder", "output"]:
            d = filedialog.askdirectory(title="Select Folder")
            res = d if d else None
        else:
            f = filedialog.askopenfilename(
                title="Select Audio Reference File",
                filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac"), ("All Files", "*.*")]
            )
            res = f if f else None

        root.destroy()
        if res:
            return res
    except Exception as e:
        pass

    # Windows PowerShell fallback
    try:
        import subprocess
        if browse_type == "files":
            cmd = "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Multiselect = $true; $f.Filter = 'Video Files|*.mp4;*.mkv;*.avi;*.mov;*.flv;*.ts;*.webm;*.3gp|All Files|*.*'; if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $f.FileNames -join '||' }"
        elif browse_type in ["folder", "output"]:
            cmd = "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $f.SelectedPath }"
        else:
            cmd = "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Filter = 'Audio Files|*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac|All Files|*.*'; if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $f.FileName }"

        proc = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True, timeout=120)
        out = proc.stdout.strip()
        if out:
            if browse_type == "files":
                return [p.strip() for p in out.split("||") if p.strip()]
            return out
    except Exception:
        pass

    return None

if __name__ == "__main__":
    b_type = sys.argv[1] if len(sys.argv) > 1 else "files"
    result = pick(b_type)
    print("PICKER_RESULT:" + json.dumps(result))
