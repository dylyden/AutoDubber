import http.server
import json
import webbrowser
import os
import tempfile
import base64
import io
import soundfile as sf
from dubber import get_local_voxcpm_model

PORT = 8080

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoxCPM Private Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --input-bg: #334155;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #475569;
            --success-color: #10b981;
            --error-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-color);
        }

        header {
            margin-bottom: 2rem;
            text-align: center;
        }

        header h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        label {
            display: block;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }

        textarea, input[type="text"] {
            width: 100%;
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-color);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus, input[type="text"]:focus {
            border-color: var(--accent-color);
        }

        textarea {
            resize: vertical;
            min-height: 120px;
        }

        .row {
            display: flex;
            gap: 1rem;
        }

        .row > .form-group {
            flex: 1;
        }

        .file-upload {
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            background: var(--input-bg);
            cursor: pointer;
            transition: border-color 0.2s;
        }

        .file-upload:hover {
            border-color: var(--accent-color);
        }

        .file-upload input {
            display: none;
        }

        .file-name {
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: var(--accent-color);
        }

        .range-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .range-container input[type="range"] {
            flex: 1;
            accent-color: var(--accent-color);
        }

        .range-value {
            width: 45px;
            text-align: right;
            font-weight: 600;
            color: var(--accent-color);
        }

        button.btn-primary {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-color), var(--accent-hover));
            color: white;
            border: none;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.1s, opacity 0.2s;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }

        button.btn-primary:hover {
            opacity: 0.95;
        }

        button.btn-primary:active {
            transform: scale(0.99);
        }

        button.btn-primary:disabled {
            background: var(--border-color);
            cursor: not-allowed;
            box-shadow: none;
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: var(--input-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            display: none;
        }

        .result-card.active {
            display: block;
        }

        audio {
            width: 100%;
            margin-top: 1rem;
        }

        .status-msg {
            margin-top: 1rem;
            font-size: 0.9rem;
            text-align: center;
        }

        .status-msg.error {
            color: var(--error-color);
        }

        .status-msg.success {
            color: var(--success-color);
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 640px) {
            .row {
                flex-direction: column;
                gap: 0;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>VoxCPM Private Studio</h1>
            <p>Multilingual Text-to-Speech & Voice Design Interface</p>
        </header>

        <div class="form-group">
            <label for="targetText">Target Text (Content to synthesize)</label>
            <textarea id="targetText" placeholder="Enter text here... VoxCPM2 generates highly expressive speech."></textarea>
        </div>

        <div class="form-group">
            <label for="controlInstruction">Control Instruction (Voice design description — Optional)</label>
            <input type="text" id="controlInstruction" placeholder="e.g. A warm young woman / 年轻女性，温柔甜美 / Excited narrator">
        </div>

        <div class="row">
            <div class="form-group">
                <label>Reference Audio (Optional for Voice Cloning)</label>
                <div class="file-upload" onclick="document.getElementById('refAudio').click()">
                    <span>📁 Upload Reference WAV/MP3</span>
                    <input type="file" id="refAudio" accept="audio/*" onchange="handleFileSelect(this)">
                    <div class="file-name" id="fileName"></div>
                </div>
            </div>

            <div class="form-group">
                <label for="cfgValue">CFG Scale (Guidance Strength)</label>
                <div class="range-container" style="margin-top: 0.5rem;">
                    <input type="range" id="cfgValue" min="1.0" max="5.0" step="0.1" value="2.0" oninput="document.getElementById('cfgValText').innerText = this.value">
                    <span class="range-value" id="cfgValText">2.0</span>
                </div>
            </div>
        </div>

        <button class="btn-primary" id="btnGenerate" onclick="generateSpeech()">
            <div class="spinner" id="btnSpinner"></div>
            <span id="btnText">🔊 Generate Speech</span>
        </button>

        <div class="status-msg" id="statusMsg"></div>

        <div class="result-card" id="resultCard">
            <label>Generated Audio Result</label>
            <audio id="audioPlayer" controls></audio>
            <div style="margin-top: 1rem; text-align: right;">
                <a id="downloadBtn" download="voxcpm_output.mp3" style="color: var(--accent-color); font-weight: 600; text-decoration: none;">⬇️ Download MP3</a>
            </div>
        </div>
    </div>

    <script>
        let refAudioBase64 = null;
        let refAudioFormat = 'wav';

        function handleFileSelect(input) {
            const file = input.files[0];
            if (!file) return;
            document.getElementById('fileName').innerText = file.name;
            refAudioFormat = file.name.split('.').pop().toLowerCase() || 'wav';

            const reader = new FileReader();
            reader.onload = function(e) {
                const arrayBuffer = e.target.result;
                const bytes = new Uint8Array(arrayBuffer);
                let binary = '';
                for (let i = 0; i < bytes.byteLength; i++) {
                    binary += String.fromCharCode(bytes[i]);
                }
                refAudioBase64 = btoa(binary);
            };
            reader.readAsArrayBuffer(file);
        }

        async function generateSpeech() {
            const text = document.getElementById('targetText').value.trim();
            const control = document.getElementById('controlInstruction').value.trim();
            const cfg = parseFloat(document.getElementById('cfgValue').value);

            const btn = document.getElementById('btnGenerate');
            const spinner = document.getElementById('btnSpinner');
            const btnText = document.getElementById('btnText');
            const statusMsg = document.getElementById('statusMsg');
            const resultCard = document.getElementById('resultCard');
            const audioPlayer = document.getElementById('audioPlayer');
            const downloadBtn = document.getElementById('downloadBtn');

            if (!text) {
                statusMsg.className = 'status-msg error';
                statusMsg.innerText = 'Please enter target text to synthesize.';
                return;
            }

            btn.disabled = true;
            spinner.style.display = 'block';
            btnText.innerText = 'Generating...';
            statusMsg.className = 'status-msg';
            statusMsg.innerText = 'Synthesizing audio on VoxCPM server...';
            resultCard.classList.remove('active');

            let finalText = text;
            if (control) {
                finalText = `(${control})${text}`;
            }

            const payload = {
                target_text: finalText,
                cfg_value: cfg
            };

            if (refAudioBase64) {
                payload.ref_audio_wav_base64 = refAudioBase64;
                payload.ref_audio_wav_format = refAudioFormat;
            }

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);
                audioPlayer.src = audioUrl;
                downloadBtn.href = audioUrl;
                resultCard.classList.add('active');
                audioPlayer.play();

                statusMsg.className = 'status-msg success';
                statusMsg.innerText = 'Speech generated successfully!';
            } catch (err) {
                statusMsg.className = 'status-msg error';
                statusMsg.innerText = `Generation failed: ${err.message}`;
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.innerText = '🔊 Generate Speech';
            }
        }
    </script>
</body>
</html>
"""

class StudioHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                payload = json.loads(post_data.decode('utf-8'))
                target_text = payload.get("target_text", "")
                cfg_value = payload.get("cfg_value", 2.0)
                ref_base64 = payload.get("ref_audio_wav_base64")
                
                control_inst = ""
                if target_text.startswith("(") and ")" in target_text:
                    idx = target_text.index(")")
                    control_inst = target_text[1:idx]
                    target_text = target_text[idx+1:]
                
                tmp_file = None
                if ref_base64:
                    fmt = payload.get("ref_audio_wav_format", "wav")
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}")
                    tmp_file.write(base64.b64decode(ref_base64))
                    tmp_file.close()

                model = get_local_voxcpm_model(device="cuda")
                full_text = f"({control_inst}){target_text}" if control_inst else target_text
                gen_kwargs = {
                    "text": full_text,
                    "cfg_value": cfg_value,
                    "inference_timesteps": 10
                }
                if tmp_file and os.path.exists(tmp_file.name):
                    gen_kwargs["reference_wav_path"] = tmp_file.name

                wav = model.generate(**gen_kwargs)

                if tmp_file:
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass

                buf = io.BytesIO()
                sf.write(buf, wav, 48000, format='WAV')
                audio_data = buf.getvalue()

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(audio_data)

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    print(f"VoxCPM Studio Server running on http://127.0.0.1:{PORT}")
    webbrowser.open(f"http://127.0.0.1:{PORT}")
    server = http.server.HTTPServer(("127.0.0.1", PORT), StudioHandler)
    server.serve_forever()
