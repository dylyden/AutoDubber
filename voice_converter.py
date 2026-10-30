import os
import json
import threading
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import numpy as np
import librosa

LRELU_SLOPE = 0.1


def _init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def _sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


@torch.jit.script
def _fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class _WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
            self.in_layers.append(weight_norm(in_layer, name="weight"))
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            self.res_skip_layers.append(weight_norm(res_skip_layer, name="weight"))

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = _fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask


class _ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=_get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=_get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=_get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(_init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(_init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class _Generator(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(_ResBlock1(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(_init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class _PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = _WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, tau=1.0):
        x_mask = torch.unsqueeze(_sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class _Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        return x


class _ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = _WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class _ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(_ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(_Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class _ReferenceEncoder(nn.Module):
    def __init__(self, spec_channels, gin_channels=0, layernorm=True):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            for i in range(K)
        ])
        out_channels = self._calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels, hidden_size=128, batch_first=True)
        self.proj = nn.Linear(128, gin_channels)
        self.layernorm = nn.LayerNorm(self.spec_channels) if layernorm else None

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)
        if self.layernorm is not None:
            out = self.layernorm(out)
        for conv in self.convs:
            out = F.relu(conv(out))
        out = out.transpose(1, 2)
        out = out.contiguous().view(out.size(0), out.size(1), -1)
        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return self.proj(out.squeeze(0))

    @staticmethod
    def _calculate_channels(L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class OpenVoiceConverterModel(nn.Module):
    def __init__(self, spec_channels, model_cfg):
        super().__init__()
        self.inter_channels = model_cfg["inter_channels"]
        self.hidden_channels = model_cfg["hidden_channels"]
        self.gin_channels = model_cfg["gin_channels"]
        self.zero_g = model_cfg.get("zero_g", True)

        self.dec = _Generator(
            self.inter_channels,
            model_cfg["resblock"],
            model_cfg["resblock_kernel_sizes"],
            model_cfg["resblock_dilation_sizes"],
            model_cfg["upsample_rates"],
            model_cfg["upsample_initial_channel"],
            model_cfg["upsample_kernel_sizes"],
            gin_channels=self.gin_channels,
        )
        self.enc_q = _PosteriorEncoder(
            spec_channels,
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            16,
            gin_channels=self.gin_channels,
        )
        self.flow = _ResidualCouplingBlock(
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            4,
            gin_channels=self.gin_channels,
        )
        self.ref_enc = _ReferenceEncoder(spec_channels, self.gin_channels)

    @torch.inference_mode()
    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, tau=0.3):
        g_src = sid_src
        g_tgt = sid_tgt
        z, _, _, y_mask = self.enc_q(
            y,
            y_lengths,
            g=g_src if not self.zero_g else torch.zeros_like(g_src),
            tau=tau,
        )
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt if not self.zero_g else torch.zeros_like(g_tgt))
        return o_hat


class ToneColorConverterEngine:
    """
    Lightweight Zero-Shot Tone Color Converter (OpenVoiceV2 architecture, 131 MB VRAM).
    Converts fast Edge-TTS Khmer speech into any target reference .wav voice in ~0.15s on GTX 1050 Ti.
    """

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device if (torch.cuda.is_available() and device != "cpu") else "cpu"
        config_path = os.path.join(model_dir, "converter", "config.json")
        ckpt_path = os.path.join(model_dir, "converter", "checkpoint.pth")

        if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
            from huggingface_hub import hf_hub_download
            os.makedirs(model_dir, exist_ok=True)
            config_path = hf_hub_download("myshell-ai/OpenVoiceV2", "converter/config.json", local_dir=model_dir)
            ckpt_path = hf_hub_download("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", local_dir=model_dir)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.sr = cfg["data"]["sampling_rate"]
        self.n_fft = cfg["data"]["filter_length"]
        self.hop_length = cfg["data"]["hop_length"]
        self.win_length = cfg["data"]["win_length"]

        spec_channels = self.n_fft // 2 + 1
        self.model = OpenVoiceConverterModel(spec_channels, cfg["model"]).to(self.device).eval()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model"], strict=True)

        self._hann_window = torch.hann_window(self.win_length).to(self.device)
        self._se_cache = {}
        self._resamplers = {}

    @torch.inference_mode()
    def _fast_load_and_resample(self, audio_path: str) -> torch.Tensor:
        import soundfile as sf
        import torchaudio.functional as AF
        try:
            data, orig_sr = sf.read(audio_path, dtype="float32", always_2d=True)
            wav_1d = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        except Exception:
            wav_1d, orig_sr = librosa.load(audio_path, sr=None, mono=True)
        t = torch.from_numpy(wav_1d).to(self.device, dtype=torch.float32)
        if orig_sr != self.sr:
            t = AF.resample(t.unsqueeze(0), int(orig_sr), int(self.sr)).squeeze(0)
        return t

    @torch.inference_mode()
    def _audio_to_spec(self, wav_in) -> torch.Tensor:
        if isinstance(wav_in, torch.Tensor):
            y = wav_in.float().unsqueeze(0).to(self.device)
        else:
            y = torch.from_numpy(wav_in).float().unsqueeze(0).to(self.device)
        pad = int((self.n_fft - self.hop_length) / 2)
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-6)
        return spec

    @torch.inference_mode()
    def extract_se(self, wav_path: str, cache: bool = True) -> torch.Tensor:
        abs_p = os.path.abspath(wav_path)
        mtime = os.path.getmtime(abs_p) if os.path.exists(abs_p) else 0.0
        cache_key = (abs_p, mtime)
        if cache and cache_key in self._se_cache:
            return self._se_cache[cache_key]

        y, _ = librosa.load(abs_p, sr=self.sr, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        if len(y_trimmed) > self.sr // 2:
            y = y_trimmed

        # Split long references into 8s chunks and average tone color embedding for stability
        chunk_len = self.sr * 8
        chunks = [y[i : i + chunk_len] for i in range(0, len(y), chunk_len) if len(y[i : i + chunk_len]) > self.sr // 2]
        if not chunks:
            chunks = [y]

        gs = []
        for ch in chunks:
            spec = self._audio_to_spec(ch)
            g = self.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
            gs.append(g)

        se = torch.stack(gs, dim=0).mean(dim=0)
        if cache:
            self._se_cache[cache_key] = se
        return se

    @torch.inference_mode()
    def convert_file(self, src_audio_path: str, ref_audio_path: str, tau: float = 0.3) -> Tuple[np.ndarray, int]:
        y_src = self._fast_load_and_resample(src_audio_path)
        spec_src = self._audio_to_spec(y_src)
        y_lengths = torch.LongTensor([spec_src.size(-1)]).to(self.device)

        # Extract source speaker embedding from the generated Edge-TTS clip
        g_src = self.model.ref_enc(spec_src.transpose(1, 2)).unsqueeze(-1)
        # Extract (or retrieve cached) target speaker embedding from reference .wav
        g_tgt = self.extract_se(ref_audio_path, cache=True)

        audio_out = self.model.voice_conversion(spec_src, y_lengths, sid_src=g_src, sid_tgt=g_tgt, tau=tau)
        wav = audio_out[0, 0].data.cpu().float().numpy()

        # Peak normalize softly to match clean broadcast loudness
        peak = np.abs(wav).max()
        if peak > 0.01:
            wav = wav / peak * 0.90
        return wav, self.sr


# ==============================================================================
# TEST 1: VoxCPM2 4GB Low-VRAM Engine (INT8 Linear + FP16 Pascal + Prompt Cache)
# ==============================================================================

class Int8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, has_bias: bool = False, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.empty((out_features, 1), dtype=dtype))
        if has_bias:
            self.register_buffer("bias", torch.empty((out_features,), dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.scale.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


def _replace_linears_with_int8_skeleton(module: nn.Module, dtype=torch.float16):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                Int8Linear(child.in_features, child.out_features, child.bias is not None, dtype=dtype),
            )
        else:
            _replace_linears_with_int8_skeleton(child, dtype=dtype)


def build_or_get_int8_safetensors(model_dir: str) -> str:
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(model_dir, exist_ok=True)
    int8_path = os.path.join(model_dir, "model_int8_4gb.safetensors")
    if os.path.exists(int8_path) and os.path.getsize(int8_path) > 100_000_000:
        return int8_path

    src_path = os.path.join(model_dir, "model.safetensors")
    required_files = [
        "config.json",
        "audiovae.pth",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    missing_meta = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    need_weights = not (os.path.exists(src_path) and os.path.getsize(src_path) > 1_000_000_000)

    if missing_meta or need_weights:
        from huggingface_hub import hf_hub_download, snapshot_download
        for fname in missing_meta:
            try:
                hf_hub_download(repo_id="openbmb/VoxCPM2", filename=fname, local_dir=model_dir)
            except Exception:
                pass
        if need_weights:
            print(f"[VoxCPM2-4GB] Downloading openbmb/VoxCPM2 weights to {model_dir} (first-time setup)...")
            try:
                hf_hub_download(repo_id="openbmb/VoxCPM2", filename="model.safetensors", local_dir=model_dir)
            except Exception:
                snapshot_download(repo_id="openbmb/VoxCPM2", local_dir=model_dir)

    print(f"[VoxCPM2-4GB] Quantizing model.safetensors -> model_int8_4gb.safetensors...")
    out_tensors = {}
    q_device = "cuda" if torch.cuda.is_available() else "cpu"
    with safe_open(src_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            is_target_linear = (
                k.endswith(".weight")
                and t.ndim == 2
                and any(
                    k.startswith(pfx)
                    for pfx in (
                        "base_lm.layers.",
                        "residual_lm.layers.",
                        "feat_encoder.",
                        "feat_decoder.",
                        "enc_to_lm_proj",
                        "lm_to_dit_proj",
                        "res_to_dit_proj",
                        "fusion_concat_proj",
                        "stop_proj",
                        "stop_head",
                    )
                )
            )
            if is_target_linear:
                t_dev = t.to(q_device, dtype=torch.float32)
                max_val = t_dev.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scale = (max_val / 127.0).to(torch.float16)
                w_int8 = torch.round(t_dev / scale.float()).clamp(-127, 127).to(torch.int8)
                out_tensors[k] = w_int8.cpu()
                out_tensors[k[:-7] + ".scale"] = scale.cpu()
                del t_dev, max_val, scale, w_int8
            else:
                out_tensors[k] = t.to(torch.float16) if t.is_floating_point() else t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_file(out_tensors, int8_path)
    # Remove the 4.02 GB raw FP32 model.safetensors after INT8 quantization to save client disk space
    try:
        if os.path.exists(int8_path) and os.path.getsize(int8_path) > 100_000_000 and os.path.exists(src_path):
            os.remove(src_path)
            print(f"[VoxCPM2-4GB] Reclaimed 4.02 GB disk space (removed raw model.safetensors).")
    except Exception:
        pass
    return int8_path


def _promote_int8_to_fp16_linear(module: nn.Module):
    """Converts Int8Linear submodules into native FP16 nn.Linear on CUDA for zero-dequantization inference."""
    for name, child in list(module.named_children()):
        if isinstance(child, Int8Linear):
            lin = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=torch.float16,
            )
            with torch.no_grad():
                lin.weight.copy_(child.weight.to(torch.float16) * child.scale.to(torch.float16))
                if child.bias is not None:
                    lin.bias.copy_(child.bias.to(torch.float16))
            setattr(module, name, lin)
        else:
            _promote_int8_to_fp16_linear(child)


class VoxCPM4GBEngine:
    """
    4GB GTX 1050 Ti Low-VRAM Wrapper for VoxCPM2.
    - Uses INT8 weight-only quantization + FP16 compute (2.28 GB steady VRAM instead of 5.13 GB).
    - Keeps embed_tokens (300 MB) and AudioVAE (360 MB) on CPU while running Transformer + DiT on CUDA.
    - Caches reference voice latents (_encode_wav) in memory so reference .wav is only encoded once.
    """

    def __init__(self, model_dir: str, device: str = "cuda"):
        from safetensors.torch import load_file
        from transformers import LlamaTokenizerFast
        from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
        from voxcpm.modules.audiovae import AudioVAEV2
        from voxcpm.modules.minicpm4.model import MiniCPMLongRoPE
        from voxcpm import VoxCPM

        self.run_device = "cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu"
        if self.run_device == "cpu":
            self._vox = VoxCPM(
                voxcpm_model_path=model_dir,
                enable_denoiser=False,
                optimize=False,
                device="cpu",
            )
            return

        int8_path = build_or_get_int8_safetensors(model_dir)
        for req_meta in ("config.json", "audiovae.pth", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenization_voxcpm2.py"):
            if not os.path.exists(os.path.join(model_dir, req_meta)):
                try:
                    from huggingface_hub import hf_hub_download
                    print(f"[VoxCPM2-4GB] Fetching missing {req_meta} from openbmb/VoxCPM2...")
                    hf_hub_download(repo_id="openbmb/VoxCPM2", filename=req_meta, local_dir=model_dir)
                except Exception:
                    try:
                        import urllib.request
                        url = f"https://huggingface.co/openbmb/VoxCPM2/resolve/main/{req_meta}"
                        urllib.request.urlretrieve(url, os.path.join(model_dir, req_meta))
                    except Exception:
                        pass
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            config = VoxCPMConfig.model_validate_json(f.read())

        config.dtype = "float16"
        config.max_length = 1024
        config.device = "cpu"

        tokenizer = LlamaTokenizerFast.from_pretrained(model_dir)
        audio_vae_config = getattr(config, "audio_vae_config", None)

        with torch.device("meta"):
            audio_vae_meta = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()
            model = VoxCPM2Model(config, tokenizer, audio_vae_meta, lora_config=None, device="cpu")
            _replace_linears_with_int8_skeleton(model.base_lm, dtype=torch.float16)
            _replace_linears_with_int8_skeleton(model.residual_lm, dtype=torch.float16)
            _replace_linears_with_int8_skeleton(model.feat_encoder, dtype=torch.float16)
            _replace_linears_with_int8_skeleton(model.feat_decoder, dtype=torch.float16)
            for proj_name in (
                "enc_to_lm_proj",
                "lm_to_dit_proj",
                "res_to_dit_proj",
                "fusion_concat_proj",
                "stop_proj",
                "stop_head",
            ):
                child = getattr(model, proj_name)
                setattr(
                    model,
                    proj_name,
                    Int8Linear(child.in_features, child.out_features, child.bias is not None, dtype=torch.float16),
                )

        sd = load_file(int8_path, device="cpu")
        model.load_state_dict(sd, strict=False, assign=True)
        del sd

        total_vram_gb = 0.0
        if torch.cuda.is_available():
            try:
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                total_vram_gb = 4.0
        self._vram_tier_gb = total_vram_gb

        audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()
        vae_ckpt = torch.load(os.path.join(model_dir, "audiovae.pth"), map_location="cpu", weights_only=True)
        audio_vae.load_state_dict(vae_ckpt.get("state_dict", vae_ckpt), strict=True)

        if total_vram_gb >= 5.5:
            # 6GB+ GPU (e.g., GTX 1660 Ti 6GB, RTX 3060+): run AudioVAEV2 directly on CUDA
            audio_vae = audio_vae.to(device="cuda", dtype=torch.float32).eval()
            orig_vae_encode = audio_vae.encode
            orig_vae_decode = audio_vae.decode
            audio_vae.encode = lambda x, sr: orig_vae_encode(x.to("cuda", dtype=torch.float32), sr)
            audio_vae.decode = lambda z: orig_vae_decode(z.to("cuda", dtype=torch.float32))
        else:
            # 4GB GPU (e.g., GTX 1050 Ti 4GB): keep AudioVAEV2 on CPU to prevent WDDM paging
            audio_vae = audio_vae.to(device="cpu", dtype=torch.float32).eval()
            orig_vae_encode = audio_vae.encode
            orig_vae_decode = audio_vae.decode
            audio_vae.encode = lambda x, sr: orig_vae_encode(x.cpu().float(), sr)
            audio_vae.decode = lambda z: orig_vae_decode(z.cpu().float())

        embed_cpu = model.base_lm.embed_tokens.to(device="cpu", dtype=torch.float16)
        orig_embed_forward = embed_cpu.forward
        embed_cpu.forward = lambda x: orig_embed_forward(x.cpu()).to("cuda", dtype=torch.float16)

        model.device = "cuda"
        model.config.device = "cuda"

        for name, mod in model.named_children():
            if name == "audio_vae":
                continue
            elif name in ("base_lm", "residual_lm"):
                mod.layers.to("cuda")
                mod.norm.to("cuda", dtype=torch.float16)
                if hasattr(mod, "rope_emb") and mod.rope_emb is not None:
                    mod.rope_emb = MiniCPMLongRoPE(mod.config).to("cuda")
                    mod.rope_emb.cos_cached = mod.rope_emb.cos_cached.to("cuda")
                    mod.rope_emb.sin_cached = mod.rope_emb.sin_cached.to("cuda")
            else:
                for sub in mod.modules():
                    if hasattr(sub, "rope_emb") and sub.rope_emb is not None:
                        sub.rope_emb = MiniCPMLongRoPE(sub.config).to("cuda")
                        sub.rope_emb.cos_cached = sub.rope_emb.cos_cached.to("cuda")
                        sub.rope_emb.sin_cached = sub.rope_emb.sin_cached.to("cuda")
                mod.to("cuda")
                for sub in mod.modules():
                    if hasattr(sub, "rope_emb") and sub.rope_emb is not None:
                        sub.rope_emb.cos_cached = sub.rope_emb.cos_cached.to("cuda")
                        sub.rope_emb.sin_cached = sub.rope_emb.sin_cached.to("cuda")

        for p_name, p in model.named_parameters():
            if "audio_vae" in p_name:
                continue
            if p.is_cuda and p.is_floating_point() and p.dtype != torch.float16:
                p.data = p.data.to(torch.float16)
        for b_name, buf in model.named_buffers():
            if buf.is_cuda and buf.is_floating_point() and buf.dtype != torch.float16 and "rope" not in b_name and "audio_vae" not in b_name:
                buf.data = buf.data.to(torch.float16)

        # Tiered FP16 promotion based on detected GPU VRAM:
        # - 4GB GPU (GTX 1050 Ti): promote feat_decoder + feat_encoder (~2.58 GB VRAM)
        # - 6GB GPU (GTX 1660 Ti): also promote residual_lm + AudioVAE on CUDA (~3.10 GB VRAM)
        # - 8GB+ GPU (RTX 3060/4060+): promote entire model including base_lm to native FP16 (~4.25 GB VRAM)
        _promote_int8_to_fp16_linear(model.feat_decoder)
        _promote_int8_to_fp16_linear(model.feat_encoder)
        if total_vram_gb >= 5.5:
            _promote_int8_to_fp16_linear(model.residual_lm)
        if total_vram_gb >= 7.5:
            _promote_int8_to_fp16_linear(model.base_lm)

        model.base_lm.embed_tokens = embed_cpu
        model.audio_vae = audio_vae
        model.base_lm.setup_cache(1, 1024, "cuda", torch.float16)
        model.residual_lm.setup_cache(1, 1024, "cuda", torch.float16)

        # Cache _encode_wav per (wav_path, mtime, padding_mode, trim_silence_vad)
        self._wav_feat_cache = {}
        orig_encode_wav = model._encode_wav

        def cached_encode_wav(wav_path: str, padding_mode: str = "right", trim_silence_vad: bool = False):
            abs_p = os.path.abspath(wav_path)
            mtime = os.path.getmtime(abs_p) if os.path.exists(abs_p) else 0.0
            key = (abs_p, mtime, padding_mode, trim_silence_vad)
            if key not in self._wav_feat_cache:
                self._wav_feat_cache[key] = orig_encode_wav(wav_path, padding_mode=padding_mode, trim_silence_vad=trim_silence_vad)
            return self._wav_feat_cache[key]

        model._encode_wav = cached_encode_wav
        model.eval()
        torch.cuda.empty_cache()

        wrapper = VoxCPM.__new__(VoxCPM)
        wrapper.tts_model = model
        wrapper.text_normalizer = None
        wrapper.denoiser = None
        self._vox = wrapper
        self._on_gpu = True
        self._gpu_mutex = threading.RLock()

    def offload_to_ram(self):
        """Parks CUDA layers in System RAM in ~0.8s without touching disk or freeing the object."""
        with self._gpu_mutex:
            if getattr(self, "run_device", "cpu") != "cuda" or not getattr(self, "_on_gpu", False):
                return
            model = self._vox.tts_model
            for name, mod in model.named_children():
                if name == "audio_vae":
                    if getattr(self, "_vram_tier_gb", 4.0) >= 5.5:
                        mod.to("cpu")
                    continue
                elif name in ("base_lm", "residual_lm"):
                    mod.layers.to("cpu")
                    mod.norm.to("cpu")
                    if hasattr(mod, "rope_emb") and mod.rope_emb is not None:
                        mod.rope_emb.to("cpu")
                        mod.rope_emb.cos_cached = mod.rope_emb.cos_cached.to("cpu")
                        mod.rope_emb.sin_cached = mod.rope_emb.sin_cached.to("cpu")
                    if hasattr(mod, "kv_cache") and mod.kv_cache is not None and hasattr(mod.kv_cache, "kv_cache"):
                        mod.kv_cache.kv_cache = mod.kv_cache.kv_cache.to("cpu")
                else:
                    mod.to("cpu")
                    for sub in mod.modules():
                        if hasattr(sub, "rope_emb") and sub.rope_emb is not None:
                            sub.rope_emb.cos_cached = sub.rope_emb.cos_cached.to("cpu")
                            sub.rope_emb.sin_cached = sub.rope_emb.sin_cached.to("cpu")
            self._on_gpu = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def reload_to_gpu(self):
        """Restores parked System RAM layers back to CUDA in ~0.7s without reading disk."""
        with self._gpu_mutex:
            if getattr(self, "run_device", "cpu") != "cuda":
                return
            model = self._vox.tts_model
            if not getattr(self, "_on_gpu", False):
                for name, mod in model.named_children():
                    if name == "audio_vae":
                        if getattr(self, "_vram_tier_gb", 4.0) >= 5.5:
                            mod.to("cuda", dtype=torch.float32)
                        continue
                    elif name in ("base_lm", "residual_lm"):
                        mod.layers.to("cuda")
                        mod.norm.to("cuda", dtype=torch.float16)
                        if hasattr(mod, "rope_emb") and mod.rope_emb is not None:
                            mod.rope_emb.to("cuda")
                            mod.rope_emb.cos_cached = mod.rope_emb.cos_cached.to("cuda")
                            mod.rope_emb.sin_cached = mod.rope_emb.sin_cached.to("cuda")
                        if hasattr(mod, "kv_cache") and mod.kv_cache is not None and hasattr(mod.kv_cache, "kv_cache"):
                            mod.kv_cache.kv_cache = mod.kv_cache.kv_cache.to("cuda", dtype=torch.float16)
                    else:
                        mod.to("cuda")
                        for sub in mod.modules():
                            if hasattr(sub, "rope_emb") and sub.rope_emb is not None:
                                sub.rope_emb.cos_cached = sub.rope_emb.cos_cached.to("cuda")
                                sub.rope_emb.sin_cached = sub.rope_emb.sin_cached.to("cuda")
                self._on_gpu = True

            # Ensure all MiniCPMLongRoPE cos_cached/sin_cached and StaticKVCache tensors are on CUDA
            for sub in model.modules():
                if hasattr(sub, "cos_cached") and isinstance(sub.cos_cached, torch.Tensor) and not sub.cos_cached.is_cuda:
                    sub.cos_cached = sub.cos_cached.to("cuda")
                if hasattr(sub, "sin_cached") and isinstance(sub.sin_cached, torch.Tensor) and not sub.sin_cached.is_cuda:
                    sub.sin_cached = sub.sin_cached.to("cuda")
                if hasattr(sub, "kv_cache") and sub.kv_cache is not None and hasattr(sub.kv_cache, "kv_cache"):
                    if not sub.kv_cache.kv_cache.is_cuda:
                        sub.kv_cache.kv_cache = sub.kv_cache.kv_cache.to("cuda", dtype=torch.float16)

    def generate(self, **kwargs):
        with self._gpu_mutex:
            self.reload_to_gpu()
            return self._vox.generate(**kwargs)

