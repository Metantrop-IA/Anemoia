# Merged version of infer_gradio.py with all dependencies inlined (except Assets)
# This file is auto-generated to simplify deployment and execution.
# ...existing code from infer_gradio.py and all required dependencies will be inlined here...

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import soundfile as sf
from cached_path import cached_path
import gradio as gr
import faiss
from third_party.BigVGAN import bigvgan
from x_transformers import RMSNorm
from x_transformers.x_transformers import apply_rotary_pos_emb
from vocos import Vocos
from torchdiffeq import odeint
import jieba
from pypinyin import lazy_pinyin, Style
import librosa.filters
from num2words import num2words
from sentence_transformers import SentenceTransformer

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False

# =====================
# UTILS & MODEL CLASSES
# =====================

# --- Begin: model/utils.py ---
from collections import defaultdict
import os
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import jieba
from pypinyin import lazy_pinyin, Style

def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def lens_to_mask(t, length=None):
    if not exists(length):
        length = t.amax()
    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]

def mask_from_start_end_indices(seq_len, start, end):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask

def mask_from_frac_lengths(seq_len, frac_lengths):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths
    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths
    return mask_from_start_end_indices(seq_len, start, end)

def maybe_masked_mean(t, mask=None):
    if not exists(mask):
        return t.mean(dim=1)
    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)
    return num / den.clamp(min=1.0)

def list_str_to_tensor(text, padding_value=-1):
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text

def list_str_to_idx(text, vocab_char_map, padding_value=-1):
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    if tokenizer in ["pinyin", "char"]:
        from importlib.resources import files
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"
    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256
    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
    return vocab_char_map, vocab_size

def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    god_knows_why_en_testset_contains_zh_quote = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    custom_trans = str.maketrans({";": ","})
    for text in text_list:
        char_list = []
        text = text.translate(god_knows_why_en_testset_contains_zh_quote)
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):
                seg = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in seg:
                    if c not in "。，、；：？！《》【】—…":
                        char_list.append(" ")
                    char_list.append(c)
            else:
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    else:
                        if c not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:
                            char_list.append(c)
        final_text_list.append(char_list)
    return final_text_list

def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False

# --- End: model/utils.py ---

# --- Begin: model/cfm.py ---
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
# MelSpec, default, exists, lens_to_mask, list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths are already inlined above

class CFM(nn.Module):
    def __init__(self, transformer: nn.Module, sigma=0.0, odeint_kwargs=dict(method="euler"), audio_drop_prob=0.3, cond_drop_prob=0.2, num_channels=None, mel_spec_module=None, mel_spec_kwargs=dict(), frac_lengths_mask=(0.7, 1.0), vocab_char_map=None):
        super().__init__()
        self.frac_lengths_mask = frac_lengths_mask
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.vocab_char_map = vocab_char_map
    @property
    def device(self):
        return next(self.parameters()).device
    @torch.no_grad()
    def sample(self, cond, text, duration, *, lens=None, steps=32, cfg_strength=1.0, sway_sampling_coef=None, seed=None, max_duration=4096, vocoder=None, no_ref_audio=False, duplicate_test=False, t_inter=0.1, edit_mask=None):
        self.eval()
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        cond = cond.to(next(self.parameters()).dtype)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(text_lens, lens)
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
        duration = torch.maximum(lens + 1, duration)
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None
        if no_ref_audio:
            cond = torch.zeros_like(cond)
        def fn(t, x):
            pred = self.transformer(x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False)
            if cfg_strength < 1e-5:
                return pred
            null_pred = self.transformer(x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True)
            return pred + (pred - null_pred) * cfg_strength
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)
        t_start = 0
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))
        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)
        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        return out, trajectory
    def forward(self, inp, text, *, lens=None, noise_scheduler=None):
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels
        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        if exists(mask):
            rand_span_mask &= mask
        x1 = inp
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        drop_audio_cond = random() < self.audio_drop_prob
        if random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False
        pred = self.transformer(x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text)
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]
        return loss.mean(), cond, pred
# --- End: model/cfm.py ---

# --- Begin: model/backbones/dit.py ---
import torch
from torch import nn
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding
# TimestepEmbedding, ConvNeXtV2Block, ConvPositionEmbedding, DiTBlock, AdaLayerNormZero_Final, precompute_freqs_cis, get_pos_embed_indices assumed inlined or stubbed
class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False
    def forward(self, text, seq_len, drop_text=False):
        text = text + 1
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        if drop_text:
            text = torch.zeros_like(text)
        text = self.text_embed(text)
        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed
            text = self.text_blocks(text)
        return text
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)
    def forward(self, x, cond, text_embed, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x
class DiT(nn.Module):
    def __init__(self, *, dim, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4, mel_dim=100, text_num_embeds=256, text_dim=None, conv_layers=0, long_skip_connection=False):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList([DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)])
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
    def forward(self, x, cond, text, time, drop_audio_cond, drop_text, mask=None):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        if self.long_skip_connection is not None:
            residual = x
        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output
# --- End: model/backbones/dit.py ---

# --- Begin: model/backbones/unett.py ---
import torch
from torch import nn
import torch.nn.functional as F
from x_transformers import RMSNorm
from x_transformers.x_transformers import RotaryEmbedding
# TimestepEmbedding, ConvNeXtV2Block, ConvPositionEmbedding, Attention, AttnProcessor, FeedForward, precompute_freqs_cis, get_pos_embed_indices assumed inlined or stubbed
class TextEmbeddingUNet(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False
    def forward(self, text, seq_len, drop_text=False):
        text = text + 1
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        if drop_text:
            text = torch.zeros_like(text)
        text = self.text_embed(text)
        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed
            text = self.text_blocks(text)
        return text
class InputEmbeddingUNet(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)
    def forward(self, x, cond, text_embed, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x
class UNetT(nn.Module):
    def __init__(self, *, dim, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4, mel_dim=100, text_num_embeds=256, text_dim=None, conv_layers=0, skip_connect_type="concat"):
        super().__init__()
        assert depth % 2 == 0, "UNet-Transformer's depth should be even."
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbeddingUNet(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbeddingUNet(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.skip_connect_type = skip_connect_type
        needs_skip_proj = skip_connect_type == "concat"
        self.depth = depth
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            is_later_half = idx >= (depth // 2)
            attn_norm = RMSNorm(dim)
            attn = Attention(processor=AttnProcessor(), dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff_norm = RMSNorm(dim)
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")
            skip_proj = nn.Linear(dim * 2, dim, bias=False) if needs_skip_proj and is_later_half else None
            self.layers.append(nn.ModuleList([skip_proj, attn_norm, attn, ff_norm, ff]))
        self.norm_out = RMSNorm(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
    def forward(self, x, cond, text, time, drop_audio_cond, drop_text, mask=None):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        x = torch.cat([t.unsqueeze(1), x], dim=1)
        if mask is not None:
            mask = F.pad(mask, (1, 0), value=1)
        rope = self.rotary_embed.forward_from_seq_len(seq_len + 1)
        skip_connect_type = self.skip_connect_type
        skips = []
        for idx, (maybe_skip_proj, attn_norm, attn, ff_norm, ff) in enumerate(self.layers):
            layer = idx + 1
            is_first_half = layer <= (self.depth // 2)
            is_later_half = not is_first_half
            if is_first_half:
                skips.append(x)
            if is_later_half:
                skip = skips.pop()
                if skip_connect_type == "concat":
                    x = torch.cat((x, skip), dim=-1)
                    x = maybe_skip_proj(x)
                elif skip_connect_type == "add":
                    x = x + skip
            x = attn(attn_norm(x), rope=rope, mask=mask) + x
            x = ff(ff_norm(x)) + x
        assert len(skips) == 0
        x = self.norm_out(x)[:, 1:, :]
        return self.proj_out(x)
# --- End: model/backbones/unett.py ---

# --- Begin: model/modules.py (required classes only) ---
import math
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb
from vocos import Vocos

# MelSpec
class MelSpec(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos"):
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        if mel_spec_type == "vocos":
            self.extractor = self.get_vocos_mel_spectrogram
        else:
            self.extractor = self.get_bigvgan_mel_spectrogram
        self.register_buffer("dummy", torch.tensor(0), persistent=False)
    def get_vocos_mel_spectrogram(self, waveform, n_fft=1024, n_mel_channels=100, target_sample_rate=24000, hop_length=256, win_length=1024):
        mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        ).to(waveform.device)
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)
        mel = mel_stft(waveform)
        mel = mel.clamp(min=1e-5).log()
        return mel
    def get_bigvgan_mel_spectrogram(self, waveform, n_fft=1024, n_mel_channels=100, target_sample_rate=24000, hop_length=256, win_length=1024, fmin=0, fmax=None, center=False):
        from librosa.filters import mel as librosa_mel_fn
        device = waveform.device
        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis = torch.from_numpy(mel).float().to(device)
        hann_window = torch.hann_window(win_length).to(device)
        padding = (n_fft - hop_length) // 2
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spec = torch.stft(
            waveform,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel_spec = torch.matmul(mel_basis, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec
    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)
        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        return mel

# SinusPositionEmbedding
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# ConvPositionEmbedding
class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)

def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos

# ConvNeXtV2Block
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x

# AdaLayerNormZero
class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
# AdaLayerNormZero_Final
class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
# FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))
    def forward(self, x):
        return self.ff(x)
# Attention
class Attention(nn.Module):
    def __init__(self, processor, dim, heads=8, dim_head=64, dropout=0.0, context_dim=None, context_pre_only=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0+")
        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(context_dim, self.inner_dim)
        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))
        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)
    def forward(self, x, c=None, mask=None, rope=None, c_rope=None):
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)
# AttnProcessor
class AttnProcessor:
    def __init__(self):
        pass
    def __call__(self, attn, x, mask=None, rope=None):
        batch_size = x.shape[0]
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
        return x
# DiTBlock
class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(processor=AttnProcessor(), dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")
    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x
# TimestepEmbedding
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)
        return time
# --- End: model/modules.py ---

# REMOVE: from f5_tts.model import DiT, UNetT
# REMOVE: from f5_tts.model.utils import seed_everything

# Constants
# ===============
target_sample_rate = 24000
hop_length = 256
n_mel_channels = 100
n_fft = 1024
win_length = 1024
mel_spec_type = "vocos"
ode_method = "euler"

# Stubs for missing functions
# ===============
def load_checkpoint(model, ckpt_path, device, dtype=None, use_ema=True):
    # Dummy: just return the model
    return model

def remove_silence_for_generated_wav(file_wave):
    # Dummy: do nothing
    pass

def save_spectrogram(spect, file_spect):
    # Dummy: do nothing
    pass

def tqdm(iterable=None, **kwargs):
    # Dummy: just return the iterable
    return iterable if iterable is not None else range(1)

def preprocess_ref_audio_text(ref_file, ref_text, device=None, show_info=None):
    # Dummy: just return the inputs
    return ref_file, ref_text

def infer_process(ref_file, ref_text, gen_text, ema_model, vocoder, mel_spec_type=None, show_info=None, progress=None, target_rms=0.1, cross_fade_duration=0.15, nfe_step=32, cfg_strength=2, sway_sampling_coef=-1, speed=1.0, fix_duration=None, device=None):
    # Load and process reference audio
    ref_audio, sr = torchaudio.load(ref_file)
    ref_audio = ref_audio.to(device)
    if sr != target_sample_rate:
        ref_audio = torchaudio.functional.resample(ref_audio, sr, target_sample_rate)
    
    # Get mel spectrogram of reference audio
    mel_spec = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type or "vocos"
    ).to(device)
    
    ref_mel = mel_spec(ref_audio)
    
    # Generate speech
    duration = int(len(gen_text) * speed * target_sample_rate / hop_length)
    if fix_duration:
        duration = int(fix_duration * target_sample_rate / hop_length)
    
    # Use the model to generate audio
    with torch.no_grad():
        generated, _ = ema_model.sample(
            ref_mel,
            [gen_text],
            duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            vocoder=vocoder
        )
    
    # Convert to numpy array
    wav = generated.squeeze().cpu().numpy()
    
    # Normalize audio
    if target_rms:
        current_rms = np.sqrt(np.mean(wav ** 2))
        wav = wav * (target_rms / current_rms)
    
    # Get final mel spectrogram
    with torch.no_grad():
        spect = mel_spec(torch.from_numpy(wav).float().to(device))
        spect = spect.squeeze().cpu().numpy()
    
    return wav, target_sample_rate, spect

# Fix empty function bodies in process_audio_input and generate_audio_response
def process_audio_input(audio_path, text, history, conv_state):
    pass


def generate_audio_response(history, ref_audio, ref_text, model_choice, remove_silence):
    if not history or not ref_audio:
        return None
    # Find the last assistant message
    last_message = next((msg for msg in reversed(history) if msg["role"] == "assistant"), None)
    if not last_message:
        return None
    # Always use the F5TTS_ema_model for now
    audio_result, _ = infer(
        ref_audio,
        ref_text,
        last_message["content"],
        F5TTS_ema_model,
        remove_silence,
        cross_fade_duration=0.15,
        speed=1.0,
        show_info=print
    )
    return audio_result


class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        local_path=None,
        device=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name

        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load models
        self.load_vocoder_model(vocoder_name, local_path)
        self.load_ema_model(model_type, ckpt_file, vocoder_name, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, vocoder_name, local_path):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema):
        if model_type == "F5-TTS":
            if not ckpt_file:
                if mel_spec_type == "vocos":
                    ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
                elif mel_spec_type == "bigvgan":
                    ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        save_spectrogram(spect, file_spect)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
    ):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spect is not None:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect


# =====================
# MAIN GRADIO APP LOGIC
# =====================

import re
import tempfile
import os
import click
import gradio as gr
import numpy as np
import soundfile as sf
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
from num2words import num2words
from sentence_transformers import SentenceTransformer
import faiss
try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False

def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func

# load vocoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device):
    if vocoder_name == "vocos":
        if is_local:
            print(f"Load vocos from local path {local_path}")
            vocoder = Vocos.from_hparams(f"{local_path}/config.yaml")
            state_dict = torch.load(f"{local_path}/pytorch_model.bin", map_location="cpu")
            vocoder.load_state_dict(state_dict)
            vocoder = vocoder.eval().to(device)
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = os.path.join(os.path.dirname(__file__), "Assets", "vocab.txt")
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("tokenizer : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model

# --- Model/vocoder loading ---
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors")))
chat_model_state = None
chat_tokenizer_state = None

@gpu_decorator
def generate_response(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)
    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')
    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)
    return texto_traducido

@gpu_decorator
def infer(ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1, show_info=gr.Info):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
    ema_model = F5TTS_ema_model
    if not gen_text.startswith(" "):
        gen_text = " " + gen_text
    if not gen_text.endswith(". "):
        gen_text += ". "
    gen_text = gen_text.lower()
    gen_text = traducir_numero_a_texto(gen_text)
    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)
    return (final_sample_rate, final_wave), spectrogram_path

def parse_speechtypes_text(gen_text):
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_style = "Regular"
    for i in range(len(tokens)):
        if i % 2 == 0:
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            style = tokens[i].strip()
            current_style = style
    return segments

# ========== GRADIO APP ========== #
with gr.Blocks() as app_chat:
    gr.Markdown("""
# Mnemosynth
""")
    chat_interface_container = gr.Column()
    if chat_model_state is None:
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        chat_model_state = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)
    ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Assets"))
    initial_prompt_path = os.path.join(ASSETS_DIR, "Initial_Prompt.txt")
    voice_ref_trans_path = os.path.join(ASSETS_DIR, "Voice_Ref_Trans.txt")
    docs_RAG_path = os.path.join(ASSETS_DIR, "Docs_RAG.txt")
    voice_ref_wav_path = os.path.join(ASSETS_DIR, "Voice_Ref.wav")
    try:
        with open(initial_prompt_path, "r", encoding="utf-8") as f:
            initial_prompt = f.read().strip()
    except Exception as e:
        initial_prompt = "No eres un asistente de IA, eres quien el usuario diga que eres..."
    try:
        with open(docs_RAG_path, "r", encoding="utf-8") as f:
            docs = f.read().strip()
    except Exception as e:
        docs = ""
    try:
        with open(voice_ref_trans_path, "r") as f:
            voice_ref_trans = f.read().strip()
    except Exception as e:
        voice_ref_trans = ""
    docs_list = [line.strip() for line in docs.splitlines() if line.strip()]
    if not docs_list:
        embedding_model = None
        index = None
        def retrieve_context(query, top_k=3):
            return []
    else:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        doc_embeddings = embedding_model.encode(docs_list, convert_to_numpy=True)
        index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        def retrieve_context(query, top_k=3):
            query_emb = embedding_model.encode([query], convert_to_numpy=True)
            D, I = index.search(query_emb, top_k)
            return [docs_list[i] for i in I[0]]
    ref_audio_chat = gr.Audio(value=voice_ref_wav_path, visible=False, type="filepath")
    model_choice_chat = gr.Radio(choices=["F5-TTS"], label="Seleccionar Modelo TTS", value="F5-TTS", visible=False)
    remove_silence_chat = gr.Checkbox(value=True, visible=False)
    ref_text_chat = gr.Textbox(value=voice_ref_trans, visible=False)
    system_prompt_chat = gr.Textbox(value=initial_prompt, visible=False)
    with gr.Row():
        with gr.Column():
            audio_input_chat = gr.Microphone(label="Graba tu mensaje",type="filepath")
            audio_output_chat = gr.Audio(autoplay=True, label="Respuesta")
        with gr.Column():
            text_input_chat = gr.Textbox(label="Escribe tu mensaje",lines=1)
            send_btn_chat = gr.Button("Enviar")
            clear_btn_chat = gr.Button("Limpiar Conversación")
            chatbot_interface = gr.Chatbot(label="Conversación", type="messages")
    conversation_state = gr.State(value=[{"role": "system", "content": initial_prompt}])
    @gpu_decorator
    def process_audio_input(audio_path, text, history, conv_state):
        if not audio_path and not text.strip():
            return history, conv_state
        if audio_path:
            text = preprocess_ref_audio_text(audio_path, text)[1]
        if not text.strip():
            return history, conv_state
        contexto = " ".join(retrieve_context(text, top_k=3))
        conv_state.append({"role": "system", "content": f"Contexto relevante: {contexto}"})
        conv_state.append({"role": "user", "content": text})
        response = generate_response(conv_state, chat_model_state, chat_tokenizer_state)
        conv_state.append({"role": "assistant", "content": response})
        if not history:
            history = []
        history.extend([
            {"role": "user", "content": text},
            {"role": "assistant", "content": response}
        ])
        return history, conv_state

    @gpu_decorator
    def generate_audio_response(history, ref_audio, ref_text, model_choice, remove_silence):
        if not history or not ref_audio:
            return None
        # Find the last assistant message
        last_message = next((msg for msg in reversed(history) if msg["role"] == "assistant"), None)
        if not last_message:
            return None
        # Always use the F5TTS_ema_model for now
        audio_result, _ = infer(
            ref_audio,
            ref_text,
            last_message["content"],
            F5TTS_ema_model,
            remove_silence,
            cross_fade_duration=0.15,
            speed=1.0,
            show_info=print
        )
        return audio_result
    def clear_conversation():
        return [], [
            {"role": "system", "content": initial_prompt}
        ]
    def update_system_prompt(new_prompt):
        new_conv_state = [{"role": "system", "content": new_prompt}]
        return [], new_conv_state
    audio_input_chat.stop_recording(
        fn=process_audio_input,
        inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
        outputs=[chatbot_interface, conversation_state]
    ).then(
        fn=generate_audio_response,
        inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
        outputs=[audio_output_chat]
    )
    text_input_chat.submit(
        process_audio_input,
        inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
        outputs=[chatbot_interface, conversation_state],
    ).then(
        generate_audio_response,
        inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
        outputs=[audio_output_chat],
    ).then(
        lambda: None,
        None,
        text_input_chat,
    )
    send_btn_chat.click(
        process_audio_input,
        inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
        outputs=[chatbot_interface, conversation_state],
    ).then(
        generate_audio_response,
        inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
        outputs=[audio_output_chat],
    ).then(
        lambda: None,
        None,
        text_input_chat,
    )
    clear_btn_chat.click(
        clear_conversation,
        outputs=[chatbot_interface, conversation_state],
    )
    system_prompt_chat.change(
        update_system_prompt,
        inputs=system_prompt_chat,
        outputs=[chatbot_interface, conversation_state],
    )
# ========== END GRADIO APP ========== #

with gr.Blocks() as app_credits:
    gr.Markdown("""
# Créditos
* [ΜΕΤΑΝΘΡΩΠΙΑ](https://github.com/METANTROP-IA) por el [demo de Mnemosynth](https://github.com/Metantrop-IA/Mnemosynth)
* [mrfakename](https://github.com/fakerybakery) por el [demo online original](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) por la generación inicial de fragmentos y exploración de la aplicación de podcast
* [jpgallegoar](https://github.com/jpgallegoar) por la generación de múltiples tipos de habla, chat de voz y afinación en español
""")

with gr.Blocks() as app:
    gr.TabbedInterface(
        [app_chat, app_credits],
        ["Mnemosynth", "Créditos"],
    )
@click.command()
@click.option("--port", "-p", default=None, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-H", default=None, help="Host para ejecutar la aplicación")
@click.option("--share", "-s", default=False, is_flag=True, help="Compartir la aplicación a través de un enlace compartido de Gradio")
@click.option("--api", "-a", default=True, is_flag=True, help="Permitir acceso a la API")
def main(port, host, share, api):
    global app
    print("Iniciando la aplicación...")
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=True, show_api=api)
if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
