from __future__ import annotations

import io
import logging
import re
import subprocess
import struct
import unicodedata
import wave
from typing import Iterator

import numpy as np
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Voice mapping: OpenAI voice names → MOSS-TTS-Nano preset names
# ---------------------------------------------------------------------------

_OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy": "Junhao",    # zh_1.wav
    "echo": "Xiaoyu",     # zh_3.wav
    "fable": "Yuewen",    # zh_4.wav
    "onyx": "Adam",       # en_4.wav
    "nova": "Lingyu",     # zh_6.wav
    "shimmer": "Bella",   # en_3.wav
    "ash": "Ava",         # en_2.wav
    "sage": "Junhao",     # zh_1.wav
    "coral": "Xiaoyu",    # zh_3.wav
    "noova": "Lingyu",    # zh_6.wav
    "ballad": "男播音",    # zh_10.wav
    "yangmi": "杨幂",      # zh_11.wav
}


def resolve_voice(voice: str) -> str:
    """Map an OpenAI voice name to a MOSS-TTS-Nano preset, or pass through."""
    return _OPENAI_VOICE_MAP.get(voice, voice)


# ---------------------------------------------------------------------------
# Emoji / kaomoji stripping
# ---------------------------------------------------------------------------

# Unicode emoji ranges (pictographs, symbols, modifiers, flags, etc.)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # ZWJ
    "\U00002B50"             # star
    "\U0000203C-\U00003299"  # misc symbols
    "\U0000FE00-\U0000FEFF"  # variation selectors & BOM
    "]+",
    re.UNICODE,
)

# Common kaomoji patterns: loose match for face-like punctuation clusters
# Uses a heuristic: sequences of CJK symbols/punctuation + special chars
# that are 3+ chars long and look like faces
_KAOMOJI_RE = re.compile(
    r"(?:"
    # Shrug: ¯\_(...)_/¯
    r"¯\\?_?\(.*?\)_?/?¯"
    r"|"
    # Face with brackets: (●◡●) (╯°□°)╯ etc.
    r"\([^)]*[◉◎⊙●◡▼▽ᗜᴖᴗ◕‿◕°□※✧※][^)]*\)"
    r"|"
    # Table flip and arm-like gestures
    r"\(╯[°□◉]\)?╯\s*[︵︶╰].*?(?:╰\s*[︶╯]\s*[°□◉]\s*╰\s*\))?"
    r"|"
    # Simple arm gestures: ヽ(…)ﾉ  ヽ(。_。)ﾉ etc
    r"[ヽヾ]\(.*?\)[ﾉヾ]"
    r"|"
    # Flipping tables: ┻━┻  ┣━┫  ┳━┳
    r"[┣┻┳╚╗╔╝]\s*[━═]\s*[┫┻┳╚╗╔╝]"
    r"|"
    # Raised arms / action: (ノಠ益ಠ)ノ彡
    r"\(ノ[^)]*ಠ[^)]*\)ノ[^)]*"
    r")",
    re.UNICODE,
)


def strip_emoji(text: str) -> str:
    """Remove emoji, kaomoji, and zero-width modifiers from *text*."""
    text = _KAOMOJI_RE.sub("", text)
    out: list[str] = []
    skip_next_variation = False
    i = 0
    while i < len(text):
        ch = text[i]
        cp = ord(ch)
        cat = unicodedata.category(ch)

        # Zero-width / variation selectors / combining marks that follow emoji
        if cat == "Cf" or cp == 0x200D:  # ZWJ, variation selectors, etc.
            i += 1
            continue

        # Emoji ranges: actual pictographic codepoints only
        if (
            0x1F600 <= cp <= 0x1F64F   # emoticons
            or 0x1F300 <= cp <= 0x1F5FF  # misc symbols & pictographs
            or 0x1F680 <= cp <= 0x1F6FF  # transport & map
            or 0x1F900 <= cp <= 0x1F9FF  # supplemental symbols
            or 0x1FA00 <= cp <= 0x1FAFF  # symbols extended-A
            or 0x1F1E6 <= cp <= 0x1F1FF  # regional indicators (flags)
            or 0x1F3FB <= cp <= 0x1F3FF  # skin tone modifiers
            or 0x2600 <= cp <= 0x26FF    # misc symbols
            or 0x2702 <= cp <= 0x27B0    # dingbats
            or cp == 0x2B50              # star
            or cp == 0x3030              # wavy dash
            or cp == 0x303D              # part alternation mark
            or cp == 0x3297              # circled "congratulations"
            or cp == 0x3299              # circled "secret"
            or cp == 0xFE0F              # variation selector-16
        ):
            i += 1
            continue

        out.append(ch)
        i += 1

    text = "".join(out)
    # Clean up stray kaomoji fragments (╯ ︵ ╰ ┳ ┻ etc.)
    text = re.sub(r"[╯╰︵︶┳┻┣┫━═彡ッツ]", "", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def _number_to_chinese(n: int) -> str:
    """Convert integer 0–100 to spoken Chinese. Falls back to digits for >100."""
    _D = "零一二三四五六七八九"
    if n == 0:
        return "零"
    if n < 10:
        return _D[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + (_D[n % 10] if n % 10 else "")
    if n < 100:
        tens, ones = divmod(n, 10)
        return _D[tens] + "十" + (_D[ones] if ones else "")
    if n == 100:
        return "百"
    return str(n)


def preprocess_tts_input(text: str) -> str:
    """Strip emoji/kaomoji, normalize newlines, and convert symbols to readable text.

    Translates unit symbols (°C, km/h, %, etc.) and range operators (~) into
    Chinese text so the TTS model can pronounce them correctly.
    """
    text = strip_emoji(text)

    # --- Time format: HH:MM → H点 / H点M分 (must run before colon→。) ---
    def _time_repl(m: re.Match) -> str:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if minute == 0:
            return f"{hour}点"
        return f"{hour}点{minute}分"

    text = re.sub(r"(\d{1,2}):(\d{2})(?!\d)", _time_repl, text)

    # --- Symbol-to-text: simple 1:1 replacements, no merging ---

    # Convert ~ → 到
    text = re.sub(r"\s*~\s*", "到", text)

    # Convert units
    text = re.sub(r"(\d)\s*°C", r"\1摄氏度", text)
    text = re.sub(r"(\d)\s*°F", r"\1华氏度", text)
    text = re.sub(r"(\d)\s*℃", r"\1摄氏度", text)
    text = re.sub(r"(\d)\s*km/h\b", r"\1千米每小时", text)
    text = re.sub(r"(\d)\s*m/s\b", r"\1米每秒", text)
    text = re.sub(r"(\d)\s*mph\b", r"\1英里每小时", text)

    # Percent: N% → 百分之N  (Chinese reads "百分之" before the number)
    def _pct_repl(m: re.Match) -> str:
        val = float(m.group(1))
        if val == int(val) and 0 <= int(val) <= 100:
            return "百分之" + _number_to_chinese(int(val))
        return "百分之" + m.group(1)

    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", _pct_repl, text)

    # --- Punctuation normalization for TTS chunking ---

    # Replace ：(full-width colon) with ，(comma) so labels and values
    # stay together as one natural phrase (e.g. "气温，十八摄氏度").
    # Using 。was causing the model to repeat the last phrase at boundaries.
    text = re.sub(r"[：:]", "，", text)

    # Replace em/en dashes with commas
    text = re.sub(r"\s*[—–]\s*", "，", text)

    # --- Newline normalization ---
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = text.replace("\n", "。")
    # Clean up consecutive punctuation like ，。 or 。，
    text = re.sub(r"[，。]{2,}", lambda m: "。" if "。" in m.group() else "，", text)
    text = text.lstrip("。")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    """OpenAI-compatible ``POST /v1/audio/speech`` request body."""

    model: str = "tts-1"
    input: str
    voice: str
    response_format: str = Field(default="wav", pattern=r"^(wav|mp3|pcm|opus)$")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _audio_to_pcm16le(audio_array: np.ndarray) -> bytes:
    """Convert float32 numpy audio to raw PCM signed-16-bit little-endian."""
    audio_np = np.asarray(audio_array, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return (audio_np * 32767.0).astype(np.int16).tobytes()


def _resample_pcm(pcm: bytes, speed: float, channels: int = 1) -> bytes:
    """Resample PCM s16le audio by *speed* factor using linear interpolation.

    speed > 1.0 = faster/shorter, speed < 1.0 = slower/longer.
    Changes pitch (simple resampling). For pitch-preserving speed change,
    use the ffmpeg atempo filter (used in opus path).
    Handles multi-channel (interleaved) PCM correctly.
    """
    if speed == 1.0 or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if channels > 1:
        # De-interleave, resample each channel, re-interleave
        per_ch = len(samples) // channels
        reshaped = samples.reshape(per_ch, channels)
        new_len = max(1, int(per_ch / speed))
        indices = np.linspace(0, per_ch - 1, new_len)
        resampled = np.column_stack([
            np.interp(indices, np.arange(per_ch), reshaped[:, ch])
            for ch in range(channels)
        ])
    else:
        new_len = max(1, int(len(samples) / speed))
        indices = np.linspace(0, len(samples) - 1, new_len)
        resampled = np.interp(indices, np.arange(len(samples)), samples)
    return resampled.astype(np.int16).tobytes()


def _wav_header_bytes(sample_rate: int, channels: int, data_length: int = 0) -> bytes:
    """Build a 44-byte RIFF WAV header.

    When *data_length* is 0 the header uses ``0x7FFFFFFF`` as a placeholder
    size so that streaming clients do not reject the file for being "too short".
    """
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    # If total data length unknown (streaming), use a large placeholder.
    data_size = data_length if data_length > 0 else 0x7FFFFFFF
    file_size = 36 + data_size

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def _wav_bytes_from_pcm(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Wrap a complete PCM buffer in a WAV container."""
    header = _wav_header_bytes(sample_rate, channels, data_length=len(pcm))
    return header + pcm


# ---------------------------------------------------------------------------
# MP3 encoding (lazy lameenc import)
# ---------------------------------------------------------------------------

def _encode_pcm_to_mp3(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Encode one PCM chunk to MP3 using *lameenc*."""
    try:
        import lameenc
    except ImportError:
        raise RuntimeError(
            "MP3 encoding requires the 'lameenc' package. "
            "Install it with: pip install lameenc"
        )

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)  # high quality
    return bytes(encoder.encode(pcm)) + bytes(encoder.flush())


# ---------------------------------------------------------------------------
# Streaming generators
# ---------------------------------------------------------------------------

def iter_pcm_audio(
    events: Iterator[tuple[dict, str, int]],
) -> Iterator[tuple[bytes, int, int]]:
    """Yield ``(pcm_bytes, sample_rate, channels)`` from synthesize_stream events.

    *events* comes from ``RequestRuntimeManager.iter_with_runtime`` which yields
    ``(event_dict, execution_device, cpu_threads)`` tuples.
    """
    for item in events:
        # iter_with_runtime yields (event, device, threads) tuples
        event = item[0] if isinstance(item, tuple) else item
        event_type = str(event.get("type", ""))
        if event_type != "audio":
            continue
        waveform = np.asarray(event["waveform_numpy"], dtype=np.float32)
        sample_rate = int(event["sample_rate"])
        channels = 1 if waveform.ndim == 1 else int(waveform.shape[1])
        pcm = _audio_to_pcm16le(waveform)
        if pcm:
            yield bytes(pcm), sample_rate, channels


def generate_wav_stream(events: Iterator[dict]) -> Iterator[bytes]:
    """Yield WAV-formatted chunks: header first, then raw PCM data."""
    header_sent = False
    for pcm, sample_rate, channels in iter_pcm_audio(events):
        if not header_sent:
            yield _wav_header_bytes(sample_rate, channels)
            header_sent = True
        yield pcm


def generate_pcm_stream(events: Iterator[dict]) -> Iterator[bytes]:
    """Yield raw PCM bytes directly."""
    for pcm, _, _ in iter_pcm_audio(events):
        yield pcm


def generate_mp3_stream(events: Iterator[dict]) -> Iterator[bytes]:
    """Yield MP3 frames; encodes each PCM chunk independently."""
    encoder = None
    try:
        import lameenc
    except ImportError:
        raise RuntimeError(
            "MP3 encoding requires the 'lameenc' package. "
            "Install it with: pip install lameenc"
        )

    for pcm, sample_rate, channels in iter_pcm_audio(events):
        if encoder is None:
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_channels(channels)
            encoder.set_quality(2)
        yield bytes(encoder.encode(pcm))

    if encoder is not None:
        flush = encoder.flush()
        if flush:
            yield bytes(flush)


# ---------------------------------------------------------------------------
# Opus encoding via ffmpeg subprocess
# ---------------------------------------------------------------------------

_OPUS_FRAME_SIZE = 960  # 20ms at 48kHz, the standard Opus frame


def start_opus_encoder(sample_rate: int, channels: int, speed: float = 1.0) -> subprocess.Popen:
    """Start an ffmpeg subprocess that accepts PCM on stdin, produces Ogg/Opus on stdout."""
    audio_filters = []
    if speed != 1.0:
        # atempo range is [0.5, 100.0]; for values outside, chain multiple filters
        remaining = speed
        while remaining > 100.0:
            audio_filters.append("atempo=100.0")
            remaining /= 100.0
        while remaining < 0.5:
            audio_filters.append("atempo=0.5")
            remaining /= 0.5
        audio_filters.append(f"atempo={remaining:.4f}")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels),
        "-i", "-",
    ]
    if audio_filters:
        cmd.extend(["-af", ",".join(audio_filters)])
    cmd.extend([
        "-c:a", "libopus", "-b:a", "65536",
        "-f", "ogg", "-",
    ])
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------

def make_error_response(
    message: str,
    *,
    param: str | None = None,
    error_type: str = "invalid_request_error",
    status_code: int = 400,
) -> tuple[dict, int]:
    """Return ``(body_dict, http_status)`` following OpenAI error schema."""
    body = {
        "error": {
            "message": message,
            "type": error_type,
        }
    }
    if param is not None:
        body["error"]["param"] = param
    return body, status_code


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------

FORMAT_CONTENT_TYPE: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "pcm": "audio/pcm",
    "opus": "audio/opus",
}
