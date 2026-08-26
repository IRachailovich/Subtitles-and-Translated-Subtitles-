"""Bounded, resumable, audio-grounded long-form transcription primitives.

This module deliberately knows nothing about provider credentials or HTTP APIs.
Callers supply a callback that transcribes one bounded audio chunk.  The shared
coordinator is therefore usable by the CLI, local web UI, frozen desktop
backend, Drive batches, mobile/LAN UI, and the cloud worker.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LONGFORM_PIPELINE_VERSION = "2026-08-17-longform-v9-evidence-gated-repetition"
COVERAGE_RECOVERY_SELECTION_VERSION = (
    "2026-08-06-acceptance-blockers-v3"
)


def _hidden_subprocess_kwargs() -> Dict[str, Any]:
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_bitrate_bits_per_second(value: Any) -> int:
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kmg]?)", text)
    if not match:
        raise ValueError(f"Unsupported audio bitrate: {value!r}")
    number = float(match.group(1))
    multiplier = {"": 1, "k": 1000, "m": 1000**2, "g": 1000**3}[
        match.group(2)
    ]
    return int(number * multiplier)


def probe_media(path: Path | str) -> Dict[str, Any]:
    """Return exact media preflight information without decoding the full file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input media does not exist: {path}")
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration,format_name,bit_rate:"
                "stream=index,codec_type,codec_name,channels,sample_rate"
            ),
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_hidden_subprocess_kwargs(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FFprobe could not inspect the input media: "
            f"{completed.stderr[-2000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
        duration = float((payload.get("format") or {}).get("duration"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("FFprobe returned no valid media duration.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Input media has an invalid duration: {duration!r}")
    audio_streams = [
        stream
        for stream in payload.get("streams") or []
        if stream.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise RuntimeError("Input media contains no audio stream.")
    usage = shutil.disk_usage(path.parent)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "duration_seconds": round(duration, 6),
        "format_name": (payload.get("format") or {}).get("format_name"),
        "bit_rate": (payload.get("format") or {}).get("bit_rate"),
        "audio_streams": audio_streams,
        "available_disk_bytes": usage.free,
    }


def merge_intervals(
    intervals: Iterable[Sequence[float]],
    *,
    join_gap_seconds: float = 0.0,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
) -> List[Tuple[float, float]]:
    cleaned = []
    for interval in intervals or []:
        if len(interval) < 2:
            continue
        start = max(float(clip_start), float(interval[0]))
        end = float(interval[1])
        if clip_end is not None:
            end = min(float(clip_end), end)
        if math.isfinite(start) and math.isfinite(end) and end > start:
            cleaned.append((start, end))
    cleaned.sort()
    merged: List[List[float]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1] + float(join_gap_seconds):
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def subtract_intervals(
    reference: Iterable[Sequence[float]],
    covered: Iterable[Sequence[float]],
) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    covered_merged = merge_intervals(covered)
    for ref_start, ref_end in merge_intervals(reference):
        cursor = ref_start
        for cov_start, cov_end in covered_merged:
            if cov_end <= cursor:
                continue
            if cov_start >= ref_end:
                break
            if cov_start > cursor:
                result.append((cursor, min(cov_start, ref_end)))
            cursor = max(cursor, cov_end)
            if cursor >= ref_end:
                break
        if cursor < ref_end:
            result.append((cursor, ref_end))
    return merge_intervals(result)


def interval_overlap_seconds(
    start: float,
    end: float,
    intervals: Iterable[Sequence[float]],
) -> float:
    total = 0.0
    for interval_start, interval_end in merge_intervals(intervals):
        if interval_end <= start:
            continue
        if interval_start >= end:
            break
        total += max(0.0, min(end, interval_end) - max(start, interval_start))
    return total


def ffmpeg_silence_intervals(
    media_path: Path | str,
    duration_seconds: float,
    *,
    noise_db: float = -45.0,
    min_silence_seconds: float = 0.25,
) -> List[Tuple[float, float]]:
    """Scan signal energy with bounded memory and return low-energy intervals."""
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(media_path),
            "-map",
            "0:a:0",
            "-af",
            (
                f"silencedetect=noise={float(noise_db):g}dB:"
                f"d={float(min_silence_seconds):g}"
            ),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_hidden_subprocess_kwargs(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg silence analysis failed: "
            f"{completed.stderr[-2000:]}"
        )
    intervals: List[Tuple[float, float]] = []
    open_start: Optional[float] = None
    for line in (completed.stderr or "").splitlines():
        start_match = re.search(r"silence_start:\s*(-?[0-9.]+)", line)
        if start_match:
            open_start = max(0.0, float(start_match.group(1)))
            continue
        end_match = re.search(
            r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)",
            line,
        )
        if not end_match:
            continue
        end = min(float(duration_seconds), float(end_match.group(1)))
        if open_start is None:
            open_start = max(0.0, end - float(end_match.group(2)))
        if end > open_start:
            intervals.append((open_start, end))
        open_start = None
    if open_start is not None and open_start < duration_seconds:
        intervals.append((open_start, duration_seconds))
    return merge_intervals(intervals, clip_end=duration_seconds)


def _decode_pcm_window(
    media_path: Path | str,
    start_seconds: float,
    duration_seconds: float,
    *,
    sample_rate: int = 16000,
) -> bytes:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{float(start_seconds):.3f}",
            "-t",
            f"{float(duration_seconds):.3f}",
            "-i",
            str(media_path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(int(sample_rate)),
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        **_hidden_subprocess_kwargs(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not decode a VAD analysis window: "
            f"{completed.stderr.decode('utf-8', errors='replace')[-2000:]}"
        )
    return completed.stdout


def silero_speech_intervals(
    media_path: Path | str,
    duration_seconds: float,
    *,
    analysis_window_seconds: float = 600.0,
    analysis_overlap_seconds: float = 2.0,
    threshold: float = 0.45,
    min_speech_ms: int = 150,
    min_silence_ms: int = 350,
    speech_pad_ms: int = 350,
) -> List[Tuple[float, float]]:
    """Run Faster-Whisper's bundled Silero VAD in bounded PCM windows."""
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    sample_rate = 16000
    window_seconds = max(30.0, float(analysis_window_seconds))
    window_overlap = max(
        0.0,
        min(float(analysis_overlap_seconds), window_seconds / 4.0),
    )
    intervals: List[Tuple[float, float]] = []
    window_start = 0.0
    while window_start < duration_seconds - 0.001:
        window_duration = min(window_seconds, duration_seconds - window_start)
        pcm = _decode_pcm_window(
            media_path,
            window_start,
            window_duration,
            sample_rate=sample_rate,
        )
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        options = VadOptions(
            threshold=float(threshold),
            min_speech_duration_ms=int(min_speech_ms),
            max_speech_duration_s=60.0,
            min_silence_duration_ms=int(min_silence_ms),
            speech_pad_ms=int(speech_pad_ms),
        )
        for speech in get_speech_timestamps(
            audio,
            vad_options=options,
            sampling_rate=sample_rate,
        ):
            intervals.append(
                (
                    window_start + float(speech["start"]) / sample_rate,
                    window_start + float(speech["end"]) / sample_rate,
                )
            )
        if window_start + window_duration >= duration_seconds - 0.001:
            break
        window_start += max(1.0, window_duration - window_overlap)
    return merge_intervals(
        intervals,
        join_gap_seconds=0.15,
        clip_end=duration_seconds,
    )


def build_speech_map(
    media_path: Path | str,
    duration_seconds: float,
    *,
    engine: str = "auto",
    silence_noise_db: float = -45.0,
    min_silence_seconds: float = 0.25,
    confirmed_silence_seconds: float = 1.0,
    vad_analysis_window_seconds: float = 600.0,
    vad_analysis_overlap_seconds: float = 2.0,
    vad_threshold: float = 0.45,
    vad_min_speech_ms: int = 150,
    vad_min_silence_ms: int = 350,
    vad_speech_pad_ms: int = 350,
) -> Dict[str, Any]:
    """Build speech, possible-audio, uncertain, and confirmed-silence regions.

    Silero is the primary speech detector when available.  Energy detection is
    intentionally retained as a second, conservative detector: energy without
    VAD speech is uncertain audio, not confirmed silence.
    """
    duration_seconds = float(duration_seconds)
    silence = ffmpeg_silence_intervals(
        media_path,
        duration_seconds,
        noise_db=silence_noise_db,
        min_silence_seconds=min_silence_seconds,
    )
    full = [(0.0, duration_seconds)]
    energy_intervals = subtract_intervals(full, silence)
    requested_engine = str(engine or "auto").strip().lower()
    vad_error = None
    speech_intervals: List[Tuple[float, float]] = []
    vad_used = False
    if requested_engine in {"auto", "silero", "silero_vad"}:
        try:
            speech_intervals = silero_speech_intervals(
                media_path,
                duration_seconds,
                analysis_window_seconds=vad_analysis_window_seconds,
                analysis_overlap_seconds=vad_analysis_overlap_seconds,
                threshold=vad_threshold,
                min_speech_ms=vad_min_speech_ms,
                min_silence_ms=vad_min_silence_ms,
                speech_pad_ms=vad_speech_pad_ms,
            )
            vad_used = True
        except Exception as exc:
            vad_error = f"{type(exc).__name__}: {exc}"
            if requested_engine not in {"auto"}:
                raise
    if not vad_used:
        # Conservative fallback: all non-silent signal is treated as possible
        # speech, so an empty ASR result cannot silently pass.
        speech_intervals = list(energy_intervals)
    uncertain = (
        subtract_intervals(energy_intervals, speech_intervals)
        if vad_used
        else []
    )
    long_silence = [
        (start, end)
        for start, end in silence
        if end - start >= float(confirmed_silence_seconds)
    ]
    confirmed_silence = subtract_intervals(long_silence, speech_intervals)
    speech_seconds = sum(end - start for start, end in speech_intervals)
    uncertain_seconds = sum(end - start for start, end in uncertain)
    return {
        "schema": "subgen_speech_map_v1",
        "engine": "silero_vad_plus_ffmpeg_energy" if vad_used else "ffmpeg_energy_conservative",
        "requested_engine": requested_engine,
        "vad_available": vad_used,
        "vad_error": vad_error,
        "duration_seconds": round(duration_seconds, 6),
        "speech_seconds": round(speech_seconds, 3),
        "uncertain_audio_seconds": round(uncertain_seconds, 3),
        "speech_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in speech_intervals
        ],
        "possible_audio_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in energy_intervals
        ],
        "uncertain_audio_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in uncertain
        ],
        "confirmed_silence_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in confirmed_silence
        ],
        "parameters": {
            "silence_noise_db": float(silence_noise_db),
            "min_silence_seconds": float(min_silence_seconds),
            "confirmed_silence_seconds": float(confirmed_silence_seconds),
            "vad_analysis_window_seconds": float(vad_analysis_window_seconds),
            "vad_analysis_overlap_seconds": float(vad_analysis_overlap_seconds),
            "vad_threshold": float(vad_threshold),
            "vad_min_speech_ms": int(vad_min_speech_ms),
            "vad_min_silence_ms": int(vad_min_silence_ms),
            "vad_speech_pad_ms": int(vad_speech_pad_ms),
        },
    }


def make_adaptive_chunk_plan(
    duration_seconds: float,
    speech_map: Mapping[str, Any],
    *,
    target_seconds: float = 300.0,
    min_seconds: float = 180.0,
    max_seconds: float = 480.0,
    overlap_seconds: float = 12.0,
    boundary_search_seconds: float = 60.0,
    min_boundary_silence_seconds: float = 1.0,
    include_overlap_context: bool = True,
) -> List[Dict[str, Any]]:
    """Plan bounded chunks, preferring the midpoint of real silence."""
    duration = max(0.0, float(duration_seconds))
    if duration <= 0:
        return []
    target = max(30.0, float(target_seconds))
    minimum = max(15.0, min(float(min_seconds), target))
    maximum = max(target, float(max_seconds))
    search = max(0.0, float(boundary_search_seconds))
    context = max(0.0, float(overlap_seconds)) / 2.0 if include_overlap_context else 0.0
    candidates = []
    for start, end in speech_map.get("confirmed_silence_intervals") or []:
        start = float(start)
        end = float(end)
        if end - start >= float(min_boundary_silence_seconds):
            candidates.append((start + end) / 2.0)

    boundaries = [0.0]
    cursor = 0.0
    boundary_kinds = []
    while duration - cursor > maximum:
        desired = cursor + target
        lower = cursor + minimum
        upper = min(duration, cursor + maximum)
        nearby = [
            candidate
            for candidate in candidates
            if lower <= candidate <= upper
            and abs(candidate - desired) <= search
        ]
        if nearby:
            boundary = min(nearby, key=lambda value: abs(value - desired))
            kind = "confirmed_silence"
        else:
            boundary = min(upper, desired)
            kind = "hard_cut"
        if boundary <= cursor + 0.001:
            boundary = min(duration, cursor + target)
            kind = "hard_cut"
        boundaries.append(boundary)
        boundary_kinds.append(kind)
        cursor = boundary
    boundaries.append(duration)

    plan = []
    for index in range(len(boundaries) - 1):
        ownership_start = boundaries[index]
        ownership_end = boundaries[index + 1]
        start = max(0.0, ownership_start - context)
        end = min(duration, ownership_end + context)
        plan.append({
            "index": index + 1,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "ownership_start": round(ownership_start, 3),
            "ownership_end": round(ownership_end, 3),
            "left_boundary_kind": (
                "media_start" if index == 0 else boundary_kinds[index - 1]
            ),
            "right_boundary_kind": (
                "media_end"
                if index >= len(boundary_kinds)
                else boundary_kinds[index]
            ),
        })
    return plan


def extract_audio_chunk(
    media_path: Path | str,
    output_path: Path | str,
    start_seconds: float,
    duration_seconds: float,
    *,
    sample_rate: int = 16000,
    bitrate: str = "64k",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{float(start_seconds):.3f}",
            "-t",
            f"{float(duration_seconds):.3f}",
            "-i",
            str(media_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(sample_rate)),
            "-b:a",
            str(bitrate),
            str(output_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_hidden_subprocess_kwargs(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to extract transcription chunk {output_path}: "
            f"{completed.stderr[-2000:]}"
        )
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"FFmpeg created no usable audio chunk: {output_path}")


def _artifact_dict(artifact: Any) -> Dict[str, Any]:
    if is_dataclass(artifact):
        return asdict(artifact)
    if isinstance(artifact, Mapping):
        return dict(artifact)
    raise TypeError(
        "Chunk transcription callback must return a dataclass or mapping artifact."
    )


def _normalize_segment(segment: Mapping[str, Any], offset: float) -> Optional[Dict[str, Any]]:
    text = re.sub(r"\s+", " ", str(segment.get("text") or segment.get("word") or "")).strip()
    if not text:
        return None
    try:
        start = float(segment.get("start")) + offset
        end = float(segment.get("end")) + offset
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        return None
    normalized = {"start": start, "end": end, "text": text}
    speaker = segment.get("speaker")
    if speaker is not None:
        speaker = str(speaker).strip()
        if speaker.casefold() not in {
            "",
            "none",
            "null",
            "unknown",
            "n/a",
            "na",
            "not specified",
            "unspecified",
        }:
            normalized["speaker"] = speaker
    for key in (
        "language",
        "confidence",
        "probability",
        "overlap",
        "overlap_turns",
        "_granularity",
    ):
        if segment.get(key) is not None:
            normalized[key] = segment.get(key)
    return normalized


def offset_artifact(artifact: Mapping[str, Any], offset: float) -> Dict[str, Any]:
    result = dict(artifact)
    result["segments"] = [
        normalized
        for item in artifact.get("segments") or []
        if (normalized := _normalize_segment(item, offset)) is not None
    ]
    result["words"] = [
        normalized
        for item in artifact.get("words") or []
        if (normalized := _normalize_segment(item, offset)) is not None
    ]
    return result


def _segment_score_for_ownership(segment: Mapping[str, Any], chunk: Mapping[str, Any]) -> float:
    center = (float(segment["start"]) + float(segment["end"])) / 2.0
    ownership_center = (
        float(chunk["ownership_start"]) + float(chunk["ownership_end"])
    ) / 2.0
    return abs(center - ownership_center)


def merge_owned_timed_segments(
    chunk_results: Sequence[Mapping[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge timestamped results by non-overlapping time ownership.

    Lexical similarity is never used to delete a later occurrence.  Exact text
    is collapsed only when two chunks observed substantially the same absolute
    time interval.
    """
    owned = []
    merge_report = []
    for result in chunk_results:
        chunk = result["chunk"]
        kept = []
        rejected = []
        for segment in result.get("segments") or []:
            center = (float(segment["start"]) + float(segment["end"])) / 2.0
            is_last = bool(result.get("is_last"))
            in_window = center >= float(chunk["ownership_start"]) and (
                center < float(chunk["ownership_end"])
                or (is_last and center <= float(chunk["ownership_end"]) + 0.001)
            )
            if in_window:
                candidate = dict(segment)
                candidate["_chunk_index"] = int(chunk["index"])
                candidate["_ownership_score"] = _segment_score_for_ownership(
                    candidate,
                    chunk,
                )
                kept.append(candidate)
            else:
                rejected.append(segment)
        owned.extend(kept)
        merge_report.append({
            "chunk_index": int(chunk["index"]),
            "method": "absolute_time_ownership",
            "segments_kept": len(kept),
            "overlap_observations_rejected": len(rejected),
        })

    owned.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    deduplicated: List[Dict[str, Any]] = []
    for candidate in owned:
        duplicate_index = None
        candidate_text = str(candidate.get("text") or "").casefold()
        for index in range(max(0, len(deduplicated) - 4), len(deduplicated)):
            previous = deduplicated[index]
            if str(previous.get("text") or "").casefold() != candidate_text:
                continue
            overlap = interval_overlap_seconds(
                float(candidate["start"]),
                float(candidate["end"]),
                [(previous["start"], previous["end"])],
            )
            shorter = min(
                float(candidate["end"]) - float(candidate["start"]),
                float(previous["end"]) - float(previous["start"]),
            )
            if shorter > 0 and overlap / shorter >= 0.60:
                duplicate_index = index
                break
        if duplicate_index is None:
            deduplicated.append(candidate)
            continue
        previous = deduplicated[duplicate_index]
        if float(candidate.get("_ownership_score", math.inf)) < float(
            previous.get("_ownership_score", math.inf)
        ):
            deduplicated[duplicate_index] = candidate

    cleaned = []
    for segment in deduplicated:
        item = {
            key: value
            for key, value in segment.items()
            if not key.startswith("_ownership") and key != "_chunk_index"
        }
        cleaned.append(item)
    text = re.sub(
        r"\s+",
        " ",
        " ".join(str(segment.get("text") or "") for segment in cleaned),
    ).strip()
    return text, cleaned, merge_report


def merge_text_only_chunks(
    chunk_results: Sequence[Mapping[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Concatenate non-overlapping text-only chunks without deduplication."""
    texts = []
    report = []
    for result in chunk_results:
        text = re.sub(r"\s+", " ", str(result.get("text") or "")).strip()
        if text:
            texts.append(text)
        report.append({
            "chunk_index": int(result["chunk"]["index"]),
            "method": "non_overlapping_audio_concatenation",
            "text_chars": len(text),
            "right_boundary_kind": result["chunk"].get("right_boundary_kind"),
        })
    return re.sub(r"\s+", " ", " ".join(texts)).strip(), report


def invalid_chunk_split_boundary(
    chunk: Mapping[str, Any],
    speech_map: Mapping[str, Any],
    *,
    minimum_part_seconds: float,
) -> Optional[float]:
    ownership_start = float(chunk["ownership_start"])
    ownership_end = float(chunk["ownership_end"])
    minimum = max(1.0, float(minimum_part_seconds))
    if ownership_end - ownership_start < minimum * 2.0:
        return None
    lower = ownership_start + minimum
    upper = ownership_end - minimum
    target = (ownership_start + ownership_end) / 2.0
    candidates = [
        (float(start) + float(end)) / 2.0
        for start, end in speech_map.get("confirmed_silence_intervals") or []
        if lower <= (float(start) + float(end)) / 2.0 <= upper
    ]
    if candidates:
        return min(candidates, key=lambda value: abs(value - target))
    return min(upper, max(lower, target))


def retry_invalid_chunk_as_splits(
    media_path: Path,
    output_dir: Path,
    chunk: Mapping[str, Any],
    *,
    parent_chunk_identity: str,
    transcribe_chunk: Callable[[str, Mapping[str, Any]], Any],
    speech_map: Mapping[str, Any],
    expected_timing: bool,
    minimum_part_seconds: float,
    overlap_seconds: float,
    sample_rate: int,
    bitrate: str,
) -> Optional[Dict[str, Any]]:
    """Retry one rejected provider chunk as two smaller sequential chunks."""
    boundary = invalid_chunk_split_boundary(
        chunk,
        speech_map,
        minimum_part_seconds=minimum_part_seconds,
    )
    if boundary is None:
        return None
    overlap = max(0.0, float(overlap_seconds))
    parent_start = float(chunk["start"])
    parent_end = float(chunk["end"])
    ownership_start = float(chunk["ownership_start"])
    ownership_end = float(chunk["ownership_end"])
    split_chunks = [
        {
            "index": int(chunk["index"]),
            "validation_split_retry_part": 1,
            "start": parent_start,
            "end": min(parent_end, boundary + overlap / 2.0),
            "ownership_start": ownership_start,
            "ownership_end": boundary,
            "left_boundary_kind": chunk.get("left_boundary_kind"),
            "right_boundary_kind": "invalid_chunk_split_retry",
        },
        {
            "index": int(chunk["index"]),
            "validation_split_retry_part": 2,
            "start": max(parent_start, boundary - overlap / 2.0),
            "end": parent_end,
            "ownership_start": boundary,
            "ownership_end": ownership_end,
            "left_boundary_kind": "invalid_chunk_split_retry",
            "right_boundary_kind": chunk.get("right_boundary_kind"),
        },
    ]
    for split in split_chunks:
        split["duration"] = float(split["end"]) - float(split["start"])

    split_results = []
    audio_paths = []
    cached_part_count = 0
    completed = False
    try:
        for split in split_chunks:
            part = int(split["validation_split_retry_part"])
            split_identity = canonical_sha256({
                "parent_chunk_identity": parent_chunk_identity,
                "split": split,
            })
            result_path = output_dir / (
                f"chunk-{int(chunk['index']):04d}."
                f"validation-retry-{part:02d}.result.json"
            )
            cached_record = None
            if result_path.exists():
                try:
                    candidate = json.loads(
                        result_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    candidate = None
                if (
                    isinstance(candidate, Mapping)
                    and candidate.get("schema")
                    == "subgen_longform_validation_split_result_v1"
                    and candidate.get("split_identity") == split_identity
                    and isinstance(candidate.get("result"), Mapping)
                    and candidate["result"].get("status") == "complete"
                ):
                    cached_record = dict(candidate["result"])
            if cached_record is not None:
                split_results.append(cached_record)
                cached_part_count += 1
                continue
            audio_path = output_dir / (
                f"chunk-{int(chunk['index']):04d}."
                f"validation-retry-{part:02d}.mp3"
            )
            audio_paths.append(audio_path)
            extract_audio_chunk(
                media_path,
                audio_path,
                split["start"],
                split["duration"],
                sample_rate=sample_rate,
                bitrate=bitrate,
            )
            artifact = offset_artifact(
                _artifact_dict(
                    transcribe_chunk(str(audio_path), split)
                ),
                float(split["start"]),
            )
            text = re.sub(
                r"\s+",
                " ",
                str(artifact.get("text") or ""),
            ).strip()
            if not text:
                raise RuntimeError(
                    "Split validation retry returned no transcript text "
                    f"for chunk {chunk['index']} part {part}."
                )
            if expected_timing and not artifact.get("segments"):
                raise RuntimeError(
                    "Split validation retry returned text without timestamps "
                    f"for chunk {chunk['index']} part {part}."
                )
            split_result = {
                "chunk": split,
                "is_last": part == len(split_chunks),
                "status": "complete",
                "text": text,
                "segments": artifact.get("segments") or [],
                "words": artifact.get("words") or [],
                "language": artifact.get("language"),
                "timing_kind": artifact.get("timing_kind") or "none",
                "usage": artifact.get("usage") or {},
                "metadata": artifact.get("metadata") or {},
            }
            atomic_write_json(
                result_path,
                {
                    "schema": (
                        "subgen_longform_validation_split_result_v1"
                    ),
                    "parent_chunk_identity": parent_chunk_identity,
                    "split_identity": split_identity,
                    "split": split,
                    "result": split_result,
                },
            )
            split_results.append(split_result)
            audio_path.unlink(missing_ok=True)
        if expected_timing:
            text, segments, merge_report = merge_owned_timed_segments(
                split_results
            )
        else:
            text, merge_report = merge_text_only_chunks(split_results)
            segments = []
        words = []
        for result in split_results:
            split = result["chunk"]
            for word in result.get("words") or []:
                center = (
                    float(word["start"]) + float(word["end"])
                ) / 2.0
                if (
                    center >= float(split["ownership_start"])
                    and (
                        center < float(split["ownership_end"])
                        or (
                            result["is_last"]
                            and center
                            <= float(split["ownership_end"]) + 0.001
                        )
                    )
                ):
                    words.append(dict(word))
        languages = list(dict.fromkeys(
            result.get("language")
            for result in split_results
            if result.get("language")
        ))
        completed = True
        return {
            "text": text,
            "segments": segments,
            "words": words,
            "language": (
                languages[0]
                if len(languages) == 1
                else ("mixed" if languages else None)
            ),
            "timing_kind": (
                split_results[0].get("timing_kind", "none")
            ),
            "usage": {
                "invalid_chunk_split_retry": True,
                "cached_part_count": cached_part_count,
                "parts": [
                    result.get("usage") or {}
                    for result in split_results
                ],
            },
            "metadata": {
                "invalid_chunk_split_retry": {
                    "boundary": round(boundary, 3),
                    "part_count": len(split_results),
                    "cached_part_count": cached_part_count,
                    "overlap_seconds": overlap,
                    "merge_report": merge_report,
                },
            },
        }
    finally:
        if completed:
            for audio_path in audio_paths:
                audio_path.unlink(missing_ok=True)


def _compact_token(value: Any) -> str:
    return re.sub(r"[^\w]+", "", str(value or "").casefold(), flags=re.UNICODE)


def apply_coverage_recovery(
    segments: Sequence[Mapping[str, Any]],
    recovery_artifact: Mapping[str, Any],
    gap: Sequence[float],
    *,
    context_seconds: float = 6.0,
    min_novel_word_probability: float = 0.60,
    existing_match_tolerance_seconds: float = 0.75,
    independent_speech_intervals: Optional[
        Sequence[Sequence[float]]
    ] = None,
    allow_unscored_timed_segments: bool = False,
    min_unscored_speech_overlap_ratio: float = 0.80,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Use a bounded retry to repair missing timing coverage without deduping text."""
    repaired = [dict(item) for item in segments]
    gap_start, gap_end = float(gap[0]), float(gap[1])
    context_start = max(0.0, gap_start - float(context_seconds))
    context_end = gap_end + float(context_seconds)
    expanded = []
    added = []
    rejected_novel = []

    recovery_words = [
        dict(item)
        for item in recovery_artifact.get("words") or []
        if item.get("start") is not None
        and item.get("end") is not None
        and float(item["end"]) > gap_start
        and float(item["start"]) < gap_end
        and _compact_token(item.get("text") or item.get("word"))
    ]
    recovery_language_uncertain = bool(
        (recovery_artifact.get("metadata") or {}).get(
            "language_detection_uncertain"
        )
    )
    for word in recovery_words:
        text = re.sub(
            r"\s+",
            " ",
            str(word.get("text") or word.get("word") or ""),
        ).strip()
        key = _compact_token(text)
        word_start = float(word["start"])
        word_end = float(word["end"])
        candidates = []
        for index, segment in enumerate(repaired):
            if (
                float(segment["end"]) < context_start
                or float(segment["start"]) > context_end
            ):
                continue
            segment_keys = {
                _compact_token(token)
                for token in str(segment.get("text") or "").split()
            }
            if key in segment_keys:
                segment_start = float(segment["start"])
                segment_end = float(segment["end"])
                if word_end < segment_start:
                    interval_distance = segment_start - word_end
                elif segment_end < word_start:
                    interval_distance = word_start - segment_end
                else:
                    interval_distance = 0.0
                if interval_distance > float(
                    existing_match_tolerance_seconds
                ):
                    continue
                distance = abs(
                    (segment_start + segment_end) / 2.0
                    - (word_start + word_end) / 2.0
                )
                candidates.append((distance, index))
        if candidates:
            _, index = min(candidates)
            segment = repaired[index]
            original = (float(segment["start"]), float(segment["end"]))
            segment["start"] = min(original[0], word_start)
            segment["end"] = max(original[1], word_end)
            expanded.append({
                "segment_index": index,
                "word": text,
                "original_start": round(original[0], 3),
                "original_end": round(original[1], 3),
                "recovered_start": round(word_start, 3),
                "recovered_end": round(word_end, 3),
            })
        else:
            center = (word_start + word_end) / 2.0
            if gap_start <= center <= gap_end:
                probability = word.get("probability")
                if probability is None:
                    probability = word.get("confidence")
                duration = max(0.001, word_end - word_start)
                signal_overlap_ratio = (
                    interval_overlap_seconds(
                        word_start,
                        word_end,
                        independent_speech_intervals or [],
                    )
                    / duration
                )
                timing_kind = str(
                    recovery_artifact.get("timing_kind") or ""
                )
                independently_supported = bool(
                    allow_unscored_timed_segments
                    and not recovery_language_uncertain
                    and timing_kind
                    in {
                        "native_word",
                        "native_segment",
                        "prompted_segment",
                    }
                    and signal_overlap_ratio
                    >= float(min_unscored_speech_overlap_ratio)
                )
                confidence_supported = bool(
                    probability is not None
                    and float(probability)
                    >= float(min_novel_word_probability)
                    and not recovery_language_uncertain
                )
                if not (
                    independently_supported
                    or confidence_supported
                ):
                    rejected_novel.append({
                        "text": text,
                        "start": round(word_start, 3),
                        "end": round(word_end, 3),
                        "probability": probability,
                        "signal_overlap_ratio": round(
                            signal_overlap_ratio,
                            4,
                        ),
                        "reason": (
                            "novel_word_uncertain_recovery_language"
                            if recovery_language_uncertain
                            else "unscored_novel_word_recovery"
                            if probability is None
                            else "novel_word_below_confidence_gate"
                        ),
                    })
                    continue
                recovered = {
                    "start": word_start,
                    "end": word_end,
                    "text": text,
                    "_coverage_recovery": True,
                }
                if independently_supported:
                    recovered["_independent_signal_supported"] = True
                for key_name in (
                    "speaker",
                    "language",
                    "confidence",
                    "probability",
                ):
                    if word.get(key_name) is not None:
                        recovered[key_name] = word[key_name]
                repaired.append(recovered)
                added.append({
                    "text": text,
                    "start": round(word_start, 3),
                    "end": round(word_end, 3),
                })

    if not recovery_words:
        for item in recovery_artifact.get("segments") or []:
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
            if not text or end <= gap_start or start >= gap_end:
                continue
            neighboring_text = " ".join(
                str(segment.get("text") or "")
                for segment in repaired
                if float(segment["end"]) >= context_start
                and float(segment["start"]) <= context_end
            )
            recovered_tokens = {
                _compact_token(token)
                for token in text.split()
                if _compact_token(token)
            }
            neighbor_tokens = {
                _compact_token(token)
                for token in neighboring_text.split()
                if _compact_token(token)
            }
            if recovered_tokens and recovered_tokens.issubset(neighbor_tokens):
                candidates = [
                    (index, segment)
                    for index, segment in enumerate(repaired)
                    if float(segment["end"]) >= context_start
                    and float(segment["start"]) <= context_end
                    and not (
                        end < float(segment["start"])
                        and float(segment["start"]) - end
                        > float(existing_match_tolerance_seconds)
                    )
                    and not (
                        float(segment["end"]) < start
                        and start - float(segment["end"])
                        > float(existing_match_tolerance_seconds)
                    )
                    and any(
                        _compact_token(token) in recovered_tokens
                        for token in str(segment.get("text") or "").split()
                    )
                ]
                if candidates:
                    first_index, first = candidates[0]
                    last_index, last = candidates[-1]
                    first["start"] = min(float(first["start"]), start)
                    last["end"] = max(float(last["end"]), end)
                    expanded.append({
                        "segment_indexes": [first_index, last_index],
                        "segment_retry": True,
                        "recovered_start": round(start, 3),
                        "recovered_end": round(end, 3),
                    })
                else:
                    rejected_novel.append({
                        "text": text,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "probability": item.get("probability")
                        or item.get("confidence"),
                        "reason": "distant_lexical_segment_match",
                    })
            else:
                center = (start + end) / 2.0
                if gap_start <= center <= gap_end:
                    probability = item.get("probability")
                    if probability is None:
                        probability = item.get("confidence")
                    duration = max(0.001, end - start)
                    signal_overlap_ratio = (
                        interval_overlap_seconds(
                            start,
                            end,
                            independent_speech_intervals or [],
                        )
                        / duration
                    )
                    timing_kind = str(
                        recovery_artifact.get("timing_kind") or ""
                    )
                    independently_supported = bool(
                        allow_unscored_timed_segments
                        and not recovery_language_uncertain
                        and timing_kind
                        in {
                            "native_word",
                            "native_segment",
                            "prompted_segment",
                        }
                        and signal_overlap_ratio
                        >= float(min_unscored_speech_overlap_ratio)
                    )
                    confidence_supported = bool(
                        probability is not None
                        and float(probability)
                        >= float(min_novel_word_probability)
                        and not recovery_language_uncertain
                    )
                    if not (
                        independently_supported
                        or confidence_supported
                    ):
                        rejected_novel.append({
                            "text": text,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "probability": probability,
                            "signal_overlap_ratio": round(
                                signal_overlap_ratio,
                                4,
                            ),
                            "reason": (
                                "novel_segment_uncertain_recovery_language"
                                if recovery_language_uncertain
                                else "unscored_novel_segment_recovery"
                            ),
                        })
                        continue
                    recovered = {
                        "start": start,
                        "end": end,
                        "text": text,
                        "_coverage_recovery": True,
                    }
                    if independently_supported:
                        recovered[
                            "_independent_signal_supported"
                        ] = True
                    for key_name in (
                        "speaker",
                        "language",
                        "confidence",
                        "probability",
                    ):
                        if item.get(key_name) is not None:
                            recovered[key_name] = item[key_name]
                    repaired.append(recovered)
                    added.append({
                        "text": text,
                        "start": round(start, 3),
                        "end": round(end, 3),
                    })

    repaired.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return repaired, {
        "gap": [round(gap_start, 3), round(gap_end, 3)],
        "recovered_word_count": len(recovery_words),
        "expanded_existing": expanded,
        "added_novel": added,
        "rejected_novel": rejected_novel,
        "existing_match_tolerance_seconds": float(
            existing_match_tolerance_seconds
        ),
        "allow_unscored_timed_segments": bool(
            allow_unscored_timed_segments
        ),
        "min_unscored_speech_overlap_ratio": float(
            min_unscored_speech_overlap_ratio
        ),
        "changed": bool(expanded or added),
    }


def coalesce_coverage_recovery_gaps(
    gaps: Sequence[Sequence[float]],
    *,
    join_seconds: float,
    max_window_seconds: float,
) -> List[List[float]]:
    """Join a missing run without creating an unbounded recovery request."""
    ordered = sorted(
        (
            [float(item[0]), float(item[1])]
            for item in gaps or []
            if len(item) >= 2 and float(item[1]) > float(item[0])
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        current = merged[-1]
        proposed_end = max(float(current[1]), end)
        if (
            start - float(current[1]) <= float(join_seconds)
            and proposed_end - float(current[0])
            <= float(max_window_seconds)
        ):
            current[1] = proposed_end
        else:
            merged.append([start, end])
    return merged


def select_coverage_recovery_gaps(
    validation: Mapping[str, Any],
    *,
    max_uncovered_gap_seconds: float,
    join_seconds: float,
    max_window_seconds: float,
    max_attempts: int,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Select only gaps capable of resolving the active validation failure."""
    raw_gaps = [
        [float(item[0]), float(item[1])]
        for item in (
            validation.get("largest_uncovered_intervals")
            or validation.get("uncovered_intervals")
            or []
        )
        if len(item) >= 2 and float(item[1]) > float(item[0])
    ]
    threshold = float(max_uncovered_gap_seconds)
    blocking_gaps = [
        gap
        for gap in raw_gaps
        if float(gap[1]) - float(gap[0]) > threshold + 0.0005
    ]
    problems = set(validation.get("problems") or [])
    low_coverage = "low_speech_coverage" in problems
    if low_coverage:
        candidate_source = raw_gaps
        selection_reason = "low_speech_coverage"
    else:
        candidate_source = blocking_gaps
        selection_reason = "uncovered_speech_gap"
    coalesced = coalesce_coverage_recovery_gaps(
        candidate_source,
        join_seconds=join_seconds,
        max_window_seconds=max_window_seconds,
    )
    ordered = sorted(
        coalesced,
        key=lambda item: float(item[1]) - float(item[0]),
        reverse=True,
    )
    attempt_limit = max(0, int(max_attempts))
    selected = ordered[:attempt_limit]
    return selected, {
        "version": COVERAGE_RECOVERY_SELECTION_VERSION,
        "reason": selection_reason,
        "raw_gap_count": len(raw_gaps),
        "raw_blocking_gap_count": len(blocking_gaps),
        "blocking_gaps": [
            [round(start, 3), round(end, 3)]
            for start, end in blocking_gaps
        ],
        "coalesced_candidate_count": len(coalesced),
        "selected_candidate_count": len(selected),
        "max_attempts": attempt_limit,
    }


def validation_has_blocking_gap_within(
    validation: Mapping[str, Any],
    target_gap: Sequence[float],
    *,
    max_uncovered_gap_seconds: float,
) -> bool:
    target_start, target_end = (
        float(target_gap[0]),
        float(target_gap[1]),
    )
    threshold = float(max_uncovered_gap_seconds)
    return any(
        float(gap[1]) - float(gap[0]) > threshold + 0.0005
        and float(gap[1]) > target_start
        and float(gap[0]) < target_end
        for gap in (
            validation.get("largest_uncovered_intervals")
            or validation.get("uncovered_intervals")
            or []
        )
        if len(gap) >= 2
    )


def coverage_validation_improved(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> bool:
    """Report measurable coverage progress without trusting text alone."""
    if bool(after.get("accept")) and not bool(before.get("accept")):
        return True
    before_problems = set(before.get("problems") or [])
    after_problems = set(after.get("problems") or [])
    if after_problems < before_problems:
        return True
    for key in (
        "uncovered_speech_seconds",
        "uncovered_speech_ratio",
        "max_uncovered_gap_seconds",
    ):
        try:
            before_value = float(before.get(key))
            after_value = float(after.get(key))
        except (TypeError, ValueError):
            continue
        if after_value < before_value - 0.0005:
            return True
    return False


def supplement_timing_evidence_segments(
    segments: Sequence[Mapping[str, Any]],
    supplemental_segments: Sequence[Mapping[str, Any]],
    speech_map: Mapping[str, Any],
    *,
    target_uncovered_intervals: Sequence[Sequence[float]],
    max_existing_coverage_ratio: float = 0.95,
    min_signal_overlap_ratio: float = 0.80,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fill native-ASR blind spots with validated prompted timestamps.

    Supplemental timestamps are admitted only where native timing is almost
    absent and an independent speech map confirms signal.  They remain marked
    so downstream review can distinguish them from native word anchors.
    """
    combined = [dict(item) for item in segments]
    native_intervals = [
        (float(item["start"]), float(item["end"]))
        for item in combined
    ]
    possible_audio = [
        (float(start), float(end))
        for start, end in (
            speech_map.get("possible_audio_intervals") or []
        )
    ]
    target_intervals = [
        (float(start), float(end))
        for start, end in (target_uncovered_intervals or [])
        if float(end) > float(start)
    ]
    admitted = []
    rejected = []
    for index, item in enumerate(supplemental_segments or []):
        normalized = _normalize_segment(item, 0.0)
        if normalized is None:
            rejected.append({
                "segment_index": index,
                "reason": "invalid_segment",
            })
            continue
        start = float(normalized["start"])
        end = float(normalized["end"])
        duration = max(0.001, end - start)
        native_ratio = min(
            1.0,
            interval_overlap_seconds(
                start,
                end,
                native_intervals,
            )
            / duration,
        )
        signal_ratio = min(
            1.0,
            interval_overlap_seconds(
                start,
                end,
                possible_audio,
            )
            / duration,
        )
        target_overlap = interval_overlap_seconds(
            start,
            end,
            target_intervals,
        )
        if target_overlap <= 0.01:
            rejected.append({
                "segment_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "reason": "no_blocking_gap_overlap",
            })
            continue
        if native_ratio > float(max_existing_coverage_ratio):
            rejected.append({
                "segment_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "reason": "native_timing_already_present",
                "native_coverage_ratio": round(native_ratio, 4),
            })
            continue
        if signal_ratio < float(min_signal_overlap_ratio):
            rejected.append({
                "segment_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "reason": "insufficient_independent_audio_signal",
                "signal_overlap_ratio": round(signal_ratio, 4),
            })
            continue
        normalized["_supplemental_timing_evidence"] = True
        normalized["_timing_source"] = (
            "validated_canonical_prompted_segment"
        )
        normalized["_target_gap_overlap_seconds"] = round(
            target_overlap,
            3,
        )
        admitted.append(normalized)
        combined.append(normalized)
    combined.sort(
        key=lambda item: (
            float(item["start"]),
            float(item["end"]),
        )
    )
    return combined, {
        "schema": "subgen_supplemental_timing_evidence_v1",
        "applied": bool(admitted),
        "candidate_count": len(supplemental_segments or []),
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
        "target_uncovered_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in target_intervals
        ],
        "max_existing_coverage_ratio": float(
            max_existing_coverage_ratio
        ),
        "min_signal_overlap_ratio": float(min_signal_overlap_ratio),
        "admitted": [
            {
                "start": round(float(item["start"]), 3),
                "end": round(float(item["end"]), 3),
                "text_sha256": hashlib.sha256(
                    str(item.get("text") or "").encode("utf-8")
                ).hexdigest(),
            }
            for item in admitted
        ],
        "rejected": rejected[:100],
        "rejected_truncated": max(0, len(rejected) - 100),
    }


def filter_timing_evidence_segments(
    segments: Sequence[Mapping[str, Any]],
    speech_map: Mapping[str, Any],
    *,
    min_possible_audio_overlap_seconds: float = 0.01,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Remove timing-only observations wholly outside independent audio signal.

    This is intentionally limited to advisory timing evidence.  It must never
    be used to delete canonical transcript text.
    """
    possible_audio = [
        (float(start), float(end))
        for start, end in speech_map.get("possible_audio_intervals") or []
    ]
    if not possible_audio:
        return [dict(item) for item in segments], {
            "applied": False,
            "reason": "independent_possible_audio_unavailable",
            "kept_count": len(segments),
            "discarded_count": 0,
            "discarded": [],
        }

    kept = []
    discarded = []
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        overlap = interval_overlap_seconds(start, end, possible_audio)
        if overlap > float(min_possible_audio_overlap_seconds):
            kept.append(dict(segment))
            continue
        discarded.append({
            "segment_index": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": str(segment.get("text") or "")[:160],
            "possible_audio_overlap_seconds": round(overlap, 4),
            "reason": "timing_observation_outside_independent_audio_signal",
        })
    return kept, {
        "applied": True,
        "reason": "timing_evidence_only",
        "kept_count": len(kept),
        "discarded_count": len(discarded),
        "discarded": discarded[:100],
        "discarded_truncated": max(0, len(discarded) - 100),
    }


def validate_timed_segments(
    segments: Sequence[Mapping[str, Any]],
    speech_map: Mapping[str, Any],
    *,
    max_uncovered_gap_seconds: float = 1.5,
    max_uncovered_ratio: float = 0.03,
    coverage_padding_seconds: float = 0.35,
    max_confirmed_silence_span_seconds: float = 1.0,
    reject_confirmed_silence_spans: bool = True,
) -> Dict[str, Any]:
    speech = [
        (float(start), float(end))
        for start, end in speech_map.get("speech_intervals") or []
    ]
    possible_audio = [
        (float(start), float(end))
        for start, end in speech_map.get("possible_audio_intervals") or []
    ]
    confirmed_silence = [
        (float(start), float(end))
        for start, end in speech_map.get("confirmed_silence_intervals") or []
    ]
    covered = []
    hallucination_candidates = []
    silence_spans = []
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        covered.append(
            (
                max(0.0, start - float(coverage_padding_seconds)),
                end + float(coverage_padding_seconds),
            )
        )
        if interval_overlap_seconds(start, end, possible_audio) <= 0.01:
            hallucination_candidates.append({
                "segment_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": str(segment.get("text") or "")[:160],
            })
        for silence_start, silence_end in confirmed_silence:
            overlap = max(0.0, min(end, silence_end) - max(start, silence_start))
            if overlap >= float(max_confirmed_silence_span_seconds):
                silence_spans.append({
                    "segment_index": index,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "silence_start": round(silence_start, 3),
                    "silence_end": round(silence_end, 3),
                    "silence_overlap_seconds": round(overlap, 3),
                })
                break
    missing = subtract_intervals(speech, covered)
    speech_seconds = sum(end - start for start, end in speech)
    missing_seconds = sum(end - start for start, end in missing)
    missing_ratio = missing_seconds / speech_seconds if speech_seconds else 0.0
    max_gap = max((end - start for start, end in missing), default=0.0)
    largest_missing = sorted(
        missing,
        key=lambda item: (
            -(float(item[1]) - float(item[0])),
            float(item[0]),
        ),
    )
    problems = []
    if hallucination_candidates:
        problems.append("subtitle_text_in_confirmed_signal_silence")
    if silence_spans and bool(reject_confirmed_silence_spans):
        problems.append("subtitle_segment_spans_confirmed_long_silence")
    if max_gap > float(max_uncovered_gap_seconds):
        problems.append("uncovered_speech_gap")
    if missing_ratio > float(max_uncovered_ratio):
        problems.append("low_speech_coverage")
    return {
        "schema": "subgen_longform_validation_v1",
        "accept": not problems,
        "problems": problems,
        "speech_seconds": round(speech_seconds, 3),
        "uncovered_speech_seconds": round(missing_seconds, 3),
        "uncovered_speech_ratio": round(missing_ratio, 5),
        "max_uncovered_gap_seconds": round(max_gap, 3),
        "max_allowed_uncovered_gap_seconds": float(max_uncovered_gap_seconds),
        "max_allowed_uncovered_ratio": float(max_uncovered_ratio),
        "uncovered_interval_count": len(missing),
        "uncovered_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in missing[:20]
        ],
        "uncovered_intervals_truncated": max(0, len(missing) - 20),
        "largest_uncovered_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in largest_missing[:256]
        ],
        "largest_uncovered_intervals_truncated": max(
            0,
            len(largest_missing) - 256,
        ),
        "hallucination_candidates": hallucination_candidates[:20],
        "confirmed_silence_spans": silence_spans[:20],
        "confirmed_silence_spans_blocking": bool(
            reject_confirmed_silence_spans
        ),
    }


def _chunk_audio_seconds(
    chunk: Mapping[str, Any],
    speech_map: Mapping[str, Any],
    key: str,
) -> float:
    return interval_overlap_seconds(
        float(chunk["ownership_start"]),
        float(chunk["ownership_end"]),
        speech_map.get(key) or [],
    )


def run_longform_transcription(
    media_path: Path | str,
    output_dir: Path | str,
    *,
    provider: str,
    model: str,
    language: Optional[str],
    prompt: str,
    transcribe_chunk: Callable[[str, Mapping[str, Any]], Any],
    expected_timing: bool,
    timing_evidence_only: bool = False,
    transcription_options: Optional[Mapping[str, Any]] = None,
    supplemental_timing_segments: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
    source_sha256: Optional[str] = None,
    target_seconds: float = 300.0,
    min_seconds: float = 180.0,
    max_seconds: float = 480.0,
    overlap_seconds: float = 12.0,
    boundary_search_seconds: float = 60.0,
    min_boundary_silence_seconds: float = 1.0,
    sample_rate: int = 16000,
    bitrate: str = "64k",
    speech_map_options: Optional[Mapping[str, Any]] = None,
    precomputed_speech_map: Optional[Mapping[str, Any]] = None,
    retry_context_seconds: float = 15.0,
    invalid_chunk_split_retry_enabled: bool = True,
    invalid_chunk_split_retry_min_seconds: float = 60.0,
    invalid_chunk_split_retry_overlap_seconds: float = 6.0,
    coverage_recovery_enabled: bool = True,
    coverage_recovery_context_seconds: float = 6.0,
    coverage_recovery_max_attempts_per_chunk: int = 6,
    coverage_recovery_max_total_attempts: int = 12,
    coverage_recovery_min_novel_probability: float = 0.60,
    coverage_recovery_existing_match_tolerance_seconds: float = 0.75,
    coverage_recovery_allow_unscored_timed_segments: bool = False,
    coverage_recovery_min_unscored_speech_overlap_ratio: float = 0.80,
    coverage_recovery_max_window_seconds: float = 180.0,
    max_uncovered_gap_seconds: float = 1.5,
    max_uncovered_ratio: float = 0.03,
) -> Dict[str, Any]:
    """Transcribe an arbitrary-duration media file with bounded resources."""
    media_path = Path(media_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = probe_media(media_path)
    output_free_bytes = shutil.disk_usage(output_dir).free
    estimated_chunk_bytes = int(
        float(max_seconds)
        * parse_bitrate_bits_per_second(bitrate)
        / 8
    )
    minimum_free_bytes = max(256 * 1024**2, estimated_chunk_bytes * 4)
    preflight.update({
        "output_available_disk_bytes": output_free_bytes,
        "estimated_max_chunk_audio_bytes": estimated_chunk_bytes,
        "minimum_required_free_bytes": minimum_free_bytes,
        "bounded_memory_pcm_window_bytes": int(
            float((speech_map_options or {}).get(
                "vad_analysis_window_seconds",
                600,
            ))
            * 16000
            * 2
        ),
    })
    if output_free_bytes < minimum_free_bytes:
        raise RuntimeError(
            "Insufficient free disk space for bounded long-form processing: "
            f"available={output_free_bytes} bytes, "
            f"required={minimum_free_bytes} bytes."
        )
    source_sha256 = source_sha256 or sha256_file(media_path)
    manifest_path = output_dir / "longform-manifest.json"
    try:
        existing_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
    except (OSError, json.JSONDecodeError):
        existing_manifest = {}
    analysis_identity = canonical_sha256({
        "pipeline_version": LONGFORM_PIPELINE_VERSION,
        "source_sha256": source_sha256,
        "source_bytes": preflight["bytes"],
        "duration_seconds": preflight["duration_seconds"],
        "speech_map_options": dict(speech_map_options or {}),
    })
    if precomputed_speech_map is not None:
        speech_map = json.loads(json.dumps(dict(precomputed_speech_map)))
    elif (
        existing_manifest.get("analysis_identity") == analysis_identity
        and isinstance(existing_manifest.get("speech_map"), Mapping)
    ):
        speech_map = json.loads(json.dumps(existing_manifest["speech_map"]))
    else:
        speech_map = build_speech_map(
            media_path,
            preflight["duration_seconds"],
            **dict(speech_map_options or {}),
        )
    map_duration = float(speech_map.get("duration_seconds") or 0.0)
    if abs(map_duration - float(preflight["duration_seconds"])) > 0.05:
        raise RuntimeError(
            "Speech map does not match the input media duration "
            f"({map_duration}s vs {preflight['duration_seconds']}s)."
        )
    plan = make_adaptive_chunk_plan(
        preflight["duration_seconds"],
        speech_map,
        target_seconds=target_seconds,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        overlap_seconds=overlap_seconds,
        boundary_search_seconds=boundary_search_seconds,
        min_boundary_silence_seconds=min_boundary_silence_seconds,
        include_overlap_context=bool(expected_timing),
    )
    transcription_identity_payload = {
        "pipeline_version": LONGFORM_PIPELINE_VERSION,
        "source_sha256": source_sha256,
        "source_bytes": preflight["bytes"],
        "duration_seconds": preflight["duration_seconds"],
        "provider": provider,
        "model": model,
        "language": language,
        "prompt_sha256": hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest(),
        "expected_timing": bool(expected_timing),
        "timing_evidence_only": bool(timing_evidence_only),
        "transcription_options": dict(transcription_options or {}),
        "invalid_chunk_split_retry": {
            "enabled": bool(invalid_chunk_split_retry_enabled),
            "minimum_part_seconds": float(
                invalid_chunk_split_retry_min_seconds
            ),
            "overlap_seconds": float(
                invalid_chunk_split_retry_overlap_seconds
            ),
        },
        "plan": plan,
        "speech_map_parameters": speech_map.get("parameters"),
        "speech_map_sha256": canonical_sha256(speech_map),
    }
    transcription_identity = canonical_sha256(
        transcription_identity_payload
    )
    identity_payload = {
        **transcription_identity_payload,
        "supplemental_timing_evidence": {
            "enabled": bool(supplemental_timing_segments),
            "segments_sha256": (
                canonical_sha256({
                    "segments": [
                        dict(item)
                        for item in (
                            supplemental_timing_segments or []
                        )
                    ]
                })
                if supplemental_timing_segments
                else None
            ),
            "max_existing_coverage_ratio": 0.95,
            "min_signal_overlap_ratio": 0.80,
        },
        "coverage_recovery": {
            "enabled": bool(coverage_recovery_enabled),
            "context_seconds": float(coverage_recovery_context_seconds),
            "max_attempts_per_chunk": int(
                coverage_recovery_max_attempts_per_chunk
            ),
            "effective_max_attempts": (
                min(
                    max(1, int(coverage_recovery_max_total_attempts)),
                    max(
                        1,
                        int(
                            coverage_recovery_max_attempts_per_chunk
                        ),
                    )
                    * max(1, len(plan)),
                )
            ),
            "max_total_attempts": int(
                coverage_recovery_max_total_attempts
            ),
            "min_novel_probability": float(
                coverage_recovery_min_novel_probability
            ),
            "existing_match_tolerance_seconds": float(
                coverage_recovery_existing_match_tolerance_seconds
            ),
            "allow_unscored_timed_segments": bool(
                coverage_recovery_allow_unscored_timed_segments
            ),
            "min_unscored_speech_overlap_ratio": float(
                coverage_recovery_min_unscored_speech_overlap_ratio
            ),
            "max_window_seconds": float(
                coverage_recovery_max_window_seconds
            ),
            "candidate_selection": (
                COVERAGE_RECOVERY_SELECTION_VERSION
            ),
        },
        "finalization": {
            "max_uncovered_gap_seconds": float(
                max_uncovered_gap_seconds
            ),
            "max_uncovered_ratio": float(max_uncovered_ratio),
            "validation_schema": "subgen_longform_validation_v1",
        },
    }
    run_identity = canonical_sha256(identity_payload)
    existing_identity_payload = dict(
        existing_manifest.get("identity") or {}
    )
    existing_identity_payload.pop("coverage_recovery", None)
    existing_identity_payload.pop("finalization", None)
    existing_transcription_identity = (
        existing_manifest.get("transcription_identity")
        or (
            canonical_sha256(existing_identity_payload)
            if existing_identity_payload
            else None
        )
    )
    previous_run_identity = existing_manifest.get("run_identity")
    manifest = {
        "schema": "subgen_longform_manifest_v1",
        "pipeline_version": LONGFORM_PIPELINE_VERSION,
        "analysis_identity": analysis_identity,
        "transcription_identity": transcription_identity,
        "run_identity": run_identity,
        "identity": identity_payload,
        "preflight": preflight,
        "speech_map": speech_map,
        "plan": plan,
        "status": "processing",
        "chunks": [],
    }
    if existing_transcription_identity == transcription_identity:
        manifest = existing_manifest
        manifest.update({
            "pipeline_version": LONGFORM_PIPELINE_VERSION,
            "analysis_identity": analysis_identity,
            "transcription_identity": transcription_identity,
            "run_identity": run_identity,
            "identity": identity_payload,
            "preflight": preflight,
            "speech_map": speech_map,
            "plan": plan,
            "status": "processing",
        })
        for stale_key in (
            "coverage_recovery",
            "coverage_recovery_selection",
            "supplemental_timing_evidence",
            "merge_report",
            "validation",
        ):
            manifest.pop(stale_key, None)
    atomic_write_json(manifest_path, manifest)

    results = []
    completed_by_index = {
        int(item["index"]): item
        for item in manifest.get("chunks") or []
        if item.get("status") in {"complete", "accepted_no_speech"}
    }
    for position, chunk in enumerate(plan):
        index = int(chunk["index"])
        result_path = output_dir / f"chunk-{index:04d}.result.json"
        chunk_identity = canonical_sha256({
            "transcription_identity": transcription_identity,
            "chunk": chunk,
        })
        legacy_chunk_identity = (
            canonical_sha256({
                "run_identity": previous_run_identity,
                "chunk": chunk,
            })
            if previous_run_identity
            else None
        )
        accepted_chunk_identities = {
            value
            for value in (
                chunk_identity,
                legacy_chunk_identity,
            )
            if value
        }
        cached = completed_by_index.get(index)
        if (
            cached
            and cached.get("chunk_identity") in accepted_chunk_identities
            and result_path.exists()
        ):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = None
            if (
                result
                and result.get("chunk_identity")
                in accepted_chunk_identities
            ):
                if result.get("chunk_identity") != chunk_identity:
                    result["chunk_identity"] = chunk_identity
                    atomic_write_json(result_path, result)
                cached["chunk_identity"] = chunk_identity
                result["is_last"] = position == len(plan) - 1
                results.append(result)
                continue

        audio_path = output_dir / f"chunk-{index:04d}.mp3"
        retry_path = output_dir / f"chunk-{index:04d}.retry.mp3"
        speech_seconds = _chunk_audio_seconds(chunk, speech_map, "speech_intervals")
        uncertain_seconds = _chunk_audio_seconds(
            chunk,
            speech_map,
            "uncertain_audio_intervals",
        )
        chunk_record = {
            "index": index,
            "chunk_identity": chunk_identity,
            "status": "extracting",
            "speech_seconds": round(speech_seconds, 3),
            "uncertain_audio_seconds": round(uncertain_seconds, 3),
        }
        manifest["chunks"] = [
            item
            for item in manifest.get("chunks") or []
            if int(item.get("index", -1)) != index
        ] + [chunk_record]
        manifest["chunks"].sort(key=lambda item: int(item["index"]))
        atomic_write_json(manifest_path, manifest)

        keep_failed_audio = False
        try:
            initial_rejection_path = output_dir / (
                f"chunk-{index:04d}.initial-rejected.json"
            )
            cached_initial_rejection = None
            if (
                bool(invalid_chunk_split_retry_enabled)
                and initial_rejection_path.exists()
            ):
                try:
                    candidate_rejection = json.loads(
                        initial_rejection_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    candidate_rejection = None
                if (
                    isinstance(candidate_rejection, Mapping)
                    and candidate_rejection.get("schema")
                    == "subgen_longform_initial_rejection_v1"
                    and candidate_rejection.get("chunk_identity")
                    == chunk_identity
                    and isinstance(
                        (
                            candidate_rejection.get("diagnostics")
                            or {}
                        ).get("rejected_artifact"),
                        Mapping,
                    )
                ):
                    cached_initial_rejection = candidate_rejection

            artifact_is_absolute = False
            if cached_initial_rejection is not None:
                chunk_record.update({
                    "status": (
                        "resuming_split_retry_after_validation_rejection"
                    ),
                    "initial_rejection_path": str(
                        initial_rejection_path
                    ),
                    "initial_rejection_reused": True,
                })
                atomic_write_json(manifest_path, manifest)
                artifact = retry_invalid_chunk_as_splits(
                    media_path,
                    output_dir,
                    chunk,
                    parent_chunk_identity=chunk_identity,
                    transcribe_chunk=transcribe_chunk,
                    speech_map=speech_map,
                    expected_timing=expected_timing,
                    minimum_part_seconds=(
                        invalid_chunk_split_retry_min_seconds
                    ),
                    overlap_seconds=(
                        invalid_chunk_split_retry_overlap_seconds
                    ),
                    sample_rate=sample_rate,
                    bitrate=bitrate,
                )
                if artifact is None:
                    raise RuntimeError(
                        "Preserved rejected chunk cannot be split using "
                        "the current bounded retry configuration."
                    )
                artifact_is_absolute = True
            else:
                extract_audio_chunk(
                    media_path,
                    audio_path,
                    chunk["start"],
                    chunk["duration"],
                    sample_rate=sample_rate,
                    bitrate=bitrate,
                )
                chunk_record["status"] = "transcribing"
                atomic_write_json(manifest_path, manifest)
                try:
                    artifact = _artifact_dict(
                        transcribe_chunk(str(audio_path), chunk)
                    )
                except Exception as initial_error:
                    diagnostics = getattr(
                        initial_error,
                        "diagnostics",
                        None,
                    )
                    rejected_artifact = (
                        diagnostics.get("rejected_artifact")
                        if isinstance(diagnostics, Mapping)
                        else None
                    )
                    split_artifact = None
                    if (
                        bool(invalid_chunk_split_retry_enabled)
                        and rejected_artifact is not None
                    ):
                        atomic_write_json(
                            initial_rejection_path,
                            {
                                "schema": (
                                    "subgen_longform_initial_rejection_v1"
                                ),
                                "chunk_identity": chunk_identity,
                                "chunk": chunk,
                                "error": {
                                    "type": type(initial_error).__name__,
                                    "message": str(initial_error),
                                },
                                "diagnostics": diagnostics,
                            },
                        )
                        chunk_record.update({
                            "status": (
                                "split_retry_after_validation_rejection"
                            ),
                            "initial_rejection_path": str(
                                initial_rejection_path
                            ),
                            "initial_rejection_reused": False,
                        })
                        atomic_write_json(manifest_path, manifest)
                        split_artifact = retry_invalid_chunk_as_splits(
                            media_path,
                            output_dir,
                            chunk,
                            parent_chunk_identity=chunk_identity,
                            transcribe_chunk=transcribe_chunk,
                            speech_map=speech_map,
                            expected_timing=expected_timing,
                            minimum_part_seconds=(
                                invalid_chunk_split_retry_min_seconds
                            ),
                            overlap_seconds=(
                                invalid_chunk_split_retry_overlap_seconds
                            ),
                            sample_rate=sample_rate,
                            bitrate=bitrate,
                        )
                    if split_artifact is None:
                        raise
                    artifact = split_artifact
                    artifact_is_absolute = True
            if not artifact_is_absolute:
                artifact = offset_artifact(
                    artifact,
                    float(chunk["start"]),
                )
            text = re.sub(r"\s+", " ", str(artifact.get("text") or "")).strip()
            artifact["text"] = text
            retry = None
            if not text and speech_seconds >= 0.15:
                retry_start = max(
                    0.0,
                    float(chunk["start"]) - float(retry_context_seconds),
                )
                retry_end = min(
                    float(preflight["duration_seconds"]),
                    float(chunk["end"]) + float(retry_context_seconds),
                )
                retry_chunk = dict(chunk)
                retry_chunk.update({
                    "start": retry_start,
                    "end": retry_end,
                    "duration": retry_end - retry_start,
                    "retry_of": index,
                })
                extract_audio_chunk(
                    media_path,
                    retry_path,
                    retry_start,
                    retry_end - retry_start,
                    sample_rate=sample_rate,
                    bitrate=bitrate,
                )
                retry_artifact = _artifact_dict(
                    transcribe_chunk(str(retry_path), retry_chunk)
                )
                retry_artifact = offset_artifact(retry_artifact, retry_start)
                retry_text = re.sub(
                    r"\s+",
                    " ",
                    str(retry_artifact.get("text") or ""),
                ).strip()
                retry_artifact["text"] = retry_text
                retry = {
                    "start": round(retry_start, 3),
                    "end": round(retry_end, 3),
                    "text_chars": len(retry_text),
                }
                if retry_text:
                    artifact = retry_artifact
                    text = retry_text
            if not text and speech_seconds >= 0.15:
                keep_failed_audio = True
                chunk_record.update({
                    "status": "failed_unresolved_speech",
                    "retry": retry,
                })
                manifest["status"] = "failed"
                atomic_write_json(manifest_path, manifest)
                raise RuntimeError(
                    "Transcription returned no text for an independently detected "
                    f"speech region after bounded retry (chunk {index}, "
                    f"{chunk['ownership_start']:.1f}s-"
                    f"{chunk['ownership_end']:.1f}s)."
                )
            status = "complete" if text else "accepted_no_speech"
            result = {
                "schema": "subgen_longform_chunk_result_v1",
                "chunk_identity": chunk_identity,
                "chunk": chunk,
                "is_last": position == len(plan) - 1,
                "status": status,
                "speech_seconds": round(speech_seconds, 3),
                "uncertain_audio_seconds": round(uncertain_seconds, 3),
                "text": text,
                "segments": artifact.get("segments") or [],
                "words": artifact.get("words") or [],
                "language": artifact.get("language"),
                "timing_kind": artifact.get("timing_kind") or "none",
                "usage": artifact.get("usage") or {},
                "metadata": artifact.get("metadata") or {},
                "retry": retry,
            }
            if expected_timing and text and not result["segments"]:
                keep_failed_audio = True
                chunk_record["status"] = "failed_missing_timestamps"
                manifest["status"] = "failed"
                atomic_write_json(manifest_path, manifest)
                raise RuntimeError(
                    f"{provider}/{model} returned text without required timestamps "
                    f"for chunk {index}."
                )
            atomic_write_json(result_path, result)
            chunk_record.update({
                "status": status,
                "result_path": str(result_path),
                "text_chars": len(text),
                "segment_count": len(result["segments"]),
                "retry": retry,
            })
            atomic_write_json(manifest_path, manifest)
            results.append(result)
        except Exception as exc:
            failure_path = output_dir / f"chunk-{index:04d}.failure.json"
            failure_record = {
                "schema": "subgen_longform_chunk_failure_v1",
                "chunk_identity": chunk_identity,
                "chunk": chunk,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            diagnostics = getattr(exc, "diagnostics", None)
            if diagnostics is not None:
                failure_record["diagnostics"] = diagnostics
            try:
                atomic_write_json(failure_path, failure_record)
                chunk_record["failure_path"] = str(failure_path)
            except Exception as failure_exc:
                chunk_record["failure_record_error"] = (
                    f"{type(failure_exc).__name__}: {failure_exc}"
                )
            chunk_record["error"] = failure_record["error"]
            if chunk_record.get("status") not in {
                "failed_unresolved_speech",
                "failed_missing_timestamps",
            }:
                chunk_record["status"] = "failed"
                manifest["status"] = "failed"
            atomic_write_json(manifest_path, manifest)
            keep_failed_audio = True
            raise
        finally:
            if not keep_failed_audio:
                audio_path.unlink(missing_ok=True)
                retry_path.unlink(missing_ok=True)

    if expected_timing:
        transcript_text, segments, merge_report = merge_owned_timed_segments(results)
        timing_evidence_filter_reports = []
        if timing_evidence_only:
            segments, filter_report = filter_timing_evidence_segments(
                segments,
                speech_map,
            )
            filter_report["stage"] = "initial_merge"
            timing_evidence_filter_reports.append(filter_report)
            transcript_text = re.sub(
                r"\s+",
                " ",
                " ".join(
                    str(segment.get("text") or "")
                    for segment in segments
                ),
            ).strip()
            if supplemental_timing_segments:
                native_validation = validate_timed_segments(
                    segments,
                    speech_map,
                    max_uncovered_gap_seconds=(
                        max_uncovered_gap_seconds
                    ),
                    max_uncovered_ratio=max_uncovered_ratio,
                    reject_confirmed_silence_spans=False,
                )
                blocking_native_gaps = [
                    gap
                    for gap in (
                        native_validation.get(
                            "largest_uncovered_intervals"
                        )
                        or native_validation.get(
                            "uncovered_intervals"
                        )
                        or []
                    )
                    if (
                        float(gap[1]) - float(gap[0])
                        > float(max_uncovered_gap_seconds) + 0.0005
                    )
                ]
                segments, supplemental_report = (
                    supplement_timing_evidence_segments(
                        segments,
                        supplemental_timing_segments,
                        speech_map,
                        target_uncovered_intervals=(
                            blocking_native_gaps
                        ),
                    )
                )
                supplemental_report["validation_before"] = (
                    native_validation
                )
                manifest["supplemental_timing_evidence"] = (
                    supplemental_report
                )
                transcript_text = re.sub(
                    r"\s+",
                    " ",
                    " ".join(
                        str(segment.get("text") or "")
                        for segment in segments
                    ),
                ).strip()
        validation = validate_timed_segments(
            segments,
            speech_map,
            max_uncovered_gap_seconds=max_uncovered_gap_seconds,
            max_uncovered_ratio=max_uncovered_ratio,
            reject_confirmed_silence_spans=not timing_evidence_only,
        )
        recovery_reports = []
        recovery_audio_paths = []
        if (
            not validation["accept"]
            and bool(coverage_recovery_enabled)
            and any(
                problem in {"uncovered_speech_gap", "low_speech_coverage"}
                for problem in validation.get("problems") or []
            )
        ):
            effective_recovery_attempts = (
                min(
                    max(1, int(coverage_recovery_max_total_attempts)),
                    max(
                        1,
                        int(
                            coverage_recovery_max_attempts_per_chunk
                        ),
                    )
                    * max(1, len(plan)),
                )
            )
            gaps, recovery_selection = select_coverage_recovery_gaps(
                validation,
                max_uncovered_gap_seconds=max_uncovered_gap_seconds,
                join_seconds=max(
                    float(coverage_recovery_context_seconds) * 2.0,
                    float(max_uncovered_gap_seconds),
                ),
                max_window_seconds=float(
                    coverage_recovery_max_window_seconds
                ),
                max_attempts=effective_recovery_attempts,
            )
            manifest["coverage_recovery_selection"] = (
                recovery_selection
            )
            for recovery_index, gap in enumerate(gaps, 1):
                if validation["accept"]:
                    break
                if not any(
                    problem in {
                        "uncovered_speech_gap",
                        "low_speech_coverage",
                    }
                    for problem in validation.get("problems") or []
                ):
                    break
                targets_blocking_gap = any(
                    float(blocking[1]) > float(gap[0])
                    and float(blocking[0]) < float(gap[1])
                    for blocking in recovery_selection.get(
                        "blocking_gaps",
                        [],
                    )
                )
                recovery_start = max(
                    0.0,
                    float(gap[0]) - float(coverage_recovery_context_seconds),
                )
                recovery_end = min(
                    float(preflight["duration_seconds"]),
                    float(gap[1]) + float(coverage_recovery_context_seconds),
                )
                recovery_chunk = {
                    "index": recovery_index,
                    "start": recovery_start,
                    "end": recovery_end,
                    "duration": recovery_end - recovery_start,
                    "ownership_start": float(gap[0]),
                    "ownership_end": float(gap[1]),
                    "left_boundary_kind": "coverage_recovery",
                    "right_boundary_kind": "coverage_recovery",
                    "coverage_recovery_gap": list(gap),
                }
                recovery_audio = (
                    output_dir
                    / f"coverage-recovery-{recovery_index:04d}.mp3"
                )
                recovery_result_path = (
                    output_dir
                    / (
                        f"coverage-recovery-"
                        f"{recovery_index:04d}.result.json"
                    )
                )
                recovery_failure_path = (
                    output_dir
                    / (
                        f"coverage-recovery-"
                        f"{recovery_index:04d}.failure.json"
                    )
                )
                recovery_chunk_identity = canonical_sha256({
                    "run_identity": run_identity,
                    "chunk": recovery_chunk,
                })
                validation_before_recovery = validation
                try:
                    cached_recovery = None
                    if recovery_result_path.exists():
                        try:
                            cached_recovery = json.loads(
                                recovery_result_path.read_text(
                                    encoding="utf-8"
                                )
                            )
                        except (OSError, json.JSONDecodeError):
                            cached_recovery = None
                    reused_checkpoint = bool(
                        cached_recovery
                        and cached_recovery.get("run_identity")
                        == run_identity
                        and (
                            cached_recovery.get("chunk_identity")
                            == recovery_chunk_identity
                            or cached_recovery.get("chunk")
                            == recovery_chunk
                        )
                        and isinstance(
                            cached_recovery.get("artifact"),
                            Mapping,
                        )
                    )
                    if reused_checkpoint:
                        recovery_artifact = dict(
                            cached_recovery["artifact"]
                        )
                        recovery_failure_path.unlink(missing_ok=True)
                    else:
                        recovery_result_path.unlink(missing_ok=True)
                        recovery_failure_path.unlink(missing_ok=True)
                        recovery_audio_paths.append(recovery_audio)
                        extract_audio_chunk(
                            media_path,
                            recovery_audio,
                            recovery_start,
                            recovery_end - recovery_start,
                            sample_rate=sample_rate,
                            bitrate=bitrate,
                        )
                        recovery_artifact = offset_artifact(
                            _artifact_dict(
                                transcribe_chunk(
                                    str(recovery_audio),
                                    recovery_chunk,
                                )
                            ),
                            recovery_start,
                        )
                        atomic_write_json(
                            recovery_result_path,
                            {
                                "schema": (
                                    "subgen_longform_coverage_recovery_v1"
                                ),
                                "run_identity": run_identity,
                                "chunk_identity": (
                                    recovery_chunk_identity
                                ),
                                "chunk": recovery_chunk,
                                "artifact": recovery_artifact,
                            },
                        )
                    segments, recovery_report = apply_coverage_recovery(
                        segments,
                        recovery_artifact,
                        gap,
                        context_seconds=coverage_recovery_context_seconds,
                        min_novel_word_probability=(
                            coverage_recovery_min_novel_probability
                        ),
                        existing_match_tolerance_seconds=(
                            coverage_recovery_existing_match_tolerance_seconds
                        ),
                        independent_speech_intervals=(
                            speech_map.get("speech_intervals") or []
                        ),
                        allow_unscored_timed_segments=(
                            coverage_recovery_allow_unscored_timed_segments
                        ),
                        min_unscored_speech_overlap_ratio=(
                            coverage_recovery_min_unscored_speech_overlap_ratio
                        ),
                    )
                    if timing_evidence_only:
                        segments, filter_report = (
                            filter_timing_evidence_segments(
                                segments,
                                speech_map,
                            )
                        )
                        filter_report.update({
                            "stage": "coverage_recovery",
                            "recovery_index": recovery_index,
                        })
                        timing_evidence_filter_reports.append(
                            filter_report
                        )
                    validation = validate_timed_segments(
                        segments,
                        speech_map,
                        max_uncovered_gap_seconds=max_uncovered_gap_seconds,
                        max_uncovered_ratio=max_uncovered_ratio,
                        reject_confirmed_silence_spans=(
                            not timing_evidence_only
                        ),
                    )
                    recovery_report["validation_after"] = validation
                    recovery_report["validation_progress"] = (
                        coverage_validation_improved(
                            validation_before_recovery,
                            validation,
                        )
                    )
                    recovery_report["reused_checkpoint"] = (
                        reused_checkpoint
                    )
                    recovery_report["targeted_blocking_gap"] = bool(
                        targets_blocking_gap
                    )
                    recovery_report["blocking_gap_resolved"] = bool(
                        targets_blocking_gap
                        and not validation_has_blocking_gap_within(
                            validation,
                            gap,
                            max_uncovered_gap_seconds=(
                                max_uncovered_gap_seconds
                            ),
                        )
                    )
                    recovery_reports.append(recovery_report)
                except Exception as exc:
                    terminal_provider_error = (
                        "non-retryable daily quota"
                        in str(exc).casefold()
                    )
                    failure_report = {
                        "gap": [
                            round(float(gap[0]), 3),
                            round(float(gap[1]), 3),
                        ],
                        "changed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "validation_after": validation,
                        "validation_progress": False,
                        "reused_checkpoint": False,
                        "terminal_provider_error": (
                            terminal_provider_error
                        ),
                        "targeted_blocking_gap": bool(
                            targets_blocking_gap
                        ),
                        "blocking_gap_resolved": False,
                    }
                    recovery_reports.append(failure_report)
                    failure_record = {
                        "schema": (
                            "subgen_longform_coverage_recovery_failure_v1"
                        ),
                        "run_identity": run_identity,
                        "chunk": recovery_chunk,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "validation_before": validation,
                    }
                    diagnostics = getattr(exc, "diagnostics", None)
                    if diagnostics is not None:
                        failure_record["diagnostics"] = diagnostics
                    atomic_write_json(
                        recovery_failure_path,
                        failure_record,
                    )
                manifest["coverage_recovery"] = recovery_reports
                atomic_write_json(manifest_path, manifest)
                if (
                    recovery_reports
                    and recovery_reports[-1].get(
                        "terminal_provider_error"
                    )
                ):
                    break
            manifest["coverage_recovery"] = recovery_reports
            transcript_text = re.sub(
                r"\s+",
                " ",
                " ".join(
                    str(segment.get("text") or "")
                    for segment in segments
                ),
            ).strip()
        if not validation["accept"]:
            manifest.update({
                "status": "failed",
                "merge_report": merge_report,
                "validation": validation,
            })
            if timing_evidence_only:
                manifest["timing_evidence_filter"] = (
                    timing_evidence_filter_reports
                )
            atomic_write_json(manifest_path, manifest)
            raise RuntimeError(
                "Long-form transcription failed its independent speech/silence "
                f"validation: {validation['problems']}."
            )
        for recovery_audio in recovery_audio_paths:
            recovery_audio.unlink(missing_ok=True)
    else:
        transcript_text, merge_report = merge_text_only_chunks(results)
        segments = []
        validation = {
            "schema": "subgen_longform_validation_v1",
            "accept": bool(transcript_text) or not speech_map.get("speech_intervals"),
            "problems": [] if transcript_text else ["empty_transcript"],
            "timing_validation_deferred": True,
        }
        if not validation["accept"]:
            manifest["status"] = "failed"
            manifest["validation"] = validation
            atomic_write_json(manifest_path, manifest)
            raise RuntimeError("Long-form transcription produced no transcript text.")

    usage = {
        "longform": True,
        "pipeline_version": LONGFORM_PIPELINE_VERSION,
        "chunk_count": len(plan),
        "resumable": True,
        "sequential": True,
        "source_duration_seconds": preflight["duration_seconds"],
        "billed_duration_seconds": round(
            sum(float(chunk["duration"]) for chunk in plan),
            3,
        ),
        "chunks": [
            {
                "index": int(result["chunk"]["index"]),
                "status": result["status"],
                "usage": result.get("usage") or {},
            }
            for result in results
        ],
    }
    languages = [
        result.get("language")
        for result in results
        if result.get("language")
    ]
    unique_languages = list(dict.fromkeys(languages))
    manifest.update({
        "status": "complete",
        "merge_report": merge_report,
        "validation": validation,
        "transcript_sha256": hashlib.sha256(
            transcript_text.encode("utf-8")
        ).hexdigest(),
        "segment_count": len(segments),
    })
    if expected_timing and timing_evidence_only:
        manifest["timing_evidence_filter"] = (
            timing_evidence_filter_reports
        )
    atomic_write_json(manifest_path, manifest)
    return {
        "provider": provider,
        "model": model,
        "text": transcript_text,
        "segments": segments,
        "words": [
            word
            for result in results
            for word in result.get("words") or []
        ],
        "language": (
            unique_languages[0]
            if len(unique_languages) == 1
            else ("mixed" if unique_languages else language)
        ),
        "duration": preflight["duration_seconds"],
        "timing_kind": (
            results[0].get("timing_kind", "none")
            if results
            else "none"
        ),
        "usage": usage,
        "metadata": {
            "longform_manifest": str(manifest_path),
            "run_identity": run_identity,
            "source_sha256": source_sha256,
            "speech_map": speech_map,
            "validation": validation,
            "supplemental_timing_evidence": manifest.get(
                "supplemental_timing_evidence"
            ),
            "coverage_recovery": manifest.get(
                "coverage_recovery"
            ),
            "chunk_languages": languages,
            "chunk_metadata": [
                {
                    "index": int(result["chunk"]["index"]),
                    "language": result.get("language"),
                    "metadata": result.get("metadata") or {},
                }
                for result in results
            ],
        },
    }
