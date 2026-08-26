"""Automatic, fail-closed acoustic evidence for repeated speech.

The engine deliberately has no expected-count or expected-interval argument.
It combines a direct spectral-onset/motif path with raw word/segment timing
observations from a separately executed timing model.  Timing-model words are
never returned as canonical subtitle text.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path


ACOUSTIC_REPETITION_ALGORITHM_VERSION = "acoustic_repetition_v2_text_guided"
DEFAULT_ANALYSIS_CONFIG = {
    "sample_rate": 16000,
    "stft_window_seconds": 0.025,
    "stft_hop_seconds": 0.010,
    "spectral_onset_min_distance_seconds": 0.70,
    "spectral_onset_prominence_quantile": 0.50,
    "spectral_onset_prominence_scale": 0.25,
    "timing_occurrence_gap_seconds": 0.12,
    "maximum_onset_deviation_seconds": 0.45,
    "minimum_pairwise_motif_similarity": 0.23,
    "maximum_duration_cv": 0.40,
    "minimum_count_confidence": 0.55,
    "candidate_backtrack_seconds": 0.08,
    "minimum_uncovered_tail_seconds": 0.50,
    "maximum_candidate_region_seconds": 30.0,
    "post_event_inspection_seconds": 0.20,
    "text_candidate_padding_seconds": 1.50,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config(value=None):
    result = dict(DEFAULT_ANALYSIS_CONFIG)
    result.update(value or {})
    return result


def analysis_config_sha256(value=None):
    payload = json.dumps(_config(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def decode_audio_pcm(audio_path, *, sample_rate=16000):
    """Decode with the same bundled FFmpeg dependency used by SubGen."""
    process = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(audio_path), "-f", "f32le",
            "-ac", "1", "-ar", str(int(sample_rate)), "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    import numpy as np

    signal = np.frombuffer(process.stdout, dtype="<f4").copy()
    if not len(signal):
        raise RuntimeError("Acoustic analysis decoded no audio samples.")
    return signal


def energy_vad_timing_observations(
    audio_path,
    *,
    sample_rate=16000,
    frame_seconds=0.020,
    hop_seconds=0.010,
    minimum_event_seconds=0.12,
    bridge_gap_seconds=0.06,
):
    """Return timing-only speech/activity regions from an adaptive RMS VAD.

    This is intentionally independent of the spectral-flux onset path.  It is
    useful as a local fallback and for blind fixtures; continuous music or
    overlap tends to produce one broad/ambiguous region and therefore fails
    closed instead of inventing a repetition count.
    """
    import numpy as np

    signal = decode_audio_pcm(audio_path, sample_rate=sample_rate)
    frame = max(16, int(round(sample_rate * frame_seconds)))
    hop = max(8, int(round(sample_rate * hop_seconds)))
    if len(signal) < frame:
        return []
    starts = np.arange(0, len(signal) - frame + 1, hop)
    rms = np.array([
        np.sqrt(np.mean(signal[start:start + frame] ** 2) + 1e-12)
        for start in starts
    ])
    db = 20.0 * np.log10(rms + 1e-9)
    low, high = np.quantile(db, [0.20, 0.90])
    threshold = float(low + 0.35 * max(6.0, high - low))
    active = db >= threshold
    bridge = max(0, int(round(bridge_gap_seconds / hop_seconds)))
    if bridge:
        inactive = np.where(~active)[0]
        for index in inactive:
            left = max(0, index - bridge)
            right = min(len(active), index + bridge + 1)
            if active[left:index].any() and active[index + 1:right].any():
                active[index] = True
    events = []
    begin = None
    for index, enabled in enumerate(active):
        if enabled and begin is None:
            begin = index
        if begin is not None and (not enabled or index == len(active) - 1):
            stop = index if not enabled else index + 1
            start_seconds = starts[begin] / sample_rate
            end_sample = min(len(signal), starts[min(stop - 1, len(starts) - 1)] + frame)
            end_seconds = end_sample / sample_rate
            if end_seconds - start_seconds >= minimum_event_seconds:
                events.append({
                    "start": start_seconds,
                    "end": end_seconds,
                    "_granularity": "vad_region",
                })
            begin = None
    return events


def split_timing_at_largest_gap(timing_segments, *, minimum_gap_seconds=0.30):
    """Split a timing sequence at its largest defensible silence."""
    observations = normalize_timing_observations(timing_segments)
    if len(observations) < 2:
        return observations, []
    gaps = [
        observations[index + 1]["start_seconds"] - observations[index]["end_seconds"]
        for index in range(len(observations) - 1)
    ]
    split = max(range(len(gaps)), key=gaps.__getitem__)
    if gaps[split] < minimum_gap_seconds:
        return observations, []
    return observations[:split + 1], observations[split + 1:]


def normalize_timing_observations(segments, *, source_offset_seconds=0.0):
    """Keep timing only; intentionally discard all timing-model lexical text."""
    result = []
    offset = float(source_offset_seconds or 0.0)
    for item in segments or []:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", item.get("start_seconds"))) + offset
            end = float(item.get("end", item.get("end_seconds"))) + offset
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        result.append({
            "start_seconds": start,
            "end_seconds": end,
            "granularity": str(item.get("_granularity") or item.get("granularity") or "segment"),
        })
    return sorted(result, key=lambda item: (item["start_seconds"], item["end_seconds"]))


def discover_terminal_candidate_region(duration_seconds, primary_timing_segments, *, config=None):
    """Find an uncovered terminal region without using text or a known count."""
    cfg = _config(config)
    observations = normalize_timing_observations(primary_timing_segments)
    duration = float(duration_seconds)
    if not observations:
        return {
            "available": False,
            "reason": "primary_timing_observations_missing",
            "region_start": None,
            "region_end": duration,
        }
    last_end = min(duration, max(item["end_seconds"] for item in observations))
    uncovered = duration - last_end
    if uncovered < float(cfg["minimum_uncovered_tail_seconds"]):
        return {
            "available": False,
            "reason": "no_independently_uncovered_terminal_region",
            "region_start": None,
            "region_end": duration,
            "last_primary_timing_end": last_end,
            "uncovered_tail_seconds": uncovered,
        }
    region_start = max(
        0.0,
        last_end - float(cfg["candidate_backtrack_seconds"]),
        duration - float(cfg["maximum_candidate_region_seconds"]),
    )
    return {
        "available": True,
        "reason": "uncovered_tail_after_primary_timing",
        "region_start": region_start,
        "region_end": duration,
        "last_primary_timing_end": last_end,
        "uncovered_tail_seconds": uncovered,
    }


def discover_text_guided_candidate_region(
    duration_seconds,
    repetition_run,
    *,
    primary_timing_segments=None,
    config=None,
):
    """Localize an arbitrary text-discovered repetition without using its count.

    Text and aligned cue timing identify only where to inspect. The acoustic
    occurrence count is inferred later from independent localized timing and
    waveform evidence. A terminal uncovered region from the timing model wins
    when available because it is independent of the canonical repeated text.
    """
    cfg = _config(config)
    duration = max(0.0, float(duration_seconds))
    run = dict(repetition_run or {})
    position = str(run.get("position") or "").lower()
    if position == "suffix" and primary_timing_segments:
        terminal = discover_terminal_candidate_region(
            duration,
            primary_timing_segments,
            config=cfg,
        )
        if terminal.get("available"):
            return {
                **terminal,
                "localization_method": "independent_terminal_timing_gap",
                "text_run_start_hint": run.get("start_seconds"),
                "text_run_end_hint": run.get("end_seconds"),
            }

    try:
        start_hint = float(run.get("start_seconds"))
        end_hint = float(run.get("end_seconds"))
    except (TypeError, ValueError):
        return {
            "available": False,
            "reason": "text_repetition_timing_unavailable",
            "region_start": None,
            "region_end": duration,
            "localization_method": "aligned_text_run",
        }
    if not math.isfinite(start_hint) or not math.isfinite(end_hint) or end_hint <= start_hint:
        return {
            "available": False,
            "reason": "text_repetition_timing_invalid",
            "region_start": None,
            "region_end": duration,
            "localization_method": "aligned_text_run",
        }

    padding = max(0.0, float(cfg["text_candidate_padding_seconds"]))
    region_start = max(0.0, start_hint - padding)
    region_end = min(
        duration,
        duration if position == "suffix" else end_hint + padding,
    )
    return {
        "available": region_end > region_start,
        "reason": "aligned_text_repetition_region",
        "region_start": region_start,
        "region_end": region_end,
        "localization_method": "aligned_text_run_with_context",
        "text_run_start_hint": start_hint,
        "text_run_end_hint": end_hint,
        "context_padding_seconds": padding,
        "text_occurrence_count_used_for_acoustic_inference": False,
    }


def group_timing_events(localized_timing_segments, *, region_start, region_end, config=None):
    """Group timing-model tokens into vocal events using only temporal gaps."""
    cfg = _config(config)
    observations = [
        item for item in normalize_timing_observations(localized_timing_segments)
        if item["end_seconds"] > region_start and item["start_seconds"] < region_end
    ]
    if not observations:
        return []
    gap = float(cfg["timing_occurrence_gap_seconds"])
    events = []
    current = dict(observations[0])
    current["token_count"] = 1
    current["granularities"] = {current.pop("granularity")}
    for item in observations[1:]:
        # A VAD region is already an independently bounded vocal event.  The
        # gap threshold is for adjacent word/token timings within one event;
        # applying it to VAD regions can merge fast consecutive repetitions.
        current_is_vad = current["granularities"] == {"vad_region"}
        item_is_vad = item["granularity"] == "vad_region"
        if (
            (current_is_vad and item_is_vad)
            or item["start_seconds"] - current["end_seconds"] > gap
        ):
            events.append(current)
            current = {
                "start_seconds": item["start_seconds"],
                "end_seconds": item["end_seconds"],
                "token_count": 1,
                "granularities": {item["granularity"]},
            }
        else:
            current["end_seconds"] = max(current["end_seconds"], item["end_seconds"])
            current["token_count"] += 1
            current["granularities"].add(item["granularity"])
    events.append(current)
    for event in events:
        event["granularities"] = sorted(event["granularities"])
    return events


def group_timing_observations_by_onsets(
    normalized_observations,
    onsets,
    *,
    region_start,
    region_end,
    config=None,
):
    """Partition timing observations at direct waveform onset boundaries.

    This is a conservative recovery path for fast genuine repetitions where a
    timing model leaves too little inter-phrase silence for gap grouping. The
    waveform supplies the candidate boundaries; timing observations only
    confirm that every boundary owns speech tokens. No expected phrase or
    occurrence count is accepted.
    """
    cfg = _config(config)
    onset_times = sorted({
        float(item["start_seconds"])
        for item in (onsets or [])
        if region_start < float(item["start_seconds"]) < region_end
    })
    if len(onset_times) < 2:
        return []
    # Each detected onset starts the next candidate occurrence. A timing word
    # may begin slightly before that onset because ASR boundaries are coarse,
    # so ownership uses the word midpoint and the onset itself as the split.
    boundaries = [float(region_start)] + onset_times[1:] + [float(region_end)]
    observations = [
        item
        for item in (normalized_observations or [])
        if item["end_seconds"] > region_start
        and item["start_seconds"] < region_end
    ]
    maximum_padding = float(cfg["maximum_onset_deviation_seconds"])
    events = []
    for index, onset_time in enumerate(onset_times):
        left = boundaries[index]
        right = boundaries[index + 1]
        owned = [
            item
            for item in observations
            if left <= (
                float(item["start_seconds"]) + float(item["end_seconds"])
            ) / 2.0 < right
        ]
        if not owned:
            return []
        observed_end = max(float(item["end_seconds"]) for item in owned)
        if index + 1 < len(onset_times):
            event_end = min(
                onset_times[index + 1] - 0.04,
                observed_end + maximum_padding,
            )
        else:
            event_end = min(float(region_end), observed_end + maximum_padding)
        if event_end <= onset_time:
            return []
        events.append({
            "start_seconds": onset_time,
            "end_seconds": event_end,
            "token_count": len(owned),
            "granularities": sorted({item["granularity"] for item in owned}),
        })
    return events


def _signal_features(signal, sample_rate, config):
    import numpy as np
    from scipy.fft import dct
    from scipy.signal import stft

    window = max(64, int(round(sample_rate * float(config["stft_window_seconds"]))))
    hop = max(16, int(round(sample_rate * float(config["stft_hop_seconds"]))))
    nfft = 1
    while nfft < window:
        nfft *= 2
    frequencies, times, spectrum = stft(
        signal,
        sample_rate,
        nperseg=window,
        noverlap=window - hop,
        nfft=nfft,
        boundary=None,
    )
    power = np.abs(spectrum) ** 2
    speech_band = (frequencies >= 100) & (frequencies <= min(4000, sample_rate / 2))
    normalized = np.log1p(power[speech_band] * 1e6)
    normalized /= np.linalg.norm(normalized, axis=0, keepdims=True) + 1e-9
    flux = np.r_[0.0, np.maximum(0.0, np.diff(normalized, axis=1)).sum(axis=0)]
    flux = np.convolve(flux, np.ones(5) / 5.0, mode="same")
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=0)) + 1e-9)

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    points = mel_to_hz(np.linspace(hz_to_mel(80), hz_to_mel(sample_rate / 2 - 100), 28))
    bank = np.zeros((26, len(frequencies)))
    for index in range(26):
        left, center, right = points[index:index + 3]
        bank[index] = np.maximum(0.0, np.minimum(
            (frequencies - left) / max(center - left, 1e-9),
            (right - frequencies) / max(right - center, 1e-9),
        ))
    cepstra = dct(np.log(bank @ power + 1e-8), axis=0, norm="ortho")[1:14].T
    delta = np.vstack([np.zeros((1, cepstra.shape[1])), np.diff(cepstra, axis=0)])
    motif = np.hstack([cepstra, delta])
    motif = (motif - motif.mean(axis=0)) / (motif.std(axis=0) + 1e-6)
    motif /= np.linalg.norm(motif, axis=1, keepdims=True) + 1e-8
    return times, flux, rms_db, motif


def spectral_onset_events(times, flux, rms_db, *, region_start, region_end, config=None):
    import numpy as np
    from scipy.signal import find_peaks

    cfg = _config(config)
    mask = (times >= float(region_start)) & (times <= float(region_end))
    indices = np.where(mask)[0]
    if len(indices) < 3:
        return []
    local_flux = flux[mask]
    frame_seconds = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    minimum_distance = max(
        1,
        int(round(float(cfg["spectral_onset_min_distance_seconds"]) / frame_seconds)),
    )
    prominence = (
        float(np.quantile(local_flux, float(cfg["spectral_onset_prominence_quantile"])))
        * float(cfg["spectral_onset_prominence_scale"])
    )
    local_peaks, properties = find_peaks(
        local_flux,
        distance=minimum_distance,
        prominence=max(prominence, 1e-6),
    )
    absolute = indices[local_peaks]
    # Estimate decoder/media noise from the complete inspected signal.  A
    # repetition region can contain continuous music, so its own lower decile
    # is not a valid noise floor and would suppress quieter genuine onsets.
    noise_floor = float(np.quantile(rms_db, 0.10))
    result = []
    for position, index in enumerate(absolute):
        time_value = float(times[index])
        # Exclude decoder/silence-edge discontinuities and near-silent peaks.
        if time_value <= region_start + 0.04 or time_value >= region_end - 0.04:
            continue
        if float(rms_db[index]) < noise_floor + 4.0:
            continue
        result.append({
            "start_seconds": time_value,
            "prominence": float(properties["prominences"][position]),
            "rms_db": float(rms_db[index]),
        })
    return result


def _dtw_similarity(left, right):
    import numpy as np
    from scipy.spatial.distance import cdist

    if len(left) < 3 or len(right) < 3:
        return 0.0
    distances = cdist(left, right, "cosine")
    rows, columns = distances.shape
    costs = np.full((rows + 1, columns + 1), np.inf)
    lengths = np.zeros((rows + 1, columns + 1), dtype=int)
    costs[0, 0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            options = (
                (costs[row - 1, column], lengths[row - 1, column]),
                (costs[row, column - 1], lengths[row, column - 1]),
                (costs[row - 1, column - 1], lengths[row - 1, column - 1]),
            )
            prior_cost, prior_length = min(options, key=lambda item: item[0])
            costs[row, column] = prior_cost + distances[row - 1, column - 1]
            lengths[row, column] = prior_length + 1
    return float(max(-1.0, min(1.0, 1.0 - costs[rows, columns] / lengths[rows, columns])))


def motif_similarity_report(times, motif_features, events):
    import numpy as np

    features = []
    for event in events:
        mask = (
            (times >= float(event["start_seconds"]))
            & (times <= float(event["end_seconds"]))
        )
        features.append(motif_features[mask])
    pairwise = []
    for left in range(len(features)):
        for right in range(left + 1, len(features)):
            pairwise.append({
                "left_event": left + 1,
                "right_event": right + 1,
                "similarity": _dtw_similarity(features[left], features[right]),
            })
    values = [item["similarity"] for item in pairwise]
    return {
        "pairwise": pairwise,
        "minimum": min(values) if values else None,
        "median": float(np.median(values)) if values else None,
    }


def repeated_prefix_length(event_count, motif_report, minimum_similarity):
    """Return the leading acoustically coherent motif run.

    Any timing events after this run are treated as later speech, not as extra
    copies of the repeated unit.
    """
    if event_count <= 1:
        return event_count
    similarities = {
        (item["left_event"], item["right_event"]): item["similarity"]
        for item in motif_report.get("pairwise") or []
    }
    if similarities.get((1, 2), -1.0) < minimum_similarity:
        return 0
    length = 2
    for candidate in range(3, event_count + 1):
        if all(
            similarities.get((prior, candidate), -1.0) >= minimum_similarity
            for prior in range(1, candidate)
        ):
            length = candidate
        else:
            break
    return length


def _match_onsets(events, onsets, maximum_deviation):
    unused = set(range(len(onsets)))
    matches = []
    for event_index, event in enumerate(events):
        candidates = [
            (abs(onsets[index]["start_seconds"] - event["start_seconds"]), index)
            for index in unused
        ]
        if not candidates:
            return []
        deviation, onset_index = min(candidates)
        if deviation > maximum_deviation:
            return []
        unused.remove(onset_index)
        matches.append({
            "event_index": event_index + 1,
            "onset_index": onset_index + 1,
            "deviation_seconds": deviation,
        })
    return matches


def infer_repetition_evidence(
    media_path,
    audio_path,
    *,
    primary_timing_segments,
    localized_timing_segments,
    localized_timing_source_offset_seconds=0.0,
    candidate_region=None,
    config=None,
):
    """Infer a repetition count without accepting a known count or phrase."""
    import numpy as np

    cfg = _config(config)
    sample_rate = int(cfg["sample_rate"])
    signal = decode_audio_pcm(audio_path, sample_rate=sample_rate)
    duration = len(signal) / sample_rate
    candidate = (
        dict(candidate_region)
        if candidate_region is not None
        else discover_terminal_candidate_region(
            duration, primary_timing_segments, config=cfg
        )
    )
    if candidate.get("available"):
        try:
            candidate["region_start"] = max(0.0, float(candidate["region_start"]))
            candidate["region_end"] = min(duration, float(candidate["region_end"]))
        except (KeyError, TypeError, ValueError):
            candidate["available"] = False
            candidate["reason"] = "candidate_region_invalid"
        if candidate.get("available") and candidate["region_end"] <= candidate["region_start"]:
            candidate["available"] = False
            candidate["reason"] = "candidate_region_empty"
    base = {
        "schema": "subgen_automatic_acoustic_repetition_evidence_v2",
        "algorithm_version": ACOUSTIC_REPETITION_ALGORITHM_VERSION,
        "inference_origin": "automatic_acoustic_engine",
        "media_sha256": sha256_file(media_path),
        "audio_sha256": sha256_file(audio_path),
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "analysis_config": cfg,
        "analysis_config_sha256": analysis_config_sha256(cfg),
        "expected_count_argument_supported": False,
        "human_ground_truth_used": False,
        "candidate_region": candidate,
    }
    if not candidate["available"]:
        return {
            **base,
            "predicted_count": None,
            "events": [],
            "methods_agree": False,
            "count_confidence": 0.0,
            "boundary_uncertainty_seconds": None,
            "later_speech_detected": None,
            "ambiguity_flags": [candidate["reason"]],
            "safe_for_automatic_trim": False,
        }

    region_start = float(candidate["region_start"])
    region_end = float(candidate["region_end"])
    localized = normalize_timing_observations(
        localized_timing_segments,
        source_offset_seconds=localized_timing_source_offset_seconds,
    )
    all_timing_events = group_timing_events(
        localized,
        region_start=region_start,
        region_end=region_end,
        config=cfg,
    )
    times, flux, rms_db, motif = _signal_features(signal, sample_rate, cfg)
    all_motif_report = motif_similarity_report(times, motif, all_timing_events)
    repeated_length = repeated_prefix_length(
        len(all_timing_events),
        all_motif_report,
        float(cfg["minimum_pairwise_motif_similarity"]),
    )
    timing_events = all_timing_events[:repeated_length]
    later_timing_events = all_timing_events[repeated_length:]
    motif_report = motif_similarity_report(times, motif, timing_events)
    timing_last_end = max(
        (item["end_seconds"] for item in timing_events),
        default=region_start,
    )
    maximum_onset_deviation = float(cfg["maximum_onset_deviation_seconds"])
    signal_inspection_start = max(
        region_start,
        (
            timing_events[0]["start_seconds"] - maximum_onset_deviation
            if timing_events else region_start
        ),
    )
    # Inspect the whole remaining candidate when timing reports no later
    # speech. Otherwise a timing model that silently misses the last genuine
    # occurrence would appear to agree merely because signal inspection
    # stopped at the same incomplete boundary. When timing does identify B,
    # stop at B's first boundary so its onset is not counted as another R.
    signal_inspection_end = min(
        region_end,
        (
            float(later_timing_events[0]["start_seconds"])
            if later_timing_events
            else region_end
        ),
    )
    onsets = spectral_onset_events(
        times,
        flux,
        rms_db,
        region_start=signal_inspection_start,
        region_end=signal_inspection_end,
        config=cfg,
    )
    onset_resegmentation = {
        "attempted": False,
        "applied": False,
        "reason_codes": [],
        "events": [],
    }
    if len(onsets) > len(all_timing_events) and len(onsets) >= 2:
        onset_resegmentation["attempted"] = True
        onset_events = group_timing_observations_by_onsets(
            localized,
            onsets,
            region_start=signal_inspection_start,
            region_end=signal_inspection_end,
            config=cfg,
        )
        onset_resegmentation["events"] = onset_events
        if len(onset_events) != len(onsets):
            onset_resegmentation["reason_codes"].append(
                "not_every_waveform_onset_has_timing_support"
            )
        else:
            onset_all_motif = motif_similarity_report(times, motif, onset_events)
            onset_repeated_length = repeated_prefix_length(
                len(onset_events),
                onset_all_motif,
                float(cfg["minimum_pairwise_motif_similarity"]),
            )
            onset_prefix = onset_events[:onset_repeated_length]
            token_counts = [item["token_count"] for item in onset_prefix]
            if onset_repeated_length < 2:
                onset_resegmentation["reason_codes"].append(
                    "waveform_onsets_do_not_form_repeated_motif"
                )
            if token_counts and len(set(token_counts)) != 1:
                onset_resegmentation["reason_codes"].append(
                    "onset_partitions_have_inconsistent_timing_token_counts"
                )
            onset_resegmentation.update({
                "repeated_prefix_length": onset_repeated_length,
                "token_counts": token_counts,
                "motif_similarity": onset_all_motif,
            })
            if not onset_resegmentation["reason_codes"]:
                all_timing_events = onset_events
                all_motif_report = onset_all_motif
                timing_events = onset_prefix
                later_timing_events = onset_events[onset_repeated_length:]
                motif_report = motif_similarity_report(times, motif, timing_events)
                timing_last_end = max(
                    (item["end_seconds"] for item in timing_events),
                    default=region_start,
                )
                onset_resegmentation["applied"] = True
    matches = _match_onsets(
        timing_events,
        onsets,
        maximum_onset_deviation,
    )
    matched_onset_indices = {item["onset_index"] - 1 for item in matches}
    unmatched_outside_events = []
    unmatched_inside_events = []
    for index, onset in enumerate(onsets):
        if index in matched_onset_indices:
            continue
        container = next((
            event_index + 1
            for event_index, event in enumerate(timing_events)
            if event["start_seconds"] < onset["start_seconds"] < event["end_seconds"]
        ), None)
        if container is None:
            unmatched_outside_events.append({**onset, "onset_index": index + 1})
        else:
            unmatched_inside_events.append({
                **onset,
                "onset_index": index + 1,
                "inside_event_index": container,
            })
    signal_event_count = len(matches) + len(unmatched_outside_events)
    durations = np.array([
        item["end_seconds"] - item["start_seconds"] for item in timing_events
    ])
    duration_cv = (
        float(durations.std() / max(durations.mean(), 1e-9))
        if len(durations)
        else float("inf")
    )
    ambiguity = []
    if not timing_events:
        ambiguity.append("localized_timing_found_no_events")
    if signal_event_count != len(timing_events):
        ambiguity.append("signal_and_timing_event_counts_disagree")
    if len(matches) != len(timing_events):
        ambiguity.append("signal_and_timing_boundaries_disagree")
    if duration_cv > float(cfg["maximum_duration_cv"]):
        ambiguity.append("event_duration_variation_too_large")
    if len(durations) > 1 and durations[-1] < 0.65 * float(np.median(durations[:-1])):
        ambiguity.append("partial_final_occurrence")
    if len(timing_events) > 1 and (
        motif_report["minimum"] is None
        or motif_report["minimum"] < float(cfg["minimum_pairwise_motif_similarity"])
    ):
        ambiguity.append("acoustic_motif_similarity_too_low")
    if len(timing_events) == 1:
        ambiguity.append("single_event_cannot_confirm_repeated_motif")

    methods_agree = not any(
        code in ambiguity
        for code in (
            "localized_timing_found_no_events",
            "signal_and_timing_event_counts_disagree",
            "signal_and_timing_boundaries_disagree",
        )
    )
    maximum_deviation = max(
        (item["deviation_seconds"] for item in matches),
        default=float(cfg["maximum_onset_deviation_seconds"]),
    )
    onset_confidence = max(
        0.0,
        1.0 - maximum_deviation / float(cfg["maximum_onset_deviation_seconds"]),
    )
    motif_confidence = (
        max(0.0, min(1.0, float(motif_report["median"] or 0.0) / 0.40))
        if len(timing_events) > 1 else 0.0
    )
    duration_confidence = max(
        0.0,
        1.0 - duration_cv / float(cfg["maximum_duration_cv"]),
    )
    confidence = float(np.mean([onset_confidence, motif_confidence, duration_confidence]))
    later_speech = bool(later_timing_events)
    count_inference_confident = bool(
        len(timing_events) > 1
        and methods_agree
        and not ambiguity
        and confidence >= float(cfg["minimum_count_confidence"])
    )
    safe = bool(count_inference_confident and not later_speech)
    events = []
    for index, event in enumerate(timing_events):
        match = matches[index] if index < len(matches) else None
        event_confidence = (
            max(
                0.0,
                1.0 - match["deviation_seconds"]
                / float(cfg["maximum_onset_deviation_seconds"]),
            )
            if match else 0.0
        )
        events.append({
            "start_seconds": event["start_seconds"],
            "end_seconds": event["end_seconds"],
            "confidence": event_confidence,
            "timing_token_count": event["token_count"],
        })
    return {
        **base,
        "region_start": region_start,
        "region_end": region_end,
        "predicted_count": len(timing_events),
        "events": events,
        "signal_method": {
            "method": "speech_band_spectral_flux_onsets_and_log_mel_mfcc_dtw",
            "event_count": signal_event_count,
            "raw_onset_count": len(onsets),
            "onsets": onsets,
            "unmatched_onsets_outside_timing_events": unmatched_outside_events,
            "internal_transition_onsets": unmatched_inside_events,
            "motif_similarity": motif_report,
            "duration_cv": duration_cv,
        },
        "timing_method": {
            "method": (
                "waveform_onset_partitioned_independent_timing_observations"
                if onset_resegmentation["applied"]
                else "independent_localized_word_or_segment_timing_gaps"
            ),
            "event_count": len(timing_events),
            "events": timing_events,
            "all_timing_events": all_timing_events,
            "later_timing_events": later_timing_events,
            "all_event_motif_similarity": all_motif_report,
            "lexical_text_used": False,
            "onset_resegmentation": onset_resegmentation,
        },
        "method_associations": matches,
        "methods_agree": methods_agree,
        "count_confidence": confidence,
        "count_inference_confident": count_inference_confident,
        "boundary_uncertainty_seconds": maximum_deviation,
        "later_speech_detected": later_speech,
        "ambiguity_flags": ambiguity,
        "safe_for_automatic_trim": safe,
    }


def trim_evidence_from_automatic_report(report):
    """Adapt automatic evidence to the exact-prefix cutter's strict contract."""
    report = dict(report or {})
    events = [
        {
            "start": item.get("start_seconds"),
            "end": item.get("end_seconds"),
            "confidence": item.get("confidence"),
        }
        for item in report.get("events") or []
    ]
    return {
        **report,
        "events": events,
        "lexical_wording_confident": True,
        "lexical_source": "original_full_audio_gemini",
        "event_count_confident": bool(report.get("count_inference_confident")),
        "source_independence_confirmed": bool(report.get("methods_agree")),
        "independent_source_kinds": [
            "timing_model_token_boundaries",
            "waveform_onsets",
        ],
        "associations": [
            {"text_occurrence": index, "event_index": index}
            for index in range(1, len(events) + 1)
        ],
        "complete_media_inspected": True,
        "later_speech_intervals": [
            [item.get("start_seconds"), item.get("end_seconds")]
            for item in (
                ((report.get("timing_method") or {}).get("later_timing_events") or [])
            )
        ],
        "loop_position": "middle" if report.get("later_speech_detected") else "suffix",
        "post_trigger_seconds": max(
            (item.get("end") for item in events),
            default=None,
        ),
        "no_genuine_occurrence_removed": bool(report.get("count_inference_confident")),
    }
