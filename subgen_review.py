"""Shared fail-closed review, evidence, approval, and burn invariants.

This module intentionally has no web, cloud, Drive, database, or model SDK
dependency.  Every SubGen operating form can therefore use the same review
semantics and persist the returned JSON-compatible objects in its own durable
store.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


REVIEW_SCHEMA_VERSION = "subgen_review_v1"
REVIEW_STATES = (
    "PENDING",
    "PROCESSING",
    "READY_FOR_REVIEW",
    "NEEDS_ATTENTION",
    "IN_REVIEW",
    "APPROVED",
    "STALE_AFTER_EDIT",
    "BURNING",
    "COMPLETED",
    "FAILED",
    "BURN_FAILED",
)
ISSUE_SEVERITIES = ("info", "warning", "critical")
ISSUE_STATUSES = ("unresolved", "corrected", "accepted", "dismissed_with_reason")
RESOLVED_ISSUE_STATUSES = frozenset({"corrected", "accepted", "dismissed_with_reason"})


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _number(value, fallback=0.0):
    if isinstance(value, str) and ":" in value:
        try:
            hours, minutes, seconds = value.strip().replace(",", ".").split(":")
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except (TypeError, ValueError):
            return float(fallback)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return result if math.isfinite(result) else float(fallback)


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_cue(cue, index=None):
    result = dict(cue or {})
    result["id"] = str(result.get("id") or result.get("cue_id") or uuid.uuid4())
    result["index"] = int(result.get("index") or index or 1)
    result["start"] = round(_number(result.get("start_seconds", result.get("start"))), 6)
    result["end"] = round(_number(result.get("end_seconds", result.get("end"))), 6)
    result["text"] = str(result.get("text") or "")
    if "translation" in result:
        result["translation"] = str(result.get("translation") or "")
    result.pop("cue_id", None)
    result.pop("start_seconds", None)
    result.pop("end_seconds", None)
    return result


def normalize_cues(cues):
    return [normalize_cue(cue, index) for index, cue in enumerate(cues or [], 1)]


def format_srt_time(seconds):
    milliseconds = max(0, round(_number(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt_bytes(cues, text_field="text"):
    blocks = []
    for index, cue in enumerate(normalize_cues(cues), 1):
        blocks.append(
            f"{index}\n{format_srt_time(cue['start'])} --> {format_srt_time(cue['end'])}\n"
            f"{str(cue.get(text_field) or cue.get('text') or '').strip()}\n"
        )
    return ("\n".join(blocks) + ("\n" if blocks else "")).encode("utf-8")


def selected_burn_cues(review):
    source_language = review.get("source_language")
    target_language = review.get("target_language")
    translated = review.get("translation_draft") or []
    if target_language and target_language != source_language and translated:
        return normalize_cues(translated)
    return normalize_cues(review.get("source_draft") or [])


def draft_hash(review):
    """Hash the exact UTF-8 SRT bytes that an approved burn must consume."""
    return sha256_bytes(render_srt_bytes(selected_burn_cues(review)))


def review_revision_hash(review):
    payload = {
        "schema_version": review.get("schema_version"),
        "source_language": review.get("source_language"),
        "target_language": review.get("target_language"),
        "source_draft": normalize_cues(review.get("source_draft") or []),
        "translation_draft": normalize_cues(review.get("translation_draft") or []),
        "issues": [
            {
                "id": issue.get("id"),
                "status": issue.get("status"),
                "resolution_history": issue.get("resolution_history") or [],
            }
            for issue in review.get("issues") or []
        ],
    }
    return sha256_bytes(_canonical_json(payload).encode("utf-8"))


def new_review(
    video_id,
    source_hash,
    *,
    source_language=None,
    target_language=None,
    provider=None,
    model=None,
    prompt_version=None,
    source_draft=None,
    translation_draft=None,
    source_location=None,
):
    now = utc_now()
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_id": str(uuid.uuid4()),
        "video_id": str(video_id),
        "source_hash": str(source_hash or "").lower(),
        "source_location": copy.deepcopy(source_location),
        "source_language": source_language,
        "target_language": target_language,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "state": "PROCESSING",
        "source_draft": normalize_cues(source_draft),
        "translation_draft": normalize_cues(translation_draft),
        "field_state": {
            "source": "raw_provider_output",
            "translation": "automatically_translated" if translation_draft else None,
            "translation_stale": False,
            "stale_translation_cue_ids": [],
        },
        "issues": [],
        "coverage_report": None,
        "timestamp_report": None,
        "recovery_attempts": [],
        "approval": None,
        "edit_history": [],
        "burn": None,
        "final_artifacts": [],
        "created_at": now,
        "updated_at": now,
    }
    review["draft_hash"] = draft_hash(review)
    review["revision_hash"] = review_revision_hash(review)
    return review


def make_issue(
    video_id,
    code,
    message,
    *,
    severity="warning",
    start_seconds=None,
    end_seconds=None,
    affected_cue_ids=None,
    evidence_source=None,
    evidence=None,
    blocking=None,
    status="unresolved",
):
    if severity not in ISSUE_SEVERITIES:
        raise ValueError(f"Unknown issue severity: {severity}")
    if status not in ISSUE_STATUSES:
        raise ValueError(f"Unknown issue status: {status}")
    if blocking is None:
        blocking = severity == "critical"
    return {
        "id": str(uuid.uuid4()),
        "video_id": str(video_id),
        "severity": severity,
        "code": str(code),
        "start_seconds": None if start_seconds is None else round(_number(start_seconds), 6),
        "end_seconds": None if end_seconds is None else round(_number(end_seconds), 6),
        "affected_cue_ids": [str(value) for value in (affected_cue_ids or [])],
        "message": str(message),
        "evidence_source": list(evidence_source or []),
        "evidence": copy.deepcopy(evidence or {}),
        "blocking": bool(blocking),
        "status": status,
        "resolution_history": [],
        "created_at": utc_now(),
    }


def unresolved_blocking_issues(review):
    return [
        issue
        for issue in review.get("issues") or []
        if issue.get("blocking") and issue.get("status") not in RESOLVED_ISSUE_STATUSES
    ]


def unresolved_warning_issues(review):
    return [
        issue
        for issue in review.get("issues") or []
        if issue.get("severity") == "warning" and issue.get("status") == "unresolved"
    ]


def _refresh_review(review, *, edited=False):
    review["draft_hash"] = draft_hash(review)
    review["revision_hash"] = review_revision_hash(review)
    review["updated_at"] = utc_now()
    if edited:
        review["approval"] = None
        review["state"] = "STALE_AFTER_EDIT"
    return review


def set_ready_for_review(review):
    review["state"] = "NEEDS_ATTENTION" if unresolved_blocking_issues(review) else "READY_FOR_REVIEW"
    return _refresh_review(review)


def add_issue(review, issue):
    review.setdefault("issues", []).append(copy.deepcopy(issue))
    if issue.get("blocking") and issue.get("status") == "unresolved":
        review["state"] = "NEEDS_ATTENTION"
        review["approval"] = None
    return _refresh_review(review)


def resolve_issue(review, issue_id, status, *, actor="human", reason=None):
    if status not in RESOLVED_ISSUE_STATUSES:
        raise ValueError("An issue resolution must be corrected, accepted, or dismissed_with_reason")
    if status == "dismissed_with_reason" and not str(reason or "").strip():
        raise ValueError("A dismissal requires a reason")
    issue = next((item for item in review.get("issues") or [] if item.get("id") == issue_id), None)
    if not issue:
        raise KeyError(f"Unknown issue: {issue_id}")
    if issue.get("blocking") and status == "accepted":
        raise ValueError("A blocking critical issue must be corrected or dismissed with a reason")
    old_status = issue.get("status", "unresolved")
    issue["status"] = status
    issue.setdefault("resolution_history", []).append({
        "at": utc_now(),
        "actor": actor,
        "from": old_status,
        "to": status,
        "reason": reason,
    })
    review["approval"] = None
    review["state"] = "IN_REVIEW"
    return _refresh_review(review)


def edit_cue(review, draft_name, cue_id, changes, *, actor="human"):
    if draft_name not in {"source", "translation"}:
        raise ValueError("draft_name must be source or translation")
    key = f"{draft_name}_draft"
    cues = review.get(key) or []
    cue = next((item for item in cues if str(item.get("id")) == str(cue_id)), None)
    if not cue:
        raise KeyError(f"Unknown cue: {cue_id}")
    allowed = {"text", "start", "end", "start_seconds", "end_seconds"}
    normalized_changes = {field: value for field, value in dict(changes or {}).items() if field in allowed}
    before = copy.deepcopy(cue)
    if "start_seconds" in normalized_changes:
        normalized_changes["start"] = normalized_changes.pop("start_seconds")
    if "end_seconds" in normalized_changes:
        normalized_changes["end"] = normalized_changes.pop("end_seconds")
    cue.update(normalized_changes)
    cue.update(normalize_cue(cue, cue.get("index")))
    cue["manually_edited"] = True
    cue["previous_source"] = cue.get("canonical_source") or ("gemini" if draft_name == "source" else "automatic_translation")
    cue["canonical_source"] = "human_review"
    review.setdefault("edit_history", []).append({
        "at": utc_now(),
        "actor": actor,
        "operation": "edit_cue",
        "draft": draft_name,
        "cue_id": str(cue_id),
        "before": before,
        "after": copy.deepcopy(cue),
    })
    review.setdefault("field_state", {})[draft_name] = "manually_edited"
    if draft_name == "source":
        has_translation = bool(review.get("translation_draft"))
        review["field_state"]["translation_stale"] = has_translation
        review["field_state"]["translation"] = "stale" if has_translation else None
        stale = {
            str(value)
            for value in review["field_state"].get("stale_translation_cue_ids") or []
        }
        if has_translation:
            stale.add(str(cue_id))
        review["field_state"]["stale_translation_cue_ids"] = sorted(stale)
    return _refresh_review(review, edited=True)


def replace_draft(review, draft_name, cues, *, actor="human", operation="replace_draft"):
    if draft_name not in {"source", "translation"}:
        raise ValueError("draft_name must be source or translation")
    key = f"{draft_name}_draft"
    before = copy.deepcopy(review.get(key) or [])
    review[key] = normalize_cues(cues)
    for cue in review[key]:
        cue["manually_edited"] = True
        cue["canonical_source"] = "human_review"
    review.setdefault("edit_history", []).append({
        "at": utc_now(),
        "actor": actor,
        "operation": operation,
        "draft": draft_name,
        "before_hash": sha256_bytes(_canonical_json(before).encode("utf-8")),
        "after_hash": sha256_bytes(_canonical_json(review[key]).encode("utf-8")),
        "before": before,
        "after": copy.deepcopy(review[key]),
    })
    review.setdefault("field_state", {})[draft_name] = "manually_edited"
    if draft_name == "source":
        has_translation = bool(review.get("translation_draft"))
        review["field_state"]["translation_stale"] = has_translation
        review["field_state"]["translation"] = "stale" if has_translation else None
        review["field_state"]["stale_translation_cue_ids"] = (
            [str(cue.get("id")) for cue in review.get("source_draft") or []]
            if has_translation else []
        )
    return _refresh_review(review, edited=True)


def mark_translation_regenerated(review, cue_ids=None, *, actor="system"):
    selected = {str(value) for value in (cue_ids or [])}
    for cue in review.get("translation_draft") or []:
        if not selected or str(cue.get("id")) in selected:
            cue["manually_edited"] = False
            cue["canonical_source"] = "automatic_translation"
            cue["regenerated"] = True
    field_state = review.setdefault("field_state", {})
    stale = {
        str(value)
        for value in field_state.get("stale_translation_cue_ids") or []
    }
    stale -= selected
    if not selected:
        stale.clear()
    field_state["stale_translation_cue_ids"] = sorted(stale)
    field_state["translation_stale"] = bool(stale)
    field_state["translation"] = "stale" if stale else "regenerated"
    review.setdefault("edit_history", []).append({
        "at": utc_now(), "actor": actor, "operation": "regenerate_translation", "cue_ids": sorted(selected)
    })
    return _refresh_review(review, edited=True)


def selected_source_cues(review, cue_ids):
    selected = [str(value) for value in cue_ids or []]
    if not selected:
        raise ValueError("At least one source cue must be selected for retranslation")
    if len(selected) != len(set(selected)):
        raise ValueError("Selected cue IDs must be unique")
    by_id = {
        str(cue.get("id")): cue for cue in review.get("source_draft") or []
    }
    missing = [cue_id for cue_id in selected if cue_id not in by_id]
    if missing:
        raise KeyError(f"Unknown source cue(s): {', '.join(missing)}")
    return [copy.deepcopy(by_id[cue_id]) for cue_id in selected]


def apply_selected_cue_retranslation(
    review,
    cue_ids,
    translated_cues,
    *,
    actor="system",
    provider=None,
    model=None,
):
    """Replace only selected translations and preserve every prior version."""
    source = selected_source_cues(review, cue_ids)
    translated = list(translated_cues or [])
    if len(source) != len(translated):
        raise ValueError("Retranslation result count does not match selected cue count")
    targets = review.setdefault("translation_draft", [])
    by_index = {int(cue.get("index") or 0): cue for cue in targets}
    history_items = []
    for source_cue, result in zip(source, translated):
        text = str((result or {}).get("text") or "").strip()
        if not text:
            raise ValueError(
                f"Retranslation for source cue {source_cue.get('id')} is empty"
            )
        index = int(source_cue.get("index") or 0)
        target = by_index.get(index)
        if target is None:
            target = {
                "id": f"translation-{source_cue.get('id')}",
                "index": index,
                "start": source_cue.get("start"),
                "end": source_cue.get("end"),
                "text": "",
            }
            targets.append(target)
            by_index[index] = target
        before = copy.deepcopy(target)
        target.update({
            "start": source_cue.get("start"),
            "end": source_cue.get("end"),
            "text": text,
            "canonical_source": "automatic_translation",
            "manually_edited": False,
            "regenerated": True,
            "translation_provider": provider,
            "translation_model": model,
            "source_cue_id": str(source_cue.get("id")),
            "source_revision_sha256": sha256_bytes(
                _canonical_json({
                    "id": source_cue.get("id"),
                    "start": source_cue.get("start"),
                    "end": source_cue.get("end"),
                    "text": source_cue.get("text"),
                }).encode("utf-8")
            ),
        })
        history_items.append({
            "source_cue_id": str(source_cue.get("id")),
            "translation_cue_id": str(target.get("id")),
            "before": before,
            "after": copy.deepcopy(target),
        })
    targets.sort(key=lambda cue: int(cue.get("index") or 0))
    selected = {str(value) for value in cue_ids}
    field_state = review.setdefault("field_state", {})
    stale = {
        str(value)
        for value in field_state.get("stale_translation_cue_ids") or []
    }
    if field_state.get("translation_stale") and not stale:
        stale = {str(cue.get("id")) for cue in review.get("source_draft") or []}
    stale -= selected
    field_state["stale_translation_cue_ids"] = sorted(stale)
    field_state["translation_stale"] = bool(stale)
    field_state["translation"] = "stale" if stale else "regenerated"
    review.setdefault("translation_history", []).append({
        "at": utc_now(),
        "actor": actor,
        "operation": "retranslate_selected_cues",
        "source_cue_ids": sorted(selected),
        "provider": provider,
        "model": model,
        "items": history_items,
    })
    review.setdefault("edit_history", []).append({
        "at": utc_now(),
        "actor": actor,
        "operation": "retranslate_selected_cues",
        "cue_ids": sorted(selected),
    })
    return _refresh_review(review, edited=True)


def confirm_translation_current(review, *, actor="human", reason="reviewed against current source"):
    if not review.get("translation_draft"):
        return _refresh_review(review)
    review.setdefault("field_state", {})["translation_stale"] = False
    review["field_state"]["translation"] = "manually_reviewed"
    review.setdefault("edit_history", []).append({
        "at": utc_now(), "actor": actor, "operation": "confirm_translation_current", "reason": reason
    })
    return _refresh_review(review, edited=True)


def combined_review_cues(review):
    source = normalize_cues(review.get("source_draft") or [])
    translated = {cue.get("index"): cue for cue in normalize_cues(review.get("translation_draft") or [])}
    result = []
    for index, cue in enumerate(source, 1):
        target = translated.get(cue.get("index"))
        result.append({
            "id": cue.get("id"),
            "index": index,
            "start": cue.get("start"),
            "end": cue.get("end"),
            "text": cue.get("text", ""),
            "translation": (target or cue).get("text", ""),
            "provenance": {
                key: cue.get(key)
                for key in (
                    "canonical_source", "previous_source", "prompt_version", "generation_scope",
                    "source_offset_seconds", "independently_verified", "manually_edited",
                )
                if cue.get(key) is not None
            },
        })
    return result


def update_review_from_combined(review, segments, *, actor="human", translation_confirmed=False):
    normalized = normalize_cues(segments)
    has_translation = bool(
        review.get("target_language") and review.get("target_language") != review.get("source_language")
    )
    source = [{
        "id": str(item.get("id") or f"source-{index}"), "index": index,
        "start": item["start"], "end": item["end"], "text": item.get("text", ""),
    } for index, item in enumerate(normalized, 1)]
    translated = [{
        "id": f"translation-{item.get('id') or index}", "index": index,
        "start": item["start"], "end": item["end"], "text": item.get("translation", ""),
    } for index, item in enumerate(normalized, 1)] if has_translation else []

    def comparable(cues):
        return [
            (round(cue["start"], 6), round(cue["end"], 6), cue.get("text", ""))
            for cue in normalize_cues(cues)
        ]

    old_source = comparable(review.get("source_draft") or [])
    source_changed = comparable(source) != old_source
    timing_changed = [item[:2] for item in comparable(source)] != [item[:2] for item in old_source]
    translation_changed = has_translation and comparable(translated) != comparable(review.get("translation_draft") or [])
    if source_changed:
        replace_draft(review, "source", source, actor=actor, operation="combined_source_update")
    if translation_changed:
        replace_draft(review, "translation", translated, actor=actor, operation="combined_translation_update")
    if has_translation and translation_confirmed:
        confirm_translation_current(review, actor=actor)
    if timing_changed and (review.get("coverage_report") or {}).get("speech_intervals"):
        previous = review["coverage_report"]
        review["issues"] = [
            issue for issue in review.get("issues") or []
            if issue.get("code") not in {"uncovered_speech", "low_speech_coverage"}
        ]
        tolerances = previous.get("tolerances") or {}
        refreshed = independent_speech_coverage(
            review.get("video_id"), previous.get("speech_intervals"),
            [(cue["start"], cue["end"]) for cue in source],
            media_duration=previous.get("media_duration_seconds", 0),
            **{key: tolerances[key] for key in (
                "vad_boundary_uncertainty", "alignment_padding", "ignore_shorter_than",
                "critical_gap_seconds", "warning_gap_seconds", "min_coverage_ratio",
            ) if key in tolerances},
        )
        review["coverage_report"] = refreshed
        review["issues"].extend(refreshed.get("issues") or [])
    if source_changed or translation_changed:
        review["state"] = "IN_REVIEW"
    return _refresh_review(review)


def approve_review(review, *, actor="human", accept_warnings=False):
    blocking = unresolved_blocking_issues(review)
    if blocking:
        raise ValueError(f"Critical unresolved issues block approval: {[item['code'] for item in blocking]}")
    warnings = unresolved_warning_issues(review)
    if warnings and not accept_warnings:
        raise ValueError("Unresolved warnings require explicit acceptance before approval")
    if warnings:
        for issue in warnings:
            old = issue["status"]
            issue["status"] = "accepted"
            issue.setdefault("resolution_history", []).append({
                "at": utc_now(), "actor": actor, "from": old, "to": "accepted", "reason": "accepted_at_approval"
            })
    if review.get("field_state", {}).get("translation_stale"):
        raise ValueError("A stale translation cannot be approved")
    current_hash = draft_hash(review)
    revision = review_revision_hash(review)
    review["approval"] = {
        "approved_at": utc_now(),
        "approved_by": actor,
        "approved_draft_hash": current_hash,
        "approved_revision_hash": revision,
    }
    review["draft_hash"] = current_hash
    review["revision_hash"] = revision
    review["state"] = "APPROVED"
    review["updated_at"] = utc_now()
    return review


def assert_burn_allowed(review, source_path=None, *, source_hash=None):
    if review.get("state") != "APPROVED" or not review.get("approval"):
        raise ValueError("Burn requires an explicitly approved review")
    if unresolved_blocking_issues(review):
        raise ValueError("Burn is blocked by unresolved critical issues")
    current_hash = draft_hash(review)
    if current_hash != review["approval"].get("approved_draft_hash"):
        raise ValueError("Current draft hash does not match the approved draft hash")
    if review_revision_hash(review) != review["approval"].get("approved_revision_hash"):
        raise ValueError("Review state changed after approval")
    actual_source_hash = source_hash.lower() if source_hash else sha256_file(source_path)
    if actual_source_hash != str(review.get("source_hash") or "").lower():
        raise ValueError("Source media hash does not match the prepared media")
    return {
        "approved_draft_hash": current_hash,
        "source_hash": actual_source_hash,
        "subtitle_bytes": render_srt_bytes(selected_burn_cues(review)),
    }


def begin_burn(review, source_path=None, *, source_hash=None):
    gate = assert_burn_allowed(review, source_path, source_hash=source_hash)
    review["state"] = "BURNING"
    review["burn"] = {
        "started_at": utc_now(),
        "source_hash": gate["source_hash"],
        "approved_draft_hash": gate["approved_draft_hash"],
    }
    review["updated_at"] = utc_now()
    return gate


def complete_burn(review, subtitle_path, video_path):
    burn_record = review.get("burn")
    if review.get("state") != "BURNING" or not isinstance(burn_record, dict):
        raise ValueError("Burn completion requires a burn lifecycle started by begin_burn")
    approved_hash = review.get("approval", {}).get("approved_draft_hash")
    burned_hash = sha256_file(subtitle_path)
    if burned_hash != approved_hash:
        review["state"] = "BURN_FAILED"
        review.setdefault("burn", {})["error"] = "Burned subtitle input differs from approved draft"
        raise ValueError("Burned subtitle hash does not match approved draft hash")
    video_hash = sha256_file(video_path)
    review["state"] = "COMPLETED"
    burn_record.update({"completed_at": utc_now(), "burned_subtitle_hash": burned_hash, "video_hash": video_hash})
    review["final_artifacts"] = [
        {"kind": "approved_srt", "path": str(subtitle_path), "sha256": burned_hash},
        {"kind": "output_video", "path": str(video_path), "sha256": video_hash},
    ]
    review["updated_at"] = utc_now()
    return review


def save_review(path, review):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_review(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("Unsupported SubGen review manifest version")
    if data.get("state") not in REVIEW_STATES:
        raise ValueError("Review manifest has an invalid state")
    return data


def merge_intervals(intervals, *, padding=0.0):
    prepared = []
    for item in intervals or []:
        if isinstance(item, dict):
            start = item.get("start_seconds", item.get("start"))
            end = item.get("end_seconds", item.get("end"))
        else:
            start, end = item[:2]
        start = max(0.0, _number(start) - padding)
        end = max(start, _number(end) + padding)
        if end > start:
            prepared.append((start, end))
    prepared.sort()
    merged = []
    for start, end in prepared:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(round(start, 6), round(end, 6)) for start, end in merged]


def interval_duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def interval_intersection(left, right):
    left = merge_intervals(left)
    right = merge_intervals(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return merge_intervals(result)


def subtract_intervals(reference, coverage):
    remaining = []
    coverage = merge_intervals(coverage)
    for start, end in merge_intervals(reference):
        cursor = start
        for cov_start, cov_end in coverage:
            if cov_end <= cursor:
                continue
            if cov_start >= end:
                break
            if cov_start > cursor:
                remaining.append((cursor, min(cov_start, end)))
            cursor = max(cursor, cov_end)
            if cursor >= end:
                break
        if cursor < end:
            remaining.append((cursor, end))
    return merge_intervals(remaining)


def independent_speech_coverage(
    video_id,
    speech_intervals,
    accepted_intervals,
    *,
    media_duration,
    evidence_sources=None,
    vad_boundary_uncertainty=0.20,
    alignment_padding=0.20,
    ignore_shorter_than=0.25,
    critical_gap_seconds=0.75,
    warning_gap_seconds=0.35,
    min_coverage_ratio=0.97,
):
    """Compare independent speech S with supported canonical intervals A.

    VAD uncertainty shrinks S at each boundary; alignment padding expands A.
    Short residuals are retained in diagnostics but do not become lexical issues.
    """
    duration = max(0.0, _number(media_duration))
    raw_speech = merge_intervals(speech_intervals)
    adjusted_speech = []
    for start, end in raw_speech:
        adjusted_start = min(end, start + max(0.0, vad_boundary_uncertainty))
        adjusted_end = max(adjusted_start, end - max(0.0, vad_boundary_uncertainty))
        if adjusted_end > adjusted_start:
            adjusted_speech.append((adjusted_start, adjusted_end))
    adjusted_speech = merge_intervals(adjusted_speech)
    supported = merge_intervals(accepted_intervals, padding=max(0.0, alignment_padding))
    intersection = interval_intersection(adjusted_speech, supported)
    missing = subtract_intervals(adjusted_speech, supported)
    speech_seconds = interval_duration(adjusted_speech)
    covered_seconds = interval_duration(intersection)
    coverage_ratio = covered_seconds / speech_seconds if speech_seconds else 1.0
    meaningful = [item for item in missing if item[1] - item[0] >= ignore_shorter_than]
    sources = list(evidence_sources or ["vad", "whisper_timing", "alignment"])
    issues = []
    for start, end in meaningful:
        gap = end - start
        severity = (
            "critical" if gap >= critical_gap_seconds
            else "warning" if gap >= warning_gap_seconds
            else "info"
        )
        confidence = min(0.99, max(0.50, 0.55 + gap / max(1.0, critical_gap_seconds) * 0.25))
        issues.append(make_issue(
            video_id,
            "uncovered_speech",
            "Independent speech evidence is not covered by an accepted, aligned canonical Gemini cue.",
            severity=severity,
            start_seconds=start,
            end_seconds=end,
            evidence_source=sources,
            evidence={
                "confidence": round(confidence, 3),
                "gap_seconds": round(gap, 3),
                "media_duration_inspected": duration,
            },
            blocking=severity == "critical",
        ))
    if coverage_ratio < min_coverage_ratio and not any(issue["severity"] == "critical" for issue in issues):
        issues.append(make_issue(
            video_id,
            "low_speech_coverage",
            "Cumulative independently detected speech coverage is below the configured tolerance.",
            severity="critical",
            start_seconds=meaningful[0][0] if meaningful else None,
            end_seconds=meaningful[-1][1] if meaningful else None,
            evidence_source=sources,
            evidence={"coverage_ratio": round(coverage_ratio, 6), "minimum": min_coverage_ratio},
        ))
    return {
        "engine": "independent_speech_union_coverage_v1",
        "media_duration_seconds": duration,
        "complete_media_inspected": True,
        "speech_intervals": adjusted_speech,
        "accepted_supported_intervals": supported,
        "covered_speech_intervals": intersection,
        "uncovered_intervals": missing,
        "meaningful_uncovered_intervals": meaningful,
        "speech_seconds": round(speech_seconds, 6),
        "covered_speech_seconds": round(covered_seconds, 6),
        "coverage_ratio": round(coverage_ratio, 6),
        "tolerances": {
            "vad_boundary_uncertainty": vad_boundary_uncertainty,
            "alignment_padding": alignment_padding,
            "ignore_shorter_than": ignore_shorter_than,
            "critical_gap_seconds": critical_gap_seconds,
            "warning_gap_seconds": warning_gap_seconds,
            "min_coverage_ratio": min_coverage_ratio,
        },
        "issues": issues,
        "accept": not any(issue["blocking"] and issue["status"] == "unresolved" for issue in issues),
    }


def _normalized_text(value):
    value = unicodedata.normalize("NFKD", str(value or "")).casefold()
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def contiguous_token_repetition_runs(
    tokens,
    *,
    min_repetitions=2,
    min_repeated_tokens=2,
):
    """Find maximal exact tandem repeats without imposing a maximum count.

    A repetition exists only when a token unit occurs at least twice. The
    detector therefore has a mathematical lower bound of two, but no semantic
    upper bound for either the unit length or occurrence count. Detection is
    text-only evidence: callers must not use it to delete, rewrite, or reject
    transcript text without independent acoustic or human confirmation.
    """
    normalized = [_normalized_text(token) for token in tokens]
    normalized = [token for token in normalized if token]
    minimum_count = max(2, int(min_repetitions or 2))
    minimum_tokens = max(2, int(min_repeated_tokens or 2))
    token_count = len(normalized)
    candidates = {}

    # For a possible period P, equality between token[i] and token[i + P]
    # over P * (N - 1) consecutive positions proves N adjacent copies. NumPy
    # evaluates each shift in native code so phrase length and occurrence count
    # remain unbounded without a Python O(words^2) comparison loop.
    import numpy as np

    token_ids = {}
    encoded = np.fromiter(
        (token_ids.setdefault(token, len(token_ids)) for token in normalized),
        dtype=np.int64,
        count=token_count,
    )
    for unit_tokens in range(1, token_count // minimum_count + 1):
        equal_positions = np.flatnonzero(
            encoded[:-unit_tokens] == encoded[unit_tokens:]
        )
        if equal_positions.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(equal_positions) > 1) + 1
        group_starts = np.concatenate((np.array([0]), breaks))
        group_ends = np.concatenate((breaks, np.array([equal_positions.size])))
        required_equalities = unit_tokens * (minimum_count - 1)
        for group_start, group_end in zip(group_starts, group_ends):
            start = int(equal_positions[int(group_start)])
            equal_run = int(
                equal_positions[int(group_end) - 1]
                - equal_positions[int(group_start)]
                + 1
            )
            if equal_run < required_equalities:
                continue
            repetition_count = 1 + equal_run // unit_tokens
            repeated_tokens = repetition_count * unit_tokens
            if repeated_tokens < minimum_tokens:
                continue
            end = start + repeated_tokens
            candidate = {
                "start_token_index": start,
                "end_token_index": end,
                "unit_tokens": unit_tokens,
                "text_occurrences": repetition_count,
                "repeated_token_count": repeated_tokens,
                "unit_text": " ".join(normalized[start:start + unit_tokens]),
                "detection_method": "exact_tandem_tokens",
            }
            key = (start, end)
            existing = candidates.get(key)
            if existing is None or unit_tokens < existing["unit_tokens"]:
                candidates[key] = candidate

    # Retain maximal regions. A smaller-period representation of the same
    # region wins above; strict sub-runs add no independent review evidence.
    maximal = []
    for candidate in sorted(
        candidates.values(),
        key=lambda item: (
            item["start_token_index"],
            -item["end_token_index"],
            item["unit_tokens"],
        ),
    ):
        if any(
            candidate["start_token_index"] >= parent["start_token_index"]
            and candidate["end_token_index"] <= parent["end_token_index"]
            for parent in maximal
        ):
            continue
        maximal = [
            parent
            for parent in maximal
            if not (
                parent["start_token_index"] >= candidate["start_token_index"]
                and parent["end_token_index"] <= candidate["end_token_index"]
            )
        ]
        maximal.append(candidate)
    return maximal


def terminal_repetition_trim_report(
    transcript_text,
    repetitive_suffix,
    independent_evidence,
    *,
    require_automatic_inference=False,
):
    """Fail-closed removal of acoustically unsupported *terminal* copies.

    This function never generates or normalizes replacement text.  A successful
    result is an exact character prefix of ``transcript_text`` ending after the
    last independently supported occurrence.  The caller must supply evidence
    from both a timing model and waveform onset analysis; model-proposed text or
    timestamps are not accepted as independent evidence here.
    """
    original = str(transcript_text or "")
    suffix = dict(repetitive_suffix or {})
    evidence = dict(independent_evidence or {})
    failures = []
    if require_automatic_inference:
        if evidence.get("inference_origin") != "automatic_acoustic_engine":
            failures.append("audio_count_not_automatically_inferred")
        if evidence.get("expected_count_argument_supported") is not False:
            failures.append("expected_count_input_not_prohibited")
        if evidence.get("human_ground_truth_used") is not False:
            failures.append("human_ground_truth_leakage")
        if not evidence.get("media_sha256") or not evidence.get("audio_sha256"):
            failures.append("automatic_evidence_hash_binding_missing")
        if not evidence.get("analysis_config_sha256") or not evidence.get("algorithm_version"):
            failures.append("automatic_evidence_identity_missing")
        if evidence.get("predicted_count") != len(evidence.get("events") or []):
            failures.append("automatic_predicted_count_mismatch")
        if not evidence.get("methods_agree"):
            failures.append("automatic_methods_disagree")
        if not evidence.get("count_inference_confident"):
            failures.append("automatic_count_confidence_insufficient")

    token_matches = list(re.finditer(r"\S+", original, flags=re.UNICODE))
    tokens = [match.group(0) for match in token_matches]
    normalized_tokens = [_normalized_text(token) for token in tokens]
    word_count = len(tokens)

    unit_words = int(suffix.get("unit_word_count") or 0)
    start_word = suffix.get("start_word_index")
    end_word = suffix.get("end_word_index")
    reported_count = int(suffix.get("repetition_count") or 0)
    if not suffix.get("detected"):
        failures.append("repeated_unit_not_detected")
    if not suffix.get("high_confidence", True):
        failures.append("repeated_unit_uncertain")
    if start_word is None or end_word is None or unit_words <= 0:
        failures.append("invalid_occurrence_boundaries")
    else:
        start_word = int(start_word)
        end_word = int(end_word)
        if end_word != word_count:
            failures.append("repetition_is_not_terminal")
        repeated_words = end_word - start_word
        if start_word < 0 or repeated_words <= 0 or repeated_words % unit_words:
            failures.append("invalid_occurrence_boundaries")
        else:
            actual_count = repeated_words // unit_words
            if reported_count != actual_count:
                failures.append("reported_occurrence_count_mismatch")
            unit = normalized_tokens[start_word:start_word + unit_words]
            if not unit or any(not token for token in unit):
                failures.append("repeated_unit_uncertain")
            for occurrence in range(actual_count):
                begin = start_word + occurrence * unit_words
                if normalized_tokens[begin:begin + unit_words] != unit:
                    failures.append("mutated_occurrence_boundaries")
                    break
            # A response truncated part-way through one more repetition can
            # make an end-anchored detector select a rotated cycle.  For
            # example, ``A B C`` repeated and ending in a partial ``A`` looks
            # like a terminal run of ``B C A``.  Cutting that rotated run after
            # K events would retain the unsupported partial ``A``.  Treat any
            # matching phase immediately before the reported run as ambiguous
            # and fail closed.
            for overlap in range(1, unit_words):
                if (
                    start_word >= overlap
                    and normalized_tokens[start_word - overlap:start_word]
                    == unit[-overlap:]
                ):
                    failures.append("phase_shifted_partial_terminal_cycle")
                    break

    if not evidence.get("lexical_wording_confident"):
        failures.append("lexical_wording_uncertain")
    if evidence.get("lexical_source") != "original_full_audio_gemini":
        failures.append("lexical_source_is_not_original_gemini")
    if not evidence.get("event_count_confident"):
        failures.append("audio_event_count_uncertain")
    ambiguity = [str(value) for value in evidence.get("ambiguity_flags") or [] if str(value)]
    if ambiguity:
        failures.append("acoustic_ambiguity_present")

    source_kinds = set(evidence.get("independent_source_kinds") or [])
    if "timing_model_token_boundaries" not in source_kinds:
        failures.append("missing_independent_timing_evidence")
    if "waveform_onsets" not in source_kinds:
        failures.append("missing_waveform_onset_evidence")
    if not evidence.get("source_independence_confirmed"):
        failures.append("evidence_independence_unconfirmed")

    events = list(evidence.get("events") or [])
    audio_count = len(events)
    associations = list(evidence.get("associations") or [])
    expected_associations = [
        {"text_occurrence": index, "event_index": index}
        for index in range(1, audio_count + 1)
    ]
    normalized_associations = [
        {
            "text_occurrence": int(item.get("text_occurrence") or 0),
            "event_index": int(item.get("event_index") or 0),
        }
        for item in associations
        if isinstance(item, dict)
    ]
    if normalized_associations != expected_associations:
        failures.append("retained_occurrences_not_uniquely_associated")
    previous_end = -1.0
    for event in events:
        if not isinstance(event, dict):
            failures.append("invalid_acoustic_event")
            break
        start = _number(event.get("start"), -1)
        end = _number(event.get("end"), -1)
        if start < previous_end or end <= start:
            failures.append("invalid_acoustic_event")
            break
        previous_end = end

    if not evidence.get("complete_media_inspected"):
        failures.append("complete_media_not_inspected")
    if evidence.get("later_speech_intervals"):
        failures.append("later_speech_exists")
    if evidence.get("loop_position", "suffix") != "suffix":
        failures.append("middle_loop_requires_recovery")
    if not evidence.get("no_genuine_occurrence_removed"):
        failures.append("genuine_occurrence_safety_unconfirmed")

    text_count = reported_count
    if audio_count <= 0:
        failures.append("no_supported_audio_occurrences")
    if text_count <= audio_count:
        failures.append("no_unsupported_terminal_continuation")

    # Stable order makes reports and tests reproducible while retaining every
    # diagnostic reason discovered in one pass.
    failures = list(dict.fromkeys(failures))
    applied = not failures
    corrected = original
    cut_character_index = None
    if applied:
        retained_last_word = int(start_word) + audio_count * unit_words - 1
        cut_character_index = token_matches[retained_last_word].end()
        corrected = original[:cut_character_index]

    return {
        "engine": "exact_terminal_repetition_trim_v1",
        "applied": applied,
        "reason_codes": failures,
        "text_occurrences": text_count,
        "audio_occurrences": audio_count,
        "retained_occurrences": audio_count if applied else text_count,
        "removed_occurrences": (text_count - audio_count) if applied else 0,
        "unit_text": suffix.get("unit_text"),
        "cut_character_index": cut_character_index,
        "original_sha256": sha256_bytes(original.encode("utf-8")),
        "corrected_sha256": sha256_bytes(corrected.encode("utf-8")),
        "corrected_text": corrected,
        "original_prefix_preserved_exactly": bool(applied and original.startswith(corrected)),
        "replacement_text_generated": False,
        "evidence": evidence,
    }


def middle_repetition_trim_report(cues, repetition_run, independent_evidence):
    """Plan/perform only the deletion part of an A + R + B middle-loop repair.

    Later speech is deliberately not synthesized here.  The returned draft is
    never publishable until independently detected B has been recovered and
    validated (or review establishes that no B is missing).
    """
    original = normalize_cues(cues)
    run = dict(repetition_run or {})
    evidence = dict(independent_evidence or {})
    failures = []
    # A loop that reaches the emitted transcript end is still a middle loop
    # when independent complete-media inspection detects later speech B.
    effective_position = evidence.get("loop_position") or (
        "middle" if evidence.get("later_speech_intervals") else run.get("position")
    )
    if effective_position != "middle":
        failures.append("not_a_middle_loop")
    unit_cues = int(run.get("unit_cues") or 0)
    start = int(run.get("start_cue_index") or 0)
    end = int(run.get("end_cue_index") or 0)
    text_count = int(run.get("text_occurrences") or 0)
    events = list(evidence.get("events") or [])
    audio_count = len(events)
    if (
        unit_cues <= 0
        and run.get("detection_method") == "exact_tandem_tokens"
        and text_count > 0
        and 0 <= start < end <= len(original)
        and (end - start) % text_count == 0
    ):
        cue_token_counts = [
            len(_normalized_text(cue.get("text")).split())
            for cue in original
        ]
        cue_start_token = sum(cue_token_counts[:start])
        cue_end_token = sum(cue_token_counts[:end])
        if (
            cue_start_token == int(run.get("start_token_index") or 0)
            and cue_end_token == int(run.get("end_token_index") or 0)
        ):
            # The exact token run covers whole cues and every occurrence uses
            # the same number of cues.  Only this lossless boundary mapping is
            # safe for the existing cue-level deletion engine.
            unit_cues = (end - start) // text_count
    if unit_cues <= 0 or start < 0 or end > len(original) or end <= start:
        failures.append("invalid_occurrence_boundaries")
    elif end - start != unit_cues * text_count:
        failures.append("reported_occurrence_count_mismatch")
    else:
        normalized_unit = [_normalized_text(cue["text"]) for cue in original[start:start + unit_cues]]
        for occurrence in range(text_count):
            offset = start + occurrence * unit_cues
            candidate = [_normalized_text(cue["text"]) for cue in original[offset:offset + unit_cues]]
            if candidate != normalized_unit:
                failures.append("mutated_occurrence_boundaries")
                break
    if not evidence.get("lexical_wording_confident"):
        failures.append("lexical_wording_uncertain")
    if evidence.get("lexical_source") != "original_full_audio_gemini":
        failures.append("lexical_source_is_not_original_gemini")
    if not evidence.get("event_count_confident"):
        failures.append("audio_event_count_uncertain")
    if evidence.get("ambiguity_flags"):
        failures.append("acoustic_ambiguity_present")
    if not evidence.get("source_independence_confirmed"):
        failures.append("evidence_independence_unconfirmed")
    kinds = set(evidence.get("independent_source_kinds") or [])
    if not {"timing_model_token_boundaries", "waveform_onsets"}.issubset(kinds):
        failures.append("insufficient_independent_event_evidence")
    associations = [
        (int(item.get("text_occurrence") or 0), int(item.get("event_index") or 0))
        for item in evidence.get("associations") or [] if isinstance(item, dict)
    ]
    if associations != [(index, index) for index in range(1, audio_count + 1)]:
        failures.append("retained_occurrences_not_uniquely_associated")
    if audio_count <= 0 or text_count <= audio_count:
        failures.append("no_unsupported_repetition_copies")
    failures = list(dict.fromkeys(failures))

    trimmed = original
    if not failures:
        retained_end = start + audio_count * unit_cues
        trimmed = original[:retained_end] + original[end:]
        for index, cue in enumerate(trimmed, 1):
            cue["index"] = index
    later_speech = list(evidence.get("later_speech_intervals") or [])
    later_valid_cues = original[end:]
    recovery_required = bool(later_speech and not later_valid_cues)
    return {
        "engine": "middle_repetition_deletion_plan_v1",
        "deletion_applied": not failures,
        "reason_codes": failures,
        "text_occurrences": text_count,
        "audio_occurrences": audio_count,
        "removed_occurrences": text_count - audio_count if not failures else 0,
        "trimmed_cues": trimmed,
        "original_wording_preserved": not failures,
        "replacement_text_generated": False,
        "later_speech_detected": bool(later_speech),
        "later_speech_recovery_required": recovery_required,
        "later_valid_original_cues_preserved": len(later_valid_cues),
        "repeated_region_requires_review": bool(failures),
        "safe_to_publish": bool(not failures and not recovery_required and later_valid_cues),
    }


def middle_emission_text_trim_report(
    transcript_text,
    repetitive_suffix,
    independent_evidence,
    *,
    require_automatic_inference=False,
):
    """Delete excess R from an emitted A+R suffix while explicitly preserving missing B."""
    evidence = dict(independent_evidence or {})
    failures = []
    if evidence.get("loop_position") != "middle":
        failures.append("not_a_middle_loop")
    later_speech = list(evidence.get("later_speech_intervals") or [])
    if not later_speech:
        failures.append("later_speech_not_independently_detected")
    post_trigger = _number(evidence.get("post_trigger_seconds"), -1)
    event_end = max((_number(item.get("end"), -1) for item in evidence.get("events") or []), default=-1)
    if post_trigger < event_end or post_trigger < 0:
        failures.append("unsafe_post_trigger_boundary")

    # Reuse the exact, audited character-prefix cutter while changing only the
    # terminal/no-later-speech preconditions that do not apply to A+R+B.
    deletion_evidence = dict(evidence)
    deletion_evidence.update({
        "loop_position": "suffix",
        "later_speech_intervals": [],
        "complete_media_inspected": True,
    })
    deletion = terminal_repetition_trim_report(
        transcript_text,
        repetitive_suffix,
        deletion_evidence,
        require_automatic_inference=require_automatic_inference,
    )
    failures.extend(deletion.get("reason_codes") or [])
    failures = list(dict.fromkeys(failures))
    applied = not failures
    return {
        **deletion,
        "engine": "middle_emission_text_trim_v1",
        "applied": applied,
        "reason_codes": failures,
        "corrected_text": deletion["corrected_text"] if applied else str(transcript_text or ""),
        "later_speech_intervals": later_speech,
        "post_trigger_seconds": post_trigger if post_trigger >= 0 else None,
        "later_speech_recovery_required": True,
        "repeated_region_requires_review": bool(failures),
        "safe_to_publish": False,
    }


def timestamp_proposal_report(
    video_id,
    segments,
    *,
    media_duration,
    independent_speech_intervals,
    max_overlap_seconds=0.20,
    max_words_per_second=8.0,
    min_acoustic_overlap_ratio=0.20,
):
    proposals = []
    issues = []
    seen_intervals = {}
    previous_start = -1.0
    previous_end = -1.0
    speech = merge_intervals(independent_speech_intervals)
    for index, raw in enumerate(segments or []):
        segment = dict(raw or {})
        start = _number(segment.get("gemini_proposed_start", segment.get("start_seconds")), float("nan"))
        end = _number(segment.get("gemini_proposed_end", segment.get("end_seconds")), float("nan"))
        text = str(segment.get("text") or "")
        cue_id = str(segment.get("id") or index + 1)
        codes = []
        if not math.isfinite(start) or not math.isfinite(end) or start < 0:
            codes.append("invalid_negative_or_non_numeric_timestamp")
        if math.isfinite(start) and math.isfinite(end) and end <= start:
            codes.append("zero_or_negative_duration_timestamp")
        if math.isfinite(end) and end > media_duration:
            codes.append("timestamp_beyond_media")
        if math.isfinite(start) and start < previous_start:
            codes.append("backward_timestamp_progression")
        if math.isfinite(start) and previous_end >= 0 and previous_end - start > max_overlap_seconds:
            codes.append("excessive_timestamp_overlap")
        interval_key = (round(start, 3), round(end, 3)) if math.isfinite(start) and math.isfinite(end) else None
        if interval_key in seen_intervals:
            codes.append("duplicate_or_reused_interval")
            other_text = seen_intervals[interval_key]
            if _normalized_text(other_text) == _normalized_text(text):
                codes.append("repeated_text_reused_interval")
        if interval_key:
            seen_intervals[interval_key] = text
        segment_interval = [(start, end)] if math.isfinite(start) and math.isfinite(end) and end > start else []
        acoustic_seconds = interval_duration(interval_intersection(segment_interval, speech))
        interval_seconds = max(0.0, end - start) if segment_interval else 0.0
        acoustic_ratio = acoustic_seconds / interval_seconds if interval_seconds else 0.0
        word_count = len(text.split())
        density = word_count / interval_seconds if interval_seconds else float("inf") if word_count else 0.0
        if density > max_words_per_second:
            codes.append("impossible_interval_density")
        if interval_seconds and acoustic_ratio < min_acoustic_overlap_ratio:
            codes.append("timestamp_covers_silence_or_unsupported_audio")
        independently_verified = not codes and acoustic_ratio >= min_acoustic_overlap_ratio
        proposal = {
            **segment,
            "id": cue_id,
            "gemini_proposed_start": start,
            "gemini_proposed_end": end,
            "independently_verified": independently_verified,
            "acoustic_overlap_ratio": round(acoustic_ratio, 6),
            "validation_codes": codes,
        }
        proposals.append(proposal)
        for code in codes:
            severity = "critical" if code in {
                "invalid_negative_or_non_numeric_timestamp", "zero_or_negative_duration_timestamp",
                "timestamp_beyond_media", "backward_timestamp_progression", "duplicate_or_reused_interval",
                "repeated_text_reused_interval", "impossible_interval_density",
            } else "warning"
            issues.append(make_issue(
                video_id, code, f"Gemini advisory timestamp proposal failed validation: {code.replace('_', ' ')}.",
                severity=severity, start_seconds=start if math.isfinite(start) else None,
                end_seconds=end if math.isfinite(end) else None, affected_cue_ids=[cue_id],
                evidence_source=["gemini_timestamp", "independent_speech"],
                evidence={"text": text, "acoustic_overlap_ratio": round(acoustic_ratio, 6)},
                blocking=False,
            ))
        if math.isfinite(start):
            previous_start = start
        if math.isfinite(end):
            previous_end = end
    proposed_end = max((item["gemini_proposed_end"] for item in proposals if math.isfinite(item["gemini_proposed_end"])), default=0.0)
    later_speech = [(start, end) for start, end in speech if end > proposed_end + 0.25]
    if later_speech:
        issues.append(make_issue(
            video_id, "gemini_timestamps_end_before_speech", "Gemini timestamp proposals end before independent speech evidence.",
            severity="critical", start_seconds=later_speech[0][0], end_seconds=later_speech[-1][1],
            evidence_source=["gemini_timestamp", "independent_speech"],
            evidence={"last_proposed_end": proposed_end}, blocking=False,
        ))
    return {
        "engine": "gemini_timestamp_advisory_validation_v1",
        "advisory_only": True,
        "segments": proposals,
        "issues": issues,
        "all_compatible": all(item["independently_verified"] for item in proposals),
        "last_proposed_end": proposed_end,
    }


def repetition_anomaly_report(
    video_id,
    segments,
    *,
    acoustic_events=None,
    acoustic_reports=None,
    min_repetitions=2,
    max_unit_cues=4,
    text_similarity=0.90,
):
    cues = normalize_cues(segments)
    normalized = [_normalized_text(cue.get("text")) for cue in cues]
    runs = []
    minimum_repetitions = max(2, int(min_repetitions or 2))
    cue_unit_limit = min(
        max(1, int(max_unit_cues or 1)),
        max(1, len(cues) // minimum_repetitions),
    )
    for unit_size in range(1, cue_unit_limit + 1):
        cursor = 0
        while cursor + unit_size * minimum_repetitions <= len(cues):
            unit = normalized[cursor:cursor + unit_size]
            count = 1
            next_index = cursor + unit_size
            while next_index + unit_size <= len(cues):
                candidate = normalized[next_index:next_index + unit_size]
                similarity = SequenceMatcher(None, " ".join(unit), " ".join(candidate)).ratio()
                if similarity < text_similarity:
                    break
                count += 1
                next_index += unit_size
            if count >= minimum_repetitions:
                runs.append({
                    "start_cue_index": cursor,
                    "end_cue_index": next_index,
                    "start_seconds": cues[cursor].get("start"),
                    "end_seconds": cues[next_index - 1].get("end"),
                    "unit_cues": unit_size,
                    "text_occurrences": count,
                    "unit_text": " ".join(cue.get("text", "") for cue in cues[cursor:cursor + unit_size]),
                    "position": "suffix" if next_index == len(cues) else "middle",
                    "detection_method": "fuzzy_cue_sequence",
                })
                cursor = next_index
            else:
                cursor += 1
    # Flatten cue tokens so exact repetitions remain detectable regardless of
    # subtitle cue boundaries and regardless of phrase length.
    flat_tokens = []
    token_cue_indices = []
    token_time_bounds = []
    for cue_index, cue_text in enumerate(normalized):
        cue_tokens = cue_text.split()
        flat_tokens.extend(cue_tokens)
        token_cue_indices.extend([cue_index] * len(cue_tokens))
        cue = cues[cue_index]
        cue_start = float(cue.get("start", 0.0))
        cue_end = max(cue_start, float(cue.get("end", cue_start)))
        cue_duration = cue_end - cue_start
        token_total = max(1, len(cue_tokens))
        token_time_bounds.extend([
            (
                cue_start + cue_duration * token_index / token_total,
                cue_start + cue_duration * (token_index + 1) / token_total,
            )
            for token_index in range(len(cue_tokens))
        ])
    for token_run in contiguous_token_repetition_runs(
        flat_tokens,
        min_repetitions=minimum_repetitions,
        min_repeated_tokens=2,
    ):
        start_token = token_run["start_token_index"]
        end_token = token_run["end_token_index"]
        if not token_cue_indices or end_token <= start_token:
            continue
        start_cue = token_cue_indices[start_token]
        end_cue = token_cue_indices[end_token - 1] + 1
        runs.append({
            **token_run,
            "start_cue_index": start_cue,
            "end_cue_index": end_cue,
            "unit_cues": 0,
            "position": (
                "suffix" if end_token == len(flat_tokens) else "middle"
            ),
            "within_broad_cue": end_cue - start_cue == 1,
            "start_seconds": token_time_bounds[start_token][0],
            "end_seconds": token_time_bounds[end_token - 1][1],
            "timing_source": "aligned_cue_token_interpolation",
        })
    exact_regions = [
        run
        for run in runs
        if run.get("detection_method") == "exact_tandem_tokens"
    ]
    filtered_runs = []
    for run in runs:
        if run.get("detection_method") == "exact_tandem_tokens":
            filtered_runs.append(run)
            continue
        if any(
            int(run["start_cue_index"]) >= int(exact["start_cue_index"])
            and int(run["end_cue_index"]) <= int(exact["end_cue_index"])
            for exact in exact_regions
        ):
            continue
        filtered_runs.append(run)
    runs = filtered_runs
    unique = {}
    for run in runs:
        key = (
            run["start_cue_index"], run["end_cue_index"],
            run.get("start_token_index"), run.get("end_token_index"),
        )
        unit_size = run.get("unit_tokens") or run.get("unit_cues") or 1
        existing_size = (
            (unique[key].get("unit_tokens") or unique[key].get("unit_cues") or 1)
            if key in unique else None
        )
        if key not in unique or unit_size < existing_size:
            unique[key] = run
    events = merge_intervals(acoustic_events or [])
    reports = [
        report
        for report in (acoustic_reports or [])
        if isinstance(report, dict) and report.get("count_inference_confident")
    ]

    def report_intervals(report):
        intervals = []
        for event in report.get("events") or []:
            try:
                event_start = float(
                    event.get("start_seconds", event.get("start"))
                )
                event_end = float(event.get("end_seconds", event.get("end")))
            except (AttributeError, TypeError, ValueError):
                continue
            if event_end > event_start:
                intervals.append((event_start, event_end))
        return merge_intervals(intervals)

    def report_for_run(run, start, end):
        exact_matches = []
        overlapping = []
        for report in reports:
            text_candidate = report.get("text_candidate") or {}
            same_token_region = (
                run.get("start_token_index") is not None
                and run.get("end_token_index") is not None
                and text_candidate.get("start_token_index") == run.get("start_token_index")
                and text_candidate.get("end_token_index") == run.get("end_token_index")
            )
            if same_token_region:
                exact_matches.append(report)
                continue
            candidate = report.get("candidate_region") or {}
            try:
                region_start = float(
                    candidate.get("region_start", report.get("region_start"))
                )
                region_end = float(
                    candidate.get("region_end", report.get("region_end"))
                )
            except (TypeError, ValueError):
                continue
            if start is None or end is None:
                continue
            overlap = max(0.0, min(float(end), region_end) - max(float(start), region_start))
            if overlap > 0:
                overlapping.append((overlap, -(region_end - region_start), report))
        if exact_matches:
            return max(exact_matches, key=lambda item: float(item.get("count_confidence") or 0.0))
        if overlapping:
            return max(overlapping, key=lambda item: (item[0], item[1]))[2]
        return None

    issues = []
    confirmed_runs = []
    for run in unique.values():
        affected = cues[run["start_cue_index"]:run["end_cue_index"]]
        start = run.get("start_seconds")
        end = run.get("end_seconds")
        if start is None:
            start = affected[0]["start"] if affected else None
        if end is None:
            end = affected[-1]["end"] if affected else None
        selected_report = report_for_run(run, start, end)
        run_events = report_intervals(selected_report) if selected_report else events
        region_events = [
            event
            for event in run_events
            if start is None or (event[1] > start and event[0] < end)
        ]
        acoustic_available = selected_report is not None or acoustic_events is not None
        acoustic_count = len(region_events) if acoustic_available else None
        text_count = int(run["text_occurrences"])
        exact_wording = run.get("detection_method") == "exact_tandem_tokens"
        if acoustic_count == text_count and exact_wording:
            confirmed_runs.append({
                **run,
                "start_seconds": start,
                "end_seconds": end,
                "acoustic_events": acoustic_count,
                "confirmation": "independent_event_count_match",
            })
            continue
        excess_text = acoustic_count is not None and text_count > acoustic_count
        missing_text = acoustic_count is not None and text_count < acoustic_count
        if excess_text:
            code = "unsupported_repetition_loop"
            summary = (
                "The canonical transcript contains more repeated occurrences "
                "than independently distinguishable acoustic events."
            )
        elif missing_text:
            code = "missing_repetition_occurrences"
            summary = (
                "The canonical transcript contains fewer repeated occurrences "
                "than independently distinguishable acoustic events."
            )
        elif acoustic_count == text_count and not exact_wording:
            code = "repeated_wording_variation_requires_review"
            summary = (
                "The number of repeated acoustic events matches the transcript, "
                "but near-matching repeated wording is not lexically identical. "
                "Acoustic event counts cannot validate or repair those words."
            )
        else:
            code = "repetition_requires_confirmation"
            summary = (
                "Repeated speech requires independent acoustic or explicit human "
                "confirmation; text-only pattern detection cannot determine the "
                "true occurrence count."
            )
        issue = make_issue(
            video_id,
            code,
            summary,
            severity="critical",
            start_seconds=start,
            end_seconds=end,
            affected_cue_ids=[cue["id"] for cue in affected],
            evidence_source=["text_repetition"] + (["acoustic_events"] if acoustic_available else []),
            evidence={
                **run,
                "acoustic_events": acoustic_count,
                "acoustic_report_algorithm": (
                    selected_report.get("algorithm_version")
                    if selected_report else None
                ),
            },
            blocking=True,
        )
        issues.append(issue)
    return {
        "engine": "position_independent_repetition_v2",
        "minimum_candidate_occurrences": minimum_repetitions,
        "text_only_decision_allowed": False,
        "runs": list(unique.values()),
        "confirmed_runs": confirmed_runs,
        "issues": issues,
    }


def estimate_recovery_boundaries(coverage_report, repetition_report=None, *, uncertainty_seconds=0.30):
    supported = coverage_report.get("accepted_supported_intervals") or []
    missing = coverage_report.get("meaningful_uncovered_intervals") or []
    repetition_issues = (repetition_report or {}).get("issues") or []
    last_supported = max((end for _, end in supported), default=None)
    first_uncovered = missing[0][0] if missing else None
    trigger_ends = [issue.get("end_seconds") for issue in repetition_issues if issue.get("end_seconds") is not None]
    post_trigger = max(trigger_ends) if trigger_ends else first_uncovered

    def boundary(name, value, sources, reason, safe):
        return {
            "name": name,
            "estimated_seconds": value,
            "uncertainty_interval": None if value is None else [
                max(0.0, value - uncertainty_seconds), value + uncertainty_seconds
            ],
            "confidence": 0.85 if value is not None and safe else 0.55 if value is not None else 0.0,
            "evidence_sources": sources,
            "reason": reason,
            "safe_for_automatic_recovery": bool(value is not None and safe),
        }

    safe_first = first_uncovered is not None and uncertainty_seconds <= 0.5
    safe_post = post_trigger is not None and uncertainty_seconds <= 0.35 and bool(trigger_ends)
    return {
        "t_last_supported": boundary(
            "t_last_supported", last_supported, ["accepted_alignment"],
            "Latest end of independently supported accepted canonical cues.", False,
        ),
        "t_first_uncovered": boundary(
            "t_first_uncovered", first_uncovered, ["independent_speech", "accepted_alignment"],
            "First meaningful speech interval not covered by accepted canonical cues.", safe_first,
        ),
        "t_post_trigger": boundary(
            "t_post_trigger", post_trigger, ["repetition_issue", "independent_speech"],
            "Estimated end of the triggering repetition region, not an exact lexical boundary.", safe_post,
        ),
    }


def plan_adaptive_recovery_windows(
    coverage_report,
    independent_timing_intervals,
    *,
    media_duration,
    maximum_windows=4,
    context_seconds=0.12,
    merge_gap_seconds=0.35,
):
    """Plan the fewest bounded requests around independently detected gaps.

    Window edges are timing/VAD boundaries, not fixed-duration subdivisions.
    The plan is non-recursive and its request count is explicitly bounded.
    """
    duration = max(0.0, _number(media_duration))
    maximum = max(1, int(maximum_windows))
    missing = merge_intervals(
        coverage_report.get("meaningful_uncovered_intervals") or []
    )
    timing = merge_intervals([
        (
            item.get("start", item.get("start_seconds")),
            item.get("end", item.get("end_seconds")),
        )
        if isinstance(item, dict) else item
        for item in independent_timing_intervals or []
    ])
    raw = []
    for gap_start, gap_end in missing:
        supporting = [
            (start, end)
            for start, end in timing
            if end > gap_start and start < gap_end
        ]
        if not supporting:
            continue
        start = max(0.0, min(item[0] for item in supporting) - context_seconds)
        end = min(duration, max(item[1] for item in supporting) + context_seconds)
        raw.append({
            "start_seconds": start,
            "end_seconds": end,
            "uncovered_intervals": [[gap_start, gap_end]],
            "boundary_sources": ["independent_timing_or_vad_start", "independent_timing_or_vad_end"],
        })
    windows = []
    for item in raw:
        if (
            windows
            and item["start_seconds"] - windows[-1]["end_seconds"] <= merge_gap_seconds
        ):
            windows[-1]["end_seconds"] = max(
                windows[-1]["end_seconds"], item["end_seconds"]
            )
            windows[-1]["uncovered_intervals"].extend(item["uncovered_intervals"])
        else:
            windows.append(item)
    # Never discard a region to honor the request cap. Merge the nearest
    # neighboring pair until the bounded plan fits.
    while len(windows) > maximum:
        pair = min(
            range(len(windows) - 1),
            key=lambda index: (
                windows[index + 1]["start_seconds"] - windows[index]["end_seconds"]
            ),
        )
        left, right = windows[pair], windows[pair + 1]
        windows[pair:pair + 2] = [{
            "start_seconds": left["start_seconds"],
            "end_seconds": right["end_seconds"],
            "uncovered_intervals": (
                left["uncovered_intervals"] + right["uncovered_intervals"]
            ),
            "boundary_sources": left["boundary_sources"],
        }]
    for index, window in enumerate(windows, 1):
        window.update({
            "index": index,
            "duration_seconds": round(
                window["end_seconds"] - window["start_seconds"], 6
            ),
            "reason": (
                "Bounded recovery around independently detected uncovered speech "
                "after longer suffix strategies did not establish complete coverage."
            ),
            "generation_scope": "bounded_window",
        })
    return {
        "schema": "subgen_adaptive_recovery_window_plan_v1",
        "strategy": "independent_timing_boundaries_fewest_windows",
        "recursive": False,
        "maximum_requests": maximum,
        "uncovered_region_count": len(missing),
        "window_count": len(windows),
        "windows": windows,
        "all_uncovered_regions_planned": bool(missing) and all(
            any(
                window["end_seconds"] > start
                and window["start_seconds"] < end
                for window in windows
            )
            for start, end in missing
        ),
    }


def offset_suffix_segments(
    segments,
    source_offset_seconds,
    *,
    generation_scope="recovered_suffix",
):
    offset = _number(source_offset_seconds)
    result = []
    for raw in segments or []:
        segment = dict(raw)
        relative_start = _number(segment.get("start_seconds", segment.get("gemini_proposed_start")))
        relative_end = _number(segment.get("end_seconds", segment.get("gemini_proposed_end")))
        segment["suffix_relative_start"] = relative_start
        segment["suffix_relative_end"] = relative_end
        segment["gemini_proposed_start"] = round(offset + relative_start, 6)
        segment["gemini_proposed_end"] = round(offset + relative_end, 6)
        segment["source_offset_seconds"] = offset
        segment["generation_scope"] = generation_scope
        result.append(segment)
    return result


def merge_validated_gemini_regions(existing, recovered, *, overlap_tolerance=0.10):
    accepted = [dict(item) for item in existing or []]
    occupied = merge_intervals([
        (item.get("start", item.get("gemini_proposed_start")), item.get("end", item.get("gemini_proposed_end")))
        for item in accepted
    ])
    added = []
    rejected = []
    for raw in recovered or []:
        item = dict(raw)
        start = _number(item.get("start", item.get("gemini_proposed_start")))
        end = _number(item.get("end", item.get("gemini_proposed_end")))
        reasons = []
        if item.get("canonical_source") not in {"gemini", "gemini_recovery"}:
            reasons.append("not_gemini_canonical_text")
        if not item.get("independently_verified"):
            reasons.append("not_independently_verified")
        reused = interval_duration(interval_intersection([(start, end)], occupied))
        if reused > overlap_tolerance:
            reasons.append("accepted_interval_reuse")
        if end <= start:
            reasons.append("invalid_interval")
        if reasons:
            rejected.append({"segment": item, "reasons": reasons})
            continue
        item["start"] = start
        item["end"] = end
        accepted.append(item)
        added.append(item)
        occupied = merge_intervals(occupied + [(start, end)])
    accepted.sort(key=lambda item: (_number(item.get("start", item.get("gemini_proposed_start"))), _number(item.get("end", item.get("gemini_proposed_end")))))
    return {"segments": accepted, "added": added, "rejected": rejected, "accept": bool(added) and not rejected}
