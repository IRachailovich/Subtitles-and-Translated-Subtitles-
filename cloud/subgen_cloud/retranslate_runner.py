"""Isolated selected-cue translation runner for the cloud review API."""

import json
import sys

from subgen_pipeline import retranslate_review_selected_cues


def main():
    request = json.loads(sys.stdin.read())
    review = request["review"]
    result = retranslate_review_selected_cues(
        review,
        request["cue_ids"],
        request.get("pipeline_config") or {},
        device="cpu",
        actor=request.get("actor") or "cloud_review_worker",
    )
    public = {
        key: result[key]
        for key in (
            "selected_source_cue_ids",
            "translated_cue_count",
            "provider",
            "model",
        )
    }
    print(
        "SUBGEN_RETRANSLATION_RESULT="
        + json.dumps(
            {"review": review, "retranslation": public},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
