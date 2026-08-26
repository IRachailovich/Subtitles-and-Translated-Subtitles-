import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("result")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))

    # Import after SUBGEN_DATA_DIR is set by the worker so caches remain job-scoped.
    from subgen_pipeline import CONFIG, main as run_pipeline

    pipeline_config = CONFIG.copy()
    pipeline_config.update(request.get("pipeline_config") or {})

    output = run_pipeline(
        request["video_path"],
        srt_path_arg=request.get("srt_path"),
        target_language=request.get("target_language"),
        style_config=request.get("style_config") or {},
        pipeline_config=pipeline_config,
        keep_files=True,
        force=request.get("force", False),
        cache_action=request.get("cache_action"),
        no_burn=request.get("no_burn", False),
        output_dir=request["output_dir"],
        approved_review_path=request.get("approved_review_path"),
        source_location=request.get("source_location"),
    )
    Path(args.result).write_text(
        json.dumps({"output_path": str(output)}, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
