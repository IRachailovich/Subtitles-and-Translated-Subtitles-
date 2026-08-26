import argparse
import io
import json
import os
import sys
import traceback
from datetime import datetime, timezone

from subgen_version import __version__


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError, io.UnsupportedOperation):
                pass


configure_utf8_stdio()


def write_smoke_progress(report_path, report):
    if not report_path:
        return
    target = os.path.abspath(report_path)
    temporary = target + ".new"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, target)


def whisperx_runtime_smoke_test(report_path=None):
    report = {
        "ok": None,
        "stage": "starting",
        "stages": [],
        "subgen_version": __version__,
    }

    def stage(name, **details):
        report["stage"] = name
        report["stages"].append({
            "name": name,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **details,
        })
        write_smoke_progress(report_path, report)

    stage("import_transformers_started")
    import transformers
    stage("import_transformers_completed", version=getattr(transformers, "__version__", "unknown"))
    stage("import_whisperx_started")
    import whisperx
    stage("import_whisperx_completed", version=getattr(whisperx, "__version__", "unknown"))
    stage("import_wav2vec2_started")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    stage("import_wav2vec2_completed")
    stage("import_subgen_review_started")
    from subgen_review import terminal_repetition_trim_report
    stage("import_subgen_review_completed", terminal_engine=terminal_repetition_trim_report.__name__)
    stage("import_subgen_transcription_started")
    from subgen_transcription import (
        CURRENT_PRODUCTION_PROMPT_VERSION,
        GEMINI_TRANSCRIPTION_THINKING_CONFIG,
        GEMINI_TRANSCRIPTION_THINKING_CONFIG_VERSION,
    )
    stage(
        "import_subgen_transcription_completed",
        production_prompt_version=CURRENT_PRODUCTION_PROMPT_VERSION,
        thinking_config=GEMINI_TRANSCRIPTION_THINKING_CONFIG,
        thinking_config_version=GEMINI_TRANSCRIPTION_THINKING_CONFIG_VERSION,
    )
    stage("import_subgen_acoustic_started")
    from subgen_acoustic import (
        ACOUSTIC_REPETITION_ALGORITHM_VERSION,
        infer_repetition_evidence,
    )
    stage(
        "import_subgen_acoustic_completed",
        algorithm_version=ACOUSTIC_REPETITION_ALGORITHM_VERSION,
        inference_entrypoint=infer_repetition_evidence.__name__,
    )

    report.update({
        "ok": True,
        "stage": "completed",
        "subgen_version": __version__,
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "whisperx_version": getattr(whisperx, "__version__", "unknown"),
        "wav2vec2_model": Wav2Vec2ForCTC.__name__,
        "wav2vec2_processor": Wav2Vec2Processor.__name__,
        "terminal_repetition_engine": terminal_repetition_trim_report.__name__,
        "production_prompt_version": CURRENT_PRODUCTION_PROMPT_VERSION,
        "thinking_config": GEMINI_TRANSCRIPTION_THINKING_CONFIG,
        "thinking_config_version": GEMINI_TRANSCRIPTION_THINKING_CONFIG_VERSION,
        "acoustic_repetition_algorithm_version": (
            ACOUSTIC_REPETITION_ALGORITHM_VERSION
        ),
        "acoustic_inference_entrypoint": infer_repetition_evidence.__name__,
    })
    write_smoke_progress(report_path, report)
    return report


def exception_chain(error):
    chain = []
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({
            "type": f"{type(current).__module__}.{type(current).__name__}",
            "message": str(current),
        })
        current = current.__cause__ or current.__context__
    return chain


def main():
    parser = argparse.ArgumentParser(description="SubGen desktop backend")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--lan", action="store_true")
    parser.add_argument("--self-test-whisperx", action="store_true")
    parser.add_argument("--self-test-report")
    args = parser.parse_args()

    if args.self_test_whisperx:
        try:
            report = whisperx_runtime_smoke_test(args.self_test_report)
            exit_code = 0
        except BaseException as error:
            prior = {}
            if args.self_test_report and os.path.isfile(args.self_test_report):
                try:
                    with open(args.self_test_report, "r", encoding="utf-8") as handle:
                        prior = json.load(handle)
                except (OSError, ValueError):
                    prior = {}
            report = {
                "ok": False,
                "stage": "failed",
                "failed_during": prior.get("stage"),
                "stages": prior.get("stages") or [],
                "error_chain": exception_chain(error),
                "traceback": traceback.format_exc(),
            }
            exit_code = 1
        output = json.dumps(report, ensure_ascii=False, indent=2)
        if args.self_test_report:
            write_smoke_progress(args.self_test_report, report)
        else:
            print(output)
        raise SystemExit(exit_code)

    from web.server import run_server
    run_server(port=args.port, host=args.host, open_browser=False, lan_access=args.lan)


if __name__ == "__main__":
    main()
