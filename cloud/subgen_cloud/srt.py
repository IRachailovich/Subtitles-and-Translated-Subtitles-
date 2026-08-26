import re


TIME_PATTERN = re.compile(
    r"^(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2}),(?P<sms>\d{3})\s+-->\s+"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2}),(?P<ems>\d{3})$"
)


def parse_srt(content):
    segments = []
    for block in content.replace("\r\n", "\n").strip().split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) < 3 or not TIME_PATTERN.match(lines[1].strip()):
            continue
        segments.append({
            "index": len(segments) + 1,
            "start": lines[1].split("-->", 1)[0].strip(),
            "end": lines[1].split("-->", 1)[1].strip(),
            "text": "\n".join(lines[2:]).strip(),
        })
    return segments


def render_srt(segments):
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = str(segment.get("start", "")).strip()
        end = str(segment.get("end", "")).strip()
        if not TIME_PATTERN.match(f"{start} --> {end}"):
            raise ValueError(f"Invalid subtitle timestamps at cue {index}")
        text = str(segment.get("translation", segment.get("text", ""))).strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"
