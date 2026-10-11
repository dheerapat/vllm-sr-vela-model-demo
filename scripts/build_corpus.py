"""Build the medical RAG corpus from data/medical/*.md.

Each "## Section" becomes one chunk. The chunk text starts with the disease title
and the section name, so a chunk carries its own context.

Output: data/medical.json, with the same shape as data/bridges.json.

Run: uv run python scripts/build_corpus.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "medical"
OUT = ROOT / "data" / "medical.json"


def split_sections(md_path):
    """-> (title, [(section_name, body), ...]) for one markdown file."""
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    title = md_path.stem.replace("_", " ").title()
    sections, name, body = [], None, []
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            if name:
                sections.append((name, "\n".join(body).strip()))
            name, body = line[3:].strip(), []
        elif name:
            body.append(line)
    if name:
        sections.append((name, "\n".join(body).strip()))
    return title, sections


def build(src=SRC, out=OUT):
    paragraphs, n = [], 0
    for md in sorted(Path(src).glob("*.md")):
        title, sections = split_sections(md)
        for section, body in sections:
            if not body:
                continue
            n += 1
            paragraphs.append(
                {
                    "id": f"{md.stem}#{n}",
                    "title": title,
                    "para": section,
                    "name": f"[{len(paragraphs) + 1}] {title} \u2014 {section}",
                    "text": f"{title} \u2014 {section}\n{body}",
                }
            )
    try:
        src_label = str(Path(src).resolve().relative_to(ROOT))
    except ValueError:
        src_label = str(src)
    data = {
        "name": "Medical disease notes",
        "source": src_label,
        "paragraphs": paragraphs,
    }
    Path(out).write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"wrote {len(paragraphs)} chunks from {len(list(Path(src).glob('*.md')))} files to {out}"
    )
    return data


if __name__ == "__main__":
    build()
