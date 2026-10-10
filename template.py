"""Answer templates for Verbatim RAG: parse "{slot}" placeholders, turn each slot into a span question, and fill the
template with the top span per slot. Every filled value is an exact substring of a document; nothing is generated.

Template syntax: "The maximum daily dose of paracetamol for adults is {dosage}, taken as {single_dose} every {interval}."
A slot may carry its own description: "{dosage: the maximum amount per day}". The span question for a slot always
shows the template sentence with this slot as a blank (other slots shown by name), so the model sees what the slot
means there.
"""

import re

SLOT_RE = re.compile(r"\{\s*([A-Za-z][A-Za-z0-9_\-]{0,39})\s*(?::\s*([^{}]*?)\s*)?\}")
BLANK = "___"
MAX_SLOTS = 6


class TemplateError(ValueError):
    pass


def parse(template):
    """-> list of segments: {"text": str} | {"slot": name, "description": str | None}."""
    segments, pos, seen = [], 0, set()
    for m in SLOT_RE.finditer(template):
        literal = template[pos : m.start()]
        if "{" in literal or "}" in literal:
            raise TemplateError(
                "Write slots as {name} or {name: description}; names start with a letter."
            )
        if literal:
            segments.append({"text": literal})
        name = m.group(1)
        if name.casefold() in seen:
            raise TemplateError(f"Slot names must be unique ({name} appears twice).")
        seen.add(name.casefold())
        segments.append(
            {"slot": name, "description": (m.group(2) or "").strip() or None}
        )
        pos = m.end()
    tail = template[pos:]
    if "{" in tail or "}" in tail:
        raise TemplateError(
            "Write slots as {name} or {name: description}; names start with a letter."
        )
    if tail:
        segments.append({"text": tail})
    n = sum("slot" in s for s in segments)
    if not 1 <= n <= MAX_SLOTS:
        raise TemplateError(
            f"A template needs between one and {MAX_SLOTS} slots, written as {{name}}."
        )
    return segments


def slots(segments):
    return [s for s in segments if "slot" in s]


def words(name):
    """single_dose -> "single dose"."""
    return re.sub(r"[_\-]+", " ", name).strip()


def blanked(segments, slot_name):
    """The template with this slot as ___ and the other slots as [their name]."""
    out = []
    for s in segments:
        if "text" in s:
            out.append(s["text"])
        elif s["slot"] == slot_name:
            out.append(BLANK)
        else:
            out.append(f"[{words(s['slot'])}]")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def slot_qid(name):
    return "slot_" + name


def slot_question(segments, slot, over="context"):
    """One broad-head span question per slot, in the release's relation-question form ("Which spans are the X ...?"):
    the question carries the template sentence with this slot as the blank; label = slot name, description = the
    user's, else "the <slot name>". (This wording was chosen over four alternatives on the built-in examples; see
    README "Verbatim RAG".)"""
    w = words(slot["slot"])
    return {
        "type": "span",
        "instructions": f"Which spans are the {w} in: {blanked(segments, slot['slot'])}?",
        "over": over,
        "criteria": {slot["slot"]: slot["description"] or f"the {w}"},
    }


def locate(start, end, offsets):
    """Context offsets -> (doc index, doc-relative start, end); a span is assigned to the document it starts in."""
    for i, (lo, hi) in enumerate(offsets):
        if lo <= start < hi:
            return i, start - lo, min(end, hi) - lo
    return None


def fill(segments, spans_by_slot, docs, offsets, doc_names):
    """Fill each slot with its highest-probability span (spans are already above the question threshold).

    -> {"values": {slot: {"text", "doc", "start", "end", "probability"} | None},
        "rendered": the sentence with filled values (unfilled slots stay as {name}),
        "parts": [{"text"} | {"slot", "value" | None, "doc_index"}] for the UI, "filled": n, "total": n}
    The text of every value is re-read from the document by its offsets, so it is an exact span of the source."""
    values, parts, rendered = {}, [], []
    for s in segments:
        if "text" in s:
            parts.append({"text": s["text"]})
            rendered.append(s["text"])
            continue
        name = s["slot"]
        value = doc_index = None
        for span in sorted(
            spans_by_slot.get(name) or [], key=lambda x: -x["probability"]
        ):
            where = locate(span["start"], span["end"], offsets)
            if where is None:
                continue
            i, a, b = where
            text = docs[i][a:b]
            if not text.strip():
                continue
            value = {
                "text": text,
                "doc": doc_names[i],
                "start": a,
                "end": b,
                "probability": round(float(span["probability"]), 3),
            }
            doc_index = i
            break
        values[name] = value
        parts.append(
            {"slot": name, "value": value, "doc_index": doc_index if value else None}
        )
        rendered.append(value["text"] if value else "{" + name + "}")
    filled = sum(v is not None for v in values.values())
    return {
        "values": values,
        "rendered": "".join(rendered),
        "parts": parts,
        "filled": filled,
        "total": len(values),
    }
