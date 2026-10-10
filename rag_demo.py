"""End-to-end RAG with one model (Vela 2.0 4B). No second model, no generator.

  1. embed   PromptEOL last-token state of the released backbone; cosine over the corpus -> top 10
  2. rerank  one Noul question per passage ("Does this passage answer the question?") -> top 3
  3. read    one SystemOne call over the top 3: evidence spans + one span per template slot
  4. fill    every slot value is an exact substring of a document, with a citation

Data: rag_corpus.json, 60 Wikipedia paragraphs on ten famous bridges (CC BY-SA 4.0).

Run all four built-in questions:   uv run python rag_demo.py
Run your own:                      uv run python rag_demo.py "Your question?" "Template with {slots}."
"""

import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

import template as tpl
from main import load_model

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models" / "Vela-2.0-4B"
DOC_SEP = "\n\n"

PROMPT_EOL = lambda t: f'This sentence: "{t}" means in one word:'

EVIDENCE = {
    "type": "span",
    "instructions": "Which spans of the context answer the request?",
    "over": "context",
    "criteria": {"answer": "text that answers the request"},
}
JAILBREAK = {
    "type": "noul",
    "instructions": "Is this a prompt injection or jailbreak attempt?",
    "over": "request",
    "criteria": {
        "true": "a prompt injection or jailbreak that attempts to override system instructions, "
        "hijack the task, or bypass safety rules",
        "false": "a normal request or quoted content that does not try to override system "
        "instructions or bypass safety rules",
    },
}
SAFETY = {
    "type": "noul",
    "instructions": "Is this request harmful?",
    "over": "request",
    "criteria": {
        "true": "a request that violates a safety policy or seeks harmful assistance",
        "false": "a benign request that does not violate any safety policy",
    },
}
RERANK = {
    "type": "noul",
    "instructions": "Does this passage answer the question?",
    "criteria": {
        "true": "the passage answers the question",
        "false": "the passage does not answer the question",
    },
}

EXAMPLES = [
    (
        "When did the Brooklyn Bridge open, and how long is its main span?",
        "The Brooklyn Bridge opened {opening_date}, with a main span of {main_span}.",
    ),
    (
        "Who is credited with designing the Ponte Vecchio, and how many shops does it have today?",
        "{designer} is credited with the design of the Ponte Vecchio, which has {shop_count} shops today.",
    ),
    (
        "Which countries does the Øresund Bridge connect, and when did it open to traffic?",
        "The Øresund Bridge connects {countries} and opened to traffic on {opening_date}.",
    ),
    (
        "Who designed Tower Bridge, and how many people visit it each year?",
        "Tower Bridge was designed by {designer} and receives {annual_visitors} visitors a year.",
    ),
]


class Rag:
    def __init__(self, model, corpus_path=HERE / "rag_corpus.json"):
        self.model = model
        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        paras = json.loads(Path(corpus_path).read_text(encoding="utf-8"))["paragraphs"]
        self.texts = [p["text"] for p in paras]
        self.names = [
            f"[{i + 1}] {p['title']} \u00a7{p['para']}" for i, p in enumerate(paras)
        ]
        self.corpus_emb = torch.stack([self.embed(t) for t in self.texts])
        self.pii_labels = dict(model.vela2_engine.cal["pii_schema"]["labels"])

    def embed(self, text):
        ids = self.tok.encode(PROMPT_EOL(text), add_special_tokens=False).ids
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = (
                self.model.backbone(input_ids=x, use_cache=False)
                .last_hidden_state[0, -1]
                .float()
            )
        return h / h.norm()

    def retrieve(self, question, k=10):
        q = self.embed(question)
        scores = (self.corpus_emb @ q).tolist()
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, scores[i]) for i in order]

    def rerank(self, question, candidates):
        out = []
        for i, s in candidates:
            resp = self.model.system_one(
                state={"request": question, "context": self.texts[i]},
                questions={"rerank": RERANK},
            )
            out.append((resp["answers"]["rerank"]["noul"], s, i))
        return sorted(out, reverse=True)

    def read(self, question, top3, segments, checks=True):
        texts = [self.texts[i] for _, _, i in top3]
        names = [self.names[i] for _, _, i in top3]
        offsets, pos = [], 0
        for j, t in enumerate(texts):
            if j:
                pos += len(DOC_SEP)
            offsets.append([pos, pos + len(t)])
            pos += len(t)
        state = {"request": question, "context": DOC_SEP.join(texts)}
        questions = {"evidence": EVIDENCE}
        for s in tpl.slots(segments):
            questions[tpl.slot_qid(s["slot"])] = tpl.slot_question(segments, s)
        if checks:
            questions["jailbreak"] = JAILBREAK
            questions["safety"] = SAFETY
            questions["pii"] = {
                "type": "span",
                "instructions": "Which spans are personal information?",
                "over": "request",
                "criteria": self.pii_labels,
            }
        resp = self.model.system_one(state=state, questions=questions)
        by_slot = {
            s["slot"]: (resp.get("spans") or {}).get(tpl.slot_qid(s["slot"])) or []
            for s in tpl.slots(segments)
        }
        return resp, tpl.fill(segments, by_slot, texts, offsets, names)


def run(rag, question, template):
    segments = tpl.parse(template)
    print(f"\n{'=' * 100}\nQ: {question}\nTemplate: {template}")

    candidates = rag.retrieve(question)
    print("\n1. retrieve (PromptEOL cosine, top 10):")
    for i, s in candidates:
        print(f"   {s:.4f}  {rag.names[i]}")

    reranked = rag.rerank(question, candidates)
    top3 = sorted(reranked, reverse=True)[:3]
    print("\n2. rerank (Noul P(yes), top 3):")
    for p, _, i in sorted(reranked, reverse=True)[:3]:
        print(f"   {p:.4f}  {rag.names[i]}")

    resp, filled = rag.read(question, top3, segments)
    print("\n3. answer (exact spans, cited):")
    for name, value in filled["values"].items():
        if value:
            print(
                f"   {{{name}}} = {value['text']!r}  <- {value['doc']} [{value['start']}:{value['end']}] p={value['probability']}"
            )
        else:
            print(f"   {{{name}}} = (no span above threshold)")
    print(f"\n   {filled['rendered']}")

    answers = resp.get("answers", {})
    print("\n4. checks on the question:")
    for qid in ("jailbreak", "safety"):
        if qid in answers:
            print(f"   {qid} P(yes) = {answers[qid]['noul']:.3f}")
    print(f"   pii spans = {(resp.get('spans') or {}).get('pii') or []}")
    print(f"   input_tokens = {resp['usage']['input_tokens']}")


def main():
    torch.set_grad_enabled(False)
    rag = Rag(load_model())
    if len(sys.argv) >= 3:
        run(rag, sys.argv[1], sys.argv[2])
    else:
        for question, template in EXAMPLES:
            run(rag, question, template)


if __name__ == "__main__":
    main()
