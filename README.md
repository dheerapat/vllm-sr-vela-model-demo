# vela-demo

A local demo of [Vela 2.0 4B](https://huggingface.co/vllm-sr/Vela-2.0-4B) on a small machine.

The demo shows two things:

1. **SystemOne decisions** — one call answers choice, yes/no (Noul) and score questions.
2. **End-to-end RAG** — one model embeds, retrieves, reranks and reads. No second model.

## Layout

```text
vela-demo/
├── main_demo.py            entry: SystemOne decisions example
├── rag_demo.py             entry: end-to-end RAG demo
├── vela_demo/              the package
│   ├── loader.py           load the model on a small-RAM machine
│   ├── system_one.py       the decisions example
│   ├── rag.py              embed -> retrieve -> rerank -> read -> fill
│   └── template.py         parse and fill answer templates
├── scripts/
│   └── build_corpus.py     build data/medical.json from data/medical/*.md
├── data/
│   ├── bridges.json        built-in corpus: 60 Wikipedia paragraphs
│   ├── medical.json        built corpus: 348 medical chunks
│   └── medical/            29 disease markdown files (source)
├── models/                 downloaded weights (git-ignored)
├── pyproject.toml
└── README.md
```

## Hardware

This demo targets a machine with about **6.7 GB RAM** and a **16 GB GPU**.

The official loader casts all weights to `float32`. That needs about 17 GB of RAM and fails here.

Our loader (`vela_demo/loader.py`) streams each `safetensors` shard to the GPU. It keeps the stored data types:

| Part | Type | Size |
| --- | --- | --- |
| Backbone | bfloat16 | about 7.1 GB |
| Heads and norms | float32 | about 2.6 GB |
| Total on GPU | mixed | about 9.8 GB |

This is the bf16 parity setting of the release. See `PARITY.md` in the model.

## Setup

```bash
uv sync
uv run hf download vllm-sr/Vela-2.0-4B --local-dir models/Vela-2.0-4B
```

`uv sync` installs Python 3.13 and the dependencies. The download is about 9.7 GB.

## SystemOne decisions

```bash
uv run python main_demo.py
```

This loads the model and runs the recorded choice, Noul and score example.

## End-to-end RAG

```bash
uv run python rag_demo.py            # four bridge questions
uv run python rag_demo.py --med      # three medical questions
```

The pipeline has four steps:

1. **embed** — PromptEOL last-token state of the backbone. Cosine over the corpus gives the top `k` (default 30).
2. **rerank** — one Noul question per passage: `Does this passage answer the question?` Keep the top 3.
3. **read** — one `system_one` call over the top 3. It returns evidence spans and one span per template slot.
4. **fill** — each slot value is an exact substring of a document, with a citation.

Run your own question:

```bash
uv run python rag_demo.py "Your question?" "A template with {slots}."
uv run python rag_demo.py --med "What is the lifetime risk of peptic ulcer disease?" "The lifetime risk is {risk}."
uv run python rag_demo.py --med "how many subtypes of HPV are there?" --evidence
```

With `--evidence` the model returns the cited spans and no template. Use this for open questions. Use a template when the question has a known answer shape.

Options:

| Option | Meaning |
| --- | --- |
| `--med` | use `data/medical.json` and the medical examples |
| `--corpus PATH` | use another corpus JSON |
| `--k N` | retrieve `N` chunks before rerank (default 30) |
| `--evidence` | return cited spans only, no template |

Use a larger `k` when a question uses rare words. The rerank step is accurate, so more candidates help.

## Corpora

A corpus is a JSON file with `{"paragraphs": [{"title", "para", "text", ...}]}`.

| Corpus | Content | Source |
| --- | --- | --- |
| `data/bridges.json` | 60 paragraphs on ten bridges | English Wikipedia, CC BY-SA 4.0 |
| `data/medical.json` | 348 chunks from 29 disease notes | `data/medical/*.md` |

Build the medical corpus from the markdown:

```bash
uv run python scripts/build_corpus.py
```

A chunk is one `## Section` of one note. The chunk text starts with the disease title and the section name, so a chunk carries its own context. The script reads any folder of markdown files with `# Title` and `## Section` headings.

The first run embeds every chunk and writes a cache next to the corpus (`data/medical.emb.pt`). Later runs load the cache. The cache rebuilds when the corpus file changes.

## Answer templates

A template is a sentence with named slots. Example:

```text
The Brooklyn Bridge opened {opening_date}, with a main span of {main_span}.
```

The model fills each slot with a span. A slot without a span above the threshold stays empty. The model never writes new answer text.

For production, you can get a template in three ways:

1. Return cited spans only. No template.
2. Keep a template library and let the model pick one with a Choice question.
3. Let a generative model write the slot list. This needs a second model.

## Notes and limits

- The release does not publish the offline PromptEOL prompt. This demo uses the canonical one. The top `k` can differ a little. The rerank step repairs most of it.
- Some slots stay empty. This is safe. The answer is incomplete, but no wrong value appears.
- The `models/` directory and the `*.emb.pt` embedding caches are ignored by git.

## Attribution

`vela_demo/template.py` comes from the [Vela 2.0 Studio Space](https://huggingface.co/spaces/vllm-sr/vela2-studio) (Apache-2.0).

`data/bridges.json` contains text from English Wikipedia, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

`data/medical/` comes from [github.com/dheerapat/med-doc](https://github.com/dheerapat/med-doc).
