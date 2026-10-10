# vela-demo

A local demo of [Vela 2.0 4B](https://huggingface.co/vllm-sr/Vela-2.0-4B) on a small machine.

The demo shows two things:

1. **SystemOne decisions** — one call answers choice, yes/no (Noul) and score questions.
2. **End-to-end RAG** — one model embeds, retrieves, reranks and reads. No second model.

## Hardware

This demo targets a machine with about **6.7 GB RAM** and a **16 GB GPU**.

The official loader casts all weights to `float32`. That needs about 17 GB of RAM and fails here.

Our loader streams each `safetensors` shard to the GPU. It keeps the stored data types:

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

## Run the SystemOne example

```bash
uv run python main.py
```

This loads the model and runs the recorded choice, Noul and score example.

## Run the end-to-end RAG demo

```bash
uv run python rag_demo.py
```

This runs four built-in questions. The pipeline has four steps:

1. **embed** — PromptEOL last-token state of the backbone. Cosine over 60 paragraphs gives the top 10.
2. **rerank** — one Noul question per passage: `Does this passage answer the question?` Keep the top 3.
3. **read** — one `system_one` call over the top 3. It returns evidence spans and one span per template slot.
4. **fill** — each slot value is an exact substring of a document, with a citation.

Run your own question:

```bash
uv run python rag_demo.py "Your question?" "A template with {slots}."
```

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

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Model loader and the SystemOne example. |
| `rag_demo.py` | The end-to-end RAG pipeline. |
| `template.py` | Parse and fill answer templates. |
| `rag_corpus.json` | 60 Wikipedia paragraphs on ten bridges. |

## Notes and limits

- The release does not publish the offline PromptEOL prompt. This demo uses the canonical one. The top 10 can differ a little. The rerank step repairs most of it.
- Some slots stay empty. This is safe. The answer is incomplete, but no wrong value appears.
- The `models/` directory is ignored by git.

## Attribution

`template.py` comes from the [Vela 2.0 Studio Space](https://huggingface.co/spaces/vllm-sr/vela2-studio) (Apache-2.0).

`rag_corpus.json` contains text from English Wikipedia, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each paragraph lists its article, URL and revision. See `ask_manifest.json` upstream for details.
