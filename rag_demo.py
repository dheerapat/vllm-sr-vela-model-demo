"""Entry point: the end-to-end RAG demo.

Run: uv run python rag_demo.py            (bridges)
     uv run python rag_demo.py --med      (medical)
     uv run python rag_demo.py --med "Your question?" "A template with {slots}."
     uv run python rag_demo.py --med "Your question?" --evidence
"""

from vela_demo.rag import main

if __name__ == "__main__":
    main()
