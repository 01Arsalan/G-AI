import json
from typing import Any, Dict, List

def extract_first_json(text: str):
    """
    Try to find the first JSON object in `text`. If none found, try to json.loads the whole text.
    Returns parsed dict or None.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()

    # quick attempt: whole text as JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # fallback: scan for {...}
    s = text
    start = None
    depth = 0

    for i, ch in enumerate(s):
        if ch == "{":
            if start is None:
                start = i
            depth += 1

        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = s[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    start = None
                    depth = 0

    return None


def index_embed_texts(doc, sentences_per_chunk, overlap_sentences, max_tokens):
    from RAG_1.RAG.sentence_chunker import chunk_document_to_dicts
    from RAG_1.RAG.vector_store import init_embed_model, embed_texts, init_chroma_client


    with open(doc, "r", encoding="utf-8") as f:
        text = f.read()

    # chunk the document
    chunks = chunk_document_to_dicts(
        doc_id="data_doc",
        text=text,
        sentences_per_chunk=sentences_per_chunk,
        overlap_sentences=overlap_sentences,
        max_tokens=max_tokens,
    )

    # 3. Extract fields
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c["meta"] for c in chunks]


    # 4. Init vector DB & embedding model
    model = init_embed_model()
    client, coll = init_chroma_client()

    # 5. Embed & store in Chroma
    embs = embed_texts(model, texts)
    coll.add(ids=ids, documents=texts, embeddings=embs.tolist(), metadatas=metas)

    print(f"Ingested {len(ids)} chunks into Chroma.")

    return model, client, coll, texts, ids, metas


def setup_vector_store_search(model, client, coll):

    from RAG_1.RAG.tools import init_pipeline_handles, build_bm25_index
    # initialize _embed_model/_chroma_collection in rag_pipeline.
    init_pipeline_handles(model, client, coll)

    # Build BM25 index (tokenize + instantiate BM25Okapi)
    res = build_bm25_index()

    print("vector store search setup complete.")



def concat_vector_results(results): 
    import re
    parts = ["Relevant Dream Notes :\n"]
    for r in results:
        meta = r.get("meta", {})
        doc_id = meta.get("doc_id")
        chunk_index = meta.get("chunk_index")
        if doc_id is None or chunk_index is None:
            m = re.search(r'(.+?)__chunk__([0-9]+)', r.get("id", ""))
            if m:
                doc_id, chunk_index = m.group(1), int(m.group(2))
        if doc_id is None or chunk_index is None:
            identifier = r.get("id", "unknown")
        else:
            identifier = f"{doc_id}_chunk_[{chunk_index}]"
        parts.append(identifier)
        parts.append(r.get("text", ""))
    return "\n".join(parts)

