import requests
import time
from RAG_1.RAG.helpers import concat_vector_results

async def search_internet(query: str, top_k: int = 5, timeout: int = 6) -> dict:
    """
    Google Custom Search JSON API implementation.
    Requires:
    - GOOGLE_SEARCH_API_KEY in environment
    - GOOGLE_SEARCH_ENGINE_ID (cx) in environment
    """

    api_key = "AIzaSyBut2H80n8s0pS7TzXvG-n9HTa42GusT1o"
    cx = "f3c203ab9258a42b8"

    start = time.time()

    if not api_key or not cx:
        return {
            "query": query,
            "engine": "stub",
            "elapsed": round(time.time() - start, 3),
            "results": [
                {
                    "title": f"Stub result for '{query}'",
                    "snippet": "Missing GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_ENGINE_ID",
                    "url": ""
                }
            ]
        }

    url = "https://www.googleapis.com/customsearch/v1"

    try:
        response = requests.get(
            url,
            params={
                "key": api_key,
                "cx": cx,
                "q": query,
                "num": top_k,
            },
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])
        results = []

        for item in items[:top_k]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", "") or item.get("htmlSnippet", ""),
                "url": item.get("link", "")
            })

        return {
            "query": query,
            "engine": "google_custom_search",
            "elapsed": round(time.time() - start, 3),
            "results": results
        }

    except Exception as e:
        return {
            "query": query,
            "engine": "google_custom_search",
            "elapsed": round(time.time() - start, 3),
            "results": [],
            "error": str(e)
        }







# -----------------------------
# Module-level state (globals)
# -----------------------------
# These lists mirror the data stored in Chroma and are required for BM25 + hybrid mapping.
_CHUNK_TEXTS: list[str] = []     # list of chunk texts in the same order added to Chroma
_CHUNK_METAS: list[dict] = []    # list of metadata dicts corresponding to each chunk
_CHROMA_IDS: list[str] = []      # list of chunk ids used when adding to Chroma (e.g. "doc__chunk__0")

# BM25 index objects (populated by build_bm25_index)
_BM25 = None                     # BM25Okapi instance (or None if not available)
_BM25_TOKENIZED_CORPUS: list[list[str]] = []

# Chroma/embedding handles (set during pipeline init)
_chroma_client = None            # chroma client instance
_chroma_collection = None        # chroma collection handle used by _ann_topk
_embed_model = None              # embedding model instance (SentenceTransformer)








# ---- Hybrid + BM25 + optional reranker additions for rag_pipeline.py ----
from typing import Optional, Dict, Any, List
import time
import numpy as np  
from RAG_1.RAG.vector_store import embed_texts


try:
    from rank_bm25 import BM25Okapi
    _RANK_BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None
    _RANK_BM25_AVAILABLE = False


# optional cross-encoder reranker
try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except Exception:
    CrossEncoder = None
    _CROSS_ENCODER_AVAILABLE = False




# in rag_pipeline.py, implement an init helper (if not present)
def init_pipeline_handles(model, client, coll):

    global _embed_model, _chroma_client, _chroma_collection
    print("Initializing pipeline handles...")
    _embed_model = model
    _chroma_client = client
    _chroma_collection = coll

    print("Pipeline handles initialized.")





# ensure BM25 variables exist (these were defined earlier in rag_pipeline)
# _BM25_TOKENIZED_CORPUS: List[List[str]]
# _BM25: BM25Okapi or None
# _CHUNK_TEXTS, _CHUNK_METAS, _CHROMA_IDS exist too

def build_bm25_index():
    """
    Build or rebuild the in-memory BM25 index from the current _CHUNK_TEXTS.
    Call this after you ingest (or reload) a corpus.
    """
    global _BM25, _BM25_TOKENIZED_CORPUS
    if not _CHUNK_TEXTS:
        _BM25 = None
        _BM25_TOKENIZED_CORPUS = []
        return {"ok": True, "inserted": 0}
    tokenized = [t.split() for t in _CHUNK_TEXTS]
    _BM25_TOKENIZED_CORPUS = tokenized
    if _RANK_BM25_AVAILABLE:
        _BM25 = BM25Okapi(_BM25_TOKENIZED_CORPUS)
        return {"ok": True, "size": len(_BM25_TOKENIZED_CORPUS)}
    else:
        _BM25 = None
        return {"ok": False, "reason": "rank_bm25 not installed"}

def _bm25_topk(query: str, top_k: int = 10):
    """
    Return list of tuples (corpus_index, score) sorted desc by BM25 score.
    If BM25 not available, returns [].
    """
    if _BM25 is None:
        return []
    q_tok = query.split()
    scores = _BM25.get_scores(q_tok)
    idxs = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0.0]

def _ann_topk(query: str, top_k: int = 50):
    """
    Query chroma and return list of (corpus_index, ann_score) where corpus_index maps to _CHROMA_IDS.
    ann_score is similarity-like (higher => better). We map distances -> similarity.
    """
    print("ANN search for query:", query)
    print("Using embed model:", _embed_model)
    print("Using chroma collection:", _chroma_collection)
    if _chroma_collection is None:
        return []

    q_emb = embed_texts(_embed_model,query).tolist()
    try:
        print("Querying Chroma with embeddings...")
        res = _chroma_collection.query(query_embeddings=[q_emb], n_results=top_k, include=["ids","distances"])
    except Exception:
        # try minimal
        res = _chroma_collection.query(query_embeddings=[q_emb], n_results=top_k, include=["distances"])

    ids = res.get("ids", [[]])[0] if "ids" in res else []
    distances = res.get("distances", [[]])[0] if "distances" in res else []

    out = []
    for cid, dist in zip(ids, distances):
        try:
            idx = _CHROMA_IDS.index(cid)
        except ValueError:
            continue
        # Convert distance -> similarity; distance may be cosine-ish depending on backend; this is a common mapping:
        # similarity = 1/(1+dist) (bounded 0..1)
        sim = 1.0 / (1.0 + float(dist) if dist is not None else 1.0)
        out.append((idx, float(sim)))
    return out

def _normalize_scores(mapping: dict) -> dict:
    """
    Normalize values in mapping (idx->score) to 0..1 range. If mapping empty -> {}.
    """
    if not mapping:
        return {}
    vals = np.array(list(mapping.values()), dtype=float)
    minv, maxv = float(vals.min()), float(vals.max())
    denom = maxv - minv if maxv != minv else 1.0
    return {k: (v - minv) / denom for k, v in mapping.items()}

def hybrid_search(
    query: str,
    top_k: int = 6,
    ann_top: int = 50,
    bm25_top: int = 50,
    alpha: float = 0.6,
    rerank: bool = True,
    reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
):
    """
    Hybrid search that:
      - gets top candidates from ANN and BM25
      - normalizes and fuses scores with weight alpha for ANN
      - optionally reranks top candidates using cross-encoder
    Returns list of results with fields: idx, text, meta, chroma_id, ann_score, bm25_score, fused_score, rerank_score (optional)
    """

    print("Hybrid search: query=", query, "top_k=", top_k, "ann_top=", ann_top, "bm25_top=", bm25_top, "alpha=", alpha, "rerank=", rerank)
    # collect candidate maps
    ann = _ann_topk(query, top_k=ann_top)   # list of (idx, score)
    bm25 = _bm25_topk(query, top_k=bm25_top) # list of (idx, score)

    ann_map = {idx: sc for idx, sc in ann}
    bm25_map = {idx: sc for idx, sc in bm25}


    print("ANN map:", ann_map)
    print("BM25 map:", bm25_map)
    print("ann:", ann)
    print("bm25:", bm25)

    candidates = set(list(ann_map.keys()) + list(bm25_map.keys()))
    if not candidates:
        return []

    ann_norm = _normalize_scores(ann_map)
    bm25_norm = _normalize_scores(bm25_map)

    fused = []
    for idx in candidates:
        a = ann_norm.get(idx, 0.0)
        b = bm25_norm.get(idx, 0.0)
        fused_score = alpha * a + (1.0 - alpha) * b
        fused.append((idx, fused_score, a, b))

    # sort descending and keep some extra for reranker
    fused = sorted(fused, key=lambda x: x[1], reverse=True)
    candidate_slice = fused[: max(top_k * 4, top_k)]

    # build results list
    results = []
    for idx, fused_score, a, b in candidate_slice:
        results.append({
            "idx": idx,
            "text": _CHUNK_TEXTS[idx],
            "meta": _CHUNK_METAS[idx],
            "chroma_id": _CHROMA_IDS[idx],
            "ann_score": float(a),
            "bm25_score": float(b),
            "fused_score": float(fused_score),
        })

    # optional reranker
    if rerank and _CROSS_ENCODER_AVAILABLE:
        try:
            reranker = CrossEncoder(reranker_model)
            pairs = [(query, r["text"]) for r in results]
            rerank_scores = reranker.predict(pairs)
            for r, sc in zip(results, rerank_scores):
                r["rerank_score"] = float(sc)
            results = sorted(results, key=lambda r: r.get("rerank_score", r["fused_score"]), reverse=True)
        except Exception:
            # if reranker fails, ignore and return fused ordering
            pass

    # trim to requested top_k
    return results[:top_k]

# Update search_DB_tool to call hybrid
def search_DB_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    New hybrid search tool wrapper.
    payload keys:
      - query: str
      - k: int
      - alpha: float (0..1) weight for ANN in fusion
      - rerank: bool
    """
    q = str(payload.get("query", "")) if isinstance(payload, dict) else str(payload)
    k = int(payload.get("k", 6)) if isinstance(payload, dict) else 6
    alpha = float(payload.get("alpha", 0.6)) if isinstance(payload, dict) else 0.6
    rerank_flag = bool(payload.get("rerank", True)) if isinstance(payload, dict) else True

    start = time.time()
    hits = hybrid_search(query=q, top_k=k, ann_top=max(50, k*10), bm25_top=max(50, k*10), alpha=alpha, rerank=rerank_flag)
    elapsed = round(time.time() - start, 3)

    print("Hybrid search...", q )
    # normalize / prepare nice output
    out_results = []
    for h in hits:
        out_results.append({
            "id": h["chroma_id"],
            "text": h["text"],
            "meta": h["meta"],
            "ann_score": h.get("ann_score"),
            "bm25_score": h.get("bm25_score"),
            "fused_score": h.get("fused_score"),
            "rerank_score": h.get("rerank_score", None),
        })

    return {"query": q, "engine": "hybrid_bm25_ann", "elapsed": elapsed, "results": out_results}

def search_doc(query) -> str:
    from RAG_1.RAG.tools import search_DB_tool
    # perform hybrid search
    results = search_DB_tool({
    "query": query,
    "k": 5,
    "alpha": 0.9,
    "rerank": False
    })
    print(f"\n\n\n\n\n\1:------------------------------Hybrid search found resuls: \n {results} \n\n\n\n\n\------------------------------")

    results = concat_vector_results(results["results"])
    # send back formated results suitable for rag_pipeline usage
    print(f"\n\n\n\n\n\------------------------------Hybrid search found resuls: \n {results} \n\n\n\n\n\------------------------------")

    return results