# sentence_chunker.py
from typing import List, Dict, Any
import re
import html
import time
import uuid

# Optional: use tiktoken for accurate token counts
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None
    _TIKTOKEN_AVAILABLE = False

# Optional: nltk sentence tokenizer (fallback to regex if not installed)
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
    _NLTK_AVAILABLE = False
except Exception:
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        _NLTK_AVAILABLE = True
    except Exception:
        _NLTK_AVAILABLE = False

def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sentence_split(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    if _NLTK_AVAILABLE:
        return nltk.tokenize.sent_tokenize(text)
    return [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if s.strip()]

def count_tokens(text: str, model_name: str | None = None) -> int:
    """
    Return token count. If tiktoken available, use encoding_for_model(model_name) if provided.
    Else fallback to whitespace word count.
    """
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE:
        try:
            if model_name:
                enc = tiktoken.encoding_for_model(model_name)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except Exception:
                pass
    return len(text.split())

def _split_long_sentence_into_token_windows(sentence: str, max_tokens: int, overlap_tokens: int, model_name: str | None = None) -> List[str]:
    """
    If a sentence is longer than max_tokens, split it into windows measured by tokens (approx via words if tiktoken missing).
    Overlap measured in tokens (approx).
    """
    words = sentence.split()
    if not words:
        return []
    # approximate step size in words
    step_words = max(1, max_tokens - overlap_tokens)
    i = 0
    windows = []
    while i < len(words):
        window_words = words[i:i + max_tokens]
        window_text = " ".join(window_words).strip()
        # refine downwards if token count still > max_tokens (rare)
        while count_tokens(window_text, model_name) > max_tokens and len(window_words) > 1:
            window_words = window_words[:-1]
            window_text = " ".join(window_words).strip()
        if not window_text:
            break
        windows.append(window_text)
        i += step_words
    return windows

def chunk_by_sentences_with_overlap(
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
    max_tokens: int = 100,
    model_name_for_tokenization: str | None = None
) -> List[str]:
    """
    Primary function:
      - build chunks with up to `sentences_per_chunk` each,
      - ensure token count <= max_tokens (split long sentences if needed),
      - use `overlap_sentences` to overlap between adjacent chunks.
    Returns list of chunk strings.
    """
    text = _clean_text(text)
    if not text:
        return []

    sentences = sentence_split(text)
    if not sentences:
        return []

    chunks: List[str] = []
    i = 0
    n = len(sentences)

    # defensive clamp of overlap
    overlap_sentences = min(max(0, overlap_sentences), sentences_per_chunk - 1)

    while i < n:
        # candidate block of sentences (naive window of size sentences_per_chunk)
        j = min(n, i + sentences_per_chunk)
        block = sentences[i:j]

        # If any sentence in block exceeds max_tokens, split that sentence first and
        # treat the pieces as separate "sentences" for chunking.
        expanded_block: List[str] = []
        for sent in block:
            if count_tokens(sent, model_name_for_tokenization) > max_tokens:
                pieces = _split_long_sentence_into_token_windows(sent, max_tokens, overlap_tokens=0, model_name=model_name_for_tokenization)
                expanded_block.extend(pieces)
            else:
                expanded_block.append(sent)

        # Now try to fit as many sentences from expanded_block into the chunk respecting token limit.
        current_chunk_sents: List[str] = []
        current_token_count = 0
        for s in expanded_block:
            s_tokens = count_tokens(s, model_name_for_tokenization)
            if current_chunk_sents and (current_token_count + s_tokens > max_tokens):
                # cannot add this sentence -> stop adding more sentences
                break
            # if empty chunk and single sentence already > max_tokens (rare), we allow it (it was split earlier)
            current_chunk_sents.append(s)
            current_token_count += s_tokens

        # Edge case: if nothing was added (maybe because the first expanded piece still > max_tokens),
        # then force-add the first piece (to avoid infinite loops).
        if not current_chunk_sents and expanded_block:
            current_chunk_sents.append(expanded_block[0])
            current_token_count = count_tokens(expanded_block[0], model_name_for_tokenization)

        # build chunk text
        chunk_text = " ".join(current_chunk_sents).strip()
        if chunk_text:
            chunks.append(chunk_text)

        # advance i by sentences_per_chunk minus overlap (in original sentence units)
        # We compute how many original sentences we advanced: ideally sentences_per_chunk - overlap_sentences
        # But if we split long sentences, we must advance at least 1 original sentence.
        advance = sentences_per_chunk - overlap_sentences
        if advance <= 0:
            advance = 1
        i += advance

    return chunks

def chunk_document_to_dicts(
    doc_id: str,
    text: str,
    meta: Dict[str, Any] | None = None,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
    max_tokens: int = 100,
    model_name_for_tokenization: str | None = None
) -> List[Dict[str, Any]]:
    """
    Produce chunk dictionaries with ids and metadata ready for ingestion.
    """
    meta = meta or {}
    chunk_texts = chunk_by_sentences_with_overlap(
        text,
        sentences_per_chunk=sentences_per_chunk,
        overlap_sentences=overlap_sentences,
        max_tokens=max_tokens,
        model_name_for_tokenization=model_name_for_tokenization
    )
    out = []
    for idx, ch in enumerate(chunk_texts):
        cid = f"{doc_id}__chunk__{idx}"
        cm = dict(meta)
        cm.update({"doc_id": doc_id, "chunk_index": idx, "created_at": time.time()})
        out.append({"id": cid, "text": ch, "meta": cm})
    return out

# ---------------------------
# Quick demo when run as script
if __name__ == "__main__":
    SAMPLE = (
        "Rust 1.70 release notes: improvements to borrow checker and performance. "
        "This release fixes a number of bugs and introduces small API stabilizations. "
        "The team recommends following the migration notes. "
        "In addition, the standard library saw performance improvements on Windows. "
        "See the full notes on the Rust blog. "
        "There are also updates to the compiler internals which affect macro expansion. "
        "Developers should test their code against the new toolchain. "
        "Security fixes were included in this release to address several CVEs."
    )

    print("NLTK available:", _NLTK_AVAILABLE, "Tiktoken available:", _TIKTOKEN_AVAILABLE)
    chunks = chunk_document_to_dicts("doc1", SAMPLE, sentences_per_chunk=5, overlap_sentences=1, max_tokens=100)
    for c in chunks:
        print("=== CHUNK ID:", c["id"])
        print("meta:", c["meta"])
        print("text:", c["text"])
        print("token_count:", count_tokens(c["text"]))
        print()
