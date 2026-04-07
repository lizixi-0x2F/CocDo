"""
CausalSearch — Pearl three-step vs vector retrieval comparison demo
===================================================================

Corpus : docs/ (22 chapters, full paragraphs)
Query  : arbitrary text

Pipeline
--------
Offline (build index):
  1. Split all .md into paragraphs; node = paragraph (full corpus, no sampling)
  2. BGE embed each paragraph -> E  (N, 512)
  3. CausalFFNN (low-rank bilinear, O(N*rank)) learns A matrix
  4. Build NeuralSCM(A, E, U)

Online (query):
  CausalSearch (Pearl three steps)
    Step 1  Abduction  : query_emb -> nearest-neighbor paragraph j*
    Step 2  Action     : do(j* = query_emb), inject query, recompute E_new
    Step 3  Prediction : delta-norm per node -> sort -> top-k
      + downstream activation : knowledge chain triggered by the query
      - upstream prerequisites: concepts needed to understand the query

  VectorSearch (baseline):
    cos_sim(query_emb, E) -> top-k   (standard RAG)
"""

import re
import sys
import pathlib
import logging
import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cocdo import CausalFFNN, NeuralSCM
from cocdo.model.causal_ffnn import acyclicity_loss

DOCS_ROOT = ROOT / "docs"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 · paragraph splitting (node = paragraph, no entity extraction)
# ═══════════════════════════════════════════════════════════════════════════════

def _chapter_label(path: pathlib.Path) -> str:
    for part in path.parts:
        m = re.match(r'chapter(\d+)', part)
        if m:
            return f"ch{m.group(1)}"
    return path.stem


def load_paragraphs(
    docs_root: pathlib.Path,
    min_chars: int = 40,
    max_chars: int = 600,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split every .md into paragraphs; each paragraph = one SCM node.
    Full corpus, no per-chapter cap — low-rank CausalFFNN handles N=2000+.

    Returns
    -------
    node_ids  : short label  "ch6·observation data is never enough..."
    texts     : full paragraph text (fed to BGE)
    chapters  : chapter label per paragraph
    """
    node_ids, texts, chapters = [], [], []

    for md in sorted(docs_root.rglob("*.md")):
        raw = md.read_text(encoding="utf-8")
        if "vitepress" in raw[:120]:
            continue
        chap = _chapter_label(md)

        clean = re.sub(r'```.*?```', '', raw, flags=re.DOTALL)
        clean = re.sub(r'^\s*#{1,6}\s+', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
        clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
        clean = re.sub(r'^\s*[-*>|]\s*', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'\$\$.*?\$\$', '', clean, flags=re.DOTALL)
        clean = re.sub(r'\$.+?\$', '', clean)

        for para in re.split(r'\n{2,}', clean):
            para = para.strip().replace('\n', ' ')
            para = re.sub(r'\s+', ' ', para)
            if not (min_chars <= len(para) <= max_chars):
                continue
            if re.search(r'<[^>]+>|:::', para):
                continue
            label = f"{chap}·{para[:18]}…"
            node_ids.append(label)
            texts.append(para)
            chapters.append(chap)

    return node_ids, texts, chapters


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 · BGE embedding
# ═══════════════════════════════════════════════════════════════════════════════

def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    inst  = "Generate a representation for this sentence for retrieving related articles: "
    vecs  = model.encode(
        [inst + t for t in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return vecs.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 · CausalFFNN -> NeuralSCM
# ═══════════════════════════════════════════════════════════════════════════════

def build_scm(E: np.ndarray, entities: list[str]) -> tuple[NeuralSCM, np.ndarray]:
    """Train low-rank CausalFFNN on paragraph embeddings, return (scm, A_np).

    Loss: contrastive — A[i,j] should be large when paragraphs i,j are
    semantically related (high cosine sim), small otherwise.
    This keeps A non-trivial and U = E - A^T E meaningful.

    rank=64 keeps memory O(N*64) instead of O(N^2*hidden).
    """
    D = E.shape[1]
    E_t   = torch.from_numpy(E).float()            # (N, D)
    E_raw = E_t.unsqueeze(0)                       # (1, N, D)

    # Precompute cosine similarity target (E is already L2-normed)
    with torch.no_grad():
        S = E_t @ E_t.T                            # (N, N) cosine sim target
        S.fill_diagonal_(0.0)                      # no self-loops

    ffnn  = CausalFFNN(d_embed=D, hidden=256, rank=64)
    optim = torch.optim.Adam(ffnn.parameters(), lr=1e-3)

    N = E.shape[0]
    rho, lam, h_prev = 1.0, 0.0, float("inf")

    print("  Training CausalFFNN ...")
    for epoch in range(400):
        optim.zero_grad()
        A, _ = ffnn(E_raw)                         # (N, N)

        # Contrastive: A should correlate with cosine similarity
        align  = ((A - S) ** 2).mean()
        # Normalize h by N^2 so it stays O(1) regardless of graph size
        h      = acyclicity_loss(A) / (N * N)
        loss   = align + lam * h + (rho / 2) * h ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ffnn.parameters(), 1.0)
        optim.step()

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                h_val = float(acyclicity_loss(ffnn(E_raw)[0])) / (N * N)
            lam = lam + rho * h_val
            if h_val > 0.25 * h_prev:
                rho = min(rho * 10, 1e6)
            h_prev = h_val
            print(f"    epoch {epoch+1}  align={align.item():.4f}  "
                  f"h={h_val:.5f}  rho={rho:.0f}")

    with torch.no_grad():
        A_np = ffnn(E_raw)[0].numpy()              # (N, N)

    # Build SCM with matrices only — skip add_causal_edge (COC topo guard
    # rejects edges when learned A isn't a clean DAG, which is fine here
    # because causal_search only needs _A, _E, _U numerically).
    E_rms = np.sqrt((E[None] ** 2).mean(axis=0))   # (N, D)
    U     = E_rms - A_np.T @ E_rms                  # (N, D) exogenous residual
    scm   = NeuralSCM(var_names=entities, A=A_np, E=E_rms, U=U)
    return scm, A_np


# ═══════════════════════════════════════════════════════════════════════════════
# Search engines
# ═══════════════════════════════════════════════════════════════════════════════

def vector_search(query_emb: np.ndarray, E: np.ndarray,
                  entities: list[str], top_k: int = 8) -> list[tuple[str, float]]:
    """Standard cosine similarity retrieval (RAG baseline)."""
    sims = E @ query_emb                          # (N,) — E is L2-normed
    idx  = np.argsort(sims)[::-1][:top_k]
    return [(entities[i], float(sims[i])) for i in idx]


def causal_search(
    query_emb: np.ndarray,
    E_bge: np.ndarray,        # (N, D) original BGE embeddings, L2-normed
    scm: NeuralSCM,
    A_np: np.ndarray,
    entities: list[str],
    top_k: int = 8,
) -> dict:
    """
    Pearl three-step causal retrieval.

    Step 1  Abduction
        j* = argmax cos_sim(query_emb, E_bge[j])   <- use original BGE space
        scm._E is RMS-compressed and loses direction; BGE stays clean.

    Step 2  Action
        do(j* = query_emb): inject query vector directly into E_bge,
        zero j*'s incoming edges in A.

    Step 3  Prediction
        E_next = A_do^T @ E_do + U
        delta_norm[j] = ||E_next[j]|| - ||E_base[j]||
        + = downstream activated,  - = upstream suppressed
    """
    assert scm._A is not None

    # U is calibrated for scm._E (RMS-compressed); recompute for E_bge space.
    # Structural equation: E_j = A^T @ E + U  ->  U = E_bge - A^T @ E_bge
    U = E_bge - A_np.T @ E_bge    # (N, D)  exogenous residual in BGE space

    # ── Step 1: Abduction — cosine in original BGE space ───────────────────
    sims   = E_bge @ query_emb              # (N,) — both L2-normed
    j_star = int(np.argmax(sims))
    anchor_name = entities[j_star]
    anchor_sim  = float(sims[j_star])

    # ── Step 2: Action ──────────────────────────────────────────────────────
    A_do = A_np.copy()
    A_do[:, j_star] = 0.0                   # sever incoming edges to anchor

    E_do = E_bge.copy()
    E_do[j_star] = query_emb               # inject query (already unit-norm)

    # ── Step 3: Prediction ──────────────────────────────────────────────────
    E_base_next = A_np.T @ E_bge + U       # (N, D)
    E_int_next  = A_do.T @ E_do  + U       # (N, D)

    base_norms = np.linalg.norm(E_base_next, axis=1)
    int_norms  = np.linalg.norm(E_int_next,  axis=1)
    deltas     = int_norms - base_norms     # + = activated, - = suppressed
    deltas[j_star] = 0.0

    sorted_idx = np.argsort(deltas)[::-1]
    downstream = [(entities[i], float(deltas[i])) for i in sorted_idx[:top_k]
                  if deltas[i]]
    upstream   = [(entities[i], float(-deltas[i])) for i in sorted_idx[::-1][:top_k]
                  if deltas[i]]
    all_deltas = [(entities[i], float(deltas[i])) for i in sorted_idx]

    return {
        "anchor":     (anchor_name, anchor_sim),
        "downstream": downstream,
        "upstream":   upstream,
        "all_deltas": all_deltas,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(query: str, vec_results, causal_results, chapters: list[str],
                     entities: list[str]):
    chap_of = dict(zip(entities, chapters))

    print(f"\n{'═'*65}")
    print(f"  Query: {query!r}")
    print(f"{'═'*65}")

    print("\n[Vector Search · RAG baseline]")
    for rank, (ent, score) in enumerate(vec_results, 1):
        print(f"  {rank:2d}. {ent!r:<35}  cos={score:.3f}  [{chap_of.get(ent,'')}]")

    anchor_name, anchor_sim = causal_results["anchor"]
    print(f"\n[CausalSearch · Pearl three-step]")
    print(f"  Abduction anchor -> {anchor_name!r}  (cos={anchor_sim:.3f})"
          f"  [{chap_of.get(anchor_name,'')}]")

    print(f"\n  + Downstream (knowledge chain triggered by query):")
    if causal_results["downstream"]:
        for ent, delta in causal_results["downstream"]:
            print(f"    + {ent!r:<45}  delta={delta:+.4e}  [{chap_of.get(ent,'')}]")
    else:
        print("    (none)")

    print(f"\n  - Upstream (prerequisites to understand the query):")
    if causal_results["upstream"]:
        for ent, delta in causal_results["upstream"][:6]:
            print(f"    - {ent!r:<45}  delta={-delta:+.4e}  [{chap_of.get(ent,'')}]")
    else:
        print("    (none above threshold)")

    # Diff: what CausalSearch finds that RAG misses
    vec_set   = {e for e, _ in vec_results}
    cs_set    = {e for e, _ in causal_results["downstream"] + causal_results["upstream"]}
    cs_unique = cs_set - vec_set
    if cs_unique:
        print(f"\n  * CausalSearch-only (missed by RAG):")
        for ent in cs_unique:
            print(f"    {ent!r}  [{chap_of.get(ent,'')}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Build index ────────────────────────────────────────────────────────
    print("=" * 65)
    print("Building CausalSearch index ...")
    print("=" * 65)

    print("\n[1/3] Loading paragraphs ...")
    entities, contexts, chapters = load_paragraphs(DOCS_ROOT)
    print(f"  {len(entities)} paragraphs")

    print("\n[2/3] Embedding with BGE ...")
    E = embed(contexts)
    print(f"  Embedding shape: {E.shape}")

    print("\n[3/3] Building SCM ...")
    scm, A_np = build_scm(E, entities)
    print(f"  SCM ready: {len(entities)} nodes")

    # ── Queries ────────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    bge  = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    inst = ""

    QUERIES = [
        # conceptual
        "How to identify causal relationships rather than correlations from observational data",
        # cross-chapter
        "What is the relationship between Transformer attention and Bayesian inference",
        # applied
        "How does an AI system know whether its own reasoning is reliable",
    ]

    for query in QUERIES:
        q_emb = bge.encode(
            [inst + query], normalize_embeddings=True
        )[0].astype(np.float32)

        vec_res    = vector_search(q_emb, E, entities, top_k=8)
        causal_res = causal_search(q_emb, E, scm, A_np, entities, top_k=8)

        print_comparison(query, vec_res, causal_res, chapters, entities)

    print(f"\n{'═'*65}")
    print("Done.")
