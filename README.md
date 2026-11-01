# SOP RAG Assistant

A small retrieval-augmented generation (RAG) pipeline that answers questions from a Standard Operating Procedure, and refuses to answer when the SOP doesn't cover the question.

I built it while learning RAG. The document is a hardware return and replacement SOP, modelled on the kind of procedure I followed as a System Engineer handling IT assets across delivery hubs. The goal was not a big system. It was to answer three practical questions with actual numbers instead of guesses:

1. Does splitting an SOP by section retrieve better than splitting it by character count?
2. Can the assistant tell when a question is outside the SOP and say so?
3. What does the final prompt sent to an LLM actually look like?

## What it does

1. **Chunks the SOP two ways.** A fixed-size baseline (300 characters, 50 overlap) and a section-based splitter that keeps each numbered clause whole and attaches the section name, document title and version as metadata.
2. **Embeds and indexes both** with `all-MiniLM-L6-v2` (normalized embeddings) in ChromaDB using cosine distance.
3. **Compares retrieval quality** on 8 test questions, each with a phrase the right chunk must contain.
4. **Calibrates a refusal threshold** by probing in-scope, unrelated and near-miss questions and looking at the distance scores.
5. **Builds the grounded prompt.** Chunks above the distance threshold are dropped. If nothing is left, the pipeline returns "I cannot find this in the SOP." without involving a model at all.

## Results

Hit rate on the 8-question test set (a hit means the retrieved chunks contain the expected phrase):

| Setup | k=1 | k=2 |
|---|---|---|
| Fixed-size chunks (300 chars) | 7/8 | 7/8 |
| Section-based chunks | 8/8 | 8/8 |

The fixed-size index missed "What should I do if replacement stock is missing at the central hub?", because the lateral stock transfer instruction got separated from the clause that sets it up. Keeping each clause in one chunk fixed that.

Refusal threshold probe (cosine distance of the best match, lower is closer):

| Question type | Best distance | Result at threshold 0.6 |
|---|---|---|
| In scope | 0.322, 0.516 | answer |
| Near miss (related topic, not in SOP) | 0.650, 0.678 | refuse |
| Unrelated | 0.781, 0.978 | refuse |

The threshold of 0.6 sits in the gap between the highest in-scope score and the lowest near-miss score.

## Limitations

I would rather state these than have you find them:

- **The SOP is tiny** (3 sections), and so is the test set (8 questions). With k=2 out of 3 chunks, retrieval is close to seeing the whole document. The 8/8 vs 7/8 gap shows the idea, not a proven margin.
- **The threshold is calibrated on 6 probe questions.** The gap between 0.516 and 0.650 is real but narrow. A larger document or more varied phrasing would need recalibration.
- **No LLM is called.** The notebook stops at the final prompt. Passing it to a model of your choice is the next step.
- **The hit-rate check is a phrase match**, not a judgment of answer quality.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook sop_rag_pipeline.ipynb
```

The first run downloads the embedding model (about 90 MB). The notebook writes `sop_sample.txt` itself, so it runs from top to bottom with nothing else needed. A copy is included in the repo for reference.

## Stack

Python, LangChain (text splitters, HuggingFace embeddings, Chroma integration), sentence-transformers (`all-MiniLM-L6-v2`), ChromaDB, Jupyter.

## Files

- `sop_rag_pipeline.ipynb` - the full pipeline with outputs
- `sop_sample.txt` - the sample SOP (synthetic document, not from any employer)
- `requirements.txt` - dependencies

## Next steps

- Connect an LLM to answer from the retrieved context
- Test on a longer, multi-document SOP set
- Build a larger question set, including paraphrased and adversarial questions
