Mode: Private-only (Hybrid toggle visible but OFF for Day-1).
No raw documents ever leave this workspace.
Embeddings computed locally; vectors stored under data/embeddings/.
Logs store metadata only (timestamps, doc_ids, token counts, latency) — no raw prompts or retrieved text.
Delete flow must remove objects + vectors for a selected doc_id.
Sensitivity tags: internal and confidential; Day-1 retrieval uses internal only.
Provenance required: Every answer shows doc title + page/section + snippet.

Open tests/smoke_test_plan.md and outline Day-1 tests:
12 ground-truth questions with expected doc + page/section.
1 adversarial request (“ignore previous instructions…”): expect block.
1 deletion test: select a file and confirm vector count decreases.
