#!/usr/bin/env python3
"""Serve a DiffRetriever checkpoint as a text -> vectors HTTP endpoint.

DiffRetriever (ielabgroup/diffretriever-*, arXiv:2605.07210) pins transformers 4.54 and peft,
which this library's environment does not carry, so the model runs in its own environment as a
small server, the way vLLM does for the backbone. This file imports nothing from agent_search.
The library side is `agent_search/retrievers/learned/diffretriever.py`.

It does what the model card's usage code does: `model.tokenize(texts, is_query)`, then
`model.encode(ids, mask, is_query, compute_sparse=...)`, and returns `repr_hidden`
(`[n, K, hidden]`, K = 4 for a query and 16 for a passage on the multi checkpoint) as fp16.

With `"sparse": true` the same forward pass also gives each text's sparse terms, built the way
the authors' encoder builds them (diffusion-retrieval scripts/encode_promptreps.py): the
per-position weights `log(1 + relu(logit))` are kept only on the text's content words (the
released `sparse_utils.get_content_token_ids`: NLTK words minus stopwords and punctuation), the
top `SPARSE_TOPK` per position are kept, scaled by 100 and rounded, and max-pooled over the
positions.

The card serves inputs at `max_length` 156 tokens in total, prompt included. A request may name
`max_text_tokens` to let a passage carry that many tokens of text after the prompt; a query
uses the card's length. A page is cut at a whitespace boundary before tokenising once the
prefix holds the budget, so the tokenizer never reads a megabyte tail.

    python scripts/serve_diffretriever.py --model ielabgroup/diffretriever-dream-7b-multi-q4-p16 \\
        --port-file /tmp/port --device cuda:0

POST /encode {"texts": [...], "is_query": bool, "max_text_tokens": int | null, "sparse": bool}
  -> {"shape": [n, K, H], "dtype": "float16", "data": base64, "sparse": [{"ids", "vals"}] when asked}
GET /health -> {"ok": true, "model": ..., "k_query": ..., "k_passage": ..., "hidden": ...}
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

LOCK = threading.Lock()
STATE: dict = {}
SPARSE_TOPK = 128          # terms kept per masked position (the authors' --sparse_topk)


def text_prefix(text: str, tokenizer, max_tokens: int, chars_per_token: int = 8, margin: int = 8) -> str:
    """The shortest whitespace-bounded prefix of `text` that tokenises to at least `max_tokens`
    + `margin` tokens (the same rule as agent_search.retrievers.dense.base.encoder_prefix)."""
    if not max_tokens or len(text) <= max_tokens * chars_per_token:
        return text
    window = max_tokens * chars_per_token
    while window < len(text):
        cut = text.rfind(" ", 0, window)
        prefix = text[:cut] if cut > 0 else text[:window]
        if len(tokenizer(prefix, add_special_tokens=False)["input_ids"]) >= max_tokens + margin:
            return prefix
        window *= 4
    return text


def encode(texts: list[str], is_query: bool, max_text_tokens: int | None, sparse: bool = False):
    model = STATE["model"]
    r = model.retriever
    card_length = STATE["card_length"]
    prompt = len(r._query_prefix_ids if is_query else r._passage_prefix_ids) + \
        len(r._query_suffix_ids if is_query else r._passage_suffix_ids)
    with LOCK:
        r.max_length = (max_text_tokens + prompt) if (max_text_tokens and not is_query) else card_length
        room = r.max_length - prompt
        texts = [text_prefix(t or "", r.tokenizer, room) for t in texts]
        ids, mask = model.tokenize(texts, is_query=is_query)
        dev = next(model.backbone.parameters()).device
        with torch.inference_mode():
            out = model.encode(ids.to(dev), mask.to(dev), is_query=is_query, compute_sparse=sparse)
        terms = sparse_terms(out, texts) if sparse else None
        r.max_length = card_length
    return out["repr_hidden"].to(torch.float16).cpu(), terms


def sparse_terms(out: dict, texts: list[str]) -> list[dict]:
    """Per text `{"ids": [...], "vals": [...]}`: content words only, top SPARSE_TOPK per position,
    x100 rounded, max over positions (zeros dropped)."""
    per = out["sparse_acts_per_pos"]                                         # [B, K, V]
    content = STATE["sparse_utils"].get_content_token_ids(texts, STATE["model"].retriever.tokenizer)
    mask = torch.zeros(per.shape[0], per.shape[2], device=per.device, dtype=per.dtype)
    for i, ids in enumerate(content):
        if ids:
            mask[i].scatter_(0, torch.tensor(sorted(ids), device=per.device, dtype=torch.long), 1.0)
    per = per * mask.unsqueeze(1)
    vals, idx = per.topk(min(SPARSE_TOPK, per.shape[-1]), dim=-1)                # [B, K, topk]
    vals = (vals.float() * 100).round()
    result = []
    for b in range(per.shape[0]):
        pooled: dict = {}
        for i, v in zip(idx[b].reshape(-1).tolist(), vals[b].reshape(-1).tolist()):
            if v > 0 and v > pooled.get(i, 0.0):
                pooled[i] = v
        result.append({"ids": list(pooled), "vals": list(pooled.values())})
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):           # quiet: one line per request would flood the job log
        pass

    def _reply(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            r = STATE["model"].retriever
            self._reply({"ok": True, "model": STATE["name"], "k_query": r._k(True), "k_passage": r._k(False),
                         "hidden": STATE["hidden"], "card_length": STATE["card_length"]})
        else:
            self._reply({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/encode":
            return self._reply({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            v, terms = encode(list(req["texts"]), bool(req.get("is_query")), req.get("max_text_tokens"),
                              bool(req.get("sparse")))
            reply = {"shape": list(v.shape), "dtype": "float16", "data": base64.b64encode(v.numpy().tobytes()).decode()}
            if terms is not None:
                reply["sparse"] = terms
            self._reply(reply)
        except Exception as e:  # noqa: BLE001: the client sees the error instead of a hung socket
            self._reply({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ielabgroup/diffretriever-dream-7b-multi-q4-p16")
    ap.add_argument("--port", type=int, default=0, help="0 = a port the OS says is free")
    ap.add_argument("--port-file", default=None, help="write the bound port here once ready")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(args.device).eval()
    # the released repo ships the authors' sparse helpers next to the model code
    import importlib.util
    from huggingface_hub import snapshot_download
    snap = args.model if os.path.isdir(args.model) else snapshot_download(args.model)
    spec = importlib.util.spec_from_file_location("diffretriever_sparse_utils", os.path.join(snap, "sparse_utils.py"))
    sparse_utils = importlib.util.module_from_spec(spec); spec.loader.exec_module(sparse_utils)
    STATE.update(model=model, name=args.model, card_length=int(model.retriever.max_length), sparse_utils=sparse_utils,
                 hidden=int(getattr(model.backbone.config, "hidden_size", 0)))

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    port = server.server_address[1]
    if args.port_file:
        with open(args.port_file, "w") as fh:
            fh.write(str(port))
    print(f"diffretriever serving {args.model} on 127.0.0.1:{port} ({args.device}); "
          f"K query {model.retriever._k(True)}, K passage {model.retriever._k(False)}, "
          f"card length {STATE['card_length']}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
