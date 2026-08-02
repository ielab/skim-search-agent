#!/usr/bin/env python3
"""Rebuild the multi-hop corpora (hotpotqa / 2wiki / musique — all Wikipedia, keyed by title)
from the full STRUCTURED article in `wikimedia/structured-wikipedia` (pre-parsed sections +
infobox + abstract). The Wikipedia title IS the article identity, so a title match (after
norm/strip normalization) is trusted as-is — no content check. A title with NO article
(renamed/deleted) is DROPPED; queries left with no gold are dropped too. The corpus is therefore
a SUBSET of the original — fine, since we re-run retrieval on it.

Emits a PAIR over the SAME kept docs + SAME pruned queries/qrels:
  data/<name>_structured/  corpus.jsonl = {_id, title, section, infobox, text}  (scopeable fields)
  data/<name>_flat/        corpus.jsonl = {_id, title, text}                    (same text, no fields)
bm25/dense see identical content in both arms; only BQL's IN(section/infobox,·) field-scoping
differs — so the pair isolates exactly the value of structure.

  python build.py inspect                               # SEE the schema (first rows, INSTANT)

  # FAST path — download the parquet ONCE, then scan all shards in parallel.
  # NB: the config `enwiki_namespace_0` is stored at `enwiki/data/*.parquet` in the repo:
  hf download wikimedia/structured-wikipedia --repo-type dataset \
      --include "enwiki/data/*.parquet" --local-dir ./sw
  python build.py build --dataset musique --sw-path ./sw   # parallel local scan (reuse ./sw for all 3)

  # or no pre-download (serial network stream, ~10 min/dataset):
  python build.py build --dataset musique                  # streaming
  python build.py build --dataset 2wiki --engine duckdb    # duckdb over a snapshot_download

DEPS (staging node): `datasets` (streaming) OR just `pyarrow` (+ `hf` CLI) for the --sw-path
parallel scan; optional `duckdb huggingface_hub` for --engine duckdb. The --sw-path scan reuses
the EXACT title matching, so the result is identical — just I/O-parallel from local disk.
Copy `data/*_flat/` and `data/*_structured/` to the GPU node afterwards.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import unicodedata
from urllib.parse import unquote, urlparse


def _tqdm(it, **kw):
    """tqdm if installed, else the bare iterable (builds run on a staging node where tqdm
    may be absent — never hard-fail on a progress bar)."""
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:
        return it


# ── title normalization (robust matching to structured-wikipedia `name`/`url`) ──

def title_from_url(url: str) -> str:
    p = urlparse(url or "")
    if "/wiki/" in p.path:
        seg = p.path.split("/wiki/", 1)[1].split("#", 1)[0].split("?", 1)[0]
        return unquote(seg).replace("_", " ").strip()
    return ""


def norm_title(t: str) -> str:
    """NFC + entity-unescape + underscores->spaces + casefold — so 'Sanaa', 'sanaa',
    'San%C4%81' / '&amp;' variants collide correctly."""
    t = html.unescape(t or "")
    t = unicodedata.normalize("NFC", t)
    return t.replace("_", " ").strip().casefold()


_STRIP_NONALNUM = re.compile(r"[^a-z0-9]+")


def strip_key(t: str) -> str:
    """A stronger normalization key: norm_title, then strip diacritics and drop EVERY
    non-alphanumeric char, so 'Foo, Bar' / 'Foo Bar' / 'Foo-Bar' / 'Foo–Bar' (en-dash) /
    'Fóo Bar' all collapse to 'foobar'. Two titles with the same strip_key are almost always the
    same article (punctuation/spacing/dash/accent variants), so a strip-key hit counts as a real
    match. (Exact norm_title is tried FIRST, so a genuine exact title can never lose to a rare
    strip collision.)"""
    n = unicodedata.normalize("NFKD", norm_title(t))
    return _STRIP_NONALNUM.sub("", "".join(c for c in n if not unicodedata.combining(c)))


# ── DEFENSIVE structured-wikipedia parser (lose nothing; robust to schema variants) ──
# Real shape (per the dataset card + observed rows): sections are
#   [{type:"section", name, has_parts:[{type:"paragraph", value}, {type:"list", has_parts:[
#     {type:"list_item", value|has_parts}]}, {type:"image", caption}, {type:"table", ...},
#     {type:"section", ...nested...}]}]
# `value` may be a plain string OR a structured "run" (list of {text|value, url}). We never
# assume a single shape: we try has_parts/parts, value/text, and recurse.

def _loads(x):
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str) and x.strip():
        try:
            return json.loads(x)
        except Exception:
            return []
    return []


def _children(p):
    if isinstance(p, dict):
        return p.get("has_parts") or p.get("parts") or p.get("values") or []
    return []


def _is_section(p):
    return isinstance(p, dict) and (p.get("type") == "section") and (p.get("name") or _children(p))


def coerce_value(v) -> str:
    """Text of a `value` field whether it's a string, a run (list of segments), or a dict."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return " ".join(coerce_value(s) for s in v).strip()
    if isinstance(v, dict):
        return coerce_value(v.get("text") if v.get("text") is not None else v.get("value")).strip()
    return str(v).strip()


def render_part(p, depth: int = 0) -> str:
    """Best-effort text for ANY non-section content part; recurses lists/tables/unknowns."""
    if not isinstance(p, dict):
        return coerce_value(p)
    t = p.get("type")
    txt = coerce_value(p.get("value") if p.get("value") is not None else p.get("text"))
    if t == "paragraph":
        return txt
    if t in ("list", "unordered_list", "ordered_list"):
        return "\n".join(x for x in (render_part(it, depth + 1) for it in _children(p)) if x)
    if t in ("list_item", "item"):
        own = ("  " * depth) + "- " + txt if txt else ""
        sub = "\n".join(x for x in (render_part(it, depth + 1) for it in _children(p)) if x)
        return "\n".join(x for x in (own, sub) if x)
    if t in ("image", "figure"):
        cap = coerce_value(p.get("caption") if p.get("caption") is not None else p.get("alt"))
        return f"[image: {cap}]" if cap else ""
    if t == "table":
        return render_table(p)
    if t == "field":                                  # stray infobox field inside a section
        return f"{coerce_value(p.get('name'))}: {txt}".strip(": ").strip()
    # unknown type: prefer nested content, else its own value
    kids = _children(p)
    if kids:
        return "\n".join(x for x in (render_part(it, depth) for it in kids) if x)
    return txt


def render_table(p) -> str:
    """Render a table preserving ROW structure (cells joined by ' | '), headers first."""
    rows = []
    header = p.get("headers") or p.get("header")
    if header:
        cells = [coerce_value(c) for c in (header if isinstance(header, list) else [header])]
        if any(cells):
            rows.append(" | ".join(cells))
    src = p.get("rows") or _children(p)
    for r in src:
        cells = [coerce_value(c) for c in (_children(r) or ([r] if not isinstance(r, dict) else []))]
        cells = [c for c in cells if c]
        if not cells and isinstance(r, dict):
            cv = coerce_value(r.get("value") if r.get("value") is not None else r.get("text"))
            if cv:
                cells = [cv]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _section_text(parts) -> str:
    """All DIRECT (non-sub-section) content of a section, rendered to text."""
    return "\n".join(x for x in (render_part(p) for p in parts if not _is_section(p)) if x).strip()


# structured-wikipedia labels the article LEAD as a section named "Abstract" (Wikipedia leads are
# unnamed) AND repeats it in the separate `abstract` field. Emitting both duplicated the lead (once
# as unheaded text, once under `## Abstract`). These names mean "the lead" -> treat as the (intro).
_LEAD_NAMES = {"abstract", "introduction", "intro", "lead"}


def extract_sections(sections_field, abstract: str = "") -> list:
    """structured-wikipedia `sections` -> a MATCHED, ordered [{heading, level, text}] list, with the
    article lead as a single heading-less (intro) part. Dedupes the lead: structured-wikipedia ships
    the lead BOTH as an "Abstract" section and in the `abstract` field — we keep ONE (preferring the
    fuller section), so no paragraph is emitted twice."""
    data = _loads(sections_field)
    out = []

    def walk(parts, level):
        for p in parts or []:
            if _is_section(p):
                out.append({"heading": (p.get("name") or "").strip(), "level": level,
                            "text": _section_text(_children(p))})
                walk([x for x in _children(p) if _is_section(x)], level + 1)
            elif isinstance(p, dict) and not out:           # rare: leading content before any section
                out.append({"heading": None, "level": 0, "text": render_part(p)})
    walk(data, 2)

    # Resolve the lead ONCE. If the first section IS the lead (named "Abstract"/etc. or already
    # unheaded), demote it to the heading-less (intro) and drop the redundant `abstract` field.
    # Otherwise (no lead section) fall back to the `abstract` field as the (intro).
    if out and (not out[0]["heading"] or out[0]["heading"].strip().lower() in _LEAD_NAMES):
        out[0] = {"heading": None, "level": 0, "text": out[0]["text"]}
    elif abstract and abstract.strip():
        out.insert(0, {"heading": None, "level": 0, "text": abstract.strip()})
    return [s for s in out if s["heading"] or s["text"]]


def parse_infobox(infoboxes_field) -> dict:
    """Flatten infobox(es) to {key: value}; ACCUMULATE repeats (don't drop), coerce runs."""
    data = _loads(infoboxes_field)
    kv: dict = {}

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") in (None, "field") and (node.get("name") or node.get("label")):
                k = coerce_value(node.get("name") if node.get("name") is not None else node.get("label"))
                v = coerce_value(node.get("value") if node.get("value") is not None else node.get("text"))
                if k and v:
                    if k in kv and v not in kv[k].split(" / "):
                        kv[k] = kv[k] + " / " + v
                    else:
                        kv.setdefault(k, v)
            for child in _children(node):
                walk(child)
        elif isinstance(node, list):
            for it in node:
                walk(it)
    walk(data)
    return kv


# ── assemble the structured document ────────────────────────────────────────

def build_body(sections: list, infobox: dict) -> str:
    out = []
    if infobox:
        out.append("\n".join(f"{k}: {v}" for k, v in infobox.items()))
    for s in sections:
        if s["heading"]:
            out.append("#" * max(2, s["level"]) + " " + s["heading"])
        if s["text"]:
            out.append(s["text"])
    return "\n\n".join(out).strip()


# ── structured-wikipedia index (streaming default; duckdb optional) ──────────

def build_index_streaming(titles: set, config: str) -> dict:
    from datasets import load_dataset
    want = {norm_title(t) for t in titles if t}
    ds = load_dataset("wikimedia/structured-wikipedia", config, split="train", streaming=True)
    ds = ds.select_columns(["name", "url", "sections", "abstract", "infoboxes"])
    index = {}
    bar = _tqdm(ds, desc="scan structured-wiki", unit="row", unit_scale=True)
    for row in bar:
        for key in {norm_title(row.get("name", "")), norm_title(title_from_url(row.get("url", "")))}:
            if key and key in want and key not in index:
                index[key] = row
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(f"matched {len(index)}/{len(want)}")
        if len(index) >= len(want):
            break
    return index


# --- parallel scan over a LOCALLY-DOWNLOADED copy of the parquet shards ------
# Download once (`hf download wikimedia/structured-wikipedia --repo-type dataset
# --include "<config>/*" --local-dir <dir>`), then scan all shards across CPU cores. The match
# logic is byte-for-byte the streaming one (norm_title on name AND url-derived title), so the
# result set is identical — just I/O-parallel from local disk instead of a serial network stream.

_WANT: set | None = None
_WANT_STRIP: set | None = None


def _init_worker(want: set, want_strip: set) -> None:
    global _WANT, _WANT_STRIP
    _WANT, _WANT_STRIP = want, want_strip


def _scan_shard(path: str) -> tuple:
    """Read one parquet shard, return (exact, strip): {norm_title: row} for titles it matches
    exactly, and {strip_key: row} for titles that match only after the stronger strip
    normalization. `strip` stays empty when strip-matching is off (_WANT_STRIP empty)."""
    import pyarrow.parquet as pq
    want, want_strip = _WANT or set(), _WANT_STRIP or set()
    tbl = pq.read_table(path, columns=["name", "url", "sections", "abstract", "infoboxes"])
    names = tbl.column("name").to_pylist()
    urls = tbl.column("url").to_pylist()
    sec, ab, ib = tbl.column("sections"), tbl.column("abstract"), tbl.column("infoboxes")
    exact: dict = {}
    strip: dict = {}
    for i, (name, url) in enumerate(zip(names, urls)):
        keys = {k for k in (norm_title(name or ""), norm_title(title_from_url(url or ""))) if k}
        row = None
        for key in keys:
            if key in want and key not in exact:
                row = row or {"name": name, "url": url, "sections": sec[i].as_py(),
                              "abstract": (ab[i].as_py() or ""), "infoboxes": ib[i].as_py()}
                exact[key] = row
        if want_strip:
            for key in keys:
                sk = _STRIP_NONALNUM.sub("", "".join(
                    c for c in unicodedata.normalize("NFKD", key) if not unicodedata.combining(c)))
                if sk and sk in want_strip and sk not in strip:
                    row = row or {"name": name, "url": url, "sections": sec[i].as_py(),
                                  "abstract": (ab[i].as_py() or ""), "infoboxes": ib[i].as_py()}
                    strip[sk] = row
    return exact, strip


def _find_parquet(sw_path: str, config: str) -> list:
    """Locate the config's parquet under a locally-downloaded copy, robust to nesting. In the real
    repo the config `enwiki_namespace_0` is stored at `enwiki/data/*.parquet` (the config name is
    NOT the folder), so prefer the language subtree (`enwiki/`), then config-named dirs, then any
    parquet. The language prefix lets a mixed en+fr download still resolve by --sw-config."""
    import glob as _glob
    lang = config.split("_namespace")[0]                  # enwiki_namespace_0 -> enwiki
    for pat in (os.path.join(sw_path, "**", lang, "data", "*.parquet"),   # real layout: enwiki/data/
                os.path.join(sw_path, "**", lang, "**", "*.parquet"),     # any enwiki/.../*.parquet
                os.path.join(sw_path, config, "**", "*.parquet"),         # <path>/<config>/...
                os.path.join(sw_path, "**", config, "**", "*.parquet"),   # config dir nested anywhere
                os.path.join(sw_path, "**", f"*{config}*.parquet"),       # config in the FILE name
                os.path.join(sw_path, "**", "*.parquet"),                 # any parquet (single dl)
                os.path.join(sw_path, "*.parquet")):
        files = sorted(_glob.glob(pat, recursive=True))
        if files:
            return files
    return []


def build_index_local(titles: set, sw_path: str, config: str, workers: int | None = None,
                      strip_match: bool = True) -> tuple:
    """Parallel scan of a locally-downloaded copy. Returns (exact, strip): exact is
    {norm_title: row}; strip is {strip_key: row} for the stronger-normalization match
    (empty if strip_match=False)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    files = _find_parquet(sw_path, config)
    if not files:
        sys.exit(f"no parquet under {sw_path!r} (looked for {config}/*.parquet). Download it once:\n"
                 f"  hf download wikimedia/structured-wikipedia --repo-type dataset "
                 f"--include '{config}/*' --local-dir {sw_path}")
    want = {norm_title(t) for t in titles if t}
    want_strip = {strip_key(t) for t in titles if t} if strip_match else set()
    workers = workers or min(8, (os.cpu_count() or 2))
    print(f"scanning {len(files)} local parquet shards with {workers} workers "
          f"(each shard ~hundreds of MB of RAM)...", flush=True)
    exact: dict = {}
    strip: dict = {}
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker, initargs=(want, want_strip)) as ex:
        futs = [ex.submit(_scan_shard, f) for f in files]
        bar = _tqdm(as_completed(futs), total=len(files), desc="scan shards", unit="shard")
        for fut in bar:
            ex_part, st_part = fut.result()
            for key, row in ex_part.items():
                exact.setdefault(key, row)
            for key, row in st_part.items():
                strip.setdefault(key, row)
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(f"matched {len(exact)}/{len(want)}")
    return exact, strip


def build_index_duckdb(titles: set, config: str) -> dict:
    import duckdb
    from huggingface_hub import snapshot_download
    want = {norm_title(t) for t in titles if t}
    repo = snapshot_download("wikimedia/structured-wikipedia", repo_type="dataset",
                             allow_patterns=[f"{config}/*.parquet"])
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE want(nm VARCHAR)")
    con.executemany("INSERT INTO want VALUES (?)", [(w,) for w in want])
    rows = con.execute(f"""
        SELECT name, url, sections, abstract, infoboxes
        FROM read_parquet('{repo}/{config}/*.parquet')
        WHERE strip_accents(lower(replace(trim(name),'_',' '))) IN (SELECT nm FROM want)
    """).fetchall()
    index = {}
    for name, url, sections, abstract, infoboxes in rows:
        for key in {norm_title(name or ""), norm_title(title_from_url(url or ""))}:
            if key:
                index.setdefault(key, {"name": name, "url": url, "sections": sections,
                                       "abstract": abstract or "", "infoboxes": infoboxes})
    return index


# ── commands ─────────────────────────────────────────────────────────────────

def _read_corpus(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _qid(q: dict) -> str:
    return str(q.get("_id") or q.get("id") or q.get("query_id") or q.get("qid") or "")


def _read_queries(src: str) -> list:
    p = os.path.join(src, "queries.jsonl")
    return list(_read_corpus(p)) if os.path.isfile(p) else []


def _read_qrels(src: str) -> dict:
    """query-id -> set(corpus-id) for gold (score>0). Reads BEIR qrels/test.tsv or qrels.jsonl."""
    qrels: dict = {}
    tsv = os.path.join(src, "qrels", "test.tsv")
    if os.path.isfile(tsv):
        with open(tsv, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                parts = line.rstrip("\n").split("\t")
                if i == 0 and parts[:1] == ["query-id"]:
                    continue
                if len(parts) >= 2:
                    if len(parts) >= 3:
                        try:
                            if float(parts[2]) <= 0:
                                continue
                        except ValueError:
                            pass
                    qrels.setdefault(parts[0], set()).add(parts[1])
        return qrels
    jl = os.path.join(src, "qrels.jsonl")
    if os.path.isfile(jl):
        for o in _read_corpus(jl):
            qi = str(o.get("query-id") or o.get("query_id") or o.get("qid") or "")
            cid = o.get("corpus-id") or o.get("doc_id") or o.get("docid")
            if qi and cid is not None and (o.get("score", 1) or 1) > 0:
                qrels.setdefault(qi, set()).add(str(cid))
    return qrels


def _show_row(row, chars):
    print("=" * 80)
    print("name:", row.get("name"), "| url:", row.get("url"))
    for f in ("abstract", "sections", "infoboxes"):
        v = row.get(f)
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        print(f"\n--- {f} ({len(s or '')} chars) ---\n{(s or '')[:chars]}")
    secs = extract_sections(row.get("sections"), row.get("abstract") or "")    # what OUR parser gets
    ib = parse_infobox(row.get("infoboxes"))
    print(f"\n>>> parsed: {len(secs)} sections "
          f"{[s['heading'] for s in secs[:12]]}; infobox keys: {list(ib)[:12]}")


def cmd_inspect(a):
    """Dump a few RAW structured-wikipedia rows so we can verify the schema before trusting it.

    NOTE: this is a STREAMING dataset of 7.6M rows with no server-side index, so `--title X`
    is a LINEAR scan until X is found — it can read many shards (minutes) and shows a progress
    bar so you know it's alive. To just eyeball the schema, run WITHOUT --title: the first rows
    arrive immediately."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/structured-wikipedia", a.sw_config, split="train", streaming=True)
    want = norm_title(a.title) if a.title else None
    if want:
        print(f"streaming scan for {a.title!r} (no index -> linear; Ctrl-C and drop --title to "
              f"just see the first rows). cap --max-scan={a.max_scan}.", flush=True)
    shown = scanned = 0
    bar = _tqdm(ds, desc="scan", unit="row", unit_scale=True) if want else ds
    for row in bar:
        scanned += 1
        if want:
            if norm_title(row.get("name", "")) != want:
                if scanned >= a.max_scan:
                    break
                continue
            if hasattr(bar, "close"):
                bar.close()
        _show_row(row, a.chars)
        shown += 1
        if shown >= a.n:
            break
    if shown == 0:
        if want and scanned >= a.max_scan:
            sys.exit(f"{a.title!r} not found in the first {scanned} rows. The dump isn't ordered "
                     f"alphabetically, so raise --max-scan, or just run without --title to verify "
                     f"the schema. (The real match happens in `build`, which scans for ALL your "
                     f"titles in one pass and reports the match rate.)")
        sys.exit("no rows matched (try without --title, or check --sw-config)")
    return 0


def cmd_build(a):
    src = os.path.join(a.data_dir, a.dataset)
    corpus_path = os.path.join(src, "corpus.jsonl")
    if not os.path.isfile(corpus_path):
        sys.exit(f"flat corpus not found: {corpus_path} (stage it first with scripts/stage_multihop.py)")
    docs = list(_read_corpus(corpus_path))
    if a.limit:
        docs = docs[: a.limit]
    titles = {d.get("title") or "" for d in docs}
    engine = "local" if a.sw_path else a.engine
    print(f"{a.dataset}: {len(docs)} docs, {len(titles)} unique titles -> structured-wikipedia ({engine})")
    strip_match = a.sw_path and not a.no_strip   # stronger strip-normalization only on local engine
    if a.sw_path:                          # parallel scan of a locally-downloaded copy (fastest)
        exact, strip = build_index_local(titles, a.sw_path, a.sw_config, a.workers,
                                         strip_match=strip_match)
    elif a.engine == "duckdb":
        exact, strip = build_index_duckdb(titles, a.sw_config), {}
    else:
        exact, strip = build_index_streaming(titles, a.sw_config), {}
    print(f"  exact-matched {len(exact)}/{len(titles)} titles "
          f"({100*len(exact)//max(len(titles),1)}%)")

    out_struct = os.path.join(a.out_dir, f"{a.dataset}_structured")
    out_flat = os.path.join(a.out_dir, f"{a.dataset}_flat")
    for od in (out_struct, out_flat):
        os.makedirs(os.path.join(od, "qrels"), exist_ok=True)

    qrels = _read_qrels(src)
    gold_ids = set().union(*qrels.values()) if qrels else set()
    n_kept = n_strip = n_gold = n_gold_kept = 0
    kept_ids: set = set()
    misses: list[str] = []                       # titles with NO article -> dropped
    gold_misses: list[str] = []                  # the subset of dropped docs that are GOLD
    cS = open(os.path.join(out_struct, "corpus.jsonl"), "w", encoding="utf-8")
    cF = open(os.path.join(out_flat, "corpus.jsonl"), "w", encoding="utf-8")
    with cS, cF:
        for d in _tqdm(docs, desc="match + write", unit="doc"):
            title = d.get("title") or ""
            did = str(d["_id"])
            is_gold = did in gold_ids
            n_gold += is_gold
            # match exactly first (so a true exact title can't lose to a strip-collision), then by
            # the stronger strip key. The title IS the article identity (Wikipedia titles are
            # unique), so a title match is trusted as-is — no containment check.
            row = exact.get(norm_title(title))
            via_strip = False
            if row is None and strip:
                row, via_strip = strip.get(strip_key(title)), True
            if row is None:                          # no article exists -> DROP the doc entirely
                misses.append(title)
                if is_gold:
                    gold_misses.append(title)
                continue
            secs = extract_sections(row.get("sections"), row.get("abstract") or "")
            ib = parse_infobox(row.get("infoboxes"))
            body = build_body(secs, ib) or (d.get("text") or "")   # never emit an empty doc
            # Sections ship as a MATCHED, ordered list of {heading, text} parts (each `##` section
            # kept WITH its own body), so downstream `fetch` reads a real slice and IN(section,·)
            # scopes to a named part instead of re-deriving from `##` markers. `section` (joined
            # headings) is kept for back-compat + a quick heading match; `text` is the full flattened
            # body used by bm25/dense and shared byte-identically with the flat twin.
            sections_list = [{"heading": s["heading"] or "(intro)", "text": s["text"]}
                             for s in secs if (s["heading"] or s["text"])]
            section = " ".join(s["heading"] for s in secs if s["heading"])
            infobox = "\n".join(f"{k}: {v}" for k, v in ib.items())
            # STRUCTURED exposes sections/infobox as scopeable FIELDS (BQL IN(section/infobox,·));
            # FLAT aggregates the SAME content into one text blob (no fields). Same docs, same
            # text -> bm25/dense identical across the pair; only BQL's field-scoping differs.
            cS.write(json.dumps({"_id": d["_id"], "title": title, "sections": sections_list,
                                 "section": section, "infobox": infobox, "text": body},
                                ensure_ascii=False) + "\n")
            cF.write(json.dumps({"_id": d["_id"], "title": title, "text": body},
                                ensure_ascii=False) + "\n")
            kept_ids.add(did)
            n_kept += 1
            n_strip += via_strip
            n_gold_kept += is_gold

    # prune queries/qrels to the kept docs: drop gold pointing at removed docs, then drop any
    # query left with no gold (single-gold queries whose gold vanished disappear). Write to BOTH.
    queries = _read_queries(src)
    kept_q = [q for q in queries if (qrels.get(_qid(q), set()) & kept_ids)]
    for od in (out_struct, out_flat):
        with open(os.path.join(od, "queries.jsonl"), "w", encoding="utf-8") as fh:
            for q in kept_q:
                fh.write(json.dumps(q, ensure_ascii=False) + "\n")
        with open(os.path.join(od, "qrels", "test.tsv"), "w", encoding="utf-8") as fh:
            fh.write("query-id\tcorpus-id\tscore\n")
            for q in kept_q:
                for cid in sorted(qrels.get(_qid(q), set()) & kept_ids):
                    fh.write(f"{_qid(q)}\t{cid}\t1\n")

    print(f"  kept {n_kept}/{len(docs)} docs ({n_strip} via strip-normalization); "
          f"dropped {len(misses)} with no article")
    if n_gold:
        print(f"  GOLD coverage: {n_gold_kept}/{n_gold} gold docs kept "
              f"({100*n_gold_kept//n_gold}%); {len(gold_misses)} gold docs dropped (no article)")
    if queries:
        print(f"  queries: kept {len(kept_q)}/{len(queries)} "
              f"(dropped {len(queries)-len(kept_q)} whose gold was removed)")
    print(f"  wrote {out_struct}/ and {out_flat}/ — SAME docs/text/queries; only the "
          f"section+infobox fields differ.")
    if a.dump_misses:
        with open(a.dump_misses, "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(misses)) + ("\n" if misses else ""))
        msg = f"  wrote {len(misses)} dropped titles -> {a.dump_misses}"
        if gold_misses:
            gp = a.dump_misses.rsplit(".", 1)
            gp = (gp[0] + ".gold." + gp[1]) if len(gp) == 2 else a.dump_misses + ".gold"
            with open(gp, "w", encoding="utf-8") as fh:
                fh.write("\n".join(sorted(gold_misses)) + "\n")
            msg += f"; {len(gold_misses)} of them GOLD -> {gp}"
        print(msg)
    print(f"  register `{a.dataset}_flat` + `{a.dataset}_structured` and run the paired comparison.")
    return 0


# ── push a built pair to Hugging Face (one private repo per dataset) ──────────

def _dataset_card(dataset: str, repo_id: str) -> str:
    """A minimal dataset card (no `configs:` block — the HF viewer would choke on the nested
    `sections` field and the two-arm layout; the corpus is consumed by the harness, not the viewer)."""
    return f"""---
license: cc-by-sa-4.0
tags:
- bql
- structured-retrieval
- multi-hop-qa
---

# {dataset} — flat + structured pair (BQL structured-retrieval experiment)

Rebuilt from `wikimedia/structured-wikipedia` (native sections + infobox, keyed by article
title — **no web scraping**). A **flat + structured PAIR over the same docs, text, and queries**;
only the scopeable fields differ, so BQL's structured-access benefit can be isolated. Gold is
keyed by `_id`, so it always resolves in both arms.

## Layout
```
structured/  corpus.jsonl   {{_id, title, sections:[{{heading,text}}], section, infobox, text}}
             queries.jsonl   qrels/test.tsv
flat/        corpus.jsonl   {{_id, title, text}}   # byte-identical `text`, no scopeable fields
             queries.jsonl   qrels/test.tsv
```
`text` carries `## Heading` markers in **both** arms (so bm25/dense/DCI see identical content);
the structured arm additionally exposes `sections`/`section`/`infobox` as fields for
`IN(section,·)` / `IN(infobox,·)` scope + section `fetch`.

Derived from public Wikipedia + the {dataset} multi-hop QA benchmark. Built with
`corpus_build/wikipedia/build.py`. Pull + stage:
```bash
huggingface-cli download {repo_id} --repo-type dataset --local-dir data/_hf/{dataset}
cp -r data/_hf/{dataset}/structured data/{dataset}_structured
cp -r data/_hf/{dataset}/flat       data/{dataset}_flat
```
"""


def cmd_push(a):
    """Upload a built flat+structured PAIR to a Hugging Face dataset repo — one repo per dataset,
    mirroring `browsecomp-plus-structured`: `<namespace>/<dataset>-structured` with `structured/`
    and `flat/` subdirs (each `corpus.jsonl` + `queries.jsonl` + `qrels/test.tsv`). Sections are
    NATIVE to structured-wikipedia (deterministic, no paid batch), so there is no `sections.jsonl`
    artifact (unlike browsecomp). Private by default — pass `--public` to override."""
    from huggingface_hub import HfApi
    api = HfApi()
    arms = [("structured", os.path.join(a.data_dir, f"{a.dataset}_structured")),
            ("flat", os.path.join(a.data_dir, f"{a.dataset}_flat"))]
    for _arm, d in arms:
        cp = os.path.join(d, "corpus.jsonl")
        if not os.path.isfile(cp):
            sys.exit(f"missing built corpus: {cp} — run `build --dataset {a.dataset}` first")
    if a.repo:
        repo_id = a.repo
    else:
        ns = a.namespace or api.whoami()["name"]
        repo_id = f"{ns}/{a.dataset}-structured"
    private = not a.public
    verb = "DRY-RUN — would push" if a.dry_run else "pushing"
    print(f"{verb} {a.dataset} -> {repo_id}  ({'private' if private else 'PUBLIC'})")
    for arm, d in arms:
        n = sum(1 for line in open(os.path.join(d, "corpus.jsonl")) if line.strip())
        nq = sum(1 for line in open(os.path.join(d, "queries.jsonl")) if line.strip())
        print(f"  {arm}/  <- {d}  ({n} docs, {nq} queries)")
    if a.dry_run:
        print("  (dry run — nothing uploaded; drop --dry-run to push)")
        return 0
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for arm, d in arms:
        print(f"  uploading {arm}/ ...", flush=True)
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=d,
                          path_in_repo=arm, commit_message=f"{a.dataset} {arm} corpus")
    if not a.no_card:
        api.upload_file(path_or_fileobj=_dataset_card(a.dataset, repo_id).encode("utf-8"),
                        path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
                        commit_message="dataset card")
    print(f"  done -> https://huggingface.co/datasets/{repo_id}")
    return 0


def _self_test():
    secs = extract_sections(json.dumps([
        {"type": "section", "name": "History", "has_parts": [
            {"type": "paragraph", "value": "founded long ago"},
            {"type": "list", "has_parts": [
                {"type": "list_item", "value": "first"},
                {"type": "list_item", "has_parts": [{"type": "paragraph", "value": "nested"}]}]},
            {"type": "image", "caption": "old map"},
            {"type": "table", "headers": ["Year", "Pop"], "rows": [
                {"has_parts": [{"value": "1990"}, {"value": "100"}]}]},
            {"type": "section", "name": "Antiquity", "has_parts": [
                {"type": "paragraph", "value": [{"text": "city of "}, {"text": "Sheba", "url": "/wiki/Sheba"}]}]}]}]),
        abstract="Sanaa is the capital.")
    headings = [s["heading"] for s in secs]
    assert headings == [None, "History", "Antiquity"], headings
    h = secs[1]["text"]
    assert "founded long ago" in h and "- first" in h and "nested" in h         # list + nested item
    assert "old map" in h and "Year | Pop" in h and "1990 | 100" in h           # image caption + table rows
    assert "city of Sheba" in secs[2]["text"]                                   # link-run paragraph
    # the emitted `sections` list keeps each heading MATCHED to its own text (the abstract's
    # heading-less lead -> "(intro)"), so downstream `fetch`/IN(section,·) address a real slice.
    sections_list = [{"heading": s["heading"] or "(intro)", "text": s["text"]}
                     for s in secs if (s["heading"] or s["text"])]
    assert [s["heading"] for s in sections_list] == ["(intro)", "History", "Antiquity"]
    assert sections_list[0]["text"] == "Sanaa is the capital." and "founded long ago" in sections_list[1]["text"]
    # LEAD DEDUP: structured-wikipedia ships the lead BOTH as an "Abstract" section and in the
    # `abstract` field — extract_sections must keep it ONCE (the fuller section, demoted to (intro)),
    # never emit it twice (the bug that put the lead before AND under `## Abstract`).
    dup = extract_sections(json.dumps([
        {"type": "section", "name": "Abstract", "has_parts": [{"type": "paragraph", "value": "The full lead paragraph."}]},
        {"type": "section", "name": "Plot", "has_parts": [{"type": "paragraph", "value": "Plot summary."}]}]),
        abstract="Short lead.")
    assert [s["heading"] for s in dup] == [None, "Plot"], [s["heading"] for s in dup]
    assert dup[0]["text"] == "The full lead paragraph."          # the SECTION lead kept, abstract field dropped
    assert sum("lead" in s["text"].lower() for s in dup) == 1    # exactly once — no duplication
    ib = parse_infobox(json.dumps([{"type": "infobox", "has_parts": [
        {"type": "field", "name": "office", "value": "President"},
        {"type": "field", "name": "office", "value": "Senator"},                # repeat -> accumulate
        {"type": "field", "name": "party", "value": [{"text": "Democratic"}]}]}]))   # structured value
    assert ib["office"] == "President / Senator" and ib["party"] == "Democratic"
    assert norm_title("Queen_Arwa_University") == norm_title("queen arwa university")
    assert title_from_url("https://en.wikipedia.org/wiki/Sanaa#History") == "Sanaa"
    # strip_key collapses punctuation / spacing / dashes / accents -> a strip hit is a real match
    assert strip_key("Foo, Bar") == strip_key("Foo Bar") == strip_key("Foo-Bar") == "foobar"
    assert strip_key("1912–13 Scottish Districts season") == strip_key("1912-13 scottish districts season")
    assert strip_key("Fóo") == strip_key("Foo") == "foo"
    assert strip_key("Mercury (element)") != strip_key("Mercury (planet)")   # disambiguator preserved
    print("self-test OK (sections/nested-lists/tables/images/link-runs, infobox repeats, "
          "title, strip_key)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("inspect")
    p.add_argument("--title"); p.add_argument("--n", type=int, default=2); p.add_argument("--chars", type=int, default=1800)
    p.add_argument("--max-scan", type=int, default=500_000, help="give up looking for --title after this many rows")
    p.add_argument("--sw-config", default="enwiki_namespace_0")
    p = sub.add_parser("build")
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "2wiki", "musique"])
    p.add_argument("--data-dir", default="../../data"); p.add_argument("--out-dir", default="../../data")
    p.add_argument("--sw-config", default="enwiki_namespace_0")
    p.add_argument("--sw-path", help="dir of a LOCALLY-downloaded structured-wikipedia copy "
                   "(hf download ...) -> parallel scan, fastest. Overrides --engine.")
    p.add_argument("--workers", type=int, help="parallel shards for --sw-path (default min(8, cpus))")
    p.add_argument("--engine", choices=["streaming", "duckdb"], default="streaming",
                   help="used only when --sw-path is NOT given")
    p.add_argument("--no-strip", action="store_true",
                   help="match titles by norm_title only; skip the stronger strip-normalization "
                        "(punctuation/spacing/dash/accent-insensitive) match (local engine only)")
    p.add_argument("--dump-misses", help="write titles with no candidate (renamed/deleted) to this file")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("push", help="upload a built flat+structured PAIR to a private HF dataset repo")
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "2wiki", "musique"])
    p.add_argument("--data-dir", default="../../data", help="where the built <dataset>_{structured,flat}/ live")
    p.add_argument("--repo", help="explicit repo id (default: <namespace>/<dataset>-structured)")
    p.add_argument("--namespace", help="HF namespace for the default repo id (default: your username)")
    p.add_argument("--public", action="store_true", help="make the repo PUBLIC (default: private)")
    p.add_argument("--no-card", action="store_true", help="skip uploading the README dataset card")
    p.add_argument("--dry-run", action="store_true", help="show what would be pushed; upload nothing")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.cmd == "inspect":
        return cmd_inspect(a)
    if a.cmd == "build":
        return cmd_build(a)
    if a.cmd == "push":
        return cmd_push(a)
    ap.print_help()
    return 1


if __name__ == "__main__":
    _rc = main()
    # HF `datasets` streaming (pyarrow/native reader) can call std::terminate during normal
    # interpreter teardown after we break out of the stream early ("Aborted (core dumped)" AFTER
    # the output is printed). os._exit skips that faulty native cleanup — our files are already
    # closed (with-blocks) by the time main() returns, so nothing is lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
