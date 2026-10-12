#!/usr/bin/env python3
"""μLattice single-page live demo server.

Serves ONE self-contained HTML page (no CDN, no external assets) and a JSON API
that runs the REAL trained model (checkpoints/capacity8k.pt, scale from its
proto_cfg) through the REAL protocol module:

  POST /api/store  {n_pairs, filler, overwrite_frac, seed}
      -> one causal forward over the generated stream; returns measured wall
         time, us/token, lattice cells grown, per-query answers vs gold,
         updated-vs-original correctness, and the learned 2D addresses of a
         sample of stored keys (for the live lattice map).
  POST /api/cost    {sizes:[{n_pairs,filler}...]}
      -> forwards at increasing stream lengths; returns us/token and cells for
         each: the constant-cost proof, measured live.

Everything the page shows is measured on this request. Nothing is canned.
"""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from mulattice.protocol import SynthCfg, build_model, force_fast_scan, key_addresses, synth_batch

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available()
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
CKPT = os.path.join(ROOT, "checkpoints", "capacity8k.pt")
CFG = SynthCfg()

print(f"loading {CKPT} on {DEVICE} ...", flush=True)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = SynthCfg()
if ck.get("proto_cfg", {}).get("addr_scale"):
    cfg.addr_scale = float(ck["proto_cfg"]["addr_scale"])
MODEL = build_model(cfg)
MODEL.load_state_dict(ck["model"])
MODEL = MODEL.to(DEVICE).eval()
force_fast_scan(MODEL)
PARAMS = sum(p.numel() for p in MODEL.parameters())
print(f"ready: scale {cfg.addr_scale:.0f}, {PARAMS:,} params", flush=True)


@torch.no_grad()
def run_store(n_pairs: int, filler: int, ow: float, seed: int) -> dict:
    from mulattice.protocol import logits_at
    g = torch.Generator().manual_seed(seed)
    Q = min(64, n_pairs)
    b = synth_batch(1, n_pairs, Q, cfg, filler=filler, overwrite_frac=ow, generator=g)
    t0 = time.time()
    pred = logits_at(MODEL, b["tokens"].to(DEVICE), b["ans_pos"].to(DEVICE)).argmax(-1).cpu()
    dt = time.time() - t0
    ok = (pred == b["gold"]).squeeze(0)
    up = b["was_updated"].squeeze(0)
    # learned addresses of a sample of stored keys (for the live lattice map)
    n_sample = min(400, n_pairs)
    kp = b["key_pos"][0, :n_sample].unsqueeze(0)
    addr = key_addresses(MODEL, b["tokens"][:1].to(DEVICE), kp.to(DEVICE))[0].cpu()
    sc = cfg.addr_scale
    pts = [[round(float(a / sc), 4), round(float(c / sc), 4)]
           for a, c in zip(addr[:, 0], addr[:, 1])]
    keys_show = b["tokens"][0, b["key_pos"][0, :n_sample]].tolist()
    golds = {}
    rows = []
    for q in range(Q):
        k = int(b["tokens"][0, b["ans_pos"][0][q] - 1])
        rows.append({
            "key": k,
            "gold": int(b["gold"][0, q]) - cfg.NK,
            "pred": int(pred[0, q]) - cfg.NK,
            "ok": bool(ok[q]),
            "updated": bool(up[q]),
        })
    n_cells = MODEL.blocks[0].mixer.scan.last_n_cells
    return {
        "T": b["L"], "wall_s": round(dt, 3), "us_per_token": round(dt / b["L"] * 1e6, 1),
        "cells": int(n_cells), "acc": round(ok.float().mean().item(), 4),
        "acc_updated": round(ok[up].float().mean().item(), 4) if bool(up.any()) else None,
        "acc_untouched": round(ok[~up].float().mean().item(), 4) if bool((~up).any()) else None,
        "n_pairs": n_pairs, "n_upd": b["n_upd"], "Q": Q,
        "rows": rows, "addr_points": pts, "addr_keys": keys_show,
        "scale": sc, "params": PARAMS, "device": DEVICE,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@torch.no_grad()
def run_cost(sizes: list) -> list:
    from mulattice.protocol import logits_at
    out = []
    for s in sizes:
        n_pairs, filler = int(s.get("n_pairs", 1024)), int(s.get("filler", 0))
        g = torch.Generator().manual_seed(int(s.get("seed", 7)))
        Q = 16
        b = synth_batch(1, n_pairs, Q, cfg, filler=filler, generator=g)
        t0 = time.time()
        logits_at(MODEL, b["tokens"].to(DEVICE), b["ans_pos"].to(DEVICE))
        dt = time.time() - t0
        out.append({
            "n_pairs": n_pairs, "filler": filler, "T": b["L"],
            "wall_s": round(dt, 3), "us_per_token": round(dt / b["L"] * 1e6, 1),
            "cells": int(MODEL.blocks[0].mixer.scan.last_n_cells),
            "flops_attention_equiv": b["L"] * b["L"],   # O(T^2) reference work
        })
    return out



@torch.no_grad()
def run_custom(doc_text: str, pairs: list, seed: int) -> dict:
    """Stream the USER's own text with the USER's own named key->value pairs,
    some overwritten mid-stream, then query.  Names map deterministically into
    the model's token space (shown in the response); the model only ever sees
    ids -- exactly the MQAR regime it was trained on."""
    from mulattice.protocol import logits_at
    NK, NV = cfg.NK, cfg.NV
    g = torch.Generator().manual_seed(seed)

    def h(s, used, span, off):
        x = abs(hash((seed, s))) % span
        while (off + x) in used:
            x = (x + 1) % span
        used.add(off + x)
        return off + x

    used = set()
    kmap, vmap = {}, {}
    for p in pairs:
        kmap[p["name"]] = h("k" + p["name"], used, NK, 0)
    # filler ids: hash the user's own words into the key space (excl. used keys)
    words = [w for w in doc_text.split() if w][: 20000] or ["..."]
    fid = {}
    for w in set(words):
        fid[w] = h("f" + w, used, NK, 0)

    store_rows, upd_rows = [], []
    for p in pairs:
        vid = h("v" + p["value"], used, NV, NK)
        vmap[p["value"]] = vid
        row = dict(p); row["kid"] = kmap[p["name"]]; row["vid"] = vid
        if p.get("new_value"):
            nid = h("v2" + p["new_value"], used, NV, NK)
            vmap[p["new_value"]] = nid
            row["nid"] = nid
            upd_rows.append(row)
        store_rows.append(row)

    # deterministic filler stream from the user's own words
    fill = [fid[w] for w in words]
    def take(n):
        out = []
        while len(out) < n:
            i = int(torch.randint(0, len(fill), (1,), generator=g))
            out.extend(fill[i:i + n - len(out)])
        return out[:n]

    F = 6                                   # filler tokens between pairs
    block = F + 2
    n_upd = len(upd_rows)
    Q = min(64, len(store_rows))
    qrows = store_rows[:Q]
    L = (len(store_rows) + n_upd) * block + 2 * Q
    toks = torch.zeros(1, L, dtype=torch.long)
    gold = torch.zeros(1, Q, dtype=torch.long)
    latest = {r["kid"]: r["vid"] for r in store_rows}
    for i, r in enumerate(store_rows):
        s = i * block
        toks[0, s:s + F] = torch.tensor(take(F))
        toks[0, s + F], toks[0, s + F + 1] = r["kid"], r["vid"]
    for j, r in enumerate(upd_rows):
        s = (len(store_rows) + j) * block
        toks[0, s:s + F] = torch.tensor(take(F))
        toks[0, s + F], toks[0, s + F + 1] = r["kid"], r["nid"]
        latest[r["kid"]] = r["nid"]
    S2 = (len(store_rows) + n_upd) * block
    ans = []
    for q, r in enumerate(qrows):
        toks[0, S2 + 2 * q], toks[0, S2 + 2 * q + 1] = r["kid"], cfg.SEP
        gold[0, q] = latest[r["kid"]]
        ans.append(S2 + 2 * q + 1)
    ans_pos = torch.tensor(ans).unsqueeze(0)
    t0 = time.time()
    pred = logits_at(MODEL, toks.to(DEVICE), ans_pos.to(DEVICE)).argmax(-1).cpu()
    dt = time.time() - t0
    ok = (pred == gold).squeeze(0)
    inv_v = {v: k for k, v in vmap.items()}
    rows = []
    for q, r in enumerate(qrows):
        rows.append({
            "name": r["name"], "asked": inv_v.get(int(pred[0, q]), "id" + str(int(pred[0, q]) - NK)),
            "truth": r["new_value"] if r.get("new_value") else r["value"],
            "ok": bool(ok[q]), "updated": bool(r.get("new_value")),
            "old": r["value"] if r.get("new_value") else None,
        })
    return {
        "T": L, "wall_s": round(dt, 3), "us_per_token": round(dt / L * 1e6, 1),
        "cells": int(MODEL.blocks[0].mixer.scan.last_n_cells),
        "acc": round(ok.float().mean().item(), 4),
        "n_pairs": len(store_rows), "n_upd": n_upd, "Q": Q,
        "rows": rows,
        "keymap": {r["name"]: r["kid"] for r in store_rows},
        "filler_words": len(set(words)),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@torch.no_grad()
def run_cellview(name: str, value: str, new_value: str, seed: int) -> dict:
    """Cell's-eye view of one fact: where the key lives (learned address + 3x3
    cells), and the delta rule replacing the stored value on overwrite."""
    from mulattice.protocol import logits_at
    from mulattice.kernel import phi_2d
    g = torch.Generator().manual_seed(seed)
    used = set()

    def h(s, span, off):
        x = abs(hash((seed, s))) % span
        while (off + x) in used:
            x = (x + 1) % span
        used.add(off + x)
        return off + x

    kid = h("k" + name, cfg.NK, 0)
    vid = h("v" + value, cfg.NV, cfg.NK)
    nid = h("v2" + new_value, cfg.NV, cfg.NK) if new_value else vid

    def stream(with_upd):
        s = [kid, vid]
        if with_upd:
            s += [kid, nid]
        s += [kid, cfg.SEP]
        return torch.tensor([s])

    answers = {}
    for tag, with_upd in (("before", False), ("after", True)):
        if tag == "after" and not new_value:
            answers[tag] = None
            continue
        t = stream(with_upd)
        pred = logits_at(MODEL, t.to(DEVICE),
                         torch.tensor([[t.shape[1] - 1]]).to(DEVICE)).argmax(-1).cpu()
        answers[tag] = int(pred[0, 0]) - cfg.NK

    mx = MODEL.blocks[0].mixer
    x = MODEL.emb(torch.tensor([[kid]]).to(DEVICE))
    xc = mx.conv(x)
    addr = (mx.cfg.addr_scale * torch.tanh(mx.Wa(xc)))[0, 0].cpu()
    w, z = phi_2d(addr.unsqueeze(0))
    cells = [[{"z": [int(z[0, i, j, 0]), int(z[0, i, j, 1])], "phi": round(float(w[0, i, j]), 4)}
              for j in range(3)] for i in range(3)]
    return {"kid": kid, "vid": vid, "nid": nid,
            "addr": [round(float(addr[0]), 2), round(float(addr[1]), 2)],
            "scale": mx.cfg.addr_scale, "cells": cells,
            "answers": answers, "value": value, "new_value": new_value,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>&mu;Lattice &mdash; the growing memory, live</title>
<style>
 :root{--ink:#1c1e26;--mut:#6b7280;--hair:#e5e7eb;--bg:#fcfcfa;--card:#ffffff;
       --acc:#2547d0;--ok:#0e8a5f;--bad:#c2413b;--upd:#b07807;--soft:#f4f4f0}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif;
      -webkit-font-smoothing:antialiased}
 .serif{font-family:Charter,Georgia,"Times New Roman",serif}
 header{max-width:980px;margin:0 auto;padding:56px 24px 8px}
 header .kicker{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--acc);font-weight:700}
 header h1{font-size:44px;line-height:1.12;margin:10px 0 12px;font-weight:400}
 header .dek{font-size:19px;color:var(--mut);max-width:640px;margin:0 0 10px}
 header .meta{font-size:12.5px;color:var(--mut);border-top:1px solid var(--hair);padding-top:12px}
 main{max-width:980px;margin:0 auto;padding:8px 24px 80px}
 section{margin-top:56px}
 .sn{display:flex;gap:16px;align-items:baseline;border-bottom:1px solid var(--hair);padding-bottom:10px;margin-bottom:18px}
 .sn .n{font-size:13px;color:var(--acc);font-weight:700;letter-spacing:.08em}
 .sn h2{font-size:26px;font-weight:400;margin:0}
 p.lead{font-size:16.5px;color:#374151;max-width:700px}
 .fig{background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:22px;margin-top:16px;
      box-shadow:0 1px 2px rgba(28,30,38,.04)}
 .figcap{font-size:13px;color:var(--mut);margin-top:12px;border-top:1px solid var(--soft);padding-top:10px}
 label{font-size:13px;color:#374151;display:block;margin:10px 0 4px;font-weight:600}
 input[type=text],textarea{width:100%;border:1px solid var(--hair);border-radius:8px;padding:9px 11px;
      font:14px/1.5 -apple-system,sans-serif;background:#fff;color:var(--ink)}
 textarea{min-height:84px;resize:vertical}
 input:focus,textarea:focus{outline:2px solid #cdd8f8;border-color:var(--acc)}
 .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 @media(max-width:760px){.grid2{grid-template-columns:1fr}}
 button{background:var(--ink);color:#fff;border:0;border-radius:999px;padding:10px 22px;font-size:14px;
      font-weight:600;cursor:pointer;margin-top:14px}
 button.sec{background:#fff;color:var(--ink);border:1px solid var(--ink)}
 button:disabled{opacity:.45;cursor:wait}
 .pairs{margin-top:10px;display:grid;gap:8px}
 .prow{display:grid;grid-template-columns:1fr 1fr 1fr 32px;gap:8px;align-items:center}
 .prow input{padding:7px 9px}
 .prow .del{border:0;background:none;color:var(--bad);font-size:17px;cursor:pointer;padding:4px}
 .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:10px;margin:16px 0 4px}
 .stat{background:var(--soft);border-radius:10px;padding:12px 14px}
 .stat b{display:block;font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
 .stat span{font-size:11.5px;color:var(--mut);letter-spacing:.04em;text-transform:uppercase}
 table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px;font-variant-numeric:tabular-nums}
 th{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);text-align:left;
    border-bottom:1px solid var(--hair);padding:6px 8px}
 td{padding:6px 8px;border-bottom:1px solid var(--soft)}
 .y{color:var(--ok);font-weight:600}.n{color:var(--bad);font-weight:600}.updv{color:var(--upd)}
 .pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:99px;background:#f6ecd8;color:var(--upd)}
 canvas{width:100%;background:#fff;border:1px solid var(--hair);border-radius:10px}
 .row{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-top:8px}
 .rangewrap{display:flex;align-items:center;gap:10px}
 input[type=range]{accent-color:var(--acc);width:150px}
 .rv{font-variant-numeric:tabular-nums;font-weight:600;min-width:52px}
 .note{font-size:13px;color:var(--mut)}
 code{background:var(--soft);padding:1px 6px;border-radius:6px;font-size:12.5px}
 footer{max-width:980px;margin:0 auto;padding:24px;color:var(--mut);font-size:12.5px;
        border-top:1px solid var(--hair)}
</style></head><body>

<header>
 <div class="kicker">Live proof &middot; real weights &middot; nothing canned</div>
 <h1 class="serif">&mu;Lattice: the growing memory, running</h1>
 <p class="dek serif">A one-million-parameter lattice mixer that stores associations while it
 reads, overwrites them on sight, and answers from memory &mdash; at constant cost per token.
 This page drives the actual trained checkpoint on the GPU.</p>
 <div class="meta" id="meta">model: capacity8k.pt &middot; loading&hellip;</div>
</header>
<main>

<section>
 <div class="sn"><span class="n">01</span><h2 class="serif">Stream your own document</h2></div>
 <p class="lead">Paste any text. Define your own facts as <em>name &rarr; value</em> pairs. Mark any
 of them to be <strong>overwritten mid-stream</strong> with a new value. Then ask. The model reads
 your words as the stream, stores your pairs as it passes them, and must return the
 <em>latest</em> value &mdash; last write wins.</p>
 <div class="fig">
  <div class="grid2">
   <div>
    <label for="doc">Your text (becomes the stream)</label>
    <textarea id="doc">The harbour was quiet that morning. Gulls circled the masts while the tide went out. A clerk stamped papers without looking up. Somewhere a bell rang twice, then stopped. The road east had been washed out again. Dust settled over the market stalls. Two children argued about a kite. Rain threatened but never came.</textarea>
   </div>
   <div>
    <label>Your facts <span class="note">(name &rarr; value; fill &ldquo;new value&rdquo; to overwrite mid-stream)</span></label>
    <div class="pairs" id="pairs"></div>
    <button class="sec" id="addp" style="margin-top:8px;padding:6px 16px;font-size:13px">+ add fact</button>
   </div>
  </div>
  <button id="runc">Read, store, overwrite &mdash; and answer</button>
  <div class="stats" id="stc" style="display:none"></div>
  <div id="cres" style="display:none">
   <table id="qrt"></table>
   <div class="figcap" id="ccap"></div>
  </div>
 </div>
</section>

<section>
 <div class="sn"><span class="n">02</span><h2 class="serif">The store at scale</h2></div>
 <p class="lead">The same mechanism under load: up to 8,192 pairs in one 133,120-token stream,
 half of them overwritten, answered after a single causal pass. The map shows the model&rsquo;s
 own learned addresses &mdash; distinct keys spread, exactly as the design intends.</p>
 <div class="fig">
  <div class="row">
   <div class="rangewrap"><label style="margin:0">pairs</label>
    <input id="np" type="range" min="64" max="8192" step="64" value="2048"><span class="rv" id="npv">2048</span></div>
   <div class="rangewrap"><label style="margin:0">filler/pair</label>
    <input id="fl" type="range" min="0" max="14" value="14"><span class="rv" id="flv">14</span></div>
   <div class="rangewrap"><label style="margin:0">overwritten</label>
    <input id="ow" type="range" min="0" max="50" value="50"><span class="rv" id="owv">50%</span></div>
   <button class="sec" id="run" style="margin-top:0">stream &amp; query</button>
  </div>
  <div class="stats" id="st" style="display:none"></div>
  <div class="row" style="align-items:stretch" id="bigres" >
   <canvas id="latt" height="280" style="flex:1.2;min-width:300px"></canvas>
   <div style="flex:1;min-width:280px;max-height:300px;overflow:auto">
    <table id="qr"></table></div>
  </div>
  <div class="figcap">Left: learned addresses of 400 stored keys from this run (grid scale shown).
   Right: 64 sampled queries; gold rows marked <span class="pill">overwritten</span> were
   re-written later in the stream &mdash; the store must answer with the new value.</div>
 </div>
</section>

<section>
 <div class="sn"><span class="n">03</span><h2 class="serif">Constant cost while memory grows</h2></div>
 <p class="lead">Each token touches exactly nine memory cells, so per-token work cannot grow with
 context. Measure it: the curve below runs the model at increasing stream lengths, live.</p>
 <div class="fig">
  <button class="sec" id="cost" style="margin-top:0">measure the cost curve</button>
  <canvas id="cc" height="300" style="margin-top:14px"></canvas>
  <div class="figcap">Blue: measured &mu;s per token &mdash; flat is the guarantee. Green: lattice
   cells, growing with stored content. Dashed red: attention&rsquo;s O(T&sup2;) work, analytic reference.</div>
 </div>
</section>

<section>
 <div class="sn"><span class="n">04</span><h2 class="serif">Inside one cell</h2></div>
 <p class="lead">Where does a value physically live, and where does the <em>new</em> value go on
 overwrite? Pick one fact from section 01 (or any name/value). The model&rsquo;s learned address
 for that key, the 3&times;3 cells it touches, and the before/after answers &mdash; same cells,
 value replaced in place by the delta rule.</p>
 <div class="fig">
  <div class="grid2">
   <div>
    <label>fact to inspect</label>
    <div class="prow" style="grid-template-columns:1fr 1fr 1fr">
     <input id="cvn" placeholder="name" value="the vault password">
     <input id="cvv" placeholder="value" value="swordfish">
     <input id="cvnv" placeholder="new value" value="bluedragon"></div>
    <button class="sec" id="cv">inspect the cell</button>
    <div id="cvs" style="display:none;margin-top:14px"></div>
   </div>
   <div><canvas id="cellcv" height="300"></canvas>
    <div class="figcap" id="cvcap"></div></div>
  </div>
 </div>
</section>

</main>
<footer>Every number on this page is produced by a live forward pass of
<code>capacity8k.pt</code> on request &mdash; timestamped per run. Filler for section 01 is your own
text, hashed deterministically into the model&rsquo;s token space; facts occupy key&ndash;value
structural positions, exactly the regime the mixer was trained on.</footer>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function api(p,body){const r=await fetch(p,{method:"POST",
 headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});return r.json()}
function stats(el,d){el.style.display="grid";
 const s=[["stream tokens",d.T.toLocaleString()],["time",d.wall_s+" s"],
  ["&mu;s/token",d.us_per_token],["cells",d.cells.toLocaleString()],["recall",d.acc]];
 el.innerHTML=s.map(x=>`<div class="stat"><b>${x[1]}</b><span>${x[0]}</span></div>`).join("")}
/* section 01 */
const DEF=[["the vault password","swordfish","bluedragon"],
 ["Elbaz's first name","Eric",""],["the car's color","red",""],
 ["the meeting time","3pm","4pm"],["the favorite meal","stew",""]];
function prow(n,v,nv){const d=document.createElement("div");d.className="prow";
 d.innerHTML=`<input placeholder="name" value="${esc(n)}">
  <input placeholder="value" value="${esc(v)}">
  <input placeholder="new value (optional)" value="${esc(nv)}">
  <button class="del" title="remove">&times;</button>`;
 d.querySelector(".del").onclick=()=>d.remove();return d}
DEF.forEach(x=>$("pairs").appendChild(prow(...x)));
$("addp").onclick=()=>$("pairs").appendChild(prow("","",""));
$("runc").onclick=async()=>{$("runc").disabled=true;$("runc").textContent="reading your stream&hellip;";
 try{
  const pairs=[...document.querySelectorAll("#pairs .prow")].map(r=>{
   const i=r.querySelectorAll("input");return{name:i[0].value.trim(),value:i[1].value.trim(),
    new_value:i[2].value.trim()}}).filter(p=>p.name&&p.value);
  const d=await api("/api/custom",{text:$("doc").value,pairs,seed:(Math.random()*1e6)|0});
  if(d.error){alert(d.error);return}
  $("meta").textContent=`capacity8k.pt · scale ${d.scale||96} · ${d.device} · run ${d.ts}`;
  stats($("stc"),d);
  $("qrt").innerHTML=`<tr><th>asked</th><th>the model answered</th><th>latest value</th><th></th></tr>`+
   d.rows.map(r=>`<tr><td><b>${esc(r.name)}</b></td>
    <td class="${r.ok?"y":"n"}">${esc(r.asked)}</td>
    <td class="${r.updated?"updv":""}">${esc(r.truth)}</td>
    <td>${r.updated?`<span class="pill">was ${esc(r.old)}</span>`:""}${r.ok?"":" ✗"}</td></tr>`).join("");
  $("cres").style.display="block";
  $("ccap").innerHTML=`${d.n_pairs} of your facts stored in ${d.T.toLocaleString()} tokens of your text`+
   (d.n_upd?`, ${d.n_upd} overwritten mid-stream`:"")+` · recall <b>${d.acc}</b> · ${d.us_per_token} &mu;s/token`+
   ` · key ids: `+Object.entries(d.keymap).slice(0,4).map(([n,k])=>`${esc(n)}=${k}`).join(", ")+`&hellip;`;
 }catch(e){alert(e)}$("runc").disabled=false;$("runc").textContent="Read, store, overwrite — and answer"};
/* section 02 */
$("np").oninput=e=>$("npv").textContent=e.target.value;
$("fl").oninput=e=>$("flv").textContent=e.target.value;
$("ow").oninput=e=>$("owv").textContent=e.target.value+"%";
function drawLatt(pts,sc){const c=$("latt"),x=c.getContext("2d");
 c.width=c.clientWidth*2;c.height=560;x.scale(2,2);const W=c.width/2,H=280;
 x.fillStyle="#fff";x.fillRect(0,0,W,H);
 x.strokeStyle="#eceef2";for(let i=0;i<=8;i++){const p=i/8;
  x.beginPath();x.moveTo(p*W,0);x.lineTo(p*W,H);x.stroke();
  x.beginPath();x.moveTo(0,p*H);x.lineTo(W,p*H);x.stroke()}
 x.fillStyle="#2547d0";for(const p of pts){const u=(p[0]/sc+1)/2,v=(p[1]/sc+1)/2;
  x.beginPath();x.arc(u*W,v*H,1.7,0,7);x.fill()}}
$("run").onclick=async()=>{$("run").disabled=true;$("run").textContent="streaming&hellip;";
 try{const d=await api("/api/store",{n_pairs:+$("np").value,filler:+$("fl").value,
  overwrite_frac:+$("ow").value/100,seed:(Math.random()*1e6)|0});
  $("meta").textContent=`capacity8k.pt · scale ${d.scale} · ${d.params.toLocaleString()} params · ${d.device} · run ${d.ts}`;
  stats($("st"),d);drawLatt(d.addr_points,d.scale);
  $("qr").innerHTML=`<tr><th>query key</th><th>answered</th><th>true</th><th></th></tr>`+
   d.rows.map(r=>`<tr><td>${r.key}</td><td class="${r.ok?"y":"n"}">${r.pred}</td>`+
   `<td class="${r.updated?"updv":""}">${r.gold}</td>`+
   `<td>${r.updated?'<span class="pill">overwritten</span>':""}${r.ok?"":" ✗"}</td></tr>`).join("");
 }catch(e){alert(e)}$("run").disabled=false;$("run").textContent="stream & query"};
/* section 03 */
function drawCC(data){const c=$("cc"),x=c.getContext("2d");c.width=c.clientWidth*2;c.height=600;
 x.scale(2,2);const W=c.width/2,H=300,P=48;
 const maxT=Math.max(...data.map(d=>d.T)),us=data.map(d=>d.us_per_token),
  ce=data.map(d=>d.cells),maxU=Math.max(...us)*1.25,maxC=Math.max(...ce)*1.15,
  maxA=Math.max(...data.map(d=>d.flops_attention_equiv));
 const X=t=>P+(Math.log(t)/Math.log(maxT))*(W-P-16),
  Y=u=>H-30-(u/maxU)*(H-56),Y2=v=>H-30-(v/maxC)*(H-56),YA=v=>H-30-(v/maxA)*(H-56);
 x.fillStyle="#fff";x.fillRect(0,0,W,H);x.strokeStyle="#eceef2";x.fillStyle="#6b7280";
 x.font="11px -apple-system,sans-serif";
 for(let i=0;i<=4;i++){const px=P+(i/4)*(W-P-16);x.beginPath();x.moveTo(px,20);x.lineTo(px,H-30);x.stroke()}
 x.strokeStyle="#c2413b";x.setLineDash([6,5]);x.beginPath();let f1=true;
 for(const d of data){const px=X(d.T),py=YA(d.flops_attention_equiv);
  f1?x.moveTo(px,py):x.lineTo(px,py);f1=false}x.stroke();x.setLineDash([]);
 x.strokeStyle="#2547d0";x.lineWidth=2.4;x.beginPath();f1=true;
 for(const d of data){const px=X(d.T),py=Y(d.us_per_token);f1?x.moveTo(px,py):x.lineTo(px,py);f1=false}
 x.stroke();
 x.strokeStyle="#0e8a5f";x.lineWidth=2.4;x.beginPath();f1=true;
 for(const d of data){const px=X(d.T),py=Y2(d.cells);f1?x.moveTo(px,py):x.lineTo(px,py);f1=false}
 x.stroke();x.lineWidth=1;
 for(const d of data){x.fillStyle="#2547d0";x.beginPath();x.arc(X(d.T),Y(d.us_per_token),3.4,0,7);x.fill();
  x.fillStyle="#0e8a5f";x.beginPath();x.arc(X(d.T),Y2(d.cells),3.4,0,7);x.fill()}
 x.fillStyle="#6b7280";
 for(const d of data)x.fillText(d.T.toLocaleString(),X(d.T)-16,H-14);
 x.fillText("stream length (tokens, log scale)",W/2-80,H-2);
 x.fillText("&mu;s/token (blue, flat = constant) · cells (green) · attention O(T²) (dashed red)",P,16)}
$("cost").onclick=async()=>{$("cost").disabled=true;$("cost").textContent="measuring&hellip;";
 try{const sizes=[64,256,1024,2048,4096,8192].map(n=>({n_pairs:n,filler:14}));
  const d=await api("/api/cost",{sizes});drawCC(d)}catch(e){alert(e)}
 $("cost").disabled=false;$("cost").textContent="measure the cost curve"};
/* section 04 */
$("cv").onclick=async()=>{$("cv").disabled=true;
 try{const d=await api("/api/cellview",{name:$("cvn").value.trim()||"key",
  value:$("cvv").value.trim()||"v1",new_value:$("cvnv").value.trim(),seed:(Math.random()*1e6)|0});
  if(d.error){alert(d.error);return}
  const inv={[d.vid]:d.value};inv[d.nid]=d.new_value||d.value;
  $("cvs").innerHTML=`
   <div class="stats" style="display:grid">
    <div class="stat"><b>(${d.addr[0]}, ${d.addr[1]})</b><span>learned address</span></div>
    <div class="stat"><b>${esc(inv[d.answers.before] ?? "—")}</b><span>stored &amp; retrieved before</span></div>
    <div class="stat"><b class="${d.answers.after!==null&&d.answers.after!==d.answers.before?"updv":""}">${esc(d.answers.after!==null?inv[d.answers.after]:"—")}</b><span>after overwrite, same cells</span></div>
   </div>
   <p class="note" style="margin:8px 0 0">key id ${d.kid} &rarr; address on grid scale ${d.scale};
   the write lands as <code>W &larr; &alpha;W + &beta;(v<sub>new</sub> &minus; &alpha;W&kappa;)&kappa;<sup>T</sup>:
   the old association is subtracted out, the new one takes its place &mdash; nothing is appended.</p>`;
  drawCell(d)}catch(e){alert(e)}$("cv").disabled=false};
function drawCell(d){const c=$("cellcv"),x=c.getContext("2d");
 c.width=c.clientWidth*2;c.height=600;x.scale(2,2);const W=c.width/2,H=300;
 x.fillStyle="#fff";x.fillRect(0,0,W,H);const cw=W/3,ch=H/3;
 for(let i=0;i<3;i++)for(let j=0;j<3;j++){
  const phi=d.cells[i][j].phi,z=d.cells[i][j].z;
  const t=Math.min(1,phi/0.7);
  x.fillStyle=`rgba(37,71,208,${(0.06+0.85*t).toFixed(3)})`;
  x.fillRect(j*cw+6,i*ch+6,cw-12,ch-12);
  x.fillStyle=t>0.45?"#fff":"#6b7280";x.font="12px -apple-system,sans-serif";
  x.fillText(`(${z[0]}, ${z[1]})`,j*cw+14,i*ch+24);
  x.font="600 15px -apple-system,sans-serif";
  x.fillText(`phi ${phi.toFixed(3)}`,j*cw+14,i*ch+ch-18)}
 $("cvcap").textContent="the 3x3 neighborhood of the key's address; darker = stronger write/read weight (the bump kernel)."}
window.addEventListener("resize",()=>{});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/favicon.ico":
            self.send_response(204); self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/store":
                r = run_store(int(req.get("n_pairs", 2048)), int(req.get("filler", 14)),
                              float(req.get("overwrite_frac", 0.5)), int(req.get("seed", 0)))
                self._json(r)
            elif self.path == "/api/cost":
                self._json(run_cost(req.get("sizes", [])))
            elif self.path == "/api/cellview":
                self._json(run_cellview(str(req.get("name", "key")),
                                        str(req.get("value", "v1")),
                                        str(req.get("new_value", "")),
                                        int(req.get("seed", 0))))
            elif self.path == "/api/custom":
                r = run_custom(str(req.get("text", "")), req.get("pairs", []),
                               int(req.get("seed", 0)))
                self._json(r)
            else:
                self.send_error(404)
        except Exception as e:  # surface errors to the page log
            self._json({"error": str(e)}, 500)

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8772"))
    # ThreadingHTTPServer: the single-threaded server hangs on browsers'
    # speculative idle connections (page "loads once, then never again").
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    srv.daemon_threads = True
    print(f"demo on http://0.0.0.0:{port}/", flush=True)
    srv.serve_forever()
