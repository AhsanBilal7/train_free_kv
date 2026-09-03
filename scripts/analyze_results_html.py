#!/usr/bin/env python3
"""Aggregate every experiments/<benchmark>/pred/*/result.json into one HTML report.

The runs this reads are the ones scripts/run_h100_aces.sh (via run_h100.sh ->
experiments/<EXPERIMENT>/run.sh) writes: one folder per (config x budget x M x ...)
cell, named by scripts/run_naming.py, holding the cumulative result.json that
eval.py wrote. Three result schemas are handled, detected from the JSON itself
rather than from the benchmark name:

  LongBench  flat    {"<task>": score}                       0-100
  RULER      nested  {"<length>": {"<task>": s, "average": s}, "average": s}
  NIAH       typed   {"metric", "lengths", "depths", "per_length", "average", "grid"}

Everything is folded into ONE record shape -- a run with a per-dataset score map --
so a single set of views works across benchmarks:

  * Full grid          one row per run: identity + hyperparameters + every dataset
  * Dataset comparison per dataset, technique x (budget x M) matrix, with an
                       optional "delta vs. baseline technique" mode
  * Charts             per dataset, score vs. budget (one line per technique x M)
                       and score vs. M (one line per technique x budget)
  * Sub-tasks          RULER per-length task breakdown / NIAH per-length metrics
  * NIAH heatmaps      the depth x length grid each NIAH run stores
  * Inventory          empty, partial, unparseable and colliding runs

Usage
-----
  python scripts/analyze_results_html.py                    # -> analysis/results_report.html
  python scripts/analyze_results_html.py --out /tmp/r.html
  python scripts/analyze_results_html.py --csv              # also a long-form CSV
  python scripts/analyze_results_html.py --benchmark LongBench RULER
"""
import argparse
import datetime as _dt
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from run_naming import parse_run_name  # noqa: E402  (same source of truth as run.sh)

# LongBench's canonical task order (QA -> few-shot -> summarisation -> synthetic
# -> code). Anything unknown is appended alphabetically.
LONGBENCH_ORDER = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

MODEL_ALIASES = [("Mistral-7B-Instruct-v0.2", "Mistral-7B"),
                 ("Mistral-7B", "Mistral-7B"),
                 ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
                 ("Llama-3.1-8B", "Llama-3.1-8B")]

# Hyperparameter columns shown in the full grid, in a stable order.
HYPER_COLS = ["model", "budget", "M", "cell", "brule", "sel", "T", "lL", "lT",
              "nrm", "alpha", "kappa", "tau", "lK", "lV", "red", "chunk"]


def short_model(tag):
    if not tag:
        return "?"
    for base, short in MODEL_ALIASES:
        if tag == base:
            return short
        if tag.startswith(base + "-"):        # seed runs: ...-v0.2-s0 -> Mistral-7B-s0
            return short + "-" + tag[len(base) + 1:]
    return tag


def technique_of(parsed, folder):
    """The run's IDENTITY, model stripped out (model is its own column).

    causal_lookahead -> its config-file name without the model suffix
    (lorekv-<cell>-<brule>-<sel>); every other mode -> the mode itself
    (andpro / critical / causal_forward_only), whose distinguishing knob is
    `red` and is carried as a separate column.
    """
    if parsed is None:
        return folder
    if parsed.get("mode") == "causal_lookahead" and parsed.get("cell"):
        return "lorekv-%s-%s-%s" % (parsed["cell"], parsed["brule"], parsed["sel"])
    return parsed.get("mode") or folder


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def flatten_result(obj):
    """result.json -> (per-dataset scores, per-dataset sub-task scores, niah block).

    Shape-detected, not name-detected, so a benchmark that changes its folder name
    keeps working. Scores are normalised to a 0-100 scale.
    """
    if not isinstance(obj, dict) or not obj:
        return {}, {}, None

    # --- NIAH: typed block with a depth x length grid -------------------------
    if "per_length" in obj and isinstance(obj.get("per_length"), dict):
        metric = obj.get("metric", "exact_match")
        scores, sub = {}, {}
        for length, m in obj["per_length"].items():
            if not isinstance(m, dict):
                continue
            v = _num(m.get(metric))
            if v is not None:
                scores[str(length)] = v * 100.0 if v <= 1.0 else v
            sub[str(length)] = {k: (val * 100.0 if (_num(val) or 0) <= 1.0 else val)
                                for k, val in m.items() if _num(val) is not None}
        niah = {"metric": metric, "lengths": obj.get("lengths", []),
                "depths": obj.get("depths", []),
                "grid": (obj.get("grid") or {}).get(metric)}
        return scores, sub, niah

    # --- RULER: {length: {task: score, "average": s}}, plus a scalar overall ---
    if any(isinstance(v, dict) for v in obj.values()):
        scores, sub = {}, {}
        for key, val in obj.items():
            if isinstance(val, dict):
                avg = _num(val.get("average"))
                if avg is None:
                    nums = [n for n in (_num(v) for v in val.values()) if n is not None]
                    avg = sum(nums) / len(nums) if nums else None
                if avg is not None:
                    scores[str(key)] = avg
                sub[str(key)] = {k: _num(v) for k, v in val.items()
                                 if _num(v) is not None and k != "average"}
        return scores, sub, None

    # --- LongBench: flat {task: score} ---------------------------------------
    return ({str(k): _num(v) for k, v in obj.items() if _num(v) is not None}, {}, None)


def dataset_sort_key(name):
    """Numeric datasets (RULER lengths, NIAH lengths) sort numerically; LongBench
    tasks follow the canonical order above; anything else sorts alphabetically."""
    try:
        return (0, float(name), "")
    except ValueError:
        pass
    if name in LONGBENCH_ORDER:
        return (1, LONGBENCH_ORDER.index(name), "")
    return (2, 0, name)


def collect(benchmarks):
    out, issues = {}, []
    for bm in benchmarks:
        pred = os.path.join(REPO_ROOT, "experiments", bm, "pred")
        if not os.path.isdir(pred):
            continue
        runs = []
        for folder in sorted(os.listdir(pred)):
            run_dir = os.path.join(pred, folder)
            if not os.path.isdir(run_dir):
                continue
            # EVERY pred/ folder becomes a run record, scored or not: a launched
            # cell of the grid that died mid-eval is an experiment that exists and
            # has to be visible, not a silent hole. Its state is read off the disk:
            #   done      result.json carries scores
            #   no-scores result.json is there but empty ({}) -- eval wrote nothing
            #   pending   no result.json at all -- eval never ran
            #   unreadable  result.json is corrupt
            # run.sh writes <dataset>.jsonl per finished dataset and .jsonl_tmp
            # while one is in flight, so those two counts say how far the run got.
            files = os.listdir(run_dir)
            done_ds = sorted((f[:-len(".jsonl")] for f in files if f.endswith(".jsonl")),
                             key=dataset_sort_key)
            tmp_ds = sorted((f[:-len(".jsonl_tmp")] for f in files if f.endswith(".jsonl_tmp")),
                            key=dataset_sort_key)

            rj = os.path.join(run_dir, "result.json")
            obj, status = {}, "pending"
            if os.path.isfile(rj):
                try:
                    with open(rj) as fh:
                        obj = json.load(fh) or {}
                    status = "done"
                except (ValueError, OSError) as exc:
                    status = "unreadable"
                    issues.append({"benchmark": bm, "folder": folder, "status": status,
                                   "problem": "unreadable result.json: %s" % exc})

            parsed = parse_run_name(folder)
            if parsed is None:
                issues.append({"benchmark": bm, "folder": folder, "status": "unparsed",
                               "problem": "folder name does not parse (run_naming.py)"})
                parsed = {}
            scores, sub, niah = flatten_result(obj)
            if status == "done" and not scores:
                status = "no-scores"

            if status != "done":
                issues.append({
                    "benchmark": bm, "folder": folder, "status": status,
                    "problem": {
                        "pending": "no result.json -- eval never ran",
                        "no-scores": "result.json is empty ({}) -- eval produced no score",
                        "unreadable": "result.json is corrupt",
                    }.get(status, status) + (
                        " · %d dataset(s) finished (%s)" % (len(done_ds), ", ".join(done_ds))
                        if done_ds else " · no dataset finished") + (
                        " · in flight: %s" % ", ".join(tmp_ds) if tmp_ds else "")})

            rec = {
                "folder": folder,
                "technique": technique_of(parsed or None, folder),
                "mode": parsed.get("mode"),
                "model": short_model(parsed.get("model")),
                "budget": _num(parsed.get("budget")),
                "M": _num(parsed.get("M")),
                "scores": scores,
                "sub": sub,
                "niah": niah,
                "n": len(scores),
                "mean": (round(sum(scores.values()) / len(scores), 2) if scores else None),
                "status": status,
                "doneDs": done_ds,
                "tmpDs": tmp_ds,
                "mtime": os.path.getmtime(rj if os.path.isfile(rj) else run_dir),
            }
            for k in ("cell", "brule", "sel", "T", "lL", "lT", "nrm",
                      "alpha", "kappa", "tau", "lK", "lV", "red", "chunk"):
                rec[k] = parsed.get(k)
            runs.append(rec)

        datasets = sorted({d for r in runs for d in r["scores"]}, key=dataset_sort_key)
        # A run is "complete" when it covers every dataset any run of this
        # benchmark covers -- the partial ones (a single qasper probe, say) must
        # never be averaged against a full sweep without saying so.
        for r in runs:
            r["complete"] = bool(datasets) and r["n"] == len(datasets)

        subtasks = {}
        for r in runs:
            for ds, m in r["sub"].items():
                subtasks.setdefault(ds, set()).update(m.keys())
        subtasks = {k: sorted(v, key=dataset_sort_key) for k, v in subtasks.items()}

        # Cells of the comparison matrix that more than one run lands in.
        seen = {}
        for r in runs:
            if r["status"] != "done":
                continue
            key = (r["technique"], r["model"], r["budget"], r["M"])
            seen.setdefault(key, []).append(r["folder"])
        for key, folders in seen.items():
            if len(folders) > 1:
                issues.append({
                    "benchmark": bm,
                    "folder": " | ".join(folders),
                    "problem": ("%d runs share (technique=%s, model=%s, B=%s, M=%s); "
                                "the matrix shows the one with the most datasets"
                                % (len(folders), key[0], key[1],
                                   _fmt(key[2]), _fmt(key[3])))})

        # A hyperparameter that never varies (chunk = 4 in every run ever launched,
        # for instance) is a column of identical cells. Hoist it out of the tables
        # and state it once above them; the axes are never hoisted.
        # A column whose non-None values are all identical carries no information
        # and is hoisted out of the tables. `chunk` is 4 in every run ever launched;
        # `T`/`lL`/`alpha`/... are constant but only exist on the lookahead runs, so
        # those are labelled "(where set)" rather than claimed of every run.
        axes = {"model", "budget", "M"}
        constants, varying = {}, []
        for col in HYPER_COLS:
            vals = {r.get(col) for r in runs}
            seen_vals = vals - {None}
            if col not in axes and len(seen_vals) == 1:
                constants[col] = {"value": _fmt_val(next(iter(seen_vals))),
                                  "everywhere": None not in vals}
            elif seen_vals or col in axes:
                varying.append(col)

        out[bm] = {
            "runs": runs,
            "datasets": datasets,
            "constants": constants,
            "gridCols": varying,
            "statusCounts": {k: sum(1 for r in runs if r["status"] == k)
                             for k in ("done", "no-scores", "pending", "unreadable")},
            "subtasks": subtasks,
            "models": sorted({r["model"] for r in runs}),
            "budgets": sorted({r["budget"] for r in runs if r["budget"] is not None}),
            "Ms": sorted({r["M"] for r in runs if r["M"] is not None}),
            "techniques": sorted({r["technique"] for r in runs}),
            "has_niah": any(r["niah"] for r in runs),
        }
    return out, issues


def _fmt_val(x):
    """Display form of a hyperparameter value (numbers compact, strings verbatim)."""
    n = _num(x)
    if n is None:
        return str(x)
    return str(int(n)) if n == int(n) else str(n)


def _fmt(x):
    if x is None:
        return "-"
    f = float(x)
    return str(int(f)) if f == int(f) else str(f)


def write_csv(data, path):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "technique", "model", "budget", "M", "dataset",
                    "score", "status", "complete", "folder"])
        for bm, blob in data.items():
            for r in blob["runs"]:
                base = [bm, r["technique"], r["model"], _fmt(r["budget"]), _fmt(r["M"])]
                if not r["scores"]:
                    # A launched-but-unscored cell still gets a row, so the CSV is
                    # the same inventory the HTML shows.
                    w.writerow(base + ["", "", r["status"], 0, r["folder"]])
                    continue
                for ds in sorted(r["scores"], key=dataset_sort_key):
                    w.writerow(base + [ds, r["scores"][ds], r["status"],
                                       int(r["complete"]), r["folder"]])
                w.writerow(base + ["MEAN", r["mean"], r["status"],
                                   int(r["complete"]), r["folder"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", nargs="*", default=None,
                    help="benchmarks to include (default: every experiments/*/pred)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "analysis",
                                                  "results_report.html"))
    ap.add_argument("--csv", action="store_true",
                    help="also write the long-form CSV next to the HTML")
    args = ap.parse_args()

    if args.benchmark:
        benchmarks = args.benchmark
    else:
        exp = os.path.join(REPO_ROOT, "experiments")
        benchmarks = sorted(d for d in os.listdir(exp)
                            if os.path.isdir(os.path.join(exp, d, "pred")))

    data, issues = collect(benchmarks)
    if not data:
        sys.exit("no runs found under experiments/*/pred -- nothing to report")

    payload = {
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "repo": REPO_ROOT,
        "hyperCols": HYPER_COLS,
        "benchmarks": data,
        "issues": issues,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(PAGE.replace("/*__DATA__*/null",
                              json.dumps(payload, default=str)))

    total = sum(len(b["runs"]) for b in data.values())
    print("wrote %s" % args.out)
    print("  %d runs across %d benchmark(s): %s"
          % (total, len(data), ", ".join("%s=%d" % (k, len(v["runs"]))
                                         for k, v in data.items())))
    if issues:
        print("  %d inventory issue(s) -- see the Inventory tab" % len(issues))
    if args.csv:
        csv_path = os.path.splitext(args.out)[0] + "_long.csv"
        write_csv(data, csv_path)
        print("wrote %s" % csv_path)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KV-cache results grid</title>
<style>
  :root{
    --bg:#f7f7f5; --panel:#fff; --ink:#1c1c1a; --muted:#6c6c66; --line:#e2e2dc;
    --accent:#2f5f9e; --pos:#1f7a4d; --neg:#b03030; --chip:#eeeee8;
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --bg:#16161a; --panel:#1e1e23; --ink:#eceae4; --muted:#9a988f; --line:#31313a;
      --accent:#7fa9e0; --pos:#5fc48d; --neg:#e08585; --chip:#2a2a31;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  header{padding:18px 22px 10px;border-bottom:1px solid var(--line);background:var(--panel)}
  h1{margin:0 0 3px;font-size:17px;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:12px}
  .wrap{padding:16px 22px 60px;max-width:1700px}
  .tabs{display:flex;gap:4px;flex-wrap:wrap;margin:12px 0 0}
  .tab{padding:6px 12px;border:1px solid var(--line);border-bottom:none;
       border-radius:6px 6px 0 0;background:var(--chip);cursor:pointer;font-size:12.5px}
  .tab.on{background:var(--panel);font-weight:600;color:var(--accent)}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
        padding:14px 16px;margin:0 0 16px}
  h2{font-size:14px;margin:0 0 10px;letter-spacing:-.01em}
  h3{font-size:12.5px;margin:16px 0 6px;color:var(--muted);font-weight:600;
     text-transform:uppercase;letter-spacing:.04em}
  .bar{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;
       padding:10px 14px;background:var(--panel);border:1px solid var(--line);
       border-radius:8px;margin:0 0 14px}
  .grp{display:flex;flex-direction:column;gap:4px}
  .grp>label:first-child{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;
       color:var(--muted);font-weight:600}
  .opts{display:flex;gap:4px;flex-wrap:wrap;max-width:640px}
  .opt{padding:2px 8px;border:1px solid var(--line);border-radius:99px;cursor:pointer;
       font-size:11.5px;background:transparent;color:var(--muted);white-space:nowrap}
  .opt.on{background:var(--accent);border-color:var(--accent);color:#fff}
  select{background:var(--panel);color:var(--ink);border:1px solid var(--line);
         border-radius:5px;padding:3px 6px;font-size:12px}
  .scroll{overflow-x:auto;max-width:100%}
  table{border-collapse:collapse;font-size:11.5px;font-variant-numeric:tabular-nums}
  th,td{border:1px solid var(--line);padding:3px 7px;text-align:right;white-space:nowrap}
  th{background:var(--chip);font-weight:600;position:sticky;top:0;z-index:2}
  th.l,td.l{text-align:left}
  td.l{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}
  tbody tr:hover td{outline:1px solid var(--accent);outline-offset:-1px}
  .best{font-weight:700;box-shadow:inset 0 0 0 2px var(--accent)}
  .dim{color:var(--muted)}
  /* spacer column between two budget blocks -- the blocks are separate pred/
     folders of ONE config, so they must read as separate blocks, not one run */
  th.sp,td.sp{border:none;background:transparent;padding:0;width:12px;min-width:12px}
  th.gsep,td.gsep{border-left:2px solid var(--accent)}
  th.gend,td.gend{border-right:2px solid var(--accent)}
  /* one config identity = one row block in the full grid; its budgets sit under it */
  tr.ghead td{background:var(--chip);border-top:2px solid var(--accent);
              text-align:left;font-weight:600;font-size:11.5px;padding:5px 8px}
  tr.ghead .bl{color:var(--muted);font-weight:400}
  tr.bsep td{border-top:1px dashed var(--line)}
  tr.gap td{height:9px;border:none;background:transparent;padding:0}
  /* a launched cell that produced no score yet -- visible, but never averaged */
  tr.unscored td{opacity:.62;font-style:italic}
  .badge{display:inline-block;padding:0 5px;border-radius:99px;font-size:10px;
         border:1px solid var(--line);color:var(--muted);white-space:nowrap}
  .badge.done{color:var(--pos);border-color:var(--pos)}
  .badge.pending{color:var(--muted)}
  .badge.nos{color:var(--neg);border-color:var(--neg)}
  .consts{font-size:11.5px;color:var(--muted);margin:0 0 8px}
  .consts b{color:var(--ink);font-weight:600}
  .sortable{cursor:pointer}
  .sortable:after{content:"\2195";opacity:.3;margin-left:3px}
  .sorted-a:after{content:"\2191";opacity:1}
  .sorted-d:after{content:"\2193";opacity:1}
  .charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
  .chart{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px}
  .chart h4{margin:0 0 2px;font-size:12px}
  .legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;font-size:10.5px;color:var(--muted)}
  .legend i{width:9px;height:2px;display:inline-block;vertical-align:middle;margin-right:3px}
  .stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
  .stat b{display:block;font-size:19px;font-weight:650}
  .stat span{color:var(--muted);font-size:11px}
  .note{color:var(--muted);font-size:11.5px;margin:6px 0 0}
  code{font-family:ui-monospace,Menlo,monospace;font-size:11px;background:var(--chip);
       padding:1px 4px;border-radius:3px}
  .hm td{width:34px;height:18px;padding:0;text-align:center;font-size:9.5px}
</style>
</head>
<body>
<header>
  <h1>KV-cache evaluation grid</h1>
  <div class="sub" id="hdr"></div>
</header>
<div class="wrap">
  <div class="tabs" id="tabs"></div>
  <div id="body"></div>
</div>
<script>
const D = /*__DATA__*/null;
const BMS = Object.keys(D.benchmarks);
const PALETTE = ["#2f5f9e","#c8642a","#1f7a4d","#8c3f8c","#b03030","#0e7c8c",
                 "#8a7a1f","#5a5a8c","#3f8c5a","#a0522d","#6a3fa0","#777"];
const STATUS = {
  "done":      ["done","done"],
  "no-scores": ["nos","result.json empty"],
  "pending":   ["pending","eval not run"],
  "unreadable":["nos","result.json corrupt"],
};
function badge(r){
  const [cls,txt] = STATUS[r.status] || ["pending", r.status];
  const prog = r.status==="done" ? "" :
    (r.doneDs.length ? ` ${r.doneDs.length} ds done` : "") +
    (r.tmpDs.length ? ` · ${r.tmpDs.join(",")} in flight` : "");
  return `<span class="badge ${cls}" title="${esc(r.folder)}${esc(prog)}">${esc(txt)}${esc(prog)}</span>`;
}
function constsLine(b){
  const ks = Object.keys(b.constants);
  if(!ks.length) return "";
  return `<p class="consts">Never varies across the ${b.runs.length} runs of this benchmark, so hoisted out of the tables: `
    + ks.map(k=>{
        const c = b.constants[k];
        return `<b>${esc(k)} = ${esc(c.value)}</b>${c.everywhere?"":' <span title="the other runs do not have this parameter at all">(where set)</span>'}`;
      }).join(" · ") + `</p>`;
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = v => v===null||v===undefined?"":(typeof v==="number"?(v===Math.round(v)?v:v.toFixed(2)):v);
const num = v => v===null||v===undefined?"–":(v===Math.round(v)?String(v):String(v));

/* ---------- per-benchmark UI state ---------- */
const S = {};
for(const bm of BMS){
  const b = D.benchmarks[bm];
  S[bm] = {
    view:"budget",
    models:new Set(b.models), budgets:new Set(b.budgets), Ms:new Set(b.Ms),
    techs:new Set(b.techniques), completeOnly:false, groupRows:true, showUnscored:true,
    baseline:"", group:"budget", xaxis:"budget",
    subDataset:b.datasets[0]||"", sort:{key:"technique",dir:1}
  };
}
let TAB = "overview";

/* ---------- filtering ---------- */
function runsOf(bm){
  const s = S[bm];
  return D.benchmarks[bm].runs.filter(r =>
    s.models.has(r.model) &&
    (r.budget===null ? true : s.budgets.has(r.budget)) &&
    (r.M===null ? true : s.Ms.has(r.M)) &&
    s.techs.has(r.technique) &&
    (!s.completeOnly || r.complete) &&
    (s.showUnscored || r.status==="done"));
}
function label(r){ return r.technique + " @ " + r.model; }

/* ---------- colour ramps ---------- */
function heat(v, lo, hi){
  if(v===null||v===undefined||!(hi>lo)) return "";
  const t = Math.max(0, Math.min(1, (v-lo)/(hi-lo)));
  return `background:hsl(212,${18+52*t}%,${94-40*t}%);color:${t>.62?"#fff":"inherit"}`;
}
function diverge(v, m){
  if(v===null||v===undefined||!(m>0)) return "";
  const t = Math.max(-1, Math.min(1, v/m));
  const h = t>=0 ? 152 : 8;
  const a = Math.abs(t);
  return `background:hsl(${h},${20+50*a}%,${95-38*a}%);color:${a>.62?"#fff":"inherit"}`;
}

/* ---------- tiny SVG line chart ---------- */
function lineChart(title, xs, series, yLabel){
  const W=340,H=210,L=42,R=10,T=12,B=30;
  const vals=[];
  series.forEach(s=>s.pts.forEach(p=>{ if(p.y!==null) vals.push(p.y); }));
  series.forEach(s=>{ if(s.flat!==null&&s.flat!==undefined) vals.push(s.flat); });
  if(!vals.length) return "";
  let lo=Math.min(...vals), hi=Math.max(...vals);
  if(hi===lo){ hi=lo+1; lo=lo-1; }
  const pad=(hi-lo)*0.12; lo-=pad; hi+=pad;
  const X = i => xs.length<2 ? (L+(W-L-R)/2) : L + i*(W-L-R)/(xs.length-1);
  const Y = v => T + (H-T-B)*(1-(v-lo)/(hi-lo));
  let g = "";
  for(let k=0;k<=4;k++){
    const v = lo + k*(hi-lo)/4, y = Y(v);
    g += `<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-width="1"/>`
       + `<text x="${L-5}" y="${y+3}" text-anchor="end" font-size="8.5" fill="var(--muted)">${v.toFixed(1)}</text>`;
  }
  xs.forEach((x,i)=>{ g += `<text x="${X(i)}" y="${H-B+13}" text-anchor="middle" font-size="9" fill="var(--muted)">${esc(x)}</text>`; });
  g += `<text x="8" y="${T+4}" font-size="8.5" fill="var(--muted)">${esc(yLabel||"")}</text>`;
  series.forEach(s=>{
    if(s.flat!==null&&s.flat!==undefined){
      g += `<line x1="${L}" x2="${W-R}" y1="${Y(s.flat)}" y2="${Y(s.flat)}" stroke="${s.color}" stroke-width="1.3" stroke-dasharray="4 3" opacity=".85"/>`;
      return;
    }
    const pts = s.pts.map((p,i)=>({x:X(i),y:p.y})).filter(p=>p.y!==null);
    if(!pts.length) return;
    g += `<polyline fill="none" stroke="${s.color}" stroke-width="1.8" stroke-linejoin="round" points="${pts.map(p=>p.x+","+Y(p.y)).join(" ")}"/>`;
    pts.forEach(p=>{ g += `<circle cx="${p.x}" cy="${Y(p.y)}" r="2.6" fill="${s.color}"/>`; });
  });
  const legend = series.map(s=>`<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("");
  return `<div class="chart"><h4>${esc(title)}</h4>
    <svg viewBox="0 0 ${W} ${H}" width="100%" role="img">${g}</svg>
    <div class="legend">${legend}</div></div>`;
}

/* ---------- filter bar ---------- */
function chips(bm, key, values, cur, fmtv){
  return values.map(v=>{
    const on = cur.has(v) ? " on" : "";
    return `<button class="opt${on}" data-bm="${bm}" data-key="${key}" data-val="${esc(v)}">${esc(fmtv?fmtv(v):v)}</button>`;
  }).join("");
}
function filterBar(bm){
  const b = D.benchmarks[bm], s = S[bm];
  const views = [["budget","Per-budget tables"],["grid","Full grid"],
                 ["matrix","Dataset × M × budget"],
                 ["charts","Charts"],["sub","Sub-tasks"]]
                 .concat(b.has_niah?[["niah","NIAH heatmaps"]]:[])
                 .concat([["inv","Inventory"]]);
  return `<div class="bar">
    <div class="grp"><label>View</label><div class="opts">
      ${views.map(([v,t])=>`<button class="opt${s.view===v?" on":""}" data-bm="${bm}" data-key="view" data-val="${v}">${t}</button>`).join("")}
    </div></div>
    <div class="grp"><label>Model</label><div class="opts">${chips(bm,"models",b.models,s.models)}</div></div>
    <div class="grp"><label>Budget B</label><div class="opts">${chips(bm,"budgets",b.budgets,s.budgets,num)}</div></div>
    <div class="grp"><label>Particles M</label><div class="opts">${chips(bm,"Ms",b.Ms,s.Ms,num)}</div></div>
    <div class="grp"><label>Technique</label><div class="opts">${chips(bm,"techs",b.techniques,s.techs)}</div></div>
    <div class="grp"><label>Only complete runs</label><div class="opts">
      <button class="opt${s.completeOnly?" on":""}" data-bm="${bm}" data-key="completeOnly" data-val="1">all ${b.datasets.length} datasets</button>
    </div></div>
    <div class="grp"><label>Grid rows</label><div class="opts">
      <button class="opt${s.groupRows?" on":""}" data-bm="${bm}" data-key="groupRows" data-val="1">group the budgets of one config</button>
    </div></div>
    <div class="grp"><label>Unscored runs</label><div class="opts">
      <button class="opt${s.showUnscored?" on":""}" data-bm="${bm}" data-key="showUnscored" data-val="1">show launched-but-unscored (${b.statusCounts["pending"]+b.statusCounts["no-scores"]})</button>
    </div></div>
  </div>`;
}

/* ---------- view: one table per budget (techniques as rows) ---------- */
function viewBudget(bm){
  const b = D.benchmarks[bm], s = S[bm];
  const runs = runsOf(bm);
  const budgets = b.budgets.filter(x=>s.budgets.has(x));
  // Non-llama models first, then the ones whose runs carry '-llama' in the folder
  // name -- so the tables come out as <4 budgets for the base model> followed by
  // <the same 4 budgets for the llama variant>.
  const isLlama = m => /llama/i.test(m);
  const models = b.models.filter(m=>s.models.has(m))
    .sort((a,c)=>(isLlama(a)?1:0)-(isLlama(c)?1:0) || a.localeCompare(c));

  const cols = b.datasets;
  let n = 0;
  const tables = [];
  for(const mdl of models){
    for(const bud of budgets){
      const rs = runs.filter(r=>r.model===mdl && r.budget===bud)
        .sort((x,y)=> x.technique.localeCompare(y.technique) || ((x.M??-1)-(y.M??-1))
                      || String(x.red).localeCompare(String(y.red)));
      if(!rs.length) continue;
      n++;
      // Best and heat scale are computed WITHIN the table, i.e. at a fixed
      // (model, budget) -- that is the comparison this view is for.
      const range = {}, best = {};
      for(const ds of cols.concat(["MEAN"])){
        const vs = rs.map(r=>ds==="MEAN"?r.mean:r.scores[ds]).filter(v=>v!==undefined&&v!==null);
        range[ds] = vs.length ? [Math.min(...vs), Math.max(...vs)] : [0,0];
        best[ds]  = vs.length ? Math.max(...vs) : null;
      }
      const showRed = rs.some(r=>r.red!==null&&r.red!==undefined);
      const body = rs.map(r=>{
        const cell = ds => {
          const v = ds==="MEAN" ? r.mean : r.scores[ds];
          if(v===undefined||v===null) return `<td class="dim">–</td>`;
          const isBest = best[ds]!==null && Math.abs(v-best[ds])<1e-9;
          const bold = ds==="MEAN" ? `<b>${v.toFixed(2)}</b>` : v.toFixed(2);
          return `<td class="${isBest?"best":""}" style="${heat(v,range[ds][0],range[ds][1])}">${bold}</td>`;
        };
        return `<tr class="${r.status==="done"?"":"unscored"}" title="${esc(r.folder)}">`
          + `<td class="l">${esc(r.technique)}</td>`
          + `<td>${r.M===null?'<span class="dim">–</span>':num(r.M)}</td>`
          + (showRed?`<td>${r.red===null||r.red===undefined?'<span class="dim">–</span>':esc(r.red)}</td>`:"")
          + cols.map(cell).join("") + cell("MEAN")
          + `<td class="l ${r.complete?"":"dim"}">${r.n}${r.status==="done"&&!r.complete?" ⚠":""}</td>`
          + `<td class="l">${badge(r)}</td></tr>`;
      }).join("");
      const nDone = rs.filter(r=>r.status==="done").length;
      tables.push(`<h3>Table ${n} — ${esc(mdl)}, B = ${num(bud)} · ${rs.length} technique run${rs.length===1?"":"s"}`
        + (nDone===rs.length?"":` <span class="dim">(${nDone} scored, ${rs.length-nDone} unscored)</span>`) + `</h3>
        <div class="scroll"><table><thead><tr>
          <th class="l">technique</th><th>M</th>${showRed?"<th>red</th>":""}
          ${cols.map(c=>`<th>${esc(c)}</th>`).join("")}<th>MEAN</th>
          <th class="l">#ds</th><th class="l">status</th>
        </tr></thead><tbody>${body}</tbody></table></div>`);
    }
  }
  return `<div class="card">
    <h2>Per-budget technique comparison — ${n} table${n===1?"":"s"}</h2>
    ${constsLine(b)}
    <p class="note">One table per (model × budget): every technique that ran at that
    budget — <code>andpro</code>, <code>critical</code>, <code>causal_forward_only</code>
    and each <code>lorekv-*</code> cell — as rows, every dataset as columns. Tables run
    base-model budgets first, then the same budgets for the variants whose folder name
    carries <code>-llama</code>. <code>M</code> is the lookahead particle count (blank for
    the methods that have none) and <code>red</code> the redundancy weight; a technique that
    ran at several M gets one row per M. The boxed value is the best in that column
    <i>within this table</i>, so every comparison here is at a fixed budget and model.
    Rows flagged ⚠ are partial — see <code>#ds</code>; italic rows are experiments that
    ran but carry no score yet, kept in place so a launched cell of the grid is never
    an invisible hole (turn them off in the filter bar).</p>
    ${tables.join("") || "<p class='note'>nothing under the current filters</p>"}</div>`;
}

/* ---------- view: full grid ---------- */
function viewGrid(bm){
  const b = D.benchmarks[bm], s = S[bm];
  let runs = runsOf(bm).slice();
  const k = s.sort.key, d = s.sort.dir;
  runs.sort((x,y)=>{
    const gv = r => k==="MEAN" ? r.mean : (k in r ? r[k] : (r.scores[k]===undefined?null:r.scores[k]));
    let a = gv(x), c = gv(y);
    if(a===null||a===undefined) return 1;
    if(c===null||c===undefined) return -1;
    if(typeof a==="number"&&typeof c==="number") return (a-c)*d;
    return String(a).localeCompare(String(c))*d;
  });
  const cols = ["technique"].concat(b.gridCols);
  const range = {};
  for(const ds of b.datasets.concat(["MEAN"])){
    const vs = runs.map(r=>ds==="MEAN"?r.mean:r.scores[ds]).filter(v=>v!==undefined&&v!==null);
    range[ds] = vs.length ? [Math.min(...vs), Math.max(...vs)] : [0,0];
  }
  const th = c => {
    const cls = s.sort.key===c ? (d>0?"sorted-a":"sorted-d") : "";
    return `<th class="sortable ${cls} ${["technique","model","cell","brule","sel"].includes(c)?"l":""}" data-bm="${bm}" data-sort="${esc(c)}">${esc(c)}</th>`;
  };
  const ncol = cols.length + b.datasets.length + 3;
  const head = cols.map(th).join("") + b.datasets.map(th).join("") + th("MEAN") +
               `<th class="l">#ds</th><th class="l">status</th>`;
  const rowHtml = (r, sep) => {
    const hy = cols.map(c=>{
      const v = c==="technique"?r.technique:r[c];
      const l = ["technique","model","cell","brule","sel"].includes(c)?" l":"";
      return `<td class="${l.trim()||""}">${v===null||v===undefined?'<span class="dim">–</span>':esc(fmt(v))}</td>`;
    }).join("");
    const sc = b.datasets.map(ds=>{
      const v = r.scores[ds];
      return v===undefined ? `<td class="dim">–</td>`
                           : `<td style="${heat(v,range[ds][0],range[ds][1])}">${v.toFixed(2)}</td>`;
    }).join("");
    const mean = r.mean===null ? `<td class="dim">–</td>`
      : `<td style="${heat(r.mean,range.MEAN[0],range.MEAN[1])}"><b>${r.mean.toFixed(2)}</b></td>`;
    const cls = (sep?"bsep ":"") + (r.status==="done"?"":"unscored");
    return `<tr class="${cls.trim()}" title="${esc(r.folder)}">${hy}${sc}${mean}`
         + `<td class="l ${r.complete?"":"dim"}">${r.n}${r.status==="done"&&!r.complete?" ⚠":""}</td>`
         + `<td class="l">${badge(r)}</td></tr>`;
  };

  let rows;
  if(s.groupRows){
    // One block per config identity (technique @ model). Its budgets are separate
    // pred/ folders that differ only in _B_<budget>, so the block header names the
    // config once and lists which of B = 128 / 256 / 512 / 1024 it actually holds --
    // a missing budget is then visible as a gap rather than as a missing row.
    const groups = new Map();
    for(const r of runs){
      const key = label(r);
      if(!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    const all = b.budgets;
    rows = [...groups.keys()].sort().map((key, gi)=>{
      const rs = groups.get(key).slice().sort((x,y)=>
        (x.budget??-1)-(y.budget??-1) || (x.M??-1)-(y.M??-1));
      const have = [...new Set(rs.map(r=>r.budget).filter(v=>v!==null))].sort((a,c)=>a-c);
      const miss = all.filter(x=>!have.includes(x));
      const cfg = rs[0].mode==="causal_lookahead"
        ? `config/${rs[0].technique}${rs[0].model.startsWith("Llama")?"-llama":""}.conf`
        : `mode ${rs[0].mode}`;
      const budgets = have.length
        ? `B = ${have.map(num).join(", ")} (${have.length===1?"one pred/ folder":have.length+" pred/ folders, one per budget"}, same config)`
        : "no budget parsed";
      const gap = gi ? `<tr class="gap"><td colspan="${ncol}"></td></tr>` : "";
      const nDone = rs.filter(r=>r.status==="done").length;
      const head = `<tr class="ghead"><td colspan="${ncol}">${esc(key)} · ${rs.length} run${rs.length===1?"":"s"}`
        + (nDone===rs.length?"":` (<b>${nDone} scored</b>, ${rs.length-nDone} unscored)`)
        + ` <span class="bl">— ${budgets}`
        + (miss.length?` · MISSING B = ${miss.map(num).join(", ")}`:"")
        + ` · ${esc(cfg)}</span></td></tr>`;
      let prevB = null;
      const body = rs.map(r=>{
        const sep = prevB !== null && r.budget !== prevB;   // thin rule at each budget change
        prevB = r.budget;
        return rowHtml(r, sep);
      }).join("");
      return gap + head + body;
    }).join("");
  } else {
    rows = runs.map(r=>rowHtml(r, false)).join("");
  }

  return `<div class="card"><h2>Full grid — ${runs.length} run(s), one row per pred/ folder</h2>
    ${constsLine(b)}
    <p class="note">Every hyperparameter the folder name encodes, then the score on every
    dataset. MEAN averages only the datasets a run actually has, so rows flagged
    ⚠ in <code>#ds</code> are partial and their MEAN is not comparable with a full sweep.
    Click a header to sort; hover a row for its folder name. Italic rows are launched
    experiments that carry no score yet — <code>status</code> says why and how far they got.
    ${s.groupRows?"Rows are blocked by config identity — the four budgets of one config are separate pred/ folders and sit together under one header, separated by a rule at each budget change.":"Rows are flat; turn on <b>group the budgets of one config</b> above to block them by config."}</p>
    <div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

/* ---------- view: dataset x M x budget matrix ---------- */
function viewMatrix(bm){
  const b = D.benchmarks[bm], s = S[bm];
  const runs = runsOf(bm);
  const budgets = b.budgets.filter(x=>s.budgets.has(x));
  const Ms = b.Ms.filter(x=>s.Ms.has(x));
  const hasBase = runs.some(r=>r.M===null);
  const MsCol = Ms.concat(hasBase?[null]:[]);          // null = the no-lookahead baselines
  const outer = s.group==="budget" ? budgets : MsCol;
  const inner = s.group==="budget" ? MsCol : budgets;

  // (technique@model) x (budget,M) -> run, keeping the run with the most datasets
  const cell = {};
  for(const r of runs){
    const key = label(r)+"|"+r.budget+"|"+r.M;
    if(!cell[key] || r.n > cell[key].n) cell[key] = r;
  }
  const rowKeys = [...new Set(runs.map(label))].sort();
  // An empty cell has two very different meanings: nothing was ever launched here
  // ("–"), or a run exists and has not produced a score yet ("·"). Never conflate them.
  const miss = (rk,bud,m) => {
    const r = cell[rk+"|"+bud+"|"+m];
    return r ? `<span title="${esc(r.folder)} — ${esc((STATUS[r.status]||["",r.status])[1])}">·</span>` : "–";
  };
  const baseSel = `<select data-bm="${bm}" data-key="baseline">
      <option value="">absolute score</option>
      ${rowKeys.map(k=>`<option value="${esc(k)}"${s.baseline===k?" selected":""}>Δ vs ${esc(k)}</option>`).join("")}
    </select>`;
  const groupSel = `<select data-bm="${bm}" data-key="group">
      <option value="budget"${s.group==="budget"?" selected":""}>group columns by budget, then M</option>
      <option value="M"${s.group==="M"?" selected":""}>group columns by M, then budget</option>
    </select>`;

  const blocks = b.datasets.map(ds=>{
    const val = (rk,bud,m) => {
      const r = cell[rk+"|"+bud+"|"+m];
      const v = r ? r.scores[ds] : undefined;
      return v===undefined ? null : v;
    };
    // is there anything at all for this dataset?
    let any = false;
    for(const rk of rowKeys) for(const o of outer) for(const i of inner){
      const [bud,m] = s.group==="budget" ? [o,i] : [i,o];
      if(val(rk,bud,m)!==null){ any = true; break; }
    }
    if(!any) return "";

    const head1 = `<th class="l" rowspan="2">technique @ model</th>` + outer.map((o,gi)=>
      (gi?`<th class="sp" rowspan="2"></th>`:"") +
      `<th class="gsep gend" colspan="${inner.length}">${s.group==="budget"?"B = "+num(o):(o===null?"no lookahead":"M = "+num(o))}</th>`).join("");
    const head2 = outer.map(()=>inner.map((i,ii)=>
      `<th class="${ii===0?"gsep":""}${ii===inner.length-1?" gend":""}">${s.group==="budget"?(i===null?"base":"M="+num(i)):"B="+num(i)}</th>`).join("")).join("");

    // per-column best + scale
    const colVals = [];
    outer.forEach(o=>inner.forEach(i=>{
      const [bud,m] = s.group==="budget" ? [o,i] : [i,o];
      colVals.push(rowKeys.map(rk=>val(rk,bud,m)).filter(v=>v!==null));
    }));
    const allV = colVals.flat();
    const lo = allV.length?Math.min(...allV):0, hi = allV.length?Math.max(...allV):0;
    let maxAbs = 0;
    if(s.baseline){
      outer.forEach(o=>inner.forEach(i=>{
        const [bud,m] = s.group==="budget" ? [o,i] : [i,o];
        const base = val(s.baseline,bud,m);
        if(base===null) return;
        rowKeys.forEach(rk=>{ const v=val(rk,bud,m); if(v!==null) maxAbs = Math.max(maxAbs, Math.abs(v-base)); });
      }));
    }

    const body = rowKeys.map(rk=>{
      let ci = 0;
      const tds = outer.map((o,gi)=>(gi?`<td class="sp"></td>`:"") + inner.map((i,ii)=>{
        const [bud,m] = s.group==="budget" ? [o,i] : [i,o];
        const edge = (ii===0?"gsep ":"") + (ii===inner.length-1?"gend ":"");
        const v = val(rk,bud,m);
        const best = colVals[ci].length ? Math.max(...colVals[ci]) : null;
        ci++;
        if(v===null) return `<td class="${edge}dim">${miss(rk,bud,m)}</td>`;
        if(s.baseline){
          const base = val(s.baseline,bud,m);
          if(base===null) return `<td class="${edge}dim" title="baseline has no score here">${v.toFixed(2)}</td>`;
          const dv = v - base;
          const txt = (dv>0?"+":"") + dv.toFixed(2);
          return `<td class="${edge}" style="${diverge(dv,maxAbs)}">${rk===s.baseline?`<span class="dim">${v.toFixed(2)}</span>`:txt}</td>`;
        }
        const isBest = best!==null && Math.abs(v-best)<1e-9;
        return `<td class="${edge}${isBest?"best":""}" style="${heat(v,lo,hi)}">${v.toFixed(2)}</td>`;
      }).join("")).join("");
      return `<tr><td class="l">${esc(rk)}</td>${tds}</tr>`;
    }).join("");

    return `<h3>${esc(ds)}</h3><div class="scroll"><table>
      <thead><tr>${head1}</tr><tr>${head2}</tr></thead><tbody>${body}</tbody></table></div>`;
  }).join("");

  // the same matrix on the run MEAN
  const meanBlock = (()=>{
    const val = (rk,bud,m) => { const r = cell[rk+"|"+bud+"|"+m]; return r&&r.mean!==null?r.mean:null; };
    const head1 = `<th class="l" rowspan="2">technique @ model</th>` + outer.map((o,gi)=>
      (gi?`<th class="sp" rowspan="2"></th>`:"") +
      `<th class="gsep gend" colspan="${inner.length}">${s.group==="budget"?"B = "+num(o):(o===null?"no lookahead":"M = "+num(o))}</th>`).join("");
    const head2 = outer.map(()=>inner.map((i,ii)=>
      `<th class="${ii===0?"gsep":""}${ii===inner.length-1?" gend":""}">${s.group==="budget"?(i===null?"base":"M="+num(i)):"B="+num(i)}</th>`).join("")).join("");
    const cv=[]; outer.forEach(o=>inner.forEach(i=>{
      const [bud,m] = s.group==="budget" ? [o,i] : [i,o];
      cv.push(rowKeys.map(rk=>val(rk,bud,m)).filter(v=>v!==null)); }));
    const allV = cv.flat(); const lo=allV.length?Math.min(...allV):0, hi=allV.length?Math.max(...allV):0;
    const body = rowKeys.map(rk=>{
      let ci=0;
      const tds = outer.map((o,gi)=>(gi?`<td class="sp"></td>`:"") + inner.map((i,ii)=>{
        const [bud,m] = s.group==="budget" ? [o,i] : [i,o];
        const edge = (ii===0?"gsep ":"") + (ii===inner.length-1?"gend ":"");
        const v = val(rk,bud,m); const best = cv[ci].length?Math.max(...cv[ci]):null; ci++;
        if(v===null) return `<td class="${edge}dim">${miss(rk,bud,m)}</td>`;
        const r = cell[rk+"|"+bud+"|"+m];
        const warn = r && !r.complete ? ` title="partial: ${r.n}/${b.datasets.length} datasets"` : "";
        const isBest = best!==null && Math.abs(v-best)<1e-9;
        return `<td class="${edge}${isBest?"best":""}"${warn} style="${heat(v,lo,hi)}">${v.toFixed(2)}${r&&!r.complete?"<span class='dim'>*</span>":""}</td>`;
      }).join("")).join("");
      return `<tr><td class="l">${esc(rk)}</td>${tds}</tr>`;
    }).join("");
    return `<h3>MEAN over the datasets each run has</h3><div class="scroll"><table>
      <thead><tr>${head1}</tr><tr>${head2}</tr></thead><tbody>${body}</tbody></table></div>
      <p class="note">* marks a partial run — its mean covers fewer datasets than the full
      sweep and is not comparable. Use <b>Only complete runs</b> above to drop them.</p>`;
  })();

  return `<div class="card">
    <h2>Dataset-level comparison — technique × budget × M</h2>
    <p class="note">One table per dataset. Rows are techniques (identity + model);
    columns cross KV budget B with lookahead particle count M, and each budget block
    is set off by a spacer column so B = 128 / 256 / 512 / 1024 read as four separate
    blocks — one pred/ folder per budget, all from the same config. The boxed cell is
    the best technique in that column. Runs with no lookahead (andpro / critical /
    causal_forward_only) have no M and sit in the <code>base</code> column.
    A cell reads <code>·</code> when a run was launched there but has produced no score
    yet, and <code>–</code> when nothing was ever launched.
    ${groupSel} &nbsp; ${baseSel}</p>
    ${meanBlock}${blocks}</div>`;
}

/* ---------- view: charts ---------- */
function viewCharts(bm){
  const b = D.benchmarks[bm], s = S[bm];
  const runs = runsOf(bm);
  const budgets = b.budgets.filter(x=>s.budgets.has(x));
  const Ms = b.Ms.filter(x=>s.Ms.has(x));
  const cell = {};
  for(const r of runs){
    const key = label(r)+"|"+r.budget+"|"+r.M;
    if(!cell[key] || r.n > cell[key].n) cell[key] = r;
  }
  const rowKeys = [...new Set(runs.map(label))].sort();
  const colorOf = {}; rowKeys.forEach((k,i)=>colorOf[k]=PALETTE[i%PALETTE.length]);
  const xIsBudget = s.xaxis==="budget";
  const xs = (xIsBudget?budgets:Ms).map(num);
  const facets = xIsBudget?Ms:budgets;
  const sel = `<select data-bm="${bm}" data-key="xaxis">
      <option value="budget"${xIsBudget?" selected":""}>x = KV budget B (one chart per M)</option>
      <option value="M"${!xIsBudget?" selected":""}>x = particles M (one chart per B)</option>
    </select>`;

  const charts = [];
  for(const ds of ["MEAN"].concat(b.datasets)){
    const get = (rk,bud,m) => {
      const r = cell[rk+"|"+bud+"|"+m];
      if(!r) return null;
      const v = ds==="MEAN" ? r.mean : r.scores[ds];
      return v===undefined||v===null ? null : v;
    };
    for(const f of facets){
      const series = [];
      for(const rk of rowKeys){
        const pts = (xIsBudget?budgets:Ms).map(x=>{
          const [bud,m] = xIsBudget ? [x,f] : [f,x];
          return {x, y:get(rk,bud,m)};
        });
        if(pts.some(p=>p.y!==null)) series.push({name:rk, color:colorOf[rk], pts, flat:null});
      }
      // baselines (no M) as dashed reference lines when x = M
      if(!xIsBudget){
        for(const rk of rowKeys){
          const v = get(rk,f,null);
          if(v!==null && !series.some(x=>x.name===rk))
            series.push({name:rk+" (no lookahead)", color:colorOf[rk], pts:[], flat:v});
        }
      }
      if(!series.length) continue;
      const title = ds + " — " + (xIsBudget ? "M = "+num(f) : "B = "+num(f));
      charts.push(lineChart(title, xs, series, "score"));
    }
  }
  return `<div class="card"><h2>Charts</h2>
    <p class="note">One small multiple per (dataset × facet); a line per technique.
    Dashed lines are runs without lookahead, which have no M. ${sel}</p>
    <div class="charts">${charts.join("")||"<p class='note'>nothing to plot under the current filters</p>"}</div></div>`;
}

/* ---------- view: sub-tasks ---------- */
function viewSub(bm){
  const b = D.benchmarks[bm], s = S[bm];
  const dss = Object.keys(b.subtasks);
  if(!dss.length) return `<div class="card"><h2>Sub-tasks</h2>
    <p class="note">This benchmark's result.json has no per-dataset sub-task breakdown.</p></div>`;
  const ds = b.subtasks[s.subDataset] ? s.subDataset : dss[0];
  const tasks = b.subtasks[ds];
  const sel = `<select data-bm="${bm}" data-key="subDataset">
      ${dss.map(d=>`<option value="${esc(d)}"${d===ds?" selected":""}>${esc(d)}</option>`).join("")}</select>`;
  const runs = runsOf(bm).filter(r=>r.sub[ds]);
  const range = {};
  for(const t of tasks){
    const vs = runs.map(r=>r.sub[ds][t]).filter(v=>v!==undefined&&v!==null);
    range[t] = vs.length?[Math.min(...vs),Math.max(...vs)]:[0,0];
  }
  const rows = runs.map(r=>{
    const tds = tasks.map(t=>{
      const v = r.sub[ds][t];
      return (v===undefined||v===null) ? `<td class="dim">–</td>`
        : `<td style="${heat(v,range[t][0],range[t][1])}">${v.toFixed(2)}</td>`;
    }).join("");
    return `<tr><td class="l">${esc(label(r))}</td><td>${num(r.budget)}</td><td>${num(r.M)}</td>${tds}</tr>`;
  }).join("");
  return `<div class="card"><h2>Sub-task breakdown — ${esc(ds)}</h2>
    <p class="note">The per-task numbers behind the dataset score. Dataset: ${sel}</p>
    <div class="scroll"><table><thead><tr>
      <th class="l">technique @ model</th><th>B</th><th>M</th>
      ${tasks.map(t=>`<th>${esc(t)}</th>`).join("")}
    </tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

/* ---------- view: NIAH heatmaps ---------- */
function viewNiah(bm){
  const runs = runsOf(bm).filter(r=>r.niah&&r.niah.grid);
  if(!runs.length) return `<div class="card"><h2>NIAH heatmaps</h2>
    <p class="note">No run under the current filters stores a depth × length grid.</p></div>`;
  const cards = runs.map(r=>{
    const g = r.niah;
    const rows = (g.grid||[]).map((row,di)=>{
      const tds = row.map(v=>{
        const pct = v===null||v===undefined?null:(v<=1?v*100:v);
        return pct===null?`<td class="dim"></td>`
          :`<td style="${heat(pct,0,100)}">${Math.round(pct)}</td>`;
      }).join("");
      return `<tr><td class="l">${fmt(g.depths[di])}%</td>${tds}</tr>`;
    }).join("");
    return `<div class="chart"><h4>${esc(label(r))} — B=${num(r.budget)} M=${num(r.M)}</h4>
      <table class="hm"><thead><tr><th class="l">depth</th>
        ${(g.lengths||[]).map(l=>`<th>${esc(l)}</th>`).join("")}</tr></thead>
        <tbody>${rows}</tbody></table>
      <p class="note">metric: ${esc(g.metric)} · overall ${r.mean===null?"–":r.mean.toFixed(2)}</p></div>`;
  }).join("");
  return `<div class="card"><h2>NIAH heatmaps — needle depth × context length</h2>
    <div class="charts">${cards}</div></div>`;
}

/* ---------- view: inventory ---------- */
function viewInv(bm){
  const b = D.benchmarks[bm];
  const iss = D.issues.filter(i=>i.benchmark===bm);
  const partial = b.runs.filter(r=>r.status==="done"&&r.n>0&&!r.complete)
    .sort((a,c)=>a.n-c.n)
    .map(r=>{
      const missing = b.datasets.filter(d=>r.scores[d]===undefined);
      return `<tr><td class="l">${esc(r.folder)}</td><td>${r.n}/${b.datasets.length}</td>
        <td class="l">${esc(missing.join(", "))}</td></tr>`;
    }).join("");
  const issRows = iss.map(i=>`<tr><td class="l">${esc(i.folder)}</td>`
    + `<td class="l">${i.status?`<span class="badge ${(STATUS[i.status]||["pending"])[0]}">${esc(i.status)}</span>`:""}</td>`
    + `<td class="l">${esc(i.problem)}</td></tr>`).join("");
  return `<div class="card"><h2>Inventory — what is missing</h2>
    <h3>Problems (${iss.length})</h3>
    ${iss.length?`<div class="scroll"><table><thead><tr><th class="l">folder</th><th class="l">status</th><th class="l">problem</th></tr></thead>
      <tbody>${issRows}</tbody></table></div>`:`<p class="note">None — every pred/ folder has a readable, parseable result.json.</p>`}
    <h3>Partial runs — scored, but not on every dataset (${b.runs.filter(r=>r.status==="done"&&r.n>0&&!r.complete).length})</h3>
    ${partial?`<div class="scroll"><table><thead><tr><th class="l">folder</th><th>coverage</th><th class="l">missing datasets</th></tr></thead>
      <tbody>${partial}</tbody></table></div>`:`<p class="note">None — every run covers all ${b.datasets.length} datasets.</p>`}
  </div>`;
}

/* ---------- overview ---------- */
function viewOverview(){
  const cards = BMS.map(bm=>{
    const b = D.benchmarks[bm];
    const complete = b.runs.filter(r=>r.complete).length;
    const sc = b.statusCounts;
    const unscored = sc["pending"] + sc["no-scores"] + sc["unreadable"];
    return `<div class="stat"><b>${b.runs.length}</b><span>${esc(bm)} pred/ folders<br>
      <b style="font-size:11px">${sc["done"]} scored</b> (${complete} on all datasets)
      · ${unscored} unscored<br>
      ${b.techniques.length} techniques · ${b.models.length} models<br>
      B ∈ {${b.budgets.map(num).join(", ")}}<br>
      M ∈ {${b.Ms.map(num).join(", ")}}<br>
      ${b.datasets.length} datasets
      ${Object.keys(b.constants).length?`<br>const: ${Object.entries(b.constants).map(([k,v])=>k+"="+v.value+(v.everywhere?"":"*")).join(", ")}`:""}
      </span></div>`;
  }).join("");

  const tops = BMS.map(bm=>{
    const b = D.benchmarks[bm];
    const rows = b.runs.filter(r=>r.complete&&r.mean!==null)
      .sort((a,c)=>c.mean-a.mean).slice(0,10)
      .map((r,i)=>`<tr><td>${i+1}</td><td class="l">${esc(r.technique)}</td><td class="l">${esc(r.model)}</td>
        <td>${num(r.budget)}</td><td>${num(r.M)}</td><td><b>${r.mean.toFixed(2)}</b></td></tr>`).join("");
    if(!rows) return `<h3>${esc(bm)}</h3><p class="note">No complete run yet — every
      folder covers a subset of the ${b.datasets.length} datasets. See its Inventory tab.</p>`;
    return `<h3>${esc(bm)} — best complete runs by MEAN</h3><div class="scroll"><table><thead><tr>
      <th>#</th><th class="l">technique</th><th class="l">model</th><th>B</th><th>M</th><th>MEAN</th>
      </tr></thead><tbody>${rows}</tbody></table></div>`;
  }).join("");

  return `<div class="card"><h2>Overview</h2>
    <p class="note">Read from <code>${esc(D.repo)}/experiments/&lt;benchmark&gt;/pred/&lt;run&gt;/result.json</code>.
    Folder names are parsed with <code>scripts/run_naming.py</code>, the same module
    <code>run.sh</code> uses to write them, so the hyperparameters shown are the ones
    actually run. Only runs covering every dataset are ranked below.</p>
    <div class="stats">${cards}</div>${tops}</div>
    <div class="card"><h2>All problems (${D.issues.length})</h2>
    ${D.issues.length?`<div class="scroll"><table><thead><tr><th class="l">benchmark</th>
      <th class="l">folder</th><th class="l">status</th><th class="l">problem</th></tr></thead><tbody>
      ${D.issues.map(i=>`<tr><td class="l">${esc(i.benchmark)}</td><td class="l">${esc(i.folder)}</td>
        <td class="l">${i.status?`<span class="badge ${(STATUS[i.status]||["pending"])[0]}">${esc(i.status)}</span>`:""}</td>
        <td class="l">${esc(i.problem)}</td></tr>`).join("")}</tbody></table></div>`
      :`<p class="note">None.</p>`}</div>`;
}

/* ---------- shell ---------- */
function render(){
  document.getElementById("hdr").textContent =
    `${D.repo} · generated ${D.generated} · ` +
    BMS.map(b=>`${b}: ${D.benchmarks[b].runs.length} runs`).join(" · ");
  document.getElementById("tabs").innerHTML =
    [["overview","Overview"]].concat(BMS.map(b=>[b,b]))
      .map(([k,t])=>`<div class="tab${TAB===k?" on":""}" data-tab="${esc(k)}">${esc(t)}</div>`).join("");
  const body = document.getElementById("body");
  if(TAB==="overview"){ body.innerHTML = viewOverview(); return; }
  const s = S[TAB];
  const v = {budget:viewBudget, grid:viewGrid, matrix:viewMatrix, charts:viewCharts,
             sub:viewSub, niah:viewNiah, inv:viewInv}[s.view] || viewBudget;
  body.innerHTML = filterBar(TAB) + v(TAB);
}

document.addEventListener("click", e=>{
  const tab = e.target.closest(".tab");
  if(tab){ TAB = tab.dataset.tab; render(); return; }
  const th = e.target.closest("th.sortable");
  if(th){
    const s = S[th.dataset.bm];
    s.sort = s.sort.key===th.dataset.sort ? {key:s.sort.key,dir:-s.sort.dir}
                                          : {key:th.dataset.sort,dir:1};
    render(); return;
  }
  const opt = e.target.closest(".opt");
  if(!opt) return;
  const s = S[opt.dataset.bm], key = opt.dataset.key, raw = opt.dataset.val;
  if(key==="view"){ s.view = raw; }
  else if(key==="completeOnly"||key==="groupRows"||key==="showUnscored"){ s[key] = !s[key]; }
  else {
    const set = s[key];
    const val = (key==="models"||key==="techs") ? raw : Number(raw);
    if(set.has(val)) set.delete(val); else set.add(val);
  }
  render();
});
document.addEventListener("change", e=>{
  const sel = e.target.closest("select");
  if(!sel||!sel.dataset.bm) return;
  S[sel.dataset.bm][sel.dataset.key] = sel.value;
  render();
});
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
