#!/usr/bin/env python3
"""
Interactive web app for visualizing log-likelihood dynamics from checkpoint summaries.

Usage:
    cd interactive_app
    python app.py
    # Then open http://localhost:8080 in your browser

Dependencies:
    pip install flask numpy
"""

import json
import re
import fnmatch
from pathlib import Path
from collections import defaultdict

import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "prob_tracking" / "results"

SOURCE_COLORS = {
    "base":    "#1f77b4",
    "qwen":    "#ff7f0e",
    "gemma":   "#2ca02c",
    "highest": "#d62728",
    "lowest":  "#9467bd",
}

PATTERN_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def compute_avg_curve(summaries: list, pattern: str) -> tuple:
    """
    For a single fnmatch pattern, compute per-step (mean, std) across all
    matching IDs in all loaded summaries.  Mirrors visualize_avg_llh_by_pattern.py.
    """
    if not summaries:
        return [], [], []
    reference_steps = summaries[0][1]["steps"]
    n_steps = len(reference_steps)
    values_per_step: list = [[] for _ in range(n_steps)]

    for _, data in summaries:
        results = data.get("results_by_type", {})
        matched = [k for k in results if fnmatch.fnmatch(k, pattern)]
        for key in matched:
            series = results[key]["avg_log_likelihood"]
            for i in range(min(n_steps, len(series))):
                if series[i] is not None:
                    values_per_step[i].append(series[i])

    means, stds, valid_steps = [], [], []
    for step, vals in zip(reference_steps, values_per_step):
        if vals:
            arr = np.array(vals)
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
            valid_steps.append(step)

    return valid_steps, means, stds

# ---------------------------------------------------------------------------
# Data helpers (mirror logic from visualize_llh_dynamics.py)
# ---------------------------------------------------------------------------

def detect_source(type_id: str) -> str:
    type_id_lower = type_id.lower()
    for source in ("lowest", "highest", "gemma", "qwen", "base"):
        if source in type_id_lower:
            return source
    return "unknown"


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_by_id_patterns(summary: dict, patterns: list) -> dict:
    filtered = {
        k: v for k, v in summary["results_by_type"].items()
        if any(fnmatch.fnmatch(k, p) for p in patterns)
    }
    return {**summary, "results_by_type": filtered}


def filter_by_questions(summary: dict, question_ids: list) -> dict:
    prefixes = tuple(f"id_{q_id}_" for q_id in question_ids)
    filtered = {
        k: v for k, v in summary["results_by_type"].items()
        if k.startswith(prefixes)
    }
    return {**summary, "results_by_type": filtered}


def parse_type_string(type_str: str):
    m = re.match(r"id_(\d+)_(base|cp(\d+))_(correct|incorrect)", type_str)
    if m:
        question_id = int(m.group(1))
        source = "base" if m.group(2) == "base" else m.group(3)
        correctness = m.group(4)
        return question_id, source, correctness
    return None, None, None


def organize_by_question(summaries: list) -> dict:
    """Mirrors organize_by_question from visualize_llh_dynamics.py."""
    if isinstance(summaries, dict):
        summaries = [("", summaries)]

    def source_to_step(source):
        return 0 if source == "base" else int(source)

    all_organized = {}

    for _, summary in summaries:
        steps = summary["steps"]
        results_by_type = summary["results_by_type"]

        questions = defaultdict(lambda: {
            "llh_by_step": defaultdict(list),
            "correctness_by_source": {},
        })

        for type_str, metrics in results_by_type.items():
            question_id, source, correctness = parse_type_string(type_str)
            if question_id is None:
                continue
            avg_llh = metrics["avg_log_likelihood"]
            for i, step in enumerate(steps):
                if i < len(avg_llh) and avg_llh[i] is not None:
                    questions[question_id]["llh_by_step"][step].append(avg_llh[i])
            src_step = source_to_step(source)
            questions[question_id]["correctness_by_source"][src_step] = correctness

        for q_id, data in questions.items():
            sorted_steps = sorted(data["llh_by_step"].keys())
            avg_llhs = [float(np.mean(data["llh_by_step"][s])) for s in sorted_steps]

            correctness_by_source = data["correctness_by_source"]
            correctness_list = []
            for s in sorted_steps:
                if s in correctness_by_source:
                    correctness_list.append(correctness_by_source[s])
                else:
                    earlier = [src for src in correctness_by_source if src <= s]
                    correctness_list.append(
                        correctness_by_source[max(earlier)] if earlier else "unknown"
                    )

            all_organized[q_id] = {
                "steps": sorted_steps,
                "avg_llh": avg_llhs,
                "correctness": correctness_list,
            }

    return all_organized


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/files")
def list_files():
    if not RESULTS_DIR.exists():
        return jsonify([])
    files = sorted(f.name for f in RESULTS_DIR.glob("*_summary.json"))
    return jsonify(files)


@app.route("/api/data")
def get_data():
    files      = request.args.getlist("files")
    plot_type  = request.args.get("plot_type", "llh")
    questions  = request.args.get("questions", "").strip()
    id_pats    = request.args.getlist("id_patterns")
    color_by   = request.args.get("color_by", "default")

    if not files:
        return jsonify({"error": "No files selected"}), 400

    summaries = []
    for fname in files:
        path = RESULTS_DIR / fname
        if not path.exists():
            return jsonify({"error": f"File not found: {fname}"}), 404
        summaries.append((Path(fname).stem, load_summary(path)))

    if questions:
        requested_ids = [int(q.strip()) for q in questions.split(",") if q.strip()]
        summaries = [(lbl, filter_by_questions(s, requested_ids)) for lbl, s in summaries]

    if id_pats:
        summaries = [(lbl, filter_by_id_patterns(s, id_pats)) for lbl, s in summaries]

    # ---- LLH / Perplexity ----
    if plot_type in ("llh", "perplexity"):
        metric_key = "avg_log_likelihood" if plot_type == "llh" else "avg_perplexity"
        traces = []

        for label, summary in summaries:
            steps = summary["steps"]
            for sample_type, metrics in summary["results_by_type"].items():
                vals = metrics[metric_key]
                valid_idx = [i for i, v in enumerate(vals) if v is not None]
                xs = [steps[i] for i in valid_idx]
                ys = [vals[i] for i in valid_idx]
                if not ys:
                    continue

                display = f"{label} / {sample_type}" if label else sample_type

                if color_by == "source":
                    color = SOURCE_COLORS.get(detect_source(sample_type), "#7f7f7f")
                elif color_by == "correctness":
                    color = "#d62728" if "_incorrect" in sample_type else "#2ca02c"
                else:
                    color = None

                trace = {
                    "x": xs, "y": ys,
                    "name": display,
                    "mode": "lines+markers",
                    "type": "scatter",
                    "hovertemplate": (
                        f"<b>{display}</b><br>"
                        "Step: %{x}<br>"
                        "Value: %{y:.4f}<extra></extra>"
                    ),
                }
                if color:
                    trace["line"] = {"color": color}
                    trace["marker"] = {"color": color}
                traces.append(trace)

        ylabel = "Average Log-Likelihood" if plot_type == "llh" else "Average Perplexity"
        return jsonify({
            "traces": traces,
            "title": f"{ylabel} Dynamics Across Training",
            "xlabel": "Training Step",
            "ylabel": ylabel,
        })

    # ---- Per-question ----
    elif plot_type == "questions":
        q_data = organize_by_question(summaries)
        traces = []

        for q_id, data in sorted(q_data.items()):
            steps       = data["steps"]
            avg_llh     = data["avg_llh"]
            correctness = data["correctness"]

            if color_by == "correctness":
                final_ok = correctness[-1] == "correct" if correctness else True
                color = "#2ca02c" if final_ok else "#d62728"
            else:
                color = None

            trace = {
                "x": steps, "y": avg_llh,
                "name": f"Q{q_id}",
                "mode": "lines+markers",
                "type": "scatter",
                "customdata": correctness,
                "hovertemplate": (
                    f"<b>Q{q_id}</b><br>"
                    "Step: %{x}<br>"
                    "LLH: %{y:.4f}<br>"
                    "Correctness: %{customdata}<extra></extra>"
                ),
            }
            if color:
                trace["line"] = {"color": color}
                trace["marker"] = {"color": color}
            traces.append(trace)

            # Overlay X markers for incorrect points (when not coloring by correctness)
            if color_by != "correctness":
                inc_steps = [s for s, c in zip(steps, correctness) if c == "incorrect"]
                inc_llh   = [l for l, c in zip(avg_llh, correctness) if c == "incorrect"]
                if inc_steps:
                    traces.append({
                        "x": inc_steps, "y": inc_llh,
                        "mode": "markers",
                        "type": "scatter",
                        "marker": {"symbol": "x", "color": "red", "size": 10},
                        "name": f"Q{q_id} (incorrect)",
                        "showlegend": False,
                        "hovertemplate": (
                            f"<b>Q{q_id} — incorrect</b><br>"
                            "Step: %{x}<br>LLH: %{y:.4f}<extra></extra>"
                        ),
                    })

        return jsonify({
            "traces": traces,
            "title": "Log-Likelihood Dynamics by Question",
            "xlabel": "Training Step",
            "ylabel": "Average Log-Likelihood",
        })

    # ---- Avg log-likelihood by pattern ----
    elif plot_type == "avg_by_pattern":
        avg_patterns  = request.args.getlist("avg_patterns")
        avg_notations = request.args.getlist("avg_notations")
        shade_std     = request.args.get("shade_std", "true").lower() != "false"

        if not avg_patterns:
            return jsonify({"error": "No patterns provided for avg_by_pattern mode"}), 400
        if len(avg_patterns) != len(avg_notations):
            return jsonify({"error": "avg_patterns and avg_notations counts must match"}), 400

        traces = []
        for idx, (pattern, notation) in enumerate(zip(avg_patterns, avg_notations)):
            color = PATTERN_COLORS[idx % len(PATTERN_COLORS)]
            steps, means, stds = compute_avg_curve(summaries, pattern)
            if not steps:
                continue

            if shade_std:
                upper = [m + s for m, s in zip(means, stds)]
                lower = [m - s for m, s in zip(means, stds)]
                fill_color = hex_to_rgba(color, 0.15)
                # Upper bound (invisible line — Plotly fills between this and next trace)
                traces.append({
                    "x": steps, "y": upper,
                    "mode": "lines", "type": "scatter",
                    "line": {"width": 0, "color": color},
                    "showlegend": False, "hoverinfo": "skip",
                    "name": f"_upper_{notation}",
                })
                # Lower bound with fill back to upper
                traces.append({
                    "x": steps, "y": lower,
                    "mode": "lines", "type": "scatter",
                    "fill": "tonexty", "fillcolor": fill_color,
                    "line": {"width": 0, "color": color},
                    "showlegend": False, "hoverinfo": "skip",
                    "name": f"_lower_{notation}",
                })

            # Mean line
            traces.append({
                "x": steps, "y": means,
                "mode": "lines+markers", "type": "scatter",
                "name": notation,
                "line": {"color": color, "width": 2},
                "marker": {"color": color, "size": 5},
                "hovertemplate": (
                    f"<b>{notation}</b><br>"
                    "Step: %{x}<br>"
                    "Mean LLH: %{y:.4f}<extra></extra>"
                ),
            })

        return jsonify({
            "traces": traces,
            "title": "Average Log-Likelihood by Pattern",
            "xlabel": "Training Step",
            "ylabel": "Average Log-Likelihood",
        })

    return jsonify({"error": "Invalid plot_type"}), 400


# ---------------------------------------------------------------------------
# Embedded HTML (single-file app, no template folder needed)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>LLH Dynamics Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ---- Header ---- */
  header {
    background: #1a1d27;
    border-bottom: 1px solid #2d3148;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }
  header h1 { font-size: 1.1rem; font-weight: 600; color: #a5b4fc; letter-spacing: .03em; }
  header .subtitle { font-size: .8rem; color: #64748b; }

  /* ---- Layout ---- */
  .layout {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ---- Sidebar ---- */
  aside {
    width: 310px;
    min-width: 260px;
    background: #1a1d27;
    border-right: 1px solid #2d3148;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    flex-shrink: 0;
  }

  .section {
    padding: 14px 16px;
    border-bottom: 1px solid #2d3148;
  }
  .section-title {
    font-size: .7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #64748b;
    margin-bottom: 10px;
  }

  /* File list */
  #file-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 240px;
    overflow-y: auto;
  }
  #file-list li label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px;
    border-radius: 6px;
    cursor: pointer;
    font-size: .78rem;
    color: #94a3b8;
    transition: background .15s;
    word-break: break-all;
  }
  #file-list li label:hover { background: #252840; color: #e2e8f0; }
  #file-list li input[type=checkbox]:checked + span { color: #a5b4fc; font-weight: 500; }

  /* Toggle button group */
  .btn-group {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  .btn-group button {
    flex: 1;
    padding: 6px 8px;
    border: 1px solid #2d3148;
    border-radius: 6px;
    background: #252840;
    color: #94a3b8;
    font-size: .78rem;
    cursor: pointer;
    transition: background .15s, color .15s;
  }
  .btn-group button.active {
    background: #4f46e5;
    border-color: #4f46e5;
    color: #fff;
  }
  .btn-group button:hover:not(.active) { background: #2d3158; color: #e2e8f0; }

  /* Inputs */
  label.field-label {
    display: block;
    font-size: .75rem;
    color: #64748b;
    margin-bottom: 4px;
    margin-top: 10px;
  }
  input[type=text] {
    width: 100%;
    padding: 6px 10px;
    background: #0f1117;
    border: 1px solid #2d3148;
    border-radius: 6px;
    color: #e2e8f0;
    font-size: .82rem;
    outline: none;
    transition: border-color .15s;
  }
  input[type=text]:focus { border-color: #4f46e5; }
  input[type=text]::placeholder { color: #374151; }

  /* Apply button */
  #apply-btn {
    width: 100%;
    padding: 9px;
    background: #4f46e5;
    border: none;
    border-radius: 8px;
    color: #fff;
    font-size: .88rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .15s;
    margin-top: 4px;
  }
  #apply-btn:hover { background: #4338ca; }
  #apply-btn:active { background: #3730a3; }

  /* Stats panel */
  #stats-box {
    font-size: .75rem;
    color: #64748b;
    line-height: 1.6;
    padding: 4px 0;
    min-height: 20px;
  }
  #stats-box span { color: #a5b4fc; font-weight: 500; }

  /* ---- Main chart area ---- */
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #0f1117;
  }

  #chart {
    flex: 1;
    min-height: 0;
  }

  /* Loading overlay */
  #loading {
    display: none;
    position: absolute;
    inset: 0;
    background: rgba(15,17,23,.7);
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    color: #a5b4fc;
    z-index: 10;
  }
  #loading.visible { display: flex; }

  /* Error toast */
  #error-msg {
    display: none;
    padding: 10px 14px;
    background: #450a0a;
    border-left: 3px solid #ef4444;
    color: #fca5a5;
    font-size: .82rem;
    flex-shrink: 0;
  }

  /* Legend color dot helper */
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }

  /* Pattern rows */
  .pattern-row {
    display: flex;
    gap: 4px;
    margin-bottom: 6px;
    align-items: center;
  }
  .pattern-row input[type=text] { flex: 1; font-size: .76rem; padding: 5px 8px; }
  .remove-row-btn {
    flex-shrink: 0;
    width: 24px; height: 24px;
    background: #3b1212;
    border: 1px solid #6b2020;
    border-radius: 5px;
    color: #f87171;
    font-size: .75rem;
    cursor: pointer;
    line-height: 1;
  }
  .remove-row-btn:hover { background: #6b2020; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2d3148; border-radius: 3px; }
</style>
</head>
<body>

<header>
  <div>
    <h1>LLH Dynamics Explorer</h1>
    <div class="subtitle">Interactive log-likelihood &amp; perplexity visualizer</div>
  </div>
</header>

<div class="layout">

  <!-- ===== Sidebar ===== -->
  <aside>

    <!-- File selector -->
    <div class="section">
      <div class="section-title">Summary Files</div>
      <ul id="file-list"><li style="color:#555;font-size:.8rem">Loading…</li></ul>
    </div>

    <!-- Plot type -->
    <div class="section">
      <div class="section-title">Plot Type</div>
      <div class="btn-group" id="plot-type-group">
        <button data-val="llh" class="active">Log-Likelihood</button>
        <button data-val="perplexity">Perplexity</button>
        <button data-val="questions">By Question</button>
        <button data-val="avg_by_pattern">Avg by Pattern</button>
      </div>
    </div>

    <!-- Color scheme -->
    <div class="section">
      <div class="section-title">Color Scheme</div>
      <div class="btn-group" id="color-group">
        <button data-val="default" class="active">Default</button>
        <button data-val="correctness">Correctness</button>
        <button data-val="source">Source</button>
      </div>
      <div id="color-legend" style="margin-top:8px;font-size:.72rem;color:#64748b;line-height:1.8;display:none"></div>
    </div>

    <!-- Filters -->
    <div class="section">
      <div class="section-title">Filters</div>
      <label class="field-label">Question IDs (comma-separated, e.g. 1,2,5)</label>
      <input type="text" id="filter-questions" placeholder="leave blank for all">

      <label class="field-label">ID Patterns (space-separated glob, e.g. id_9* id_4*)</label>
      <input type="text" id="filter-patterns" placeholder="leave blank for all">
    </div>

    <!-- Pattern Groups (avg_by_pattern mode only) -->
    <div class="section" id="pattern-section" style="display:none">
      <div class="section-title">Pattern Groups</div>
      <div id="pattern-rows"></div>
      <button id="add-pattern-btn" style="width:100%;padding:6px;background:#252840;border:1px dashed #2d3148;border-radius:6px;color:#64748b;font-size:.78rem;cursor:pointer;margin-top:4px;">+ Add Pattern</button>
      <label style="display:flex;align-items:center;gap:6px;margin-top:10px;font-size:.78rem;color:#94a3b8;cursor:pointer">
        <input type="checkbox" id="shade-std-cb" checked>
        Show ±1 std shading
      </label>
    </div>

    <!-- Apply -->
    <div class="section">
      <button id="apply-btn">Apply &amp; Refresh</button>
    </div>

    <!-- Stats -->
    <div class="section">
      <div class="section-title">Stats</div>
      <div id="stats-box">No data loaded yet.</div>
    </div>

  </aside>

  <!-- ===== Chart area ===== -->
  <main style="position:relative">
    <div id="loading" class="visible">Loading files…</div>
    <div id="error-msg"></div>
    <div id="chart"></div>
  </main>

</div>

<script>
// ============================================================
// State
// ============================================================
const state = {
  plotType: "llh",
  colorBy:  "default",
};

// ============================================================
// Plotly layout defaults (dark theme)
// ============================================================
function makeLayout(title, xlabel, ylabel) {
  return {
    title: { text: title, font: { color: "#e2e8f0", size: 15 } },
    paper_bgcolor: "#0f1117",
    plot_bgcolor:  "#13161f",
    font: { color: "#94a3b8", size: 11 },
    xaxis: {
      title: { text: xlabel, font: { color: "#94a3b8" } },
      gridcolor: "#1e2236",
      zerolinecolor: "#1e2236",
      tickcolor: "#2d3148",
    },
    yaxis: {
      title: { text: ylabel, font: { color: "#94a3b8" } },
      gridcolor: "#1e2236",
      zerolinecolor: "#1e2236",
      tickcolor: "#2d3148",
    },
    legend: {
      bgcolor: "#1a1d27",
      bordercolor: "#2d3148",
      borderwidth: 1,
      font: { size: 10, color: "#94a3b8" },
      tracegroupgap: 2,
    },
    margin: { t: 50, r: 20, b: 50, l: 60 },
    hovermode: "closest",
    hoverlabel: {
      bgcolor: "#1a1d27",
      bordercolor: "#4f46e5",
      font: { color: "#e2e8f0", size: 12 },
    },
    modebar: { bgcolor: "transparent", color: "#64748b", activecolor: "#a5b4fc" },
  };
}

const plotConfig = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["select2d","lasso2d","autoScale2d"],
};

// ============================================================
// Helpers
// ============================================================
function showError(msg) {
  const el = document.getElementById("error-msg");
  el.textContent = msg;
  el.style.display = msg ? "block" : "none";
}

function setLoading(on) {
  document.getElementById("loading").classList.toggle("visible", on);
}

function selectedFiles() {
  return [...document.querySelectorAll("#file-list input[type=checkbox]:checked")]
    .map(cb => cb.value);
}

function updateStats(traces) {
  const box = document.getElementById("stats-box");
  if (!traces || traces.length === 0) { box.innerHTML = "No traces."; return; }
  const visible = traces.filter(t => t.showlegend !== false);
  box.innerHTML = `<span>${visible.length}</span> series loaded.`;
}

// ============================================================
// Color legend
// ============================================================
const CORRECTNESS_LEGEND = `
  <span class="dot" style="background:#2ca02c"></span>correct &nbsp;
  <span class="dot" style="background:#d62728"></span>incorrect
`;
const SOURCE_LEGEND = `
  <span class="dot" style="background:#1f77b4"></span>base &nbsp;
  <span class="dot" style="background:#ff7f0e"></span>qwen &nbsp;
  <span class="dot" style="background:#2ca02c"></span>gemma<br>
  <span class="dot" style="background:#d62728"></span>highest &nbsp;
  <span class="dot" style="background:#9467bd"></span>lowest
`;

function updateColorLegend() {
  const div = document.getElementById("color-legend");
  if (state.colorBy === "correctness") { div.innerHTML = CORRECTNESS_LEGEND; div.style.display = "block"; }
  else if (state.colorBy === "source") { div.innerHTML = SOURCE_LEGEND; div.style.display = "block"; }
  else { div.style.display = "none"; }
}

// ============================================================
// Pattern rows (avg_by_pattern mode)
// ============================================================
function addPatternRow(patternVal = "", notationVal = "") {
  const container = document.getElementById("pattern-rows");
  const row = document.createElement("div");
  row.className = "pattern-row";
  row.innerHTML = `
    <input type="text" class="pat-glob" placeholder="id_1*" value="${patternVal}">
    <input type="text" class="pat-label" placeholder="Label" value="${notationVal}">
    <button class="remove-row-btn" title="Remove">✕</button>
  `;
  row.querySelector(".remove-row-btn").addEventListener("click", () => row.remove());
  container.appendChild(row);
}

function collectPatternRows() {
  const rows = document.querySelectorAll("#pattern-rows .pattern-row");
  const patterns  = [];
  const notations = [];
  rows.forEach(row => {
    const glob  = row.querySelector(".pat-glob").value.trim();
    const label = row.querySelector(".pat-label").value.trim();
    if (glob) {
      patterns.push(glob);
      notations.push(label || glob);
    }
  });
  return { patterns, notations };
}

function togglePatternSection() {
  const show = state.plotType === "avg_by_pattern";
  document.getElementById("pattern-section").style.display = show ? "" : "none";
  // Seed one empty row if none exist
  if (show && document.querySelectorAll("#pattern-rows .pattern-row").length === 0) {
    addPatternRow();
  }
}

// ============================================================
// Main fetch + render
// ============================================================
async function fetchAndRender() {
  const files = selectedFiles();
  if (files.length === 0) {
    showError("Please select at least one summary file.");
    return;
  }
  showError("");
  setLoading(true);

  const questions = document.getElementById("filter-questions").value.trim();
  const patsRaw   = document.getElementById("filter-patterns").value.trim();
  const patterns  = patsRaw ? patsRaw.split(/\s+/) : [];

  const params = new URLSearchParams();
  files.forEach(f => params.append("files", f));
  params.set("plot_type", state.plotType);
  params.set("color_by",  state.colorBy);
  if (questions) params.set("questions", questions);
  patterns.forEach(p => params.append("id_patterns", p));

  if (state.plotType === "avg_by_pattern") {
    const { patterns: avgPats, notations: avgNots } = collectPatternRows();
    if (avgPats.length === 0) {
      showError("Add at least one pattern in the Pattern Groups section.");
      setLoading(false);
      return;
    }
    avgPats.forEach(p => params.append("avg_patterns", p));
    avgNots.forEach(n => params.append("avg_notations", n));
    const shadeStd = document.getElementById("shade-std-cb").checked;
    params.set("shade_std", shadeStd ? "true" : "false");
  }

  try {
    const resp = await fetch(`/api/data?${params}`);
    const data = await resp.json();
    if (!resp.ok) { showError(data.error || "Server error"); setLoading(false); return; }

    const layout = makeLayout(data.title, data.xlabel, data.ylabel);
    Plotly.react("chart", data.traces, layout, plotConfig);
    updateStats(data.traces);
  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    setLoading(false);
  }
}

// ============================================================
// Initialise file list
// ============================================================
async function loadFileList() {
  const ul = document.getElementById("file-list");
  try {
    const resp = await fetch("/api/files");
    const files = await resp.json();
    if (files.length === 0) {
      ul.innerHTML = '<li style="color:#555;font-size:.8rem">No summary files found in prob_tracking/results/</li>';
      setLoading(false);
      return;
    }
    ul.innerHTML = files.map(f => `
      <li>
        <label>
          <input type="checkbox" value="${f}">
          <span>${f.replace(/_all_checkpoints_summary\.json$/, "")}</span>
        </label>
      </li>
    `).join("");
    setLoading(false);
  } catch (e) {
    ul.innerHTML = '<li style="color:#ef4444;font-size:.8rem">Failed to load file list</li>';
    setLoading(false);
  }
}

// ============================================================
// Button group helper
// ============================================================
function bindBtnGroup(groupId, stateKey) {
  const group = document.getElementById(groupId);
  group.querySelectorAll("button").forEach(btn => {
    btn.addEventListener("click", () => {
      group.querySelectorAll("button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state[stateKey] = btn.dataset.val;
      if (stateKey === "colorBy") updateColorLegend();
      if (stateKey === "plotType") togglePatternSection();
    });
  });
}

// ============================================================
// Boot
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  bindBtnGroup("plot-type-group", "plotType");
  bindBtnGroup("color-group",     "colorBy");

  document.getElementById("apply-btn").addEventListener("click", fetchAndRender);
  document.getElementById("add-pattern-btn").addEventListener("click", () => addPatternRow());

  // Allow Enter key in filter inputs
  ["filter-questions","filter-patterns"].forEach(id => {
    document.getElementById(id).addEventListener("keydown", e => {
      if (e.key === "Enter") fetchAndRender();
    });
  });

  // Resize chart when window resizes
  window.addEventListener("resize", () => {
    const chart = document.getElementById("chart");
    if (chart && chart.data) Plotly.relayout("chart", {});
  });

  loadFileList();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override path to prob_tracking/results/ directory")
    args = parser.parse_args()

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    print(f"Results directory : {RESULTS_DIR}")
    print(f"Starting server   : http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
