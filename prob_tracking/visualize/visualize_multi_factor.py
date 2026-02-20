#!/usr/bin/env python3
"""
Visualize log-likelihood dynamics from multiple summary files with separate
legend sections for each of four factors encoded in the type ID:

  Factor 2.1 – Problem type  : original problem vs. problem_paraphrased
  Factor 2.2 – Model source  : base | gemma | qwen | cp<N> (immediate checkpoint)
  Factor 2.3 – Decoding      : greedy (no temp / *greedy*) vs. temperature-0.7 (*0.7*)
  Factor 2.4 – Correctness   : *correct (ends with 'correct') vs. *incorrect

Visual encoding
---------------
  Color      → model source   (blue=base, green=gemma, orange=qwen,
                                unique colormap color per cp checkpoint, grey=unknown)
  Linestyle  → problem type   (solid=original, dashed=paraphrased)
  Marker     → decoding       (o=greedy, ^=0.7)
  Marker fill→ correctness    (filled=correct, hollow=incorrect)

Legend is split into four independent sub-sections via blank proxy handles.
The cp-checkpoint sub-section is built dynamically from the data.

Usage example
-------------
  python prob_tracking/visualize/visualize_multi_factor.py \\
      --summary_path prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json \\
                     prob_tracking/results/test_amc_high_3B_all_checkpoints_summary.json \\
      --output_dir prob_tracking/image/multi_factor \\
      --no_show
"""

import json
import re
import fnmatch
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path


# ── Visual encoding constants ──────────────────────────────────────────────────

# Fixed colors for named sources
SOURCE_COLORS = {
    'base':    '#1f77b4',   # blue
    'gemma':   '#2ca02c',   # green
    'qwen':    '#ff7f0e',   # orange
    'highest': '#d62728',   # red
    'lowest':  '#9467bd',   # purple
    'unknown': '#7f7f7f',   # grey
}


def _build_cp_palette(threshold: float = 0.25) -> list:
    """
    Build a palette for cp checkpoint colors from tab20, excluding any color
    that is perceptually close (Euclidean RGB distance < threshold) to a
    named source color (base / gemma / qwen / highest / lowest).

    With threshold=0.25 this reliably removes tab20 hues that overlap with
    the fixed source colors.
    """
    reserved = [
        mcolors.to_rgba(c)
        for k, c in SOURCE_COLORS.items()
        if k != 'unknown'
    ]
    palette = []
    for i in range(20):
        c = plt.cm.tab20(i / 20)
        min_dist = min(
            sum((c[j] - r[j]) ** 2 for j in range(3)) ** 0.5
            for r in reserved
        )
        if min_dist > threshold:
            palette.append(c)
    return palette


# Pre-built at import time; guaranteed not to overlap Base / Gemma / Qwen colors
_CP_PALETTE = _build_cp_palette()

LINESTYLES = {
    'original':    '-',
    'paraphrased': '--',
}

MARKERS = {
    'greedy': 'o',
    '0.7':    '^',
}

_CP_RE = re.compile(r'(?<![a-zA-Z])cp(\d+)(?!\d)', re.IGNORECASE)


# ── Factor detection ───────────────────────────────────────────────────────────

def detect_factors(type_id: str) -> dict:
    """
    Detect the four factors from a type ID string.

    Returns a dict with keys:
        paraphrased (bool)   – True if 'problem_paraphrased' is in the ID
        source      (str)    – 'highest' | 'lowest' | 'base' | 'gemma' | 'qwen'
                               | 'cp<N>' | 'unknown'
                               highest/lowest are checked first, then other named
                               sources, then cp<N> pattern, then unknown.
        decoding    (str)    – 'greedy' | '0.7'
        correct     (bool)   – True if ID ends with 'correct' (not 'incorrect')
    """
    lower = type_id.lower()

    # 2.1 Problem type
    paraphrased = 'problem_paraphrased' in lower

    # 2.2 Model source
    #   Priority: highest/lowest first, then gemma/qwen/base, then cp<N>, then unknown
    source = 'unknown'
    for s in ('highest', 'lowest', 'gemma', 'qwen', 'base'):
        if s in lower:
            source = s
            break
    if source == 'unknown':
        m = _CP_RE.search(type_id)
        if m:
            source = f'cp{m.group(1)}'

    # 2.3 Decoding strategy
    if '0.7' in type_id:
        decoding = '0.7'
    else:
        decoding = 'greedy'

    # 2.4 Correctness – False only if ends with 'incorrect'; default True when neither keyword present
    correct = not lower.endswith('incorrect')

    return {
        'paraphrased': paraphrased,
        'source':      source,
        'decoding':    decoding,
        'correct':     correct,
    }


def collect_cp_sources(summaries: list) -> list:
    """
    Scan all type IDs across the given summaries and return a sorted list
    of unique 'cp<N>' source strings found (e.g. ['cp1110', 'cp2220', ...]).
    """
    cp_set = set()
    for _, summary in summaries:
        for type_id in summary['results_by_type']:
            f = detect_factors(type_id)
            if f['source'].startswith('cp'):
                cp_set.add(f['source'])
    # Sort numerically by the checkpoint number
    return sorted(cp_set, key=lambda s: int(s[2:]))


def make_cp_color_map(cp_sources: list) -> dict:
    """
    Assign a unique color from _CP_PALETTE to each cp source string.
    Colors are guaranteed not to overlap with Base, Gemma, or Qwen.
    Returns a dict mapping 'cp<N>' -> rgba color.
    """
    return {
        src: _CP_PALETTE[i % len(_CP_PALETTE)]
        for i, src in enumerate(cp_sources)
    }


def get_color(source: str, cp_color_map: dict) -> object:
    """Return the color for a given source string."""
    if source in SOURCE_COLORS:
        return SOURCE_COLORS[source]
    if source in cp_color_map:
        return cp_color_map[source]
    return SOURCE_COLORS['unknown']


def line_props(factors: dict, cp_color_map: dict) -> dict:
    """Return matplotlib line/marker kwargs for a given set of factors."""
    color     = get_color(factors['source'], cp_color_map)
    linestyle = LINESTYLES['paraphrased' if factors['paraphrased'] else 'original']
    marker    = MARKERS[factors['decoding']]
    # Correct → filled marker; incorrect → hollow (unfilled) marker
    if factors['correct']:
        markerfacecolor = color
        markeredgecolor = color
    else:
        markerfacecolor = 'none'
        markeredgecolor = color

    return dict(
        color           = color,
        linestyle       = linestyle,
        marker          = marker,
        markerfacecolor = markerfacecolor,
        markeredgecolor = markeredgecolor,
        linewidth       = 1.8,
        markersize      = 6,
        alpha           = 0.85,
    )


# ── Data loading / filtering ───────────────────────────────────────────────────

def load_summary(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_by_id_patterns(summary: dict, patterns: list) -> dict:
    filtered = {
        k: v for k, v in summary['results_by_type'].items()
        if any(fnmatch.fnmatch(k, p) for p in patterns)
    }
    return {**summary, 'results_by_type': filtered}


def filter_by_questions(summary: dict, question_ids: list) -> dict:
    prefixes = tuple(f'id_{q}_' for q in question_ids)
    filtered = {
        k: v for k, v in summary['results_by_type'].items()
        if k.startswith(prefixes)
    }
    return {**summary, 'results_by_type': filtered}


# ── Compound legend builders ───────────────────────────────────────────────────

def _blank() -> mlines.Line2D:
    """Invisible spacer handle for the legend."""
    return mlines.Line2D([], [], color='none', label='')


def build_legend_handles(cp_color_map: dict = None) -> list:
    """
    Build a flat list of legend handles that groups the four factors
    with blank spacers as section headers.

    Args:
        cp_color_map: mapping of 'cp<N>' -> color for immediate checkpoints.
                      If provided, a sub-section listing each cp is added
                      to the Model source section.
    """
    if cp_color_map is None:
        cp_color_map = {}

    handles = []

    # ── 2.1 Problem type ──────────────────────────────────────────────────────
    handles.append(mpatches.Patch(color='none', label='── Problem type ──'))
    handles.append(mlines.Line2D(
        [], [], color='grey', linestyle=LINESTYLES['original'],
        linewidth=2, label='Original problem'))
    handles.append(mlines.Line2D(
        [], [], color='grey', linestyle=LINESTYLES['paraphrased'],
        linewidth=2, label='Paraphrased problem'))

    handles.append(_blank())

    # ── 2.2 Model source ──────────────────────────────────────────────────────
    handles.append(mpatches.Patch(color='none', label='── Model source ──'))
    for src, clr in SOURCE_COLORS.items():
        if src == 'unknown':
            continue
        handles.append(mlines.Line2D(
            [], [], color=clr, linestyle='-', linewidth=2,
            label=src.capitalize()))

    # Immediate checkpoints – one entry per unique cp value, sorted numerically
    if cp_color_map:
        handles.append(mpatches.Patch(color='none', label='  Immediate checkpoints:'))
        for cp_src, clr in sorted(cp_color_map.items(), key=lambda x: int(x[0][2:])):
            step_num = cp_src[2:]   # strip 'cp' prefix
            handles.append(mlines.Line2D(
                [], [], color=clr, linestyle='-', linewidth=2,
                label=f'  cp step {step_num}'))

    handles.append(_blank())

    # ── 2.3 Decoding strategy ─────────────────────────────────────────────────
    handles.append(mpatches.Patch(color='none', label='── Decoding ──'))
    handles.append(mlines.Line2D(
        [], [], color='grey', linestyle='none',
        marker=MARKERS['greedy'], markersize=8,
        markerfacecolor='grey', label='Greedy'))
    handles.append(mlines.Line2D(
        [], [], color='grey', linestyle='none',
        marker=MARKERS['0.7'], markersize=8,
        markerfacecolor='grey', label='Temperature 0.7'))

    handles.append(_blank())

    # ── 2.4 Correctness ───────────────────────────────────────────────────────
    handles.append(mpatches.Patch(color='none', label='── Correctness ──'))
    handles.append(mlines.Line2D(
        [], [], color='grey', linestyle='none',
        marker='s', markersize=8,
        markerfacecolor='grey', markeredgecolor='grey',
        label='Correct (filled)'))
    handles.append(mlines.Line2D(
        [], [], color='grey', linestyle='none',
        marker='s', markersize=8,
        markerfacecolor='none', markeredgecolor='grey',
        label='Incorrect (hollow)'))

    return handles


# ── Plotting functions ─────────────────────────────────────────────────────────

def _plot_lines(ax, summaries: list, metric_key: str, cp_color_map: dict):
    """Shared inner loop: plot one line per type ID, encoding all four factors."""
    for file_label, summary in summaries:
        steps = summary['steps']
        for type_id, metrics in summary['results_by_type'].items():
            values = metrics[metric_key]
            valid_idx = [i for i, v in enumerate(values) if v is not None]
            if not valid_idx:
                continue

            xs = [steps[i] for i in valid_idx]
            ys = [values[i] for i in valid_idx]

            factors = detect_factors(type_id)
            props   = line_props(factors, cp_color_map)

            label = f'[{file_label}] {type_id}' if file_label else type_id
            ax.plot(xs, ys, label=label, **props)


def plot_avg_log_likelihood(summaries: list, output_path: str = None, show: bool = True,
                            title: str = 'Average Log-Likelihood Dynamics (Multi-Factor)'):
    """
    Plot avg log-likelihood for every type in every summary,
    encoding the four factors via color / linestyle / marker / fill.
    Immediate checkpoint sources (cp<N>) each receive a unique color.
    """
    cp_sources   = collect_cp_sources(summaries)
    cp_color_map = make_cp_color_map(cp_sources)

    _, ax = plt.subplots(figsize=(14, 8))
    _plot_lines(ax, summaries, 'avg_log_likelihood', cp_color_map)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Log-Likelihood', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    legend_handles = build_legend_handles(cp_color_map)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    if show:
        plt.show()
    plt.close()


def plot_perplexity(summaries: list, output_path: str = None, show: bool = True,
                    title: str = 'Average Perplexity Dynamics (Multi-Factor)'):
    """
    Plot avg perplexity for every type in every summary with multi-factor encoding.
    Immediate checkpoint sources (cp<N>) each receive a unique color.
    """
    cp_sources   = collect_cp_sources(summaries)
    cp_color_map = make_cp_color_map(cp_sources)

    _, ax = plt.subplots(figsize=(14, 8))
    _plot_lines(ax, summaries, 'avg_perplexity', cp_color_map)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Perplexity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    legend_handles = build_legend_handles(cp_color_map)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    if show:
        plt.show()
    plt.close()


def plot_by_question(summaries: list, output_path: str = None, show: bool = True,
                     title: str = 'Log-Likelihood by Question (Multi-Factor)'):
    """
    Plot one line per (question, type) combination; each line's visual style
    encodes the four factors.  Questions are identified by the leading 'id_N_'
    prefix; the line label shows the question ID and type ID suffix.
    Immediate checkpoint sources (cp<N>) each receive a unique color.
    """
    cp_sources   = collect_cp_sources(summaries)
    cp_color_map = make_cp_color_map(cp_sources)

    _, ax = plt.subplots(figsize=(16, 9))
    _plot_lines(ax, summaries, 'avg_log_likelihood', cp_color_map)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Log-Likelihood', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    legend_handles = build_legend_handles(cp_color_map)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    if show:
        plt.show()
    plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Visualize log-likelihood dynamics from multiple summary files '
            'with separate legend sections for paraphrased/source/decoding/correctness.'
        )
    )
    parser.add_argument(
        '--summary_path', nargs='+', required=True,
        help='One or more checkpoint summary JSON files.'
    )
    parser.add_argument(
        '--output_dir', default='prob_tracking/image/multi_factor',
        help='Directory to save plots (default: prob_tracking/image/multi_factor)'
    )
    parser.add_argument(
        '--no_show', action='store_true',
        help="Don't open interactive windows; only save files."
    )
    parser.add_argument(
        '--plot_type', choices=['all', 'llh', 'perplexity', 'questions'],
        default='all',
        help='Which plot(s) to produce (default: all).'
    )
    parser.add_argument(
        '--questions', '--question', dest='questions', default=None,
        help="Comma-separated question IDs to include, e.g. '1,2,3'."
    )
    parser.add_argument(
        '--id_patterns', nargs='+', default=None,
        help="Glob patterns to filter type IDs (e.g. 'id_9*')."
    )

    args = parser.parse_args()
    show = not args.no_show
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load all summaries
    summaries = []
    for path in args.summary_path:
        print(f"Loading: {path}")
        s = load_summary(path)
        label = Path(path).stem
        summaries.append((label, s))

    # Optional: filter by question IDs
    if args.questions is not None:
        q_ids = [int(q.strip()) for q in args.questions.split(',')]
        summaries = [(lbl, filter_by_questions(s, q_ids)) for lbl, s in summaries]
        n = sum(len(s['results_by_type']) for _, s in summaries)
        print(f"Filtered to question(s) {q_ids}: {n} types total")

    # Optional: filter by glob patterns on type IDs
    if args.id_patterns is not None:
        summaries = [(lbl, filter_by_id_patterns(s, args.id_patterns))
                     for lbl, s in summaries]
        n = sum(len(s['results_by_type']) for _, s in summaries)
        print(f"Filtered by patterns {args.id_patterns}: {n} types total")

    out = args.output_dir

    if args.plot_type in ('all', 'llh'):
        print("Generating log-likelihood plot …")
        plot_avg_log_likelihood(
            summaries,
            output_path=f"{out}/avg_llh_multi_factor.png",
            show=show,
        )

    if args.plot_type in ('all', 'perplexity'):
        print("Generating perplexity plot …")
        plot_perplexity(
            summaries,
            output_path=f"{out}/avg_perplexity_multi_factor.png",
            show=show,
        )

    if args.plot_type in ('all', 'questions'):
        print("Generating per-question plot …")
        plot_by_question(
            summaries,
            output_path=f"{out}/llh_by_question_multi_factor.png",
            show=show,
        )

    print("Done.")


if __name__ == '__main__':
    main()
