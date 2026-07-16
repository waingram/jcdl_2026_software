#!/usr/bin/env python3
"""Build a compact paper table for ETD transfer point estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / 'outputs/analysis/etd_transfer_three_way_v1/condition_summary.csv'
DEFAULT_OUTPUT_DIR = ROOT / 'outputs/analysis/etd_transfer_three_way_v1'

ROW_LABELS = {
    'Zero-shot ETD test': 'Zero-shot source',
    'ETD-adapted test': 'ETD-adapted',
    'ETD-scratch test': 'ETD-scratch',
}
ROW_ORDER = [
    'ETD-scratch test',
    'Zero-shot ETD test',
    'ETD-adapted test',
]
METRICS = [
    ('macro_auroc', 'Macro AUROC', True, True),
    ('macro_f1', 'Macro F1', True, False),
    ('soft_rmse_p1', 'Soft RMSE', False, True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--stem', default='condition_summary_paper')
    return parser.parse_args()


def _format_value(value: float, *, bold: bool) -> str:
    rendered = f'{float(value):.4f}'
    return rf'\textbf{{{rendered}}}' if bold else rendered


def _best_labels(df: pd.DataFrame) -> dict[str, str]:
    best: dict[str, str] = {}
    for metric, _, higher_is_better, _ in METRICS:
        values = pd.to_numeric(df[metric], errors='raise')
        best_idx = values.idxmax() if higher_is_better else values.idxmin()
        best[metric] = str(df.loc[best_idx, 'label'])
    return best


def _build_compact_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {'label', *[metric for metric, _, _, _ in METRICS]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    missing_rows = [label for label in ROW_ORDER if label not in set(df['label'])]
    if missing_rows:
        raise ValueError(f'Missing expected rows: {missing_rows}')

    source_df = df.set_index('label', drop=False).loc[ROW_ORDER].reset_index(drop=True)
    best = _best_labels(source_df)

    rows: list[dict[str, str]] = []
    for row in source_df.itertuples(index=False):
        row_label = str(row.label)
        out_row = {'Regime': ROW_LABELS.get(row_label, row_label)}
        for metric, display, _, bold_best in METRICS:
            value = getattr(row, metric)
            out_row[display] = _format_value(value, bold=(bold_best and best[metric] == row_label))
        rows.append(out_row)
    return pd.DataFrame(rows)


def _render_latex(compact_df: pd.DataFrame) -> str:
    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \caption[Teacher-referenced ETD transfer performance]{Teacher-referenced \ac{ETD} transfer performance for an \ac{ETD}-only scratch baseline, the zero-shot publication-domain student, and the source-initialized model after continued distillation on \ac{ETD} data. Macro \ac{AUROC} and macro F1 are computed against teacher-derived binary labels obtained by thresholding $p_1$ at 0.5; soft \ac{RMSE} is computed against the full teacher probability vector. Higher is better for macro \ac{AUROC} and macro F1; lower is better for soft \ac{RMSE}.}',
        r'  \label{tab:etd-transfer-performance}',
        r'  \small',
        r'  \begin{tabular}{@{} l c c c @{}}',
        r'    \toprule',
        r'    \textbf{Regime} & \textbf{Macro AUROC} & \textbf{Macro F1} & \textbf{Soft RMSE} \\',
        r'    \midrule',
    ]
    for row in compact_df.itertuples(index=False):
        lines.append('    ' + ' & '.join(str(value) for value in row) + r' \\')
    lines.extend([
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
        '',
    ])
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    compact_df = _build_compact_df(df)
    csv_path = args.output_dir / f'{args.stem}.csv'
    tex_path = args.output_dir / f'{args.stem}.tex'
    compact_df.to_csv(csv_path, index=False)
    tex_path.write_text(_render_latex(compact_df), encoding='utf-8')
    print(f'Wrote {csv_path}')
    print(f'Wrote {tex_path}')


if __name__ == '__main__':
    main()
