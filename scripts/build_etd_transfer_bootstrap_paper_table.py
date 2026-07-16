#!/usr/bin/env python3
"""Build a compact paper table from ETD transfer bootstrap deltas."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / 'outputs/analysis/etd_bootstrap_transfer_v1/pairwise_bootstrap_deltas.csv'
DEFAULT_OUTPUT_DIR = ROOT / 'outputs/analysis/etd_bootstrap_transfer_v1'

METRIC_COLUMNS = [
    ('macro_auroc', r'Macro AUROC $\Delta$ (95\% CI)'),
    ('macro_f1', r'Macro F1 $\Delta$ (95\% CI)'),
    ('soft_rmse_p1', r'Soft RMSE $\Delta$ (95\% CI)'),
]
COMPARISON_ORDER = [
    'Adapted - Zero-shot',
    'Adapted - Scratch',
    'Zero-shot - Scratch',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--stem', default='pairwise_bootstrap_deltas_paper')
    return parser.parse_args()


def _format_signed(value: float) -> str:
    return f"{float(value):+.4f}"


def _cell(row: pd.Series) -> str:
    point = _format_signed(row['point_delta'])
    ci = f"[{_format_signed(row['ci_low'])}, {_format_signed(row['ci_high'])}]"
    return f"{point} {ci}"


def _build_compact_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {'comparison_display', 'metric', 'point_delta_fmt', 'ci_fmt'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    rows: list[dict[str, str]] = []
    for comparison in COMPARISON_ORDER:
        comparison_df = df.loc[df['comparison_display'] == comparison]
        if comparison_df.empty:
            raise ValueError(f'Missing comparison: {comparison}')
        out_row: dict[str, str] = {'Comparison': comparison}
        for metric_key, metric_display in METRIC_COLUMNS:
            metric_df = comparison_df.loc[comparison_df['metric'] == metric_key]
            if len(metric_df) != 1:
                raise ValueError(f'Expected one row for {comparison} / {metric_key}, found {len(metric_df)}')
            out_row[metric_display] = _cell(metric_df.iloc[0])
        rows.append(out_row)
    return pd.DataFrame(rows)


def _render_latex(compact_df: pd.DataFrame) -> str:
    headers = ['Comparison', *[display for _, display in METRIC_COLUMNS]]
    lines = [
        r'\begin{table*}[t]',
        r'  \centering',
        r'  \caption[Compact paired-bootstrap ETD transfer deltas]{Compact paired document-level bootstrap deltas for \ac{ETD} transfer. Each cell reports the point estimate and 95\% confidence interval. Positive $\Delta$ values indicate better discrimination or thresholded decision quality for macro \ac{AUROC} and macro F1; negative $\Delta$ values indicate lower soft \ac{RMSE} and stronger teacher-probability fidelity.}',
        r'  \label{tab:etd-transfer-bootstrap-compact}',
        r'  \small',
        r'  \setlength{\tabcolsep}{5pt}',
        r'  \begin{tabular}{@{} l c c c @{}}',
        r'    \toprule',
        '    ' + ' & '.join([fr'\textbf{{{header}}}' for header in headers]) + r' \\',
        r'    \midrule',
    ]
    for row in compact_df.itertuples(index=False):
        values = [str(value) for value in row]
        lines.append('    ' + ' & '.join(values) + r' \\')
    lines.extend([
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table*}',
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
