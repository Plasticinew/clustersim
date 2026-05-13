#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/codex-matplotlib')

import matplotlib.pyplot as plt

from test_1to1_traces import REPO_ROOT, run_command
from test_random_traces import build_command, generate_trace, normalize_arrival_args, normalize_workloads, STRATEGY_TO_POLICY
from run_random_trace_curve_gallery import override_ratio_settings


DEFAULT_OUTPUT_DIR = REPO_ROOT / 'res' / 'maxfar_distribution'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sweep max_far values and summarize per-trace metric distributions.'
    )
    parser.add_argument('--num-traces', type=int, default=20, help='number of random traces to evaluate')
    parser.add_argument('--tasks-per-trace', type=int, default=200)
    parser.add_argument('--until', type=int, default=4000, help='arrival window in seconds')
    parser.add_argument('--num-servers', type=int, default=8)
    parser.add_argument('--seed-base', type=int, default=5000)
    parser.add_argument(
        '--workloads',
        type=lambda s: s.split(','),
        required=True,
        help='comma-separated workload names to sample from',
    )
    parser.add_argument(
        '--strategy',
        choices=tuple(STRATEGY_TO_POLICY.keys()),
        default='optimal',
    )
    parser.add_argument(
        '--arrival-profile',
        choices=('uniform', 'piecewise'),
        default='uniform',
    )
    parser.add_argument(
        '--piecewise-weights',
        type=lambda s: s.split(','),
        help='comma-separated positive bucket weights for piecewise arrivals',
    )
    parser.add_argument(
        '--max-far-gb-per-node',
        type=lambda s: [int(x) for x in s.split(',') if x.strip()],
        default='8,16,24,32,40,48,56,64',
        help='comma-separated per-node remote memory caps in GB',
    )
    parser.add_argument(
        '--workload-local-ratio-percent',
        type=float,
        default=50.0,
        help='local-memory percentage to assign to every active workload',
    )
    parser.add_argument(
        '--uniform-min-ratio',
        type=float,
        default=0.5,
        help='global minimum local-memory ratio used by uniform policies',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(values: Sequence[float]) -> Dict[str, float]:
    return {
        'mean': mean(values),
        'median': median(values),
        'p25': percentile(values, 0.25),
        'p75': percentile(values, 0.75),
        'min': min(values),
        'max': max(values),
    }


def plot_distribution_lines(
    output_path: Path,
    x_values: Sequence[int],
    uswap_values: Sequence[Sequence[float]],
    fastswap_values: Sequence[Sequence[float]],
    ylabel: str,
    title: str,
    xlabel: str,
    scale: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    us_mean = [mean(vals) * scale for vals in uswap_values]
    fs_mean = [mean(vals) * scale for vals in fastswap_values]
    us_p25 = [percentile(vals, 0.25) * scale for vals in uswap_values]
    us_p75 = [percentile(vals, 0.75) * scale for vals in uswap_values]
    fs_p25 = [percentile(vals, 0.25) * scale for vals in fastswap_values]
    fs_p75 = [percentile(vals, 0.75) * scale for vals in fastswap_values]

    ax.fill_between(x_values, us_p25, us_p75, color='#1f77b4', alpha=0.18, label='uswap IQR')
    ax.fill_between(x_values, fs_p25, fs_p75, color='#d62728', alpha=0.18, label='fastswap IQR')
    ax.plot(x_values, us_mean, marker='o', linewidth=2.2, label='uswap mean', color='#1f77b4')
    ax.plot(x_values, fs_mean, marker='o', linewidth=2.2, label='fastswap mean', color='#d62728')
    ax.set_xticks(list(x_values))
    ax.set_xticklabels([str(x) for x in x_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_traces <= 0:
        raise ValueError('--num-traces must be positive')
    if not (0 < args.workload_local_ratio_percent <= 100):
        raise ValueError('--workload-local-ratio-percent must be in the range (0, 100]')
    if not (0 < args.uniform_min_ratio <= 1):
        raise ValueError('--uniform-min-ratio must be in the range (0, 1]')

    workload_names = normalize_workloads(args.workloads)
    arrival_weights = normalize_arrival_args(args.arrival_profile, args.piecewise_weights)
    policy = STRATEGY_TO_POLICY[args.strategy]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    traces = [
        generate_trace(trace_id, args.tasks_per_trace, workload_names, args.seed_base)
        for trace_id in range(1, args.num_traces + 1)
    ]

    samples_path = args.output_dir / 'samples.tsv'
    summary_path = args.output_dir / 'summary.tsv'

    samples: List[Dict[str, float | int | str]] = []

    with samples_path.open('w') as handle:
        handle.write(
            'trace_id\tmax_far_gb_per_node\tengine\tarrival_profile\tstrategy\t'
            'avg_turnaround_slowdown\tavg_remote_mem_util\tavg_queue_time_ms\n'
        )
        for max_far_gb in args.max_far_gb_per_node:
            total_mb = max_far_gb * 1024 * args.num_servers
            for trace in traces:
                for fastswap in (False, True):
                    engine = 'fastswap' if fastswap else 'uswap'
                    command = build_command(
                        trace,
                        args.num_servers,
                        args.until,
                        policy,
                        fastswap,
                        args.arrival_profile,
                        arrival_weights,
                    )
                    override_ratio_settings(
                        command,
                        len(trace.active_workloads),
                        args.workload_local_ratio_percent,
                        args.uniform_min_ratio,
                        total_mb,
                    )
                    metrics = run_command(command)
                    row = {
                        'trace_id': trace.trace_id,
                        'max_far_gb_per_node': max_far_gb,
                        'engine': engine,
                        'arrival_profile': args.arrival_profile,
                        'strategy': args.strategy,
                        'avg_turnaround_slowdown': metrics['avg_turnaround_slowdown'],
                        'avg_remote_mem_util': metrics['avg_remote_mem_util'],
                        'avg_queue_time_ms': metrics['avg_queue_time'],
                    }
                    samples.append(row)
                    handle.write(
                        '{trace_id}\t{max_far_gb_per_node}\t{engine}\t{arrival_profile}\t{strategy}\t'
                        '{avg_turnaround_slowdown:.6f}\t{avg_remote_mem_util:.6f}\t{avg_queue_time_ms:.6f}\n'.format(
                            **row
                        )
                    )
                    handle.flush()
                    print(
                        '[trace {trace_id:03d}/{num_traces:03d}] max_far={max_far_gb_per_node}G/node engine={engine} '
                        'avg_turnaround={avg_turnaround_slowdown:.3f}x avg_remote_util={avg_remote_mem_util:.2%}'.format(
                            num_traces=args.num_traces,
                            **row,
                        )
                    )

    by_key: Dict[tuple[int, str], List[Dict[str, float | int | str]]] = {}
    for row in samples:
        key = (int(row['max_far_gb_per_node']), str(row['engine']))
        by_key.setdefault(key, []).append(row)

    with summary_path.open('w') as handle:
        handle.write(
            'max_far_gb_per_node\tengine\tavg_turnaround_mean\tavg_turnaround_median\tavg_turnaround_p25\tavg_turnaround_p75\t'
            'avg_turnaround_min\tavg_turnaround_max\tavg_remote_util_mean\tavg_remote_util_median\tavg_remote_util_p25\t'
            'avg_remote_util_p75\tavg_remote_util_min\tavg_remote_util_max\n'
        )
        for max_far_gb in args.max_far_gb_per_node:
            for engine in ('uswap', 'fastswap'):
                rows = by_key[(max_far_gb, engine)]
                turnaround_values = [float(r['avg_turnaround_slowdown']) for r in rows]
                remote_values = [float(r['avg_remote_mem_util']) for r in rows]
                turn_stats = summarize(turnaround_values)
                remote_stats = summarize(remote_values)
                handle.write(
                    '{max_far}\t{engine}\t{turn_mean:.6f}\t{turn_median:.6f}\t{turn_p25:.6f}\t{turn_p75:.6f}\t'
                    '{turn_min:.6f}\t{turn_max:.6f}\t{remote_mean:.6f}\t{remote_median:.6f}\t{remote_p25:.6f}\t'
                    '{remote_p75:.6f}\t{remote_min:.6f}\t{remote_max:.6f}\n'.format(
                        max_far=max_far_gb,
                        engine=engine,
                        turn_mean=turn_stats['mean'],
                        turn_median=turn_stats['median'],
                        turn_p25=turn_stats['p25'],
                        turn_p75=turn_stats['p75'],
                        turn_min=turn_stats['min'],
                        turn_max=turn_stats['max'],
                        remote_mean=remote_stats['mean'],
                        remote_median=remote_stats['median'],
                        remote_p25=remote_stats['p25'],
                        remote_p75=remote_stats['p75'],
                        remote_min=remote_stats['min'],
                        remote_max=remote_stats['max'],
                    )
                )

    us_turn = [[float(r['avg_turnaround_slowdown']) for r in by_key[(g, 'uswap')]] for g in args.max_far_gb_per_node]
    fs_turn = [[float(r['avg_turnaround_slowdown']) for r in by_key[(g, 'fastswap')]] for g in args.max_far_gb_per_node]
    us_remote = [[float(r['avg_remote_mem_util']) for r in by_key[(g, 'uswap')]] for g in args.max_far_gb_per_node]
    fs_remote = [[float(r['avg_remote_mem_util']) for r in by_key[(g, 'fastswap')]] for g in args.max_far_gb_per_node]

    plot_distribution_lines(
        args.output_dir / 'avg_turnaround_slowdown_distribution_vs_maxfar.png',
        args.max_far_gb_per_node,
        us_turn,
        fs_turn,
        ylabel='Average Turnaround Slowdown (x)',
        title='Uniform Arrival + {} Policy ({}s)\nPer-Trace Turnaround Slowdown Distribution'.format(
            args.strategy, args.until
        ),
        xlabel='Average Available Remote Memory per Node (GB)',
    )
    plot_distribution_lines(
        args.output_dir / 'avg_remote_mem_util_distribution_vs_maxfar.png',
        args.max_far_gb_per_node,
        us_remote,
        fs_remote,
        ylabel='Average Remote Memory Utilization (%)',
        title='Uniform Arrival + {} Policy ({}s)\nPer-Trace Remote Utilization Distribution'.format(
            args.strategy, args.until
        ),
        xlabel='Average Available Remote Memory per Node (GB)',
        scale=100.0,
    )

    print('wrote {}'.format(samples_path))
    print('wrote {}'.format(summary_path))
    print('wrote {}'.format(args.output_dir / 'avg_turnaround_slowdown_distribution_vs_maxfar.png'))
    print('wrote {}'.format(args.output_dir / 'avg_remote_mem_util_distribution_vs_maxfar.png'))


if __name__ == '__main__':
    main()
