#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/codex-matplotlib')

import matplotlib.pyplot as plt

from test_1to1_traces import REPO_ROOT, WORKLOAD_NAMES, run_command
from test_random_traces import (
    STRATEGY_TO_POLICY,
    build_command,
    generate_trace,
    normalize_arrival_args,
    normalize_workloads,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / 'res' / 'random_curve_gallery'
DEFAULT_STRATEGIES = ('optimal', 'uniform', 'fix')
DEFAULT_ARRIVAL_PROFILES = ('uniform', 'piecewise')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a small random-trace gallery and plot uswap/directswap vs fastswap memory curves.'
    )
    parser.add_argument('--trace-ids', type=lambda s: [int(x) for x in s.split(',')], default=[1, 2])
    parser.add_argument('--tasks-per-trace', type=int, default=120)
    parser.add_argument('--until', type=int, default=120, help='arrival window in seconds')
    parser.add_argument('--num-servers', type=int, default=8)
    parser.add_argument('--seed-base', type=int, default=5000)
    parser.add_argument(
        '--strategies',
        type=lambda s: [x.strip() for x in s.split(',') if x.strip()],
        default=list(DEFAULT_STRATEGIES),
        help='comma-separated subset of: optimal,uniform,fix',
    )
    parser.add_argument(
        '--arrival-profiles',
        type=lambda s: [x.strip() for x in s.split(',') if x.strip()],
        default=list(DEFAULT_ARRIVAL_PROFILES),
        help='comma-separated subset of: uniform,piecewise',
    )
    parser.add_argument(
        '--piecewise-weights',
        type=lambda s: s.split(','),
        default=['1', '2', '5', '2', '1'],
        help='bucket weights to use when piecewise arrivals are enabled',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        '--workloads',
        type=lambda s: s.split(','),
        help='comma-separated workload names to sample from; defaults to the built-in workload set',
    )
    parser.add_argument(
        '--workload-local-ratio-percent',
        type=float,
        default=66.0,
        help='local-memory percentage to assign to every active workload, e.g. 1 means 99%% can be swapped out',
    )
    parser.add_argument(
        '--uniform-min-ratio',
        type=float,
        default=0.5,
        help='global minimum local-memory ratio used by uniform policies',
    )
    parser.add_argument(
        '--max-far-mb',
        type=int,
        help='override the total remote-memory cap passed to simulation_one_time.py',
    )
    return parser.parse_args()


def read_curve(path: Path) -> List[Dict[str, float]]:
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        return [{key: float(value) for key, value in row.items()} for row in reader]


def strategy_label(strategy: str) -> str:
    return {
        'optimal': 'optimal',
        'uniform': 'uniform',
        'fix': 'fix',
    }[strategy]


def engine_label(fastswap: bool) -> str:
    return 'fastswap' if fastswap else 'uswap'


def override_ratio_settings(
    command: List[str],
    num_active_workloads: int,
    workload_local_ratio_percent: float,
    uniform_min_ratio: float,
    max_far_mb: int | None,
) -> None:
    workload_ratios_index = command.index('--workload_ratios') + 1
    command[workload_ratios_index] = ','.join(
        ['{:g}'.format(workload_local_ratio_percent)] * num_active_workloads
    )
    min_ratio_index = command.index('--min_ratio') + 1
    command[min_ratio_index] = '{:g}'.format(uniform_min_ratio)
    if max_far_mb is not None:
        max_far_index = command.index('--max_far') + 1
        command[max_far_index] = str(max_far_mb)


def plot_overview(
    output_path: Path,
    trace_id: int,
    arrival_profile: str,
    strategy_results: Sequence[Dict[str, object]],
) -> None:
    fig, axes = plt.subplots(len(strategy_results), 1, figsize=(11, 3.6 * len(strategy_results)), sharex=False)
    if len(strategy_results) == 1:
        axes = [axes]

    for axis, result in zip(axes, strategy_results):
        uswap_curve = result['uswap_curve']
        fastswap_curve = result['fastswap_curve']
        uswap_metrics = result['uswap_metrics']
        fastswap_metrics = result['fastswap_metrics']
        strategy = result['strategy']

        uswap_times = [point['time_s'] for point in uswap_curve]
        fastswap_times = [point['time_s'] for point in fastswap_curve]
        axis.step(
            uswap_times,
            [point['remote_mem_utilization'] * 100.0 for point in uswap_curve],
            where='post',
            linewidth=2.0,
            label='uswap',
        )
        axis.step(
            fastswap_times,
            [point['remote_mem_utilization'] * 100.0 for point in fastswap_curve],
            where='post',
            linewidth=2.0,
            label='fastswap',
        )
        axis.set_ylabel('Remote Util. (%)')
        axis.set_title(
            '{} remote memory utilization\nuswap avg_turnaround_slowdown={:.3f}x, fastswap avg_turnaround_slowdown={:.3f}x'.format(
                strategy_label(strategy),
                uswap_metrics['avg_turnaround_slowdown'],
                fastswap_metrics['avg_turnaround_slowdown'],
            )
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        axis.set_xlabel('Time (s)')

    fig.suptitle('Random trace {:03d} | arrival={}'.format(trace_id, arrival_profile), fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    strategies = list(args.strategies)
    arrival_profiles = list(args.arrival_profiles)
    workload_names = normalize_workloads(args.workloads)
    for strategy in strategies:
        if strategy not in STRATEGY_TO_POLICY:
            raise ValueError('unknown strategy: {}'.format(strategy))
    for arrival_profile in arrival_profiles:
        if arrival_profile not in ('uniform', 'piecewise'):
            raise ValueError('unknown arrival profile: {}'.format(arrival_profile))

    piecewise_weights = normalize_arrival_args('piecewise', args.piecewise_weights)
    if not (0 < args.workload_local_ratio_percent <= 100):
        raise ValueError('--workload-local-ratio-percent must be in the range (0, 100]')
    if not (0 < args.uniform_min_ratio <= 1):
        raise ValueError('--uniform-min-ratio must be in the range (0, 1]')
    if args.max_far_mb is not None and args.max_far_mb <= 0:
        raise ValueError('--max-far-mb must be positive when provided')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / 'manifest.tsv'
    with manifest_path.open('w') as manifest:
        manifest.write(
            'trace_id\tarrival_profile\tstrategy\tengine\tpolicy\tworkload_local_ratio_percent\tuniform_min_ratio\tmax_far_mb\tcurve_tsv\tcurve_png\tmakespan\tavg_remote_mem_util\tavg_queue_time\tavg_runtime_slowdown\tavg_turnaround_slowdown\n'
        )

        for trace_id in args.trace_ids:
            trace = generate_trace(trace_id, args.tasks_per_trace, workload_names, args.seed_base)

            for arrival_profile in arrival_profiles:
                arrival_weights = None if arrival_profile == 'uniform' else piecewise_weights
                strategy_results = []

                for strategy in strategies:
                    policy = STRATEGY_TO_POLICY[strategy]
                    per_engine = {}

                    for fastswap in (False, True):
                        engine = engine_label(fastswap)
                        prefix = args.output_dir / 'curves' / 'trace{:03d}_{}_{}_{}'.format(
                            trace_id,
                            arrival_profile,
                            strategy,
                            engine,
                        )
                        command = build_command(
                            trace,
                            args.num_servers,
                            args.until,
                            policy,
                            fastswap,
                            arrival_profile,
                            arrival_weights,
                        )
                        override_ratio_settings(
                            command,
                            len(trace.active_workloads),
                            args.workload_local_ratio_percent,
                            args.uniform_min_ratio,
                            args.max_far_mb,
                        )
                        command.extend(['--memory-curve-prefix', str(prefix)])
                        metrics = run_command(command)
                        curve_tsv = Path(str(prefix) + '.tsv')
                        curve_png = Path(str(prefix) + '.png')
                        curve = read_curve(curve_tsv)
                        per_engine[engine] = {
                            'metrics': metrics,
                            'curve': curve,
                            'curve_tsv': curve_tsv,
                            'curve_png': curve_png,
                        }
                        manifest.write(
                            'trace{trace_id:03d}\t{arrival_profile}\t{strategy}\t{engine}\t{policy}\t{workload_local_ratio_percent}\t{uniform_min_ratio}\t{max_far_mb}\t{curve_tsv}\t{curve_png}\t'
                            '{makespan:.6f}\t{avg_remote_mem_util:.6f}\t{avg_queue_time:.6f}\t{avg_runtime_slowdown:.6f}\t{avg_turnaround_slowdown:.6f}\n'.format(
                                trace_id=trace_id,
                                arrival_profile=arrival_profile,
                                strategy=strategy,
                                engine=engine,
                                policy=policy,
                                workload_local_ratio_percent='{0:g}'.format(args.workload_local_ratio_percent),
                                uniform_min_ratio='{0:g}'.format(args.uniform_min_ratio),
                                max_far_mb='' if args.max_far_mb is None else args.max_far_mb,
                                curve_tsv=curve_tsv,
                                curve_png=curve_png,
                                makespan=metrics['makespan'],
                                avg_remote_mem_util=metrics['avg_remote_mem_util'],
                                avg_queue_time=metrics['avg_queue_time'],
                                avg_runtime_slowdown=metrics['avg_runtime_slowdown'],
                                avg_turnaround_slowdown=metrics['avg_turnaround_slowdown'],
                            )
                        )
                        manifest.flush()
                        print(
                            '[trace {trace_id:03d}] arrival={arrival_profile} strategy={strategy} engine={engine} '
                            'avg_turnaround={avg_turnaround:.3f}x avg_remote_util={avg_remote_util:.2%} avg_queue={avg_queue:.0f}ms'.format(
                                trace_id=trace_id,
                                arrival_profile=arrival_profile,
                                strategy=strategy,
                                engine=engine,
                                avg_turnaround=metrics['avg_turnaround_slowdown'],
                                avg_remote_util=metrics['avg_remote_mem_util'],
                                avg_queue=metrics['avg_queue_time'],
                            )
                        )

                    strategy_results.append(
                        {
                            'strategy': strategy,
                            'uswap_curve': per_engine['uswap']['curve'],
                            'fastswap_curve': per_engine['fastswap']['curve'],
                            'uswap_metrics': per_engine['uswap']['metrics'],
                            'fastswap_metrics': per_engine['fastswap']['metrics'],
                        }
                    )

                overview_path = args.output_dir / 'overview_trace{:03d}_{}.png'.format(trace_id, arrival_profile)
                plot_overview(overview_path, trace_id, arrival_profile, strategy_results)
                print('wrote {}'.format(overview_path))

    print('wrote {}'.format(manifest_path))


if __name__ == '__main__':
    main()
