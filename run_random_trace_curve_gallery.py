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


def plot_overview(
    output_path: Path,
    trace_id: int,
    arrival_profile: str,
    strategy_results: Sequence[Dict[str, object]],
) -> None:
    fig, axes = plt.subplots(len(strategy_results), 2, figsize=(14, 4 * len(strategy_results)), sharex=False)
    if len(strategy_results) == 1:
        axes = [axes]

    for row_axes, result in zip(axes, strategy_results):
        uswap_curve = result['uswap_curve']
        fastswap_curve = result['fastswap_curve']
        uswap_metrics = result['uswap_metrics']
        fastswap_metrics = result['fastswap_metrics']
        strategy = result['strategy']

        uswap_times = [point['time_s'] for point in uswap_curve]
        fastswap_times = [point['time_s'] for point in fastswap_curve]

        row_axes[0].step(
            uswap_times,
            [point['local_used_mem_mb'] for point in uswap_curve],
            where='post',
            linewidth=2.0,
            label='uswap',
        )
        row_axes[0].step(
            fastswap_times,
            [point['local_used_mem_mb'] for point in fastswap_curve],
            where='post',
            linewidth=2.0,
            label='fastswap',
        )
        row_axes[0].set_ylabel('Memory (MB)')
        row_axes[0].set_title(
            '{} actual occupied local memory\nuswap makespan={:.0f}, fastswap makespan={:.0f}'.format(
                strategy_label(strategy),
                uswap_metrics['makespan'],
                fastswap_metrics['makespan'],
            )
        )
        row_axes[0].grid(True, alpha=0.3)
        row_axes[0].legend()

        row_axes[1].step(
            uswap_times,
            [point['active_task_demand_mem_mb'] for point in uswap_curve],
            where='post',
            linewidth=2.0,
            label='uswap',
        )
        row_axes[1].step(
            fastswap_times,
            [point['active_task_demand_mem_mb'] for point in fastswap_curve],
            where='post',
            linewidth=2.0,
            label='fastswap',
        )
        row_axes[1].set_ylabel('Memory (MB)')
        row_axes[1].set_title(
            '{} current task total memory demand\nuswap queue={:.0f} ms, fastswap queue={:.0f} ms'.format(
                strategy_label(strategy),
                uswap_metrics['avg_queue_time'],
                fastswap_metrics['avg_queue_time'],
            )
        )
        row_axes[1].grid(True, alpha=0.3)
        row_axes[1].legend()

    for row_axes in axes:
        row_axes[0].set_xlabel('Time (s)')
        row_axes[1].set_xlabel('Time (s)')

    fig.suptitle('Random trace {:03d} | arrival={}'.format(trace_id, arrival_profile), fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    strategies = list(args.strategies)
    arrival_profiles = list(args.arrival_profiles)
    for strategy in strategies:
        if strategy not in STRATEGY_TO_POLICY:
            raise ValueError('unknown strategy: {}'.format(strategy))
    for arrival_profile in arrival_profiles:
        if arrival_profile not in ('uniform', 'piecewise'):
            raise ValueError('unknown arrival profile: {}'.format(arrival_profile))

    piecewise_weights = normalize_arrival_args('piecewise', args.piecewise_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / 'manifest.tsv'
    with manifest_path.open('w') as manifest:
        manifest.write(
            'trace_id\tarrival_profile\tstrategy\tengine\tpolicy\tcurve_tsv\tcurve_png\tmakespan\tavg_mem_util\tavg_queue_time\tavg_runtime_slowdown\n'
        )

        for trace_id in args.trace_ids:
            trace = generate_trace(trace_id, args.tasks_per_trace, WORKLOAD_NAMES, args.seed_base)

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
                            'trace{trace_id:03d}\t{arrival_profile}\t{strategy}\t{engine}\t{policy}\t{curve_tsv}\t{curve_png}\t'
                            '{makespan:.6f}\t{avg_mem_util:.6f}\t{avg_queue_time:.6f}\t{avg_runtime_slowdown:.6f}\n'.format(
                                trace_id=trace_id,
                                arrival_profile=arrival_profile,
                                strategy=strategy,
                                engine=engine,
                                policy=policy,
                                curve_tsv=curve_tsv,
                                curve_png=curve_png,
                                makespan=metrics['makespan'],
                                avg_mem_util=metrics['avg_mem_util'],
                                avg_queue_time=metrics['avg_queue_time'],
                                avg_runtime_slowdown=metrics['avg_runtime_slowdown'],
                            )
                        )
                        manifest.flush()
                        print(
                            '[trace {trace_id:03d}] arrival={arrival_profile} strategy={strategy} engine={engine} '
                            'makespan={makespan:.0f} avg_mem_util={avg_mem_util:.2%} avg_queue={avg_queue:.0f}ms'.format(
                                trace_id=trace_id,
                                arrival_profile=arrival_profile,
                                strategy=strategy,
                                engine=engine,
                                makespan=metrics['makespan'],
                                avg_mem_util=metrics['avg_mem_util'],
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
