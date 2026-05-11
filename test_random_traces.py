#!/usr/bin/env python3

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import statistics
import sys
from typing import List, Sequence, Tuple

from test_1to1_traces import (
    CPU_PER_NODE,
    LOCAL_MEM_MB,
    REMOTE_MEM_MB,
    REPO_ROOT,
    WORKLOAD_NAMES,
    run_command,
    write_command_file,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / 'res' / 'random_trace_test'
DEFAULT_LOCAL_RATIO = 66
DEFAULT_MIN_RATIO = 0.5
STRATEGY_TO_POLICY = {
    'uniform': 'auto-shrink',
    'fix': 'fixed-ratio',
    'optimal': 'nonuniform-optimal',
}


@dataclass(frozen=True)
class RandomTrace:
    trace_id: int
    trace_seed: int
    simulation_seed: int
    sampled_tasks: Tuple[str, ...]
    active_workloads: Tuple[str, ...]
    active_counts: Tuple[int, ...]
    counts_by_name: Tuple[Tuple[str, int], ...]

    @property
    def size(self) -> int:
        return len(self.sampled_tasks)

    @property
    def workloads_arg(self) -> str:
        return ','.join(self.active_workloads)

    @property
    def ratios_arg(self) -> str:
        return ':'.join(str(count) for count in self.active_counts)

    @property
    def workload_ratios_arg(self) -> str:
        return ','.join([str(DEFAULT_LOCAL_RATIO)] * len(self.active_workloads))

    @property
    def counts_text(self) -> str:
        return ','.join('{}={}'.format(name, count) for name, count in self.counts_by_name)

    @property
    def sampled_tasks_text(self) -> str:
        return ','.join(self.sampled_tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate purely random task traces and compare directswap against fastswap.'
    )
    parser.add_argument('--num-servers', type=int, default=8, help='number of compute nodes to simulate')
    parser.add_argument('--num-traces', type=int, default=200, help='how many random traces to generate')
    parser.add_argument('--tasks-per-trace', type=int, default=200, help='how many tasks each trace contains')
    parser.add_argument('--until', type=int, default=200, help='max arrival time in seconds')
    parser.add_argument(
        '--arrival-profile',
        choices=('uniform', 'piecewise'),
        default='uniform',
        help='arrival-time distribution passed through to simulation_one_time.py',
    )
    parser.add_argument(
        '--arrival-weights',
        type=lambda s: s.split(','),
        help='comma-separated positive bucket weights for piecewise arrivals, e.g. 1,2,5,2,1',
    )
    parser.add_argument(
        '--strategy',
        choices=tuple(STRATEGY_TO_POLICY.keys()),
        default='optimal',
        help='uniform maps to auto-shrink, fix maps to fixed-ratio, optimal maps to nonuniform-optimal',
    )
    parser.add_argument('--seed-base', type=int, default=5000, help='base seed used for trace generation')
    parser.add_argument(
        '--workloads',
        type=lambda s: s.split(','),
        help='comma-separated workload names to sample uniformly from; defaults to the built-in workload set',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='directory for generated traces, commands, and results',
    )
    return parser.parse_args()


def normalize_arrival_args(arrival_profile: str, arrival_weights: Sequence[str] | None) -> List[float] | None:
    if arrival_profile == 'uniform':
        if arrival_weights is not None:
            raise ValueError('--arrival-weights can only be used with --arrival-profile piecewise')
        return None

    if arrival_weights is None:
        raise ValueError('--arrival-weights is required when --arrival-profile piecewise is used')

    weights = [float(raw_weight) for raw_weight in arrival_weights]
    if not weights:
        raise ValueError('--arrival-weights must not be empty')
    if any(weight <= 0 for weight in weights):
        raise ValueError('--arrival-weights must all be positive')
    return weights


def normalize_workloads(selected: Sequence[str] | None) -> List[str]:
    if selected is None:
        return list(WORKLOAD_NAMES)

    normalized: List[str] = []
    seen = set()
    for raw_name in selected:
        name = raw_name.strip()
        if not name or name in seen:
            continue
        if name not in WORKLOAD_NAMES:
            raise ValueError('unknown workload: {}'.format(name))
        normalized.append(name)
        seen.add(name)

    if not normalized:
        raise ValueError('at least one valid workload must be specified')
    return normalized


def generate_trace(trace_id: int, tasks_per_trace: int, workload_names: Sequence[str], seed_base: int) -> RandomTrace:
    trace_seed = seed_base + trace_id
    simulation_seed = seed_base + 100000 + trace_id
    rng = random.Random(trace_seed)
    sampled_tasks = tuple(rng.choice(workload_names) for _ in range(tasks_per_trace))
    counts = Counter(sampled_tasks)
    active_workloads = tuple(name for name in workload_names if counts[name] > 0)
    active_counts = tuple(counts[name] for name in active_workloads)
    counts_by_name = tuple((name, counts[name]) for name in workload_names)
    return RandomTrace(
        trace_id=trace_id,
        trace_seed=trace_seed,
        simulation_seed=simulation_seed,
        sampled_tasks=sampled_tasks,
        active_workloads=active_workloads,
        active_counts=active_counts,
        counts_by_name=counts_by_name,
    )


def build_command(
    trace: RandomTrace,
    num_servers: int,
    until: int,
    policy: str,
    fastswap: bool,
    arrival_profile: str,
    arrival_weights: Sequence[float] | None,
) -> List[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / 'simulation_one_time.py'),
        str(trace.simulation_seed),
        '--num_servers',
        str(num_servers),
        '--cpus',
        str(CPU_PER_NODE),
        '--mem',
        str(LOCAL_MEM_MB),
        '--workload',
        trace.workloads_arg,
        '--ratios',
        trace.ratios_arg,
        '--workload_ratios',
        trace.workload_ratios_arg,
        '--remotemem',
        '--until',
        str(until),
        '--size',
        str(trace.size),
        '--max_far',
        str(num_servers * REMOTE_MEM_MB),
        '--arrival-profile',
        arrival_profile,
        '--policy',
        policy,
        '--min_ratio',
        str(DEFAULT_MIN_RATIO),
    ]
    if arrival_weights is not None:
        command.extend(['--arrival-weights', ','.join('{:g}'.format(weight) for weight in arrival_weights)])
    if policy == 'nonuniform-optimal':
        command.append('--use_shrink')
    if fastswap:
        command.append('--use_fastswap')
    return command


def winner_from_ratio(fastswap_over_direct: float, tolerance: float = 1e-9) -> str:
    if fastswap_over_direct > 1.0 + tolerance:
        return 'directswap'
    if fastswap_over_direct < 1.0 - tolerance:
        return 'fastswap'
    return 'tie'


def write_trace_metadata(
    path: Path,
    traces: Sequence[RandomTrace],
    strategy: str,
    policy: str,
    num_servers: int,
    until: int,
    arrival_profile: str,
    arrival_weights: Sequence[float] | None,
) -> None:
    arrival_weights_text = '' if arrival_weights is None else ','.join('{:g}'.format(weight) for weight in arrival_weights)
    with path.open('w') as handle:
        handle.write(
            'trace_id\tstrategy\tpolicy\tarrival_profile\tarrival_weights\ttrace_seed\tsimulation_seed\tnum_servers\tuntil_s\tjobs\t'
            'cpu_per_node\tlocal_mem_mb\tremote_mem_per_node_mb\tworkloads\tratios\tcounts\tsampled_tasks\n'
        )
        for trace in traces:
            handle.write(
                'trace{idx:03d}\t{strategy}\t{policy}\t{arrival_profile}\t{arrival_weights}\t{trace_seed}\t{simulation_seed}\t{servers}\t{until}\t{jobs}\t'
                '{cpu}\t{local_mem}\t{remote_mem}\t{workloads}\t{ratios}\t{counts}\t{sampled_tasks}\n'.format(
                    idx=trace.trace_id,
                    strategy=strategy,
                    policy=policy,
                    arrival_profile=arrival_profile,
                    arrival_weights=arrival_weights_text,
                    trace_seed=trace.trace_seed,
                    simulation_seed=trace.simulation_seed,
                    servers=num_servers,
                    until=until,
                    jobs=trace.size,
                    cpu=CPU_PER_NODE,
                    local_mem=LOCAL_MEM_MB,
                    remote_mem=REMOTE_MEM_MB,
                    workloads=trace.workloads_arg,
                    ratios=trace.ratios_arg,
                    counts=trace.counts_text,
                    sampled_tasks=trace.sampled_tasks_text,
                )
            )


def write_summary(
    path: Path,
    strategy: str,
    policy: str,
    num_traces: int,
    tasks_per_trace: int,
    arrival_profile: str,
    arrival_weights: Sequence[float] | None,
    direct_metrics: Sequence[float],
    fast_metrics: Sequence[float],
    ratios: Sequence[float],
    winners: Sequence[str],
) -> None:
    direct_wins = sum(1 for winner in winners if winner == 'directswap')
    fast_wins = sum(1 for winner in winners if winner == 'fastswap')
    ties = sum(1 for winner in winners if winner == 'tie')
    arrival_weights_text = '' if arrival_weights is None else ','.join('{:g}'.format(weight) for weight in arrival_weights)

    with path.open('w') as handle:
        handle.write(
            'strategy\tpolicy\tarrival_profile\tarrival_weights\tnum_traces\ttasks_per_trace\tdirectswap_wins\tfastswap_wins\tties\t'
            'avg_directswap_makespan\tavg_fastswap_makespan\tavg_fastswap_over_direct\t'
            'median_fastswap_over_direct\tbest_fastswap_over_direct\tworst_fastswap_over_direct\n'
        )
        handle.write(
            '{strategy}\t{policy}\t{arrival_profile}\t{arrival_weights}\t{num_traces}\t{tasks_per_trace}\t{direct_wins}\t{fast_wins}\t{ties}\t'
            '{avg_direct:.6f}\t{avg_fast:.6f}\t{avg_ratio:.6f}\t{median_ratio:.6f}\t{best_ratio:.6f}\t{worst_ratio:.6f}\n'.format(
                strategy=strategy,
                policy=policy,
                arrival_profile=arrival_profile,
                arrival_weights=arrival_weights_text,
                num_traces=num_traces,
                tasks_per_trace=tasks_per_trace,
                direct_wins=direct_wins,
                fast_wins=fast_wins,
                ties=ties,
                avg_direct=statistics.mean(direct_metrics),
                avg_fast=statistics.mean(fast_metrics),
                avg_ratio=statistics.mean(ratios),
                median_ratio=statistics.median(ratios),
                best_ratio=max(ratios),
                worst_ratio=min(ratios),
            )
        )


def main() -> None:
    args = parse_args()
    if args.num_traces <= 0:
        raise ValueError('--num-traces must be positive')
    if args.tasks_per_trace <= 0:
        raise ValueError('--tasks-per-trace must be positive')
    arrival_weights = normalize_arrival_args(args.arrival_profile, args.arrival_weights)

    workload_names = normalize_workloads(args.workloads)
    policy = STRATEGY_TO_POLICY[args.strategy]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    traces = [
        generate_trace(trace_id, args.tasks_per_trace, workload_names, args.seed_base)
        for trace_id in range(1, args.num_traces + 1)
    ]

    traces_path = args.output_dir / 'traces.tsv'
    results_path = args.output_dir / 'results.tsv'
    summary_path = args.output_dir / 'summary.tsv'
    direct_path = args.output_dir / 'run_directswap.sh'
    fastswap_path = args.output_dir / 'run_fastswap.sh'

    direct_commands: List[List[str]] = []
    fastswap_commands: List[List[str]] = []
    direct_makespans: List[float] = []
    fastswap_makespans: List[float] = []
    fastswap_over_direct_values: List[float] = []
    winners: List[str] = []

    write_trace_metadata(
        traces_path,
        traces,
        args.strategy,
        policy,
        args.num_servers,
        args.until,
        args.arrival_profile,
        arrival_weights,
    )

    with results_path.open('w') as handle:
        handle.write(
            'trace_id\tstrategy\tpolicy\tarrival_profile\tarrival_weights\ttrace_seed\tsimulation_seed\tworkloads\tratios\tjobs\t'
            'directswap_makespan\tdirectswap_avg_mem_util\tdirectswap_avg_queue_ms\t'
            'directswap_avg_runtime_slowdown\tfastswap_makespan\tfastswap_avg_mem_util\t'
            'fastswap_avg_queue_ms\tfastswap_avg_runtime_slowdown\tfastswap_over_direct\twinner\n'
        )

        for trace in traces:
            direct_cmd = build_command(
                trace,
                args.num_servers,
                args.until,
                policy,
                False,
                args.arrival_profile,
                arrival_weights,
            )
            fastswap_cmd = build_command(
                trace,
                args.num_servers,
                args.until,
                policy,
                True,
                args.arrival_profile,
                arrival_weights,
            )
            direct_commands.append(direct_cmd)
            fastswap_commands.append(fastswap_cmd)

            direct_metrics = run_command(direct_cmd)
            fastswap_metrics = run_command(fastswap_cmd)
            direct_makespan = direct_metrics['makespan']
            fastswap_makespan = fastswap_metrics['makespan']
            fastswap_over_direct = fastswap_makespan / direct_makespan
            winner = winner_from_ratio(fastswap_over_direct)

            direct_makespans.append(direct_makespan)
            fastswap_makespans.append(fastswap_makespan)
            fastswap_over_direct_values.append(fastswap_over_direct)
            winners.append(winner)

            print(
                '[trace {idx:03d}/{total:03d}] strategy={strategy} workloads={workloads} ratios={ratios} jobs={jobs} '
                'arrival={arrival_profile} direct={direct:.0f} direct_mem_util={direct_mem_util:.2%} '
                'fast={fast:.0f} fast_mem_util={fast_mem_util:.2%} fast/direct={ratio:.4f} winner={winner}'.format(
                    idx=trace.trace_id,
                    total=len(traces),
                    strategy=args.strategy,
                    workloads=trace.workloads_arg,
                    ratios=trace.ratios_arg,
                    jobs=trace.size,
                    arrival_profile=args.arrival_profile,
                    direct=direct_makespan,
                    direct_mem_util=direct_metrics['avg_mem_util'],
                    fast=fastswap_makespan,
                    fast_mem_util=fastswap_metrics['avg_mem_util'],
                    ratio=fastswap_over_direct,
                    winner=winner,
                )
            )

            handle.write(
                'trace{idx:03d}\t{strategy}\t{policy}\t{arrival_profile}\t{arrival_weights}\t{trace_seed}\t{simulation_seed}\t{workloads}\t{ratios}\t{jobs}\t'
                '{direct:.6f}\t{direct_mem_util:.6f}\t{direct_queue:.6f}\t{direct_slowdown:.6f}\t'
                '{fast:.6f}\t{fast_mem_util:.6f}\t{fast_queue:.6f}\t{fast_slowdown:.6f}\t{ratio:.6f}\t{winner}\n'.format(
                    idx=trace.trace_id,
                    strategy=args.strategy,
                    policy=policy,
                    arrival_profile=args.arrival_profile,
                    arrival_weights='' if arrival_weights is None else ','.join('{:g}'.format(weight) for weight in arrival_weights),
                    trace_seed=trace.trace_seed,
                    simulation_seed=trace.simulation_seed,
                    workloads=trace.workloads_arg,
                    ratios=trace.ratios_arg,
                    jobs=trace.size,
                    direct=direct_makespan,
                    direct_mem_util=direct_metrics['avg_mem_util'],
                    direct_queue=direct_metrics['avg_queue_time'],
                    direct_slowdown=direct_metrics['avg_runtime_slowdown'],
                    fast=fastswap_makespan,
                    fast_mem_util=fastswap_metrics['avg_mem_util'],
                    fast_queue=fastswap_metrics['avg_queue_time'],
                    fast_slowdown=fastswap_metrics['avg_runtime_slowdown'],
                    ratio=fastswap_over_direct,
                    winner=winner,
                )
            )
            handle.flush()

    write_summary(
        summary_path,
        args.strategy,
        policy,
        args.num_traces,
        args.tasks_per_trace,
        args.arrival_profile,
        arrival_weights,
        direct_makespans,
        fastswap_makespans,
        fastswap_over_direct_values,
        winners,
    )
    write_command_file(direct_path, direct_commands)
    write_command_file(fastswap_path, fastswap_commands)

    print('wrote {}'.format(traces_path))
    print('wrote {}'.format(results_path))
    print('wrote {}'.format(summary_path))
    print('wrote {}'.format(direct_path))
    print('wrote {}'.format(fastswap_path))


if __name__ == '__main__':
    main()
