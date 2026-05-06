#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

from test_1to1_traces import (
    active_entries,
    build_command,
    build_specs,
    enumerate_candidates,
    run_command,
    write_command_file,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH_OUTPUT_DIR = REPO_ROOT / 'res' / 'directswap_best_cases'
SEARCH_POLICY = 'nonuniform-optimal'


@dataclass(frozen=True)
class RankedCase:
    rank: int
    candidate_index: int
    workloads: str
    ratios: str
    jobs: int
    cpu_per_node: int
    mem_per_node_mb: int
    remote_per_node_mb: int
    directswap_makespan: float
    fastswap_makespan: float
    fastswap_over_direct: float
    makespan_delta: float
    directswap_avg_mem_util: float
    fastswap_avg_mem_util: float
    directswap_avg_queue_ms: float
    fastswap_avg_queue_ms: float
    directswap_avg_runtime_slowdown: float
    fastswap_avg_runtime_slowdown: float
    reasons: str
    seed: int


def mem_util_gain_pct(case: RankedCase) -> float:
    if case.fastswap_avg_mem_util <= 0:
        return 0.0
    return (case.directswap_avg_mem_util / case.fastswap_avg_mem_util - 1) * 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Search workload mixes where directswap outperforms fastswap and keep the top cases.'
    )
    parser.add_argument(
        '--policy',
        choices=('auto-shrink', 'fixed-ratio', 'hybrid-fixed-ratio', 'nonuniform', 'nonuniform-optimal'),
        default='nonuniform-optimal',
        help='memory placement policy to use during the search',
    )
    parser.add_argument('--num-servers', type=int, default=8, help='number of compute nodes to simulate')
    parser.add_argument('--repeats', type=int, default=4, help='deprecated alias for workload percent in 100%% steps')
    parser.add_argument(
        '--workload-percent',
        type=int,
        help='total workload size as a percentage of one per-node pack across the cluster; e.g. 1000 means 10x',
    )
    parser.add_argument('--until', type=int, default=200, help='max arrival time in seconds')
    parser.add_argument(
        '--candidate-limit',
        type=int,
        default=200,
        help='how many top memory-fit candidates to evaluate before ranking directswap wins',
    )
    parser.add_argument('--top-k', type=int, default=20, help='how many winning cases to keep')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_SEARCH_OUTPUT_DIR,
        help='directory for ranked cases, commands, and results',
    )
    return parser.parse_args()


def write_ranked_results(path: Path, ranked_cases: List[RankedCase]) -> None:
    with path.open('w') as handle:
        handle.write(
            'rank\tcandidate_index\tpolicy\treasons\tworkloads\tratios\tjobs\tcpu_per_node\tmem_per_node_mb\t'
            'remote_per_node_mb\tdirectswap_makespan\tfastswap_makespan\tfastswap_over_direct\t'
            'makespan_delta\tdirectswap_avg_mem_util\tfastswap_avg_mem_util\t'
            'directswap_mem_util_gain_pct\t'
            'directswap_avg_queue_ms\tfastswap_avg_queue_ms\t'
            'directswap_avg_runtime_slowdown\tfastswap_avg_runtime_slowdown\tseed\n'
        )
        for case in ranked_cases:
            handle.write(
                '{rank}\t{candidate_index}\t{policy}\t{reasons}\t{workloads}\t{ratios}\t{jobs}\t{cpu}\t{mem}\t'
                '{remote}\t{direct:.6f}\t{fast:.6f}\t{ratio:.6f}\t{delta:.6f}\t'
                '{direct_mem_util:.6f}\t{fast_mem_util:.6f}\t{mem_gain:.6f}\t'
                '{direct_queue:.6f}\t{fast_queue:.6f}\t'
                '{direct_slowdown:.6f}\t{fast_slowdown:.6f}\t{seed}\n'.format(
                    rank=case.rank,
                    candidate_index=case.candidate_index,
                    policy=SEARCH_POLICY,
                    reasons=case.reasons,
                    workloads=case.workloads,
                    ratios=case.ratios,
                    jobs=case.jobs,
                    cpu=case.cpu_per_node,
                    mem=case.mem_per_node_mb,
                    remote=case.remote_per_node_mb,
                    direct=case.directswap_makespan,
                    fast=case.fastswap_makespan,
                    ratio=case.fastswap_over_direct,
                    delta=case.makespan_delta,
                    direct_mem_util=case.directswap_avg_mem_util,
                    fast_mem_util=case.fastswap_avg_mem_util,
                    mem_gain=mem_util_gain_pct(case),
                    direct_queue=case.directswap_avg_queue_ms,
                    fast_queue=case.fastswap_avg_queue_ms,
                    direct_slowdown=case.directswap_avg_runtime_slowdown,
                    fast_slowdown=case.fastswap_avg_runtime_slowdown,
                    seed=case.seed,
                )
            )


def write_all_results(path: Path, evaluated_cases: List[RankedCase]) -> None:
    with path.open('w') as handle:
        handle.write(
            'candidate_index\tpolicy\treasons\tworkloads\tratios\tjobs\tcpu_per_node\tmem_per_node_mb\t'
            'remote_per_node_mb\tdirectswap_makespan\tfastswap_makespan\tfastswap_over_direct\t'
            'directswap_gain_pct\tdirectswap_avg_mem_util\tfastswap_avg_mem_util\t'
            'directswap_mem_util_gain_pct\tdirectswap_avg_queue_ms\tfastswap_avg_queue_ms\t'
            'directswap_avg_runtime_slowdown\tfastswap_avg_runtime_slowdown\tseed\n'
        )
        for case in evaluated_cases:
            handle.write(
                '{candidate_index}\t{policy}\t{reasons}\t{workloads}\t{ratios}\t{jobs}\t{cpu}\t{mem}\t'
                '{remote}\t{direct:.6f}\t{fast:.6f}\t{ratio:.6f}\t{gain:.6f}\t'
                '{direct_mem_util:.6f}\t{fast_mem_util:.6f}\t{mem_gain:.6f}\t'
                '{direct_queue:.6f}\t{fast_queue:.6f}\t'
                '{direct_slowdown:.6f}\t{fast_slowdown:.6f}\t{seed}\n'.format(
                    candidate_index=case.candidate_index,
                    policy=SEARCH_POLICY,
                    reasons=case.reasons,
                    workloads=case.workloads,
                    ratios=case.ratios,
                    jobs=case.jobs,
                    cpu=case.cpu_per_node,
                    mem=case.mem_per_node_mb,
                    remote=case.remote_per_node_mb,
                    direct=case.directswap_makespan,
                    fast=case.fastswap_makespan,
                    ratio=case.fastswap_over_direct,
                    gain=(case.fastswap_over_direct - 1) * 100,
                    direct_mem_util=case.directswap_avg_mem_util,
                    fast_mem_util=case.fastswap_avg_mem_util,
                    mem_gain=mem_util_gain_pct(case),
                    direct_queue=case.directswap_avg_queue_ms,
                    fast_queue=case.fastswap_avg_queue_ms,
                    direct_slowdown=case.directswap_avg_runtime_slowdown,
                    fast_slowdown=case.fastswap_avg_runtime_slowdown,
                    seed=case.seed,
                )
            )


def write_cases_metadata(path: Path, ranked_cases: List[RankedCase]) -> None:
    with path.open('w') as handle:
        handle.write(
            'rank\tcandidate_index\tpolicy\treasons\tworkloads\tratios\tjobs\tcpu_per_node\tmem_per_node_mb\t'
            'remote_per_node_mb\tdirectswap_gain_pct\tdirectswap_avg_mem_util\t'
            'fastswap_avg_mem_util\tdirectswap_mem_util_gain_pct\tseed\n'
        )
        for case in ranked_cases:
            handle.write(
                '{rank}\t{candidate_index}\t{policy}\t{reasons}\t{workloads}\t{ratios}\t{jobs}\t{cpu}\t{mem}\t'
                '{remote}\t{gain:.4f}\t{direct_mem_util:.6f}\t{fast_mem_util:.6f}\t{mem_gain:.6f}\t{seed}\n'.format(
                    rank=case.rank,
                    candidate_index=case.candidate_index,
                    policy=SEARCH_POLICY,
                    reasons=case.reasons,
                    workloads=case.workloads,
                    ratios=case.ratios,
                    jobs=case.jobs,
                    cpu=case.cpu_per_node,
                    mem=case.mem_per_node_mb,
                    remote=case.remote_per_node_mb,
                    gain=(case.fastswap_over_direct - 1) * 100,
                    direct_mem_util=case.directswap_avg_mem_util,
                    fast_mem_util=case.fastswap_avg_mem_util,
                    mem_gain=mem_util_gain_pct(case),
                    seed=case.seed,
                )
            )


def main() -> None:
    global SEARCH_POLICY
    args = parse_args()
    SEARCH_POLICY = args.policy
    if args.workload_percent is None:
        args.workload_percent = args.repeats * 100
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs()
    candidates = enumerate_candidates(specs)
    if args.candidate_limit > 0:
        candidates = candidates[:args.candidate_limit]

    winner_commands: Dict[int, Dict[str, List[str]]] = {}
    evaluated_cases: List[RankedCase] = []
    winners: List[RankedCase] = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        seed = 2000 + candidate_index
        active = active_entries(candidate, specs)
        direct_cmd, size = build_command(
            candidate,
            specs,
            seed,
            args.num_servers,
            args.workload_percent,
            args.until,
            False,
            args.policy,
        )
        fastswap_cmd, _ = build_command(
            candidate,
            specs,
            seed,
            args.num_servers,
            args.workload_percent,
            args.until,
            True,
            args.policy,
        )
        direct_metrics = run_command(direct_cmd)
        fastswap_metrics = run_command(fastswap_cmd)
        directswap_makespan = direct_metrics['makespan']
        fastswap_makespan = fastswap_metrics['makespan']
        fastswap_over_direct = fastswap_makespan / directswap_makespan

        print(
            '[candidate {idx:03d}/{total:03d}] policy={policy} workload_percent={workload_percent} workloads={workloads} ratios={ratios} '
            'cpu={cpu} mem={mem} direct={direct:.0f} direct_mem_util={direct_mem_util:.2%} '
            'fast={fast:.0f} fast_mem_util={fast_mem_util:.2%} mem_util_gain={mem_gain:.2f}% '
            'fast/direct={ratio:.4f}'.format(
                idx=candidate_index,
                total=len(candidates),
                policy=args.policy,
                workload_percent=args.workload_percent,
                workloads=','.join(name for name, _ in active),
                ratios=':'.join(str(count) for _, count in active),
                cpu=candidate.cpu,
                mem=candidate.mem,
                direct=directswap_makespan,
                direct_mem_util=direct_metrics['avg_mem_util'],
                fast=fastswap_makespan,
                fast_mem_util=fastswap_metrics['avg_mem_util'],
                mem_gain=((direct_metrics['avg_mem_util'] / fastswap_metrics['avg_mem_util'] - 1) * 100)
                if fastswap_metrics['avg_mem_util'] > 0 else 0.0,
                ratio=fastswap_over_direct,
            )
        )

        evaluated_case = RankedCase(
            rank=0,
            candidate_index=candidate_index,
            workloads=','.join(name for name, _ in active),
            ratios=':'.join(str(count) for _, count in active),
            jobs=size,
            cpu_per_node=candidate.cpu,
            mem_per_node_mb=candidate.mem,
            remote_per_node_mb=candidate.remote_mem,
            directswap_makespan=directswap_makespan,
            fastswap_makespan=fastswap_makespan,
            fastswap_over_direct=fastswap_over_direct,
            makespan_delta=fastswap_makespan - directswap_makespan,
            directswap_avg_mem_util=direct_metrics['avg_mem_util'],
            fastswap_avg_mem_util=fastswap_metrics['avg_mem_util'],
            directswap_avg_queue_ms=direct_metrics['avg_queue_time'],
            fastswap_avg_queue_ms=fastswap_metrics['avg_queue_time'],
            directswap_avg_runtime_slowdown=direct_metrics['avg_runtime_slowdown'],
            fastswap_avg_runtime_slowdown=fastswap_metrics['avg_runtime_slowdown'],
            reasons='directswap-beats-fastswap' if fastswap_over_direct > 1 else 'fastswap-beats-or-ties',
            seed=seed,
        )
        evaluated_cases.append(evaluated_case)

        if fastswap_over_direct <= 1:
            continue

        winner_commands[candidate_index] = {
            'direct': direct_cmd,
            'fast': fastswap_cmd,
        }
        winners.append(evaluated_case)

    evaluated_cases.sort(
        key=lambda case: (
            -case.fastswap_over_direct,
            -case.makespan_delta,
            -mem_util_gain_pct(case),
            case.candidate_index,
        )
    )

    winners.sort(
        key=lambda case: (
            -case.fastswap_over_direct,
            -case.makespan_delta,
            -case.remote_per_node_mb,
            case.workloads,
        )
    )
    winners = [replace(case, rank=index) for index, case in enumerate(winners[:args.top_k], start=1)]

    direct_commands = [winner_commands[case.candidate_index]['direct'] for case in winners]
    fastswap_commands = [winner_commands[case.candidate_index]['fast'] for case in winners]

    results_path = args.output_dir / 'ranked_results.tsv'
    all_results_path = args.output_dir / 'all_results.tsv'
    cases_path = args.output_dir / 'ranked_cases.tsv'
    direct_path = args.output_dir / 'run_directswap.sh'
    fastswap_path = args.output_dir / 'run_fastswap.sh'

    write_all_results(all_results_path, evaluated_cases)
    write_ranked_results(results_path, winners)
    write_cases_metadata(cases_path, winners)
    write_command_file(direct_path, direct_commands)
    write_command_file(fastswap_path, fastswap_commands)

    print('kept {} directswap-winning cases'.format(len(winners)))
    print('wrote {}'.format(cases_path))
    print('wrote {}'.format(all_results_path))
    print('wrote {}'.format(results_path))
    print('wrote {}'.format(direct_path))
    print('wrote {}'.format(fastswap_path))


if __name__ == '__main__':
    main()
