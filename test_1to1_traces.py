#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Dict, List, Tuple

import workloads_simu as workloads


CPU_PER_NODE = 96
LOCAL_MEM_MB = 128 * 1024
REMOTE_MEM_MB = 32 * 1024
TOTAL_MEM_MB = LOCAL_MEM_MB + REMOTE_MEM_MB
WORKLOAD_LOCAL_RATIO = 80
WORKLOAD_NAMES = [
    'quicksort',
    'kmeans',
    'wordcount',
    'linearregression',
    'xgboost',
    'xsbench',
    'snappy',
    'pagerank',
    'redis',
    'graph500',
    'llama',
    'qs_c18_m30000',
    'qs_c18_m50000',
    'qs_c18_m80000',
    'qs_c30_m30000',
    'qs_c30_m50000',
    'qs_c30_m80000',
    'qs_c48_m30000',
    'qs_c48_m50000',
    'qs_c48_m80000',
]
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'res' / 'one_to_one_trace_test'


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    cpu_req: int
    ideal_mem: int


@dataclass(frozen=True)
class Candidate:
    counts: Tuple[int, ...]
    cpu: int
    mem: int
    score: float

    @property
    def cpu_util(self) -> float:
        return self.cpu / CPU_PER_NODE

    @property
    def mem_util(self) -> float:
        return self.mem / TOTAL_MEM_MB

    @property
    def remote_mem(self) -> int:
        return self.mem - LOCAL_MEM_MB

    @property
    def local_ratio(self) -> float:
        return LOCAL_MEM_MB / self.mem


@dataclass
class Case:
    candidate: Candidate
    reasons: List[str] = field(default_factory=list)


def build_specs() -> List[WorkloadSpec]:
    specs = []
    for name in WORKLOAD_NAMES:
        workload = workloads.get_workload_class(name)
        specs.append(WorkloadSpec(name, workload.cpu_req, workload.ideal_mem))
    return specs


def score_candidate(_cpu: int, mem: int) -> float:
    return mem / TOTAL_MEM_MB


def candidate_sort_key(candidate: Candidate) -> Tuple[float, int, Tuple[int, ...]]:
    return (-candidate.score, -candidate.mem, candidate.counts)


def active_entries(candidate: Candidate, specs: List[WorkloadSpec]) -> List[Tuple[str, int]]:
    return [
        (spec.name, count)
        for spec, count in zip(specs, candidate.counts)
        if count > 0
    ]


def enumerate_candidates(specs: List[WorkloadSpec]) -> List[Candidate]:
    candidates: List[Candidate] = []

    def dfs(index: int, counts: List[int], cpu: int, mem: int) -> None:
        if cpu > CPU_PER_NODE or mem > TOTAL_MEM_MB:
            return

        if index == len(specs):
            if cpu == 0 or mem <= LOCAL_MEM_MB:
                return
            candidates.append(Candidate(tuple(counts), cpu, mem, score_candidate(cpu, mem)))
            return

        spec = specs[index]
        max_count = min(
            (CPU_PER_NODE - cpu) // spec.cpu_req,
            (TOTAL_MEM_MB - mem) // spec.ideal_mem,
        )
        for count in range(max_count + 1):
            counts.append(count)
            dfs(index + 1, counts, cpu + count * spec.cpu_req, mem + count * spec.ideal_mem)
            counts.pop()

    dfs(0, [], 0, 0)
    candidates.sort(key=candidate_sort_key)
    return candidates


def select_cases(candidates: List[Candidate], specs: List[WorkloadSpec]) -> List[Case]:
    selected: Dict[Tuple[int, ...], Case] = {}

    def add_case(candidate: Candidate, reason: str) -> None:
        case = selected.get(candidate.counts)
        if case is None:
            selected[candidate.counts] = Case(candidate=candidate, reasons=[reason])
        elif reason not in case.reasons:
            case.reasons.append(reason)

    add_case(candidates[0], 'best-overall')
    for index, spec in enumerate(specs):
        for candidate in candidates:
            if candidate.counts[index] > 0:
                add_case(candidate, 'best-with-{}'.format(spec.name))
                break

    cases = list(selected.values())
    cases.sort(key=lambda case: candidate_sort_key(case.candidate))
    return cases


def build_command(
    candidate: Candidate,
    specs: List[WorkloadSpec],
    seed: int,
    num_servers: int,
    workload_percent: int,
    until: int,
    fastswap: bool,
    policy: str,
) -> Tuple[List[str], int]:
    active = active_entries(candidate, specs)
    base_pack_jobs = sum(count for _, count in active) * num_servers
    if workload_percent % 100 != 0:
        raise ValueError('workload_percent must be a multiple of 100')
    size = base_pack_jobs * workload_percent // 100
    command = [
        sys.executable,
        str(REPO_ROOT / 'simulation_one_time.py'),
        str(seed),
        '--num_servers',
        str(num_servers),
        '--cpus',
        str(CPU_PER_NODE),
        '--mem',
        str(LOCAL_MEM_MB),
        '--workload',
        ','.join(name for name, _ in active),
        '--ratios',
        ':'.join(str(count) for _, count in active),
        '--workload_ratios',
        ','.join([str(WORKLOAD_LOCAL_RATIO)] * len(active)),
        '--remotemem',
        '--until',
        str(until),
        '--size',
        str(size),
        '--max_far',
        str(num_servers * REMOTE_MEM_MB),
        '--policy',
        policy,
        '--min_ratio',
        str(0.5),
    ]
    if policy == 'nonuniform-optimal':
        command.append('--use_shrink')
    if fastswap:
        command.append('--use_fastswap')
    return command, size


def run_command(command: List[str]) -> Dict[str, float]:
    metrics_command = command + ['--json-metrics']
    completed = subprocess.run(
        metrics_command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            'command failed: {}\nstdout:\n{}\nstderr:\n{}'.format(
                shlex.join(metrics_command),
                completed.stdout.strip(),
                completed.stderr.strip(),
            )
        )
    output = completed.stdout.strip()
    try:
        metrics = json.loads(output)
        return {
            'makespan': float(metrics['makespan']),
            'avg_mem_util': float(metrics['avg_mem_util']),
            'avg_remote_mem_util': float(metrics['avg_remote_mem_util']),
            'avg_queue_time': float(metrics['avg_queue_time']),
            'avg_runtime_slowdown': float(metrics['avg_runtime_slowdown']),
            'avg_turnaround_slowdown': float(metrics['avg_turnaround_slowdown']),
        }
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            'unexpected simulation output for {}: {!r}'.format(shlex.join(metrics_command), output)
        ) from exc


def write_case_metadata(path: Path, cases: List[Case], specs: List[WorkloadSpec], num_servers: int, workload_percent: int, until: int, policy: str) -> None:
    with path.open('w') as handle:
        handle.write(
            'case_id\tpolicy\tworkload_percent\treasons\tworkloads\tratios\tjobs\tcpu_per_node\tmem_per_node_mb\t'
            'cpu_util\tmem_util\tremote_per_node_mb\tlocal_ratio\tseed\tuntil_s\tnum_servers\n'
        )
        for index, case in enumerate(cases, start=1):
            candidate = case.candidate
            active = active_entries(candidate, specs)
            seed = 1000 + index
            _, size = build_command(candidate, specs, seed, num_servers, workload_percent, until, False, policy)
            handle.write(
                'case{idx:02d}\t{policy}\t{workload_percent}\t{reasons}\t{workloads}\t{ratios}\t{size}\t{cpu}\t{mem}\t'
                '{cpu_util:.4f}\t{mem_util:.4f}\t{remote}\t{local_ratio:.4f}\t{seed}\t{until}\t{servers}\n'.format(
                    idx=index,
                    policy=policy,
                    workload_percent=workload_percent,
                    reasons=','.join(case.reasons),
                    workloads=','.join(name for name, _ in active),
                    ratios=':'.join(str(count) for _, count in active),
                    size=size,
                    cpu=candidate.cpu,
                    mem=candidate.mem,
                    cpu_util=candidate.cpu_util,
                    mem_util=candidate.mem_util,
                    remote=candidate.remote_mem,
                    local_ratio=candidate.local_ratio,
                    seed=seed,
                    until=until,
                    servers=num_servers,
                )
            )


def write_command_file(path: Path, commands: List[List[str]]) -> None:
    with path.open('w') as handle:
        handle.write('#!/usr/bin/env bash\n')
        handle.write('set -euo pipefail\n')
        for command in commands:
            handle.write('{}\n'.format(shlex.join(command)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate and run 1:1 local/remote trace tests near the memory limit while respecting the CPU limit.'
    )
    parser.add_argument('--num-servers', type=int, default=8, help='number of compute nodes to simulate')
    parser.add_argument('--repeats', type=int, default=4, help='deprecated alias for workload percent in 100%% steps')
    parser.add_argument(
        '--workload-percent',
        type=int,
        help='total workload size as a percentage of one per-node pack across the cluster; e.g. 1000 means 10x',
    )
    parser.add_argument('--until', type=int, default=200, help='max arrival time in seconds')
    parser.add_argument('--limit', type=int, default=0, help='limit the number of selected cases; 0 means all')
    parser.add_argument(
        '--policy',
        choices=('auto-shrink', 'fixed-ratio', 'hybrid-fixed-ratio', 'nonuniform', 'nonuniform-optimal'),
        default='auto-shrink',
        help='memory placement policy to use for both directswap and fastswap',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='directory for generated commands and results',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workload_percent is None:
        args.workload_percent = args.repeats * 100
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs()
    candidates = enumerate_candidates(specs)
    if not candidates:
        raise RuntimeError('no workload combination fits the 32 CPU / 128GB per-node target')

    cases = select_cases(candidates, specs)
    if args.limit > 0:
        cases = cases[:args.limit]

    direct_commands: List[List[str]] = []
    fastswap_commands: List[List[str]] = []
    results_path = args.output_dir / 'results.tsv'
    cases_path = args.output_dir / 'cases.tsv'
    direct_path = args.output_dir / 'run_directswap.sh'
    fastswap_path = args.output_dir / 'run_fastswap.sh'

    write_case_metadata(cases_path, cases, specs, args.num_servers, args.workload_percent, args.until, args.policy)

    with results_path.open('w') as results_file:
        results_file.write(
            'case_id\tpolicy\tworkload_percent\treasons\tworkloads\tratios\tjobs\tcpu_per_node\tmem_per_node_mb\t'
            'remote_per_node_mb\tdirectswap_makespan\tdirectswap_avg_mem_util\t'
            'directswap_avg_queue_ms\tdirectswap_avg_runtime_slowdown\t'
            'fastswap_makespan\tfastswap_avg_mem_util\tfastswap_avg_queue_ms\t'
            'fastswap_avg_runtime_slowdown\tfastswap_over_direct\n'
        )

        for index, case in enumerate(cases, start=1):
            seed = 1000 + index
            active = active_entries(case.candidate, specs)
            direct_cmd, size = build_command(
                case.candidate,
                specs,
                seed,
                args.num_servers,
                args.workload_percent,
                args.until,
                False,
                args.policy,
            )
            fastswap_cmd, _ = build_command(
                case.candidate,
                specs,
                seed,
                args.num_servers,
                args.workload_percent,
                args.until,
                True,
                args.policy,
            )

            direct_commands.append(direct_cmd)
            fastswap_commands.append(fastswap_cmd)

            print(
                '[case {idx:02d}/{total:02d}] policy={policy} workload_percent={workload_percent} workloads={workloads} ratios={ratios} '
                'cpu={cpu}/32 mem={mem}/{mem_cap}MB remote={remote}/{remote_cap}MB jobs={jobs}'.format(
                    idx=index,
                    total=len(cases),
                    policy=args.policy,
                    workload_percent=args.workload_percent,
                    workloads=','.join(name for name, _ in active),
                    ratios=':'.join(str(count) for _, count in active),
                    cpu=case.candidate.cpu,
                    mem=case.candidate.mem,
                    mem_cap=TOTAL_MEM_MB,
                    remote=case.candidate.remote_mem,
                    remote_cap=REMOTE_MEM_MB,
                    jobs=size,
                )
            )

            direct_metrics = run_command(direct_cmd)
            fastswap_metrics = run_command(fastswap_cmd)
            direct_value = direct_metrics['makespan']
            fastswap_value = fastswap_metrics['makespan']
            ratio = fastswap_value / direct_value

            print(
                '  directswap={:.4f} direct_avg_mem_util={:.2%} direct_avg_queue={:.2f}s '
                'direct_avg_runtime_slowdown={:.3f}x'.format(
                    direct_value,
                    direct_metrics['avg_mem_util'],
                    direct_metrics['avg_queue_time'] / 1000,
                    direct_metrics['avg_runtime_slowdown'],
                )
            )
            print(
                '  fastswap={:.4f} fast_avg_mem_util={:.2%} fast_avg_queue={:.2f}s '
                'fast_avg_runtime_slowdown={:.3f}x fastswap/direct={:.6f}'.format(
                    fastswap_value,
                    fastswap_metrics['avg_mem_util'],
                    fastswap_metrics['avg_queue_time'] / 1000,
                    fastswap_metrics['avg_runtime_slowdown'],
                    ratio,
                )
            )

            results_file.write(
                'case{idx:02d}\t{policy}\t{workload_percent}\t{reasons}\t{workloads}\t{ratios}\t{size}\t{cpu}\t{mem}\t'
                '{remote}\t{direct:.6f}\t{direct_mem_util:.6f}\t{direct_queue:.6f}\t'
                '{direct_slowdown:.6f}\t{fast:.6f}\t{fast_mem_util:.6f}\t'
                '{fast_queue:.6f}\t{fast_slowdown:.6f}\t{ratio:.6f}\n'.format(
                    idx=index,
                    policy=args.policy,
                    workload_percent=args.workload_percent,
                    reasons=','.join(case.reasons),
                    workloads=','.join(name for name, _ in active),
                    ratios=':'.join(str(count) for _, count in active),
                    size=size,
                    cpu=case.candidate.cpu,
                    mem=case.candidate.mem,
                    remote=case.candidate.remote_mem,
                    direct=direct_value,
                    direct_mem_util=direct_metrics['avg_mem_util'],
                    direct_queue=direct_metrics['avg_queue_time'],
                    direct_slowdown=direct_metrics['avg_runtime_slowdown'],
                    fast=fastswap_value,
                    fast_mem_util=fastswap_metrics['avg_mem_util'],
                    fast_queue=fastswap_metrics['avg_queue_time'],
                    fast_slowdown=fastswap_metrics['avg_runtime_slowdown'],
                    ratio=ratio,
                )
            )
            results_file.flush()

    write_command_file(direct_path, direct_commands)
    write_command_file(fastswap_path, fastswap_commands)

    print('wrote {}'.format(cases_path))
    print('wrote {}'.format(results_path))
    print('wrote {}'.format(direct_path))
    print('wrote {}'.format(fastswap_path))


if __name__ == '__main__':
    main()
