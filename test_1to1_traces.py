#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Dict, List, Tuple

import workloads_simu as workloads


CPU_PER_NODE = 64
LOCAL_MEM_MB = 128 * 1024
REMOTE_MEM_MB = 64 * 1024
TOTAL_MEM_MB = LOCAL_MEM_MB + REMOTE_MEM_MB
WORKLOAD_LOCAL_RATIO = 70
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


def score_candidate(cpu: int, mem: int) -> float:
    cpu_util = cpu / CPU_PER_NODE
    mem_util = mem / TOTAL_MEM_MB
    return math.sqrt(cpu_util * mem_util) - 0.35 * abs(cpu_util - mem_util)


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
    candidates.sort(key=lambda candidate: (candidate.score, candidate.cpu, candidate.mem), reverse=True)
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
    cases.sort(key=lambda case: (case.candidate.score, case.candidate.cpu, case.candidate.mem), reverse=True)
    return cases


def build_command(
    candidate: Candidate,
    specs: List[WorkloadSpec],
    seed: int,
    num_servers: int,
    repeats: int,
    until: int,
    fastswap: bool,
) -> Tuple[List[str], int]:
    active = active_entries(candidate, specs)
    size = sum(count for _, count in active) * num_servers * repeats
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
        '--use_shrink',
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
        '--uniform',
        '--min_ratio',
        str(0.5),
    ]
    if fastswap:
        command.append('--use_fastswap')
    return command, size


def run_command(command: List[str]) -> float:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            'command failed: {}\nstdout:\n{}\nstderr:\n{}'.format(
                shlex.join(command),
                completed.stdout.strip(),
                completed.stderr.strip(),
            )
        )
    output = completed.stdout.strip()
    try:
        return float(output)
    except ValueError as exc:
        raise RuntimeError(
            'unexpected simulation output for {}: {!r}'.format(shlex.join(command), output)
        ) from exc


def write_case_metadata(path: Path, cases: List[Case], specs: List[WorkloadSpec], num_servers: int, repeats: int, until: int) -> None:
    with path.open('w') as handle:
        handle.write(
            'case_id\treasons\tworkloads\tratios\tjobs\tcpu_per_node\tmem_per_node_mb\t'
            'cpu_util\tmem_util\tremote_per_node_mb\tlocal_ratio\tseed\tuntil_s\tnum_servers\n'
        )
        for index, case in enumerate(cases, start=1):
            candidate = case.candidate
            active = active_entries(candidate, specs)
            seed = 1000 + index
            _, size = build_command(candidate, specs, seed, num_servers, repeats, until, False)
            handle.write(
                'case{idx:02d}\t{reasons}\t{workloads}\t{ratios}\t{size}\t{cpu}\t{mem}\t'
                '{cpu_util:.4f}\t{mem_util:.4f}\t{remote}\t{local_ratio:.4f}\t{seed}\t{until}\t{servers}\n'.format(
                    idx=index,
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
        description='Generate and run 1:1 local/remote trace tests near the CPU and memory limits.'
    )
    parser.add_argument('--num-servers', type=int, default=8, help='number of compute nodes to simulate')
    parser.add_argument('--repeats', type=int, default=4, help='how many per-node packs to replay in each trace')
    parser.add_argument('--until', type=int, default=200, help='max arrival time in seconds')
    parser.add_argument('--limit', type=int, default=0, help='limit the number of selected cases; 0 means all')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='directory for generated commands and results',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    write_case_metadata(cases_path, cases, specs, args.num_servers, args.repeats, args.until)

    with results_path.open('w') as results_file:
        results_file.write(
            'case_id\treasons\tworkloads\tratios\tjobs\tcpu_per_node\tmem_per_node_mb\t'
            'remote_per_node_mb\tdirectswap_makespan\tfastswap_makespan\tfastswap_over_direct\n'
        )

        for index, case in enumerate(cases, start=1):
            seed = 1000 + index
            active = active_entries(case.candidate, specs)
            direct_cmd, size = build_command(
                case.candidate,
                specs,
                seed,
                args.num_servers,
                args.repeats,
                args.until,
                False,
            )
            fastswap_cmd, _ = build_command(
                case.candidate,
                specs,
                seed,
                args.num_servers,
                args.repeats,
                args.until,
                True,
            )

            direct_commands.append(direct_cmd)
            fastswap_commands.append(fastswap_cmd)

            print(
                '[case {idx:02d}/{total:02d}] workloads={workloads} ratios={ratios} '
                'cpu={cpu}/32 mem={mem}/{mem_cap}MB remote={remote}/{remote_cap}MB jobs={jobs}'.format(
                    idx=index,
                    total=len(cases),
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

            direct_value = run_command(direct_cmd)
            fastswap_value = run_command(fastswap_cmd)
            ratio = fastswap_value / direct_value

            print(
                '  directswap={:.4f} fastswap={:.4f} fastswap/direct={:.6f}'.format(
                    direct_value,
                    fastswap_value,
                    ratio,
                )
            )

            results_file.write(
                'case{idx:02d}\t{reasons}\t{workloads}\t{ratios}\t{size}\t{cpu}\t{mem}\t'
                '{remote}\t{direct:.6f}\t{fast:.6f}\t{ratio:.6f}\n'.format(
                    idx=index,
                    reasons=','.join(case.reasons),
                    workloads=','.join(name for name, _ in active),
                    ratios=':'.join(str(count) for _, count in active),
                    size=size,
                    cpu=case.candidate.cpu,
                    mem=case.candidate.mem,
                    remote=case.candidate.remote_mem,
                    direct=direct_value,
                    fast=fastswap_value,
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
