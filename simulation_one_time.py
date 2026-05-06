import argparse
import json
from simulation import simulate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('--remotemem', '-r', action='store_true',
                        help='enable remote memory')
    parser.add_argument('--num_servers','-n', type=int, help='number of servers to simulate', required=True)
    parser.add_argument('--max_far', '-s', type=int, default=0,
                        help='max size of far memory, default=0 (unlimited)')
    parser.add_argument('--cpus', '-c', type=int,
                        help='number of cpus required for each server',
                        required=True)
    parser.add_argument('--mem', '-m', type=int,
                        help='memory required for each server (MB)',
                        required=True)
    parser.add_argument('--size', type=int,
                        help='size of workload (num of tasks) ' \
                        'default=100', default=100)
    parser.add_argument('--workload', type=lambda s: s.split(','),
                        help='tasks that comprise the workload ' \
                        'default=xgboost,pagerank,redis',
                        default='xgboost,pagerank,redis')
    parser.add_argument('--ratios', type=lambda s: s.split(':'),
                        help='ratios of tasks in workload, default=2:2:1',
                        default="2:2:1")
    parser.add_argument('--until', type=int,
                        help='max arrival time in minutes default=20',
                        default=15)
    parser.add_argument('--policy', choices=('auto-shrink', 'fixed-ratio', 'hybrid-fixed-ratio', 'nonuniform', 'nonuniform-optimal'),
                        help='memory placement policy; defaults to legacy behavior when omitted')
    parser.add_argument('--uniform', action='store_true',
                        help='deprecated alias for --policy auto-shrink')
    parser.add_argument('--min_ratio', type=float,
                        help='smallest allowable memory ratio')
    parser.add_argument('--workload_ratios', type= lambda s: s.split(','),default="50,50,50",
                        help='ratios for each workload')
    parser.add_argument('--use_shrink', action='store_true', help='use optimization based shrinking')
    parser.add_argument('--use_fastswap', action='store_true', help='use fastswap')
    parser.add_argument('--json-metrics', action='store_true', help='print simulation metrics as JSON')

    cmdargs = parser.parse_args()
    if cmdargs.policy is None:
        policy = 'auto-shrink' if cmdargs.uniform else None
    else:
        if cmdargs.uniform and cmdargs.policy != 'auto-shrink':
            parser.error('--uniform cannot be combined with policies other than auto-shrink')
        policy = cmdargs.policy

    uniform_policy = False
    fixed_ratio_policy = False
    hybrid_fixed_ratio_policy = False
    use_shrink = cmdargs.use_shrink
    if policy == 'auto-shrink':
        uniform_policy = True
        use_shrink = False
    elif policy == 'fixed-ratio':
        uniform_policy = True
        fixed_ratio_policy = True
        use_shrink = False
    elif policy == 'hybrid-fixed-ratio':
        uniform_policy = True
        hybrid_fixed_ratio_policy = True
        use_shrink = False
    elif policy == 'nonuniform':
        use_shrink = False
    elif policy == 'nonuniform-optimal':
        use_shrink = True

    result = simulate(
        cmdargs.seed,
        cmdargs.mem,
        cmdargs.size,
        cmdargs.until,
        cmdargs.ratios,
        cmdargs.workload,
        cmdargs.cpus,
        cmdargs.num_servers,
        cmdargs.remotemem,
        list(map(float,cmdargs.workload_ratios)),
        max_far=cmdargs.max_far,
        use_shrink=use_shrink,
        uniform=uniform_policy,
        fixed_ratio_policy=fixed_ratio_policy,
        hybrid_fixed_ratio_policy=hybrid_fixed_ratio_policy,
        min_ratio=cmdargs.min_ratio,
        use_small_workload=False,
        use_fastswap=cmdargs.use_fastswap,
        return_metrics=cmdargs.json_metrics,
    )
    if cmdargs.json_metrics:
        print(json.dumps(result))
    else:
        print('{}'.format(result))

if __name__ == '__main__':
    main()       
