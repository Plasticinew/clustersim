# Cluster simulation with far memory

## Pre-requisites

python numpy, scipy, sortedcollections


## `start_simulations.py`
`start_simulations.py` is the start point from which you can run various rack scale simulations. It accepts multiple arguments, but you can generate the large scale simulation results we presented in our paper (Figure 7, 8 and 9) with the default configuration:
```
python3 start_simulations.py 
```
This would run the default large-scale simulation where the amount of far memory and additional local memory vary, the results would be written in a text file stored in results/results_192G_48cores.

### Other Arguments
Argument            | Description
--------------------------------|---------------------------------------------
--num_random, -n           | Number of randomly generated workloads.
--limits, -l       | Limits of m2c.
--cpu, -c | Number of cpu per machine.
--mem, -m              | Amount of memory per machine (unit is MB). 
--jps, -j            | Number of jobs per server.
--filename, -f           | Filename for final results.
--simu_name, -s            | Name of the simulation loop function.
--use_small_workload           | To use small workload.

## `simulation_one_time.py` and `test.sh`
`simulation_one_time.py` allows you to run single simulation with small workloads. `test.sh` contains examples usage of `simulation_one_time.py`. `test.sh` requires two parameters: seed to generate workload and amount of far memory. Here is an example usage:
```
./test.sh 2000 32768 
```

## `test_1to1.sh`
`test_1to1.sh` is a directly executable trace test based on `gen_traces.py`. It fixes each compute node at `32 CPUs + 64GB local memory`, sets every workload to a roughly `2:1` local/remote split via `--workload_ratios 66`, and caps Fastswap remote memory at `64GB` per node. The script automatically picks workload mixes that are as close as possible to the per-node total memory limit while keeping CPU demand within the per-node CPU limit, then runs both direct swap and Fastswap with the same traces. It supports `auto-shrink` for dynamic uniform shrink, `fixed-ratio` for a fixed local/remote split from the start, `hybrid-fixed-ratio` for local-first placement that falls back to a fixed split under pressure, and `nonuniform-optimal` for the previous optimized non-uniform scheduler.

Example usage:
```
./test_1to1.sh
```

Optional arguments:
```
./test_1to1.sh --limit 2 --repeats 1 --until 60
./test_1to1.sh --policy fixed-ratio
./test_1to1.sh --policy hybrid-fixed-ratio
./test_1to1.sh --policy nonuniform-optimal
./test_1to1.sh --workload-percent 1000
```

## `test_directswap_best.sh`
`test_directswap_best.sh` searches workload mixes where Directswap beats Fastswap and ranks the best cases. By default it evaluates the top memory-fitting candidates under the `nonuniform-optimal` policy and keeps the top 20 Directswap wins.

Example usage:
```
./test_directswap_best.sh
./test_directswap_best.sh --policy fixed-ratio
./test_directswap_best.sh --policy hybrid-fixed-ratio
./test_directswap_best.sh --candidate-limit 500 --top-k 20
./test_directswap_best.sh --workload-percent 1000
```

## `test_random_traces.sh`
`test_random_traces.sh` generates purely random traces and compares Directswap against Fastswap on each trace. By default it generates `200` traces, each containing `200` tasks sampled uniformly from the built-in workload set. It supports `uniform`, `fix`, and `optimal` strategy aliases, which map to `auto-shrink`, `fixed-ratio`, and `nonuniform-optimal` respectively.

Example usage:
```
./test_random_traces.sh
./test_random_traces.sh --strategy uniform
./test_random_traces.sh --strategy fix
./test_random_traces.sh --num-traces 50 --tasks-per-trace 200
./test_random_traces.sh --workloads quicksort,redis,llama
```
## Questions
For additional questions please contact us at cfm@lists.eecs.berkeley.edu
