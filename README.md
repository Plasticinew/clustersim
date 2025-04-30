# Overview

This simulator is an improved implementation based on [the open-source simulator](https://github.com/clusterfarmem/clustersim.git) of FastSwap (EuroSys'20), designed to conveniently reproduce the experimental results of the swap system presented in the paper. 

This README file is divided into two sections: The first section briefly introduces how the simulator works and what is the differnence when simulating FastSwap and FineMem-Swap. The second section illustrates how to reproduce our experimental results.

# Introduction

## DM swap systems simulator
The simulator employs a centralized job scheduler to allocate jobs and utilizes multiple simulated compute nodes to model job execution. Upon job arrival, the scheduler applies a customized admission policy to identify a suitable compute node for execution. If no eligible node is available, the job is queued and reconsidered for deployment upon subsequent job completion events. Each compute node dynamically adjusts memory configurations (local-remote memory ratios) for its assigned jobs based on pre-profiled performance degradation data, thereby simulating application execution progress. The accuracy of this simulation has been validated by FastSwap.

## Job Admission Policy
Our re-implementated simulator employs distinct job admission policies for FastSwap and FineMem-Swap:
* For FastSwap, the scheduler’s admission policy for unscheduled job is: a compute node exists where the job can be accommodated
without exceeding its fixed remote memory usage (the pre-registered remote swap partition size, 32GB in our setting).
* For FineMem-Swap, the scheduler’s admission policy for unscheduled job is: a compute node exists where accommodating the job keeps the total remote memory usage of the entire cluster within total remote memory size (number of compute nodes times 32GB in our setting).



# Run Experiment

## Pre-requisites
Install the required python dependencies for simulation and plotting.

```sh
sudo apt update && sudo apt install python3-pip

pip install numpy, scipy, sortedcollections, matplotlib
```

## Remote Memory Usage
To reproduce FineMem in Figure 16(a), enter `clustersim/` and run 

```sh
python draw_remote_mem_usage.py <random_seed>

# example: python draw_remote_mem_usage.py 2025
```

`random_seed` is an interger to . The script firstly generates a tarce containing 200 jobs (XGBoost:Snappy:Redis=2:2:1), simulates it on both FastSwap and FineMem-Swap with 5 compute nodes and print the corresponding remote memory usage (results are in `clustersim/remote_mem_usage`). Then, the script draws the corresponding experimental result figure in `clustersim/figures/remote_mem_usage.png`.



## Throughput Imporvement
`gen_small_traces.py` will randomly generate 500 traces, each containing 200 jobs, and simulate the processing of these traces  on both FastSwap and FineMem-Swap with 5 compute nodes. The comparative results for each trace are output in real time to `clustersim/small_traces/`. `gen_large_traces.py` operates similarly, randomly generating and simulating 500 traces with 10,000 jobs each with 40 compute nodes, and outputs the results to `clustersim/large_traces/`.

To reproduce FineMem in Figure 16(b), enter `clustersim/` and run `python gen_small_traces.py` and `python gen_large_traces.py` respectively. After execution, run `python draw_throughput_improvement_cdf.py` to draw the corresponding experimental result figure in `clustersim/figures/throughput_improvement.png`. Note that the simulation may require significant execution time (particularly for large traces), allowing partial intermediate results to be plotted before all simulations complete.

