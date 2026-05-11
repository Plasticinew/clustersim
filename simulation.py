import logging
import argparse
import random
import time
from sortedcollections import SortedDict
import numpy as np
from scipy.optimize import Bounds, minimize
import os

PRINT_ENABLE = False
use_optimal_shrink = False
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_ROOT, 'res')
os.makedirs(RESULTS_DIR, exist_ok=True)

def eq(x,mems,local_mem):
    return np.dot(x, mems) - local_mem

def eq_grad(x,mems,local_mem):
    return mems

def obj_new(x, ideal_mems, percents, profiles, gradients=None, mem_gradients=None):
    r1 = 0
    r2 = 0
    for i in range(ideal_mems.shape[0]):
        r1 += ideal_mems[i]*(1-percents[i])*(x[i]*profiles[i](x[i]) - profiles[i](1))/1000
        r2 += ideal_mems[i]*(1-percents[i])*(1-x[i])*profiles[i](x[i])/1000
    return r1/r2 

def obj_grad_new(x, ideal_mems, percents, profiles, gradients, mem_gradients):
    r1 = 0
    r2 = 0
    g1 = np.empty(ideal_mems.shape)
    g2 = np.empty(ideal_mems.shape)
    for i in range(ideal_mems.shape[0]):
        r1 += ideal_mems[i]*(1-percents[i])*(x[i]*profiles[i](x[i]) - profiles[i](1))/1000
        r2 += ideal_mems[i]*(1-percents[i])*(1-x[i])*profiles[i](x[i])/1000
        g1[i] = ideal_mems[i]*(1-percents[i])*mem_gradients[i](x[i])
        g2[i] = ideal_mems[i]*(1-percents[i])*(gradients[i](x[i]) - mem_gradients[i](x[i]))
    grads = np.empty(ideal_mems.shape)
    for i in range(ideal_mems.shape[0]):
        grads[i] = (g1[i]*r2 - r1*g2[i])/r2**2 # r3 has the same gradient as r1
    return grads

def my_print(s):
    if PRINT_ENABLE:
        print(s)

class Server:
    def __init__(self, sid, L, remotemem, max_cpus, max_mem,
                 uniform_policy, fixed_ratio_policy, hybrid_fixed_ratio_policy, use_fastswap, min_ratio, workload_ratios, reclamation_cpus, max_remote_mem):
        self.sid = sid
        self.alloc_mem = 0
        self.min_mem_sum = 0
        self.cur_ratio = 1
        self.executing = []
        self.last_time = 0 # added for simulation
        self.next_time = 0 # optimization
        self.L = L
        self.max_remote_mem = max_remote_mem
        self.checkin(max_cpus, max_mem, remotemem, uniform_policy, fixed_ratio_policy, hybrid_fixed_ratio_policy, use_fastswap,
                     min_ratio, workload_ratios, reclamation_cpus)
        filename = 'server-{}-fastswap.txt'.format(sid) if use_fastswap else 'server-{}-directswap.txt'.format(sid)
        self.path = os.path.join(RESULTS_DIR, filename)
        self.res_file = open(self.path, 'w')


    def append_job(self, workload):
        self.executing.append(workload)

    def remove_job(self, workload):
        self.executing.remove(workload)

    def checkin(self, max_cpus, max_mem, use_remote, use_uniform_policy, fixed_ratio_policy, hybrid_fixed_ratio_policy, use_fastswap,min_ratio, workload_ratios, reclamation_cpus):
        """
        the scheduler checks in with these params.
        we return whether we have enough resources to do the checkin.
        if True, this machine will start executing jobs
        """
        # the checkin used feasible num. of cpus and mem. now initialize
        # the machine resources
        self.total_mem = max_mem
        self.total_cpus = max_cpus
        self.free_cpus = max_cpus
        self.remote_mem = use_remote
        self.use_uniform_policy = use_uniform_policy
        self.fixed_ratio_policy = fixed_ratio_policy
        self.hybrid_fixed_ratio_policy = hybrid_fixed_ratio_policy
        self.use_fastswap = use_fastswap
        self.min_ratio = min_ratio
        self.workload_ratios = workload_ratios
        self.extra_cpus = reclamation_cpus*self.remote_mem 
        if self.remote_mem and (self.fixed_ratio_policy or self.hybrid_fixed_ratio_policy):
            if self.max_remote_mem > 0:
                self.fixed_ratio = self.total_mem / (self.total_mem + self.max_remote_mem)
            elif self.min_ratio is not None:
                self.fixed_ratio = self.min_ratio
            else:
                self.fixed_ratio = 1
        else:
            self.fixed_ratio = 1
        logging.info("Checkin Successful")

        return True

    def use_fixed_ratio_policy(self):
        return self.remote_mem and self.fixed_ratio_policy

    def use_hybrid_fixed_ratio_policy(self):
        return self.remote_mem and self.hybrid_fixed_ratio_policy

    def uses_ratio_based_placement(self, projected_alloc_mem=None):
        if not self.remote_mem:
            return False
        if self.use_fixed_ratio_policy():
            return True
        if self.use_hybrid_fixed_ratio_policy():
            if projected_alloc_mem is None:
                projected_alloc_mem = self.alloc_mem
            return projected_alloc_mem > self.total_mem
        return False

    def get_workload_ratio(self, workload, force_ratio_mode=False):
        if not force_ratio_mode and not self.uses_ratio_based_placement():
            return 1
        if workload.min_ratio is not None:
            return min(1, workload.min_ratio)
        return self.fixed_ratio

    def ratio_mode_usage(self, workloads):
        local_usage = 0.0
        remote_usage = 0.0
        for workload in workloads:
            workload_ratio = self.get_workload_ratio(workload, force_ratio_mode=True)
            local_usage += workload.ideal_mem * workload_ratio
            remote_usage += workload.ideal_mem * (1 - workload_ratio)
        return local_usage, remote_usage

    def get_local_usage(self):
        if self.uses_ratio_based_placement():
            local_usage, _ = self.ratio_mode_usage(self.executing)
            return local_usage
        return min(self.alloc_mem, self.total_mem)

    def get_remote_usage(self):
        if not self.remote_mem:
            return 0
        if self.uses_ratio_based_placement():
            _, remote_usage = self.ratio_mode_usage(self.executing)
            return remote_usage
        return max(self.alloc_mem - self.total_mem,0)

    def projected_fixed_ratio_usage(self, workload):
        projected_alloc_mem = self.alloc_mem + workload.ideal_mem
        if self.uses_ratio_based_placement(projected_alloc_mem):
            projected_local, projected_remote = self.ratio_mode_usage(self.executing + [workload])
        else:
            projected_local = self.get_local_usage() + workload.ideal_mem
            projected_remote = self.get_remote_usage()
        projected_used_cpus = (self.total_cpus - self.free_cpus) + workload.cpu_req
        return projected_local, projected_remote, projected_used_cpus

    def fixed_ratio_directswap_score(self, workload, soft_remote_cap):
        projected_local, projected_remote, projected_used_cpus = self.projected_fixed_ratio_usage(workload)
        safe_remote_cap = max(float(soft_remote_cap), 1.0)
        local_pressure = projected_local / self.total_mem
        remote_pressure = projected_remote / safe_remote_cap
        cpu_pressure = projected_used_cpus / self.total_cpus
        remote_over_soft_cap = max(0.0, projected_remote - soft_remote_cap) / safe_remote_cap
        pressure_gap = max(local_pressure, remote_pressure, cpu_pressure) - min(local_pressure, remote_pressure, cpu_pressure)
        return (
            remote_over_soft_cap,
            pressure_gap,
            max(local_pressure, remote_pressure, cpu_pressure),
            remote_pressure,
            local_pressure,
            cpu_pressure,
            projected_remote,
            projected_local,
            projected_used_cpus,
            self.sid,
        )
    def update_resources(self, workload, add):
        if add:
            self.free_cpus -= workload.cpu_req
            self.alloc_mem += workload.ideal_mem
            self.min_mem_sum += workload.min_mem
        else:
            self.free_cpus += workload.cpu_req
            self.alloc_mem -= workload.ideal_mem
            self.min_mem_sum -= workload.min_mem

    def finish_job(self, old_idd):
        for workload in self.executing:
            if workload.idd == old_idd:
                self.update_resources(workload, False)
                self.remove_job(workload)
                my_print("finished {} at server {}".format(workload.get_name(), self.sid))
                break

    def fill_job(self, new_workload):
        new_idd = None
        if new_workload:
            self.update_resources(new_workload, True)
            self.append_job(new_workload)
            new_idd = new_workload.idd
            my_print("started {} at server {}".format(new_workload.get_name(), self.sid))

        if self.remote_mem:
            if self.use_uniform_policy:
                ratios = self.shrink_all_uniformly(self.executing)
            else:
                ratios = self.shrink_all_proportionally(self.executing)
                if use_optimal_shrink:
                    old_ratios = ratios
                    ratios = self.shrink_optimally(self.executing,ratios, new_idd)
        else:
            assert self.alloc_mem <= self.total_mem
            ratios = None
        
        #self.res_file.write(str(max(self.alloc_mem - self.total_mem,0)) + '\n')

        self.update_all_new(self.executing, new_idd, ratios)
        self.last_time = cur_time

    def set_cur_ratio(self):
        if self.uses_ratio_based_placement():
            try:
                self.cur_ratio = self.get_local_usage() / self.alloc_mem
            except ZeroDivisionError:
                self.cur_ratio = 1
            return
        try:
            self.cur_ratio = min(1, self.total_mem / self.alloc_mem)
        except ZeroDivisionError:
            self.cur_ratio = 1

    def update_all_new(self, workloads, new_idd=None, ratios=None):
        if ratios:
            assert len(workloads) == len(ratios)
            for w, ratio in zip(workloads,ratios):
                w.update(self.L, self.sid, cur_time, self.last_time, new_idd, ratio)
        else:
            for w in workloads:
                w.update(self.L, self.sid, cur_time, self.last_time, new_idd)

    def compute_opt_ratios(self, workloads,init_ratios,new_idd):
        #ratios = init_ratios
        el_time = cur_time - self.last_time
        min_ratios = np.array([w.min_ratio for w in workloads])
        ideal_mems = np.array([w.ideal_mem for w in workloads])
        percents = np.array([(1-(w.idd==new_idd))*min(w.percent + el_time/w.profile(w.ratio), 1) for w in workloads])
        profiles = [w.profile for w in workloads]
        mem_gradients = [w.mem_gradient for w in workloads]
        gradients = [w.gradient for w in workloads]
        x0 = np.array(init_ratios)
        eq_cons = {'type': 'eq',  'fun' : eq, 'jac': eq_grad, 'args': (ideal_mems,self.total_mem)}
        bounds = Bounds(0.5, 1.0)
        res = minimize(obj_new, x0, method='SLSQP', jac=obj_grad_new, args=(ideal_mems, percents, profiles, gradients, mem_gradients), constraints=eq_cons, options={'disp': False}, bounds=bounds)
        final_ratios = res.x
        return np.round(final_ratios,3), res.fun

    def shrink_all_uniformly(self, workloads):
        if self.uses_ratio_based_placement():
            self.set_cur_ratio()
            return [self.get_workload_ratio(w, force_ratio_mode=True) for w in workloads]
        if self.use_hybrid_fixed_ratio_policy():
            self.cur_ratio = 1
            return [1 for _ in workloads]
        total_ideal_mem = sum([w.ideal_mem for w in workloads])
        try:
            local_ratio = min(1, self.total_mem / total_ideal_mem)
        except ZeroDivisionError:
            local_ratio = 1

        assert local_ratio >= self.min_ratio
        self.set_cur_ratio()
        return [local_ratio for w in workloads]

    def shrink_all_proportionally(self, workloads):
        assert self.min_mem_sum <= self.total_mem

        total_ideal_mem = sum([w.ideal_mem for w in workloads])
        total_min_mem = sum([w.min_mem for w in workloads])
        memory_pool = total_ideal_mem - total_min_mem
        excess_mem = max(0, total_ideal_mem - self.total_mem) # Prevent containers from overgrowing
        ratios = []
        # Shrink each container
        for w in workloads:
            try:
                share_of_excess = (w.ideal_mem - w.min_mem) / memory_pool * excess_mem
            except ZeroDivisionError:
                # The pool of memory allowed to be pushed to remote storage is empty
                share_of_excess = 0
            ratio = (w.ideal_mem - share_of_excess) / w.ideal_mem
            ratios.append(ratio)
        return ratios

    def shrink_optimally(self,workloads,init_ratios, new_idd):
        total_ideal_mem = sum([w.ideal_mem for w in workloads])
        excess_mem = max(0, total_ideal_mem - self.total_mem)
        if excess_mem <= 0:
            return init_ratios
        names = [w.get_name() for w in workloads]
        if excess_mem > 0:
           ratios,_ = self.compute_opt_ratios(workloads,init_ratios,new_idd)
           ratios = ratios.tolist()
        return ratios

    def fits_remotemem(self, w, avail_far_mem):
        """ assumes the workload didn't fit normally, try to fit it with
        remote memory. we only want to determine whether the workload might
        fit, but will let the server compute its own ratio (to avoid consistency
        issues)"""
        if not self.remote_mem:
            return False
        if not self.fits_cpu(w):
            return False

        if self.use_fixed_ratio_policy() or self.use_hybrid_fixed_ratio_policy():
            local_alloc_mem, remote_alloc_mem, _ = self.projected_fixed_ratio_usage(w)
            if not self.uses_ratio_based_placement(self.alloc_mem + w.ideal_mem):
                return False
            if local_alloc_mem <= self.total_mem:
                if self.use_fastswap:
                    return remote_alloc_mem <= self.max_remote_mem
                else:
                    return avail_far_mem is None or remote_alloc_mem <= avail_far_mem
            return False

        if self.use_uniform_policy:
            local_alloc_mem = self.alloc_mem + w.ideal_mem
            local_ratio = min(1, self.total_mem / local_alloc_mem)
            if local_ratio >= self.min_ratio:
                if self.use_fastswap:
                    if local_alloc_mem - self.total_mem <= self.max_remote_mem:
                        return True
                else:                
                    if local_alloc_mem - self.total_mem <= avail_far_mem:
                        return True
        else:
            local_alloc_mem = self.alloc_mem + w.ideal_mem
            local_min_mem_sum = self.min_mem_sum + w.min_mem
            if local_min_mem_sum <= self.total_mem:
                if self.use_fastswap:
                    if local_alloc_mem - self.total_mem <= self.max_remote_mem: 
                        #print(self.max_remote_mem)
                        return True
                else:
                    if avail_far_mem is None  or local_alloc_mem - self.total_mem <= avail_far_mem:
                        return True

        return False

    def fits_normally(self, w):
        free_mem = self.total_mem - self.alloc_mem
        return self.fits_all_cpu(w) and free_mem >= w.ideal_mem

    def fits_cpu(self, w):
        return self.free_cpus >= w.cpu_req

    def fits_all_cpu(self, w):
        return (self.free_cpus + self.extra_cpus) >= w.cpu_req

class Event:
    def __init__(self,sid,wname,idd,start):
        self.sid = sid
        self.wname = wname # name to get the req for scheduling
        self.idd = idd
        self.start = start
    def event_to_workload(self,workload_ratios):
        new_workload_class = workloads.get_workload_class(self.wname)
        new_workload = new_workload_class(self.idd)
        if self.wname in workload_ratios:
            new_workload.set_min_ratio(workload_ratios[self.wname])
        return new_workload
    def get_name(self):
        return self.wname + str(self.idd)

class Schedule:
    def __init__(self):
        self.sd = SortedDict() # maintain a sorted dictionary, key is time stamp, value is (sid, idd, start/end)
    def add_event(self, timestamp, sid, wname, idd, start):
        while timestamp in self.sd:
            if timestamp == cur_time:
                print('duplicate start')
                timestamp += 0.0001
            else:
                print('duplicate end')
                timestamp -= min(0.0001, (cur_time - timestamp)/2)
        self.sd[timestamp] = Event(sid, wname, idd, start)
        return timestamp
    def delete_event(self, timestamp):
        if timestamp in self.sd: # avoid key error in redundant deletes
            del self.sd[timestamp]
    def next_event(self):
        return self.sd.popitem(0) # return (key,value) tuple
    def is_empty(self):
        return len(self.sd) == 0
    def size(self):
        return len(self.sd)
    def get_next_time(self):
        if len(self.sd) > 0:
            return self.sd.peekitem(0)[0]
        return None


def find_server_fits(servers, workload, max_far_mem, server_seq):
    # seq = list(range(len(servers)))
    # random.shuffle(seq) # random iteration
    seq = server_seq
    if not servers:
        return None
    if not servers[0].use_fixed_ratio_policy():
        # first try to fit the workload normally
        for i in seq:
            s = servers[i]
            if s.fits_normally(workload):
                return s
    # normal placement didn't work, are we using remote memory?
    if not servers[0].remote_mem:
        return None

    # we are using remote memory. for every server, check if we
    # can fit it using remote mem
    total_far_mem_used = sum([s.get_remote_usage() for s in servers])
    fit_candidates = []
    soft_remote_cap = servers[0].max_remote_mem

    for i in seq:
        s = servers[i]
        others_far_mem_used = total_far_mem_used - s.get_remote_usage()
        if max_far_mem > 0: # has a limit
            avail_far_mem = max_far_mem - others_far_mem_used
        else: # no limits
            avail_far_mem = None
        if s.remote_mem and s.fits_remotemem(workload, avail_far_mem):
            if (servers[0].use_fixed_ratio_policy() or servers[0].use_hybrid_fixed_ratio_policy()) and not servers[0].use_fastswap:
                fit_candidates.append((s.fixed_ratio_directswap_score(workload, soft_remote_cap), s))
            else:
                return s

    if fit_candidates:
        fit_candidates.sort(key=lambda entry: entry[0])
        return fit_candidates[0][1]

    return None

def find_new_workload(servers, Pending, max_far_mem, server_seq):
    # sequential
    tried_names = []

    while True:
        next_workload = None
        next_time = None
        next_name = None
        for name, l in Pending.items():
            if len(l) == 0 or name in tried_names:
                continue
            workload, timestamp = l[0]
            if next_name is None or timestamp < next_time:
                next_workload = workload
                next_time = timestamp
                next_name = name
        if next_workload:
            tried_names.append(next_name)
            s = find_server_fits(servers, next_workload, max_far_mem, server_seq)
            if s:
                Pending[next_name].pop(0)
                return s, next_workload
        else:
            break

    return None, None

def get_avail_far_mem(servers, s, max_far_mem):
    if max_far_mem == 0:
        return None
    others_far_mem_used = 0
    for ss in servers:
        if ss != s:
            others_far_mem_used += ss.get_remote_usage()
    return max_far_mem - others_far_mem_used

def update_server_seq(default_seq,server_seq):
    server_seq[:] = default_seq[:]
    random.shuffle(server_seq)

def summarize_jobs(jobs_ts):
    if not jobs_ts:
        return 0.0, 0.0, 0.0

    ideal_runtime_cache = {}
    total_queue_time = 0.0
    total_runtime_slowdown = 0.0
    total_turnaround_slowdown = 0.0

    for job in jobs_ts.values():
        total_queue_time += job['exec'] - job['arrival']
        service_time = job['finish'] - job['exec']
        turnaround_time = job['finish'] - job['arrival']
        workload_name = job['workload']
        if workload_name not in ideal_runtime_cache:
            ideal_runtime_cache[workload_name] = workloads.get_workload_class(workload_name)(0).profile(1)
        ideal_runtime = ideal_runtime_cache[workload_name]
        total_runtime_slowdown += service_time / ideal_runtime
        total_turnaround_slowdown += turnaround_time / ideal_runtime

    num_jobs = len(jobs_ts)
    return (
        total_queue_time / num_jobs,
        total_runtime_slowdown / num_jobs,
        total_turnaround_slowdown / num_jobs,
    )


def pending_mem_demand(Pending):
    return sum(workload.ideal_mem for queue in Pending.values() for workload, _ in queue)


def record_memory_curve_point(memory_curve_points, timestamp, servers, Pending, remote_mem_capacity):
    running_task_demand = sum(s.alloc_mem for s in servers)
    local_used = sum(s.get_local_usage() for s in servers)
    remote_used = sum(s.get_remote_usage() for s in servers)
    pending_task_demand = pending_mem_demand(Pending)
    if remote_mem_capacity > 0:
        remote_mem_utilization = remote_used / remote_mem_capacity
    else:
        remote_mem_utilization = 0.0
    point = {
        'time_ms': float(timestamp),
        'time_s': float(timestamp) / 1000.0,
        'local_used_mem_mb': float(local_used),
        'remote_used_mem_mb': float(remote_used),
        'remote_mem_utilization': float(remote_mem_utilization),
        'total_used_mem_mb': float(local_used + remote_used),
        'running_task_demand_mem_mb': float(running_task_demand),
        'pending_task_demand_mem_mb': float(pending_task_demand),
        'active_task_demand_mem_mb': float(running_task_demand + pending_task_demand),
    }
    if memory_curve_points and abs(memory_curve_points[-1]['time_ms'] - point['time_ms']) < 1e-9:
        memory_curve_points[-1] = point
    else:
        memory_curve_points.append(point)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_memory_curve_tsv(path, memory_curve_points):
    ensure_parent_dir(path)
    with open(path, 'w') as handle:
        handle.write(
            'time_ms\ttime_s\tlocal_used_mem_mb\tremote_used_mem_mb\tremote_mem_utilization\ttotal_used_mem_mb\t'
            'running_task_demand_mem_mb\tpending_task_demand_mem_mb\tactive_task_demand_mem_mb\n'
        )
        for point in memory_curve_points:
            handle.write(
                '{time_ms:.6f}\t{time_s:.6f}\t{local_used_mem_mb:.6f}\t{remote_used_mem_mb:.6f}\t{remote_mem_utilization:.6f}\t{total_used_mem_mb:.6f}\t'
                '{running_task_demand_mem_mb:.6f}\t{pending_task_demand_mem_mb:.6f}\t{active_task_demand_mem_mb:.6f}\n'.format(
                    **point
                )
            )


def plot_memory_curve(path, memory_curve_points):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('matplotlib is required to plot the memory curve') from exc

    ensure_parent_dir(path)
    times = [point['time_s'] for point in memory_curve_points]
    remote_utilization_pct = [point['remote_mem_utilization'] * 100.0 for point in memory_curve_points]

    plt.figure(figsize=(10, 5))
    plt.step(times, remote_utilization_pct, where='post', linewidth=2.5, label='Remote memory utilization')
    plt.xlabel('Time (s)')
    plt.ylabel('Remote Memory Utilization (%)')
    plt.title('Remote Memory Utilization Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def schedule(
    servers,
    L,
    workload_ratios,
    max_far_mem,
    jobs_ts,
    use_fastswap,
    until,
    return_metrics=False,
    memory_curve_tsv=None,
    memory_curve_png=None,
):
    Pending = dict() #key is wname, value is a list of workloads and timestamp of that name in pending
    global cur_time
    default_seq = list(range(len(servers)))
    server_seq = default_seq[:]
    until = until / 1000
    id = 200 / (until/ 60)
    filename = 'fastswap_{}.txt'.format('{:g}'.format(id)) if use_fastswap else 'directswap_{}.txt'.format('{:g}'.format(id))
    mem_usage_path = os.path.join(RESULTS_DIR, filename)
    total_mem_capacity = sum(s.total_mem for s in servers)
    remote_mem_capacity = 0
    if servers and servers[0].remote_mem:
        if use_fastswap:
            remote_mem_capacity = sum(s.max_remote_mem for s in servers)
        elif max_far_mem > 0:
            remote_mem_capacity = max_far_mem
    cluster_mem_capacity = total_mem_capacity + remote_mem_capacity

    prev_time_second = 0
    prev_remote_mem_useage = 0
    last_sample_time = 0
    total_mem_time = 0.0
    remote_mem_time = 0.0
    memory_curve_points = [] if (memory_curve_tsv or memory_curve_png) else None

    if memory_curve_points is not None:
        record_memory_curve_point(memory_curve_points, 0, servers, Pending, remote_mem_capacity)

    def accumulate_usage(next_time):
        nonlocal last_sample_time
        nonlocal total_mem_time
        nonlocal remote_mem_time
        delta = next_time - last_sample_time
        if delta <= 0:
            last_sample_time = next_time
            return
        cluster_alloc_mem = sum(s.alloc_mem for s in servers)
        cluster_remote_mem = sum(s.get_remote_usage() for s in servers)
        total_mem_time += cluster_alloc_mem * delta
        remote_mem_time += cluster_remote_mem * delta
        last_sample_time = next_time

    with open(mem_usage_path, 'w') as mem_usage_file:
        while not L.is_empty():
            timestamp, event = L.next_event()
            accumulate_usage(timestamp)
            cur_time = timestamp
            cur_time_second = int(timestamp / 1000)
            for i in range(prev_time_second, cur_time_second):
                mem_usage_file.write(str(prev_remote_mem_useage) + '\n')

            my_print('timestamp: {} ms'.format(cur_time))
            if event.start: # start node
                workload = event.event_to_workload(workload_ratios)
                s = None
                if not workload.wname in Pending:
                    Pending[workload.wname] = [] # initialize
                if len(Pending[workload.wname]) == 0:
                    s = find_server_fits(servers, workload, max_far_mem, server_seq) # only need to do this when pending is empty for the class
                if s:
                    update_server_seq(default_seq,server_seq)
                    s.fill_job(workload)
                    jobs_ts[workload.idd]['exec'] = cur_time
                else:
                    my_print("job {} can't fit".format(workload.get_name()))
                    Pending[workload.wname].append((workload,cur_time))
                if memory_curve_points is not None:
                    record_memory_curve_point(memory_curve_points, cur_time, servers, Pending, remote_mem_capacity)
            else: # end node
                next_time = L.get_next_time()
                old_idd = event.idd
                old_s = servers[event.sid]
                old_s.finish_job(old_idd) # finish job will update resource
                jobs_ts[old_idd]['finish'] = cur_time
                if memory_curve_points is not None:
                    record_memory_curve_point(memory_curve_points, cur_time, servers, Pending, remote_mem_capacity)
                ids = [] # sid's that have been updated
                s, new_workload = find_new_workload(servers, Pending, max_far_mem, server_seq)
                while new_workload:
                    next_time = L.get_next_time() # next time may change as jobs are added
                    if next_time:
                        next_start_time = cur_time + random.uniform(0, min(100, (next_time - cur_time)*0.5)) # don't want to excde next time
                    else:
                        next_start_time = cur_time + random.uniform(0, 100)
                    accumulate_usage(next_start_time)
                    cur_time = next_start_time
                    s.fill_job(new_workload)
                    jobs_ts[new_workload.idd]['exec'] = cur_time
                    update_server_seq(default_seq,server_seq)
                    ids.append(s.sid)
                    if memory_curve_points is not None:
                        record_memory_curve_point(memory_curve_points, cur_time, servers, Pending, remote_mem_capacity)
                    s, new_workload = find_new_workload(servers, Pending, max_far_mem, server_seq)
                if not (old_s.sid in ids): # only when it has not been updated
                    old_s.fill_job(None)
                    if memory_curve_points is not None:
                        record_memory_curve_point(memory_curve_points, cur_time, servers, Pending, remote_mem_capacity)
            
            prev_time_second = cur_time_second
            prev_remote_mem_useage = 0
            for s in servers:
                prev_remote_mem_useage += s.get_remote_usage()

            my_print('')

    total_duration = cur_time
    makespan = round(cur_time)
    avg_mem_util = 0.0
    avg_remote_mem_util = 0.0
    if total_duration > 0 and cluster_mem_capacity > 0:
        avg_mem_util = total_mem_time / (total_duration * cluster_mem_capacity)
    if total_duration > 0 and remote_mem_capacity > 0:
        avg_remote_mem_util = remote_mem_time / (total_duration * remote_mem_capacity)
    avg_queue_time, avg_runtime_slowdown, avg_turnaround_slowdown = summarize_jobs(jobs_ts)

    my_print("avg_mem_util={:.6f}".format(avg_mem_util))
    my_print("avg_remote_mem_util={:.6f}".format(avg_remote_mem_util))
    my_print("avg_queue_time={:.6f}".format(avg_queue_time))
    my_print("avg_runtime_slowdown={:.6f}".format(avg_runtime_slowdown))
    my_print("avg_turnaround_slowdown={:.6f}".format(avg_turnaround_slowdown))

    total_pending = 0
    for v in Pending.values():
        total_pending += len(v)
    assert total_pending == 0

    if memory_curve_points is not None:
        record_memory_curve_point(memory_curve_points, cur_time, servers, Pending, remote_mem_capacity)
        if memory_curve_tsv:
            write_memory_curve_tsv(memory_curve_tsv, memory_curve_points)
        if memory_curve_png:
            plot_memory_curve(memory_curve_png, memory_curve_points)

    if return_metrics:
        return {
            'makespan': makespan,
            'avg_mem_util': avg_mem_util,
            'avg_remote_mem_util': avg_remote_mem_util,
            'avg_queue_time': avg_queue_time,
            'avg_runtime_slowdown': avg_runtime_slowdown,
            'avg_turnaround_slowdown': avg_turnaround_slowdown,
        }
    return makespan

def sample_arrival_time(max_arrival, arrival_profile, arrival_weights):
    if arrival_profile == 'uniform':
        return random.uniform(0, max_arrival)

    if arrival_profile != 'piecewise':
        raise ValueError('unknown arrival profile: {}'.format(arrival_profile))

    if not arrival_weights:
        raise ValueError('piecewise arrival profile requires arrival_weights')

    if any(weight <= 0 for weight in arrival_weights):
        raise ValueError('arrival_weights must all be positive')

    num_bins = len(arrival_weights)
    bin_index = random.choices(range(num_bins), weights=arrival_weights, k=1)[0]
    left = max_arrival * bin_index / num_bins
    right = max_arrival * (bin_index + 1) / num_bins
    return random.uniform(left, right)


def get_schedule(size, max_arrival, workloads, ratios, jobs_ts, arrival_profile='uniform', arrival_weights=None):
    L = Schedule()
    wid = 0
    def add_workload(name):
        nonlocal wid
        nonlocal jobs_ts
        ts = sample_arrival_time(max_arrival, arrival_profile, arrival_weights)
        L.add_event(ts, 0, name, wid, True) # sid doesn't matter for start nodes, set when scheduled
        assert wid not in jobs_ts
        jobs_ts[wid] = {'arrival': ts, 'exec': 0, 'finish': 0, 'workload': name}
        wid += 1

    assert len(workloads) == len(ratios)
    ratios = list(map(int, ratios))
    # this is what a ratio of 1 corresponds to
    unit = int(size / sum(ratios))
    for workload_name, ratio in zip(workloads, ratios):
        times = unit * ratio
        for _ in range(times):
            add_workload(workload_name)

    return L


def simulate(seed, mem, size, until, ratios, workload, cpus, num_servers, remotemem, workload_ratios, max_far=0, use_shrink=False, uniform=False, fixed_ratio_policy=False, hybrid_fixed_ratio_policy=False, min_ratio=None, use_small_workload=False, use_fastswap=False, return_metrics=False, arrival_profile='uniform', arrival_weights=None, memory_curve_tsv=None, memory_curve_png=None):
    global workloads
    if use_small_workload:
        import workloads_small as workloads
        reclamation_cpus = 1 # use 1 core for small workload (8 cpus, 32G)
    else:
        import workloads_simu as workloads
        reclamation_cpus = 3 # use 3 cores for large workload (48 cpus, 192G)

    global cur_time
    cur_time = 0 # gloabl current time
    global use_optimal_shrink
    use_optimal_shrink = use_shrink
    # min_ratio must be specified if the uniform mem policy is used
    try:
        assert (not uniform) or (uniform and min_ratio)
    except AssertionError:
        raise RuntimeError("If uniform policy is used, min_ratio must be specified")

    # Put the workload_ratio values in a dictionary with the corresponding name
    if workload_ratios:
        assert len(workload_ratios) == len(workload)
        workload_ratios = dict(zip(workload, workload_ratios))
        for k in workload_ratios.keys():
            workload_ratios[k] = workload_ratios[k]/100
    else:
        workload_ratios = dict()

    until = until * 1000  # seconds -> ms
    # Instantiate Servers
    random.seed(seed)
    jobs_ts = {}
    L = get_schedule(size, until, workload, ratios, jobs_ts, arrival_profile=arrival_profile, arrival_weights=arrival_weights)
    servers = []
    for sid in range(num_servers):
        servers.append(Server(sid, L, remotemem, cpus, mem,
                              uniform, fixed_ratio_policy, hybrid_fixed_ratio_policy, use_fastswap, min_ratio, workload_ratios, reclamation_cpus, max_far/num_servers))
    try:
        return schedule(
            servers,
            L,
            workload_ratios,
            max_far,
            jobs_ts,
            use_fastswap,
            until,
            return_metrics=return_metrics,
            memory_curve_tsv=memory_curve_tsv,
            memory_curve_png=memory_curve_png,
        )
    except KeyboardInterrupt:
        for s in servers[:]:
            del s
    finally:
        for s in servers:
            s.res_file.close()
