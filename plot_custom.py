#!/opt/homebrew/bin/python3.12

from plot_scripts.utils.Stats import CacheStats
from plot_scripts.utils.Plots import *
from plot_scripts.utils.Workloads import *

import os
import argparse

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d",
        type=str,
        default="",
        help="directory with run logs"
        )

parser.add_argument("--cache",
        type=str,
        default="L1D",
        help="Cache Name")

args = parser.parse_args()
result_dir = args.d

baseDir = os.getcwd()
graph="com-lj"
multiple_traces=True
utilization=True
mpki=True
ipc=True
evictions_breakdown=False
stats_64 = CacheStats(args.cache)
stats_32 = CacheStats(args.cache)
stats_16 = CacheStats(args.cache)
stats_08 = CacheStats(args.cache)
stats_64.populateStats(result_dir, graph, 64, 16, multiple_traces)
#stats_32.populateStats(result_dir, graph, 32, 32, multiple_traces)
#stats_16.populateStats(result_dir, graph, 16, 64, multiple_traces)
stats_08.populateStats(result_dir, graph, 8, 128, multiple_traces)
workloadSet["all"] = stats_64.exptList
stats_64.desc="Baseline (64B)"
#stats_32.desc="LRU2 (32B)"
#stats_16.desc="LRU2 (16B)"
stats_08.desc="LRU2 (8B)"

#hue_order = ["Baseline (64B)", "LRU2 (32B)", "LRU2 (16B)", "LRU2 (8B)"]
hue_order = ["Baseline (64B)", "LRU2 (8B)"]

#Plotting
for workload in ["PageRank", "PageRankDelta", "Radii", "Triangle", "BFS", "BFSCC", "BC", "MIS", "Components", "BellmanFord", "CF", 
                 "perlbench_s", "gcc_s", "bwaves_s", "mcf_s", "cactuBSSN_s", "lbm_s",  "omnetpp_s", "wrf_s", "xalancbmk_s", "x264_s", "cam4_s", "pop2_s", "deepsjeng_s", "imagick_s", "leela_s", "nab_s", "exchange2_s", "fotonik3d_s", "roms_s", "xz_s",
                 "perlbench", "bzip2", "gcc", "bwaves", "gamess", "mcf", "milc", "zeusmp", "gromacs", "cactusADM", "leslie3d", "namd", "gobmk", "dealII", "soplex", "povray", "calculix", "hmmer", "sjeng", "GemsFDTD", "libquantum", "h264ref", "tonto", "lbm", "omnetpp", "astar", "wrf", "sphinx3", "xalancbmk",
                "blackscholes", "bodytrack", "canneal", "dedup", "dedup", "facesim", "fluidanimate", "raytrace", "streamcluster", "swaptions", "vips"]:

    print (workload)
    for expt in workloadSet[workload]:
        print(expt, stats_08.expts[expt].num_instructions)
    if utilization:
        set_save_path(os.path.join(result_dir, "utilization", args.cache))
        plot_utilization(workload, workloadSet[workload], [stats_64, stats_08], graph, hue_order)
    if mpki:
        set_save_path(os.path.join(result_dir, "mpki", args.cache))
        plot_mpki(workload, workloadSet[workload], [stats_64, stats_08], graph, hue_order)
    if evictions_breakdown:
        set_save_path(os.path.join(result_dir, "breakdown", args.cache))
        plot_evictions_breakdown(workload, workloadSet[workload], [stats_64, stats_08], graph)
    if ipc:
        set_save_path(os.path.join(result_dir, "ipc", args.cache))
        plot_ipc(workload, workloadSet[workload], [stats_64, stats_08], graph, hue_order)

