#!/opt/homebrew/bin/python3.12

from utils.Stats import CacheStats
from utils.Plots import *
from utils.Workloads import *

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d",
        type=str,
        default="",
        help="directory with run logs"
        )

parser.add_argument("--graph",
        type=str,
        default="com-lj",
        help="Ligra input graph")

parser.add_argument("--mrc",
        action='store_true',
        help="Plot MRC")

parser.add_argument("--footprint",
        action='store_true',
        help="Plot Footprint")

parser.add_argument("--utilization",
        action='store_true',
        help="Plot Utilization")

parser.add_argument("--mpki",
        action='store_true',
        help="Plot MPKI")

parser.add_argument("--evictions-breakdown",
        action='store_true',
        help="Plot MPKI")

parser.add_argument("--workload-set",
        type=str,
        default="custom",
        help="Choose workload set to plot")

parser.add_argument("--multiple-traces",
        action="store_true",
        help="Enable plotting multiple traces per workload")

args = parser.parse_args()

baseDir = os.getcwd()

stats_64 = CacheStats("L1D")
stats_8 = CacheStats("L1D")
stats_64.populateStats(args.d, args.graph, 64, 16, args.multiple_traces)
stats_64.populateStats("ubench_" + str(args.d), args.graph, 64, 16, args.multiple_traces)
stats_8.populateStats(args.d, args.graph, 8, 128, args.multiple_traces)
stats_8.populateStats("ubench_" + str(args.d), args.graph, 8, 128, args.multiple_traces)
workloadSet["all"] = stats_64.exptList
#Plotting
set_save_path(args.d)
if args.mrc:
    plot_mrc(args.workload_set, workloadSet[args.workload_set], stats_64, args.graph)
if args.footprint:
    plot_footprint(args.workload_set, workloadSet[args.workload_set], [stats_64, stats_8], args.graph)
if args.utilization:
    plot_utilization(args.workload_set, workloadSet[args.workload_set], [stats_64, stats_8], args.graph)
if args.mpki:
    plot_mpki(args.workload_set, workloadSet[args.workload_set], [stats_64, stats_8], args.graph)
if args.evictions_breakdown:
    plot_evictions_breakdown(args.workload_set, workloadSet[args.workload_set], [stats_64, stats_8], args.graph)
