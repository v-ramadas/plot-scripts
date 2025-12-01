#!/opt/homebrew/bin/python3.12

from utils.Stats import CacheStats
from utils.Plots import *
import os
import argparse

benchmarks = {
        "ligra": ["BC", "BFS", "BFSCC", "BellmanFord", "CF", "Components", "MIS", "PageRank", "PageRankDelta", "Radii", "Triangle"],
        "spec2006": ["perlbench", "bzip2", "gcc", "bwaves", "gamess", "mcf", "milc", "zeusmp", "gromacs", "cactusADM", "leslie3d", "namd", "gobmk", "dealII", "soplex", "povray", "calculix", "hmmer", "sjeng", "GemsFDTD", "libquantum", "h264ref", "tonto", "lbm", "omnetpp", "astar", "wrf", "sphinx3", "xalancbmk"],
        "spec2017": ["perlbench_s", "gcc_s", "bwaves_s", "mcf_s", "cactuBSSN_s", "lbm_s", "omnetpp_s", "wrf_s", "xalancbmk_s", "x264_s", "cam4_s", "pop2_s", "deepsjeng_s", "imagick_s", "leela_s", "nab_s", "exchange2_s", "fotonik3d_s", "roms_s", "xz_s"],
        "ubench": ["ubench_stream", "ubench_stride_2", "ubench_stride_4", "ubench_stride_8", "spmv"],
        "custom": ["PageRankDelta", "BFS", "BC", "Radii", "ubench_stride_4", "ubench_stride_8"],
        "large": ["BC", "MIS", "Radii", "Components", "BellmanFord", "GemsFDTD", "bwaves", "lbm", "milc", "fotonik3d_s", "mcf_s", "lbm_s"],
        "medium": ["PageRank", "PageRankDelta", "BFSCC", "CF", "wrf", "leslie3d", "pop2_s"],
        "small": ["omnetpp_s", "x264_s", "deepsjeng_s", "cam4_s", "libquantum", "gromacs", "calculix", "gobmk"],
        "mixed": [],
        }

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

parser.add_argument("--util",
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
print(args.multiple_traces)
stats_64.populateStats(args.d, args.graph, 64, 16, args.multiple_traces)
stats_64.populateStats("ubench_" + str(args.d), args.graph, 64, 16, args.multiple_traces)
stats_8.populateStats(args.d, args.graph, 8, 128, args.multiple_traces)
stats_8.populateStats("ubench_" + str(args.d), args.graph, 8, 128, args.multiple_traces)
benchmarks["all"] = stats_64.exptList
#Plotting
set_save_path(args.d)
if args.mrc:
    plot_mrc(args.workload_set, benchmarks[args.workload_set], stats_64, args.graph)
if args.footprint:
    plot_footprint(args.workload_set, benchmarks[args.workload_set], [stats_64, stats_8], args.graph)
if args.util:
    plot_utilization(args.workload_set, benchmarks[args.workload_set], [stats_64, stats_8], args.graph)
if args.mpki:
    plot_mpki(args.workload_set, benchmarks[args.workload_set], [stats_64, stats_8], args.graph)
if args.evictions_breakdown:
    plot_evictions_breakdown(args.workload_set, benchmarks[args.workload_set], [stats_64, stats_8], args.graph)
