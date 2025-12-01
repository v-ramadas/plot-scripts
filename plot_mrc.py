from utils.Stats import CacheStats
from utils.Plots import *
from utils.Workloads import *

import os
import re
import argparse

import argparse

parser = argparse.ArgumentParser()

def setMRCStats(stats, expt, stat_file, num_sets, suite, graph):
    f = open(stat_file, 'r')
    if expt not in stats.mrc.keys():
        stats.mrc[expt] = stats.MRCStats()
        stats.expts[expt] = stats.ExptStats()
        stats.expts[expt].suite = suite
        stats.expts[expt].graph = graph

    num_instruction = 1
    for line in f:
        if "Instruction count" in line:
            line_list = line.split()
            num_instruction = int(line_list[5])
        elif "Partial misses" in line:
            line_list = line.split()[2:]
            for item in line_list:
                num_misses = int(item.split(':')[0])
                count = int(item.split(':')[1])
                stats.mrc[expt].partial_misses[num_misses] = (count/num_instruction)*1000
        else:
            line_list = line.split('|')
            if len(line_list) < 4 or "MPKI" in line:
                continue
            cache_size =  int(line_list[0])/(1024*1024)
            stats.mrc[expt].cache_sizes.append(cache_size)
            stats.mrc[expt].miss_rate.append(float(line_list[3]))


    return


def populateStats(stats, results_dir, graph, block_size, num_sets, multiple_traces):
    output_dir = os.path.join(results_dir, "expt", "outputs")
    for stats_dir in os.listdir(output_dir):
        pattern = r"blkSize_(\d+)_set_(\d+)"
        if (int(re.match(pattern, stats_dir).group(1)) != block_size):
            continue
        if (int(re.match(pattern, stats_dir).group(2)) != num_sets):
            continue
        num_sets = int(re.match(pattern, stats_dir).group(2))
        baseDir = os.getcwd()
        os.chdir(os.path.join(baseDir, output_dir, stats_dir))
        stats.block_size = block_size
        for expt_file in os.listdir(os.getcwd()):
            f = open(os.path.abspath(expt_file), 'r')
            pattern = r'^[^_]+'
            match = re.match(pattern, expt_file)
            if match is None:
                continue

            inst_dropped_pattern = ""
            if "ligra" in expt_file:
                pattern = pattern = r"[^.]+\.([^.]+)"
                match = re.match(pattern, expt_file)
                if match.group(1) != graph:
                    continue
                pattern = r"^ligra_ligra_(.*?)\."
                inst_dropped_pattern = r'drop_(\d+M)\.'
                suite = "ligra"
                graph = re.match(r"^[^.]+\.([^\.]+)\.", expt_file).group(1)
            elif "spec2006" in expt_file:
                pattern = r"^spec2006_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
                suite = "spec2006"
                inst_dropped_pattern = r'(\d+B)\.champsimtrace'
            elif "spec2017" in expt_file:
                pattern = r"^spec2017_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
                suite = "spec2017"
                inst_dropped_pattern = r'(\d+B)\.champsimtrace'
            elif "parsec" in expt_file:
                pattern = r"parsec_parsec_2.1.(.*?)\."
                suite = "parsec"
                inst_dropped_pattern = r'drop_(\d+M)\.'
            elif "gap" in expt_file:
                pattern = r"gap_(.*?)\."
                suite = "gap"
            else:
                pattern = r'^([^.]+)'
                suite = "ubench"
            expt = (re.match(pattern, expt_file).group(1))
            simpoints_inst_dropped = ""
            if inst_dropped_pattern != "" and multiple_traces:
                simpoints_inst_dropped = re.search(inst_dropped_pattern, expt_file).group(1) 
                expt = '_'.join([expt, simpoints_inst_dropped])
            stats.exptList.append(expt)
            setMRCStats(stats, expt, os.path.abspath(expt_file), num_sets, suite, graph)
            stats.expts[expt].simpoints_inst_dropped = simpoints_inst_dropped

            f.close()
        os.chdir(baseDir)

    return

def plot_partials(workload_set, benchmarks, stats, graph, hue_order):
    for expt in benchmarks:
        if expt not in stats.mrc.keys():
            continue
        df = pd.DataFrame(
            list(stats.mrc[expt].partial_misses.items()), columns=['MPKI', 'Number of Requests'])
        ax = sns.barplot(
            data=df,
            x="MPKI",      # The shared x-axis variable
            y="Number of Requests",     # The y-axis variable
            hue="MPKI", # The categorical variable that defines each line
            palette="pastel",
            hue_order=hue_order,
            legend =False,
            #marker='o',
        )

        #plt.ylim(0, 150.0)
        vertical_offset = 0.1
        for patch in ax.patches:
            # Get the height (value) and the center x-position of the bar
            height = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2
            # Format the label (e.g., two decimal places)
            label = f'{height:.2f}'

            if x == 0 and height == 0:
                continue

            # Simple placement slightly above the bar center:
            ax.text(
                x,                      # x-coordinate of the bar center
                height + vertical_offset, # y-coordinate (just above the bar)
                label,                  # The text label
                ha='center',            # Horizontal alignment (center the text on the bar)
                va='bottom',            # Vertical alignment (align the bottom of the text)
                fontsize=8,             # Adjust font size as needed
                rotation=45             # Rotate the text for better overlap prevention
            )

        plt.title(f"Partial Misses for {expt}")
        plt.xlabel("Benchmark")
        plt.xticks(rotation=45)
        plt.ylabel("Misses")
        plt.tight_layout()
        save_path = os.path.join(plot_save_dir, f"{expt}_partial_misses.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    return

baseDir = os.getcwd()
graph="com-lj"
result_dir="results/mrc_sweep_11_27"
num_sets = 8192
multiple_traces=True
mrc=True
footprint=True
stats_64 = CacheStats("L1D")
stats_08 = CacheStats("L1D")
populateStats(stats_64, result_dir, graph, 64, num_sets, multiple_traces)
populateStats(stats_08, result_dir, graph, 8, num_sets, multiple_traces)
workloadSet["all"] = stats_64.exptList
stats_64.desc="Baseline (64B)"
stats_08.desc="LRU2 (8B)"

out_dir=os.path.join("mrc_sweep_11_18_" + graph, "num_sets_" + str(num_sets))

hue_order = ["Baseline (64B)", "LRU2(8B)"]

workloadSet["as-Skitter"] = ["BFS_0M", "BFS_100M"]
#Plotting
for workload in ["PageRank", "PageRankDelta", "Radii", "Triangle", "BFS", "BFSCC", "BC", "MIS", "Components", "BellmanFord", "CF",
                 "perlbench_s", "gcc_s", "bwaves_s", "mcf_s", "cactuBSSN_s", "lbm_s",  "omnetpp_s", "wrf_s", "xalancbmk_s", "x264_s", "cam4_s", "pop2_s", "deepsjeng_s", "imagick_s", "leela_s", "nab_s", "exchange2_s", "fotonik3d_s", "roms_s", "xz_s",
                 "perlbench", "bzip2", "gcc", "bwaves", "gamess", "mcf", "milc", "zeusmp", "gromacs", "cactusADM", "leslie3d", "namd", "gobmk", "dealII", "soplex", "povray", "calculix", "hmmer", "sjeng", "GemsFDTD", "libquantum", "h264ref", "tonto", "lbm", "omnetpp", "astar", "wrf", "sphinx3", "xalancbmk",
                 "blackscholes", "bodytrack", "canneal", "dedup", "dedup", "facesim", "fluidanimate", "raytrace", "streamcluster", "swaptions", "vips",
                 "gap"]:
    set_save_path(os.path.join(out_dir, "mrc"))
    plot_mrc(workload, workloadSet[workload], [stats_64, stats_08], graph)
    plot_partials(workload, workloadSet[workload], stats_08, graph, hue_order)

