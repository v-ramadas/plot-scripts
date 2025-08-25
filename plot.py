#!/opt/homebrew/bin/python3.12

import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
import math

parser = argparse.ArgumentParser()

parser.add_argument("-d",
        type=str,
        default="",
        help="directory with run logs"
        )

args = parser.parse_args()


caches = ["L1D"]
block_size = [8, 16]

results_dir = os.path.join('results', args.d)

class Stats:
    class CacheStats:
        def __init__(self):
            self.hit_rate = 0
            self.total_hit_rate = 0
            self.unrealised_hit_rate = 0
            self.total_unrealised_hit_rate = 0
            self.hits = 0
            self.misses = 0
            self.accesses = 0
            self.total_misses = 0
            self.total_hits = 0
            self.total_accesses = 0
            self.unrealised_hits = 0
            self.total_unrealised_hits = 0
            self.compulsory_misses = 0
            self.capacity_misses = 0
            self.conflict_misses = 0
            self.cache_line_utilization = 0
            self.total_cache_line_utilization = 0
            self.evictions = 0
            self.total_evictions = 0
            self.mshr_merge = 0
            self.evictions_breakdown = {}
            return
        
    def __init__(self, cacheList):
        self.caches = {}
        for cache in cacheList:
            self.caches[cache] = {8: self.CacheStats(), 16: self.CacheStats()}
        return
    
    def setEvictionsBreakdown(self, cache, line: str, block_size: int):
        pattern = r'(\d+):\s+(\d+)'
        matches = re.findall(pattern, line)
        for key, value in matches:
            cache[block_size].evictions_breakdown[key] = int(value)
        return


    def setStats(self, line: str, block_size: int):
        line_list = line.split()
        if len(line_list) < 2:
            # Empty line. Skip it
            return

        match = re.match(r"cpu0->(?:cpu0_)?(.+)", line_list[0])
        if not match:
            return
        cache = match.group(1)
        if cache not in self.caches.keys():
            return
        
        if "TOTAL" != line_list[1]:
            # Not aggregate stats. Skip it
            return

        if "TOTAL_ACCESS:" == line_list[2]:
            self.caches[cache][block_size].total_accesses = int(line_list[3])
            self.caches[cache][block_size].total_hits = int(line_list[5])
            self.caches[cache][block_size].total_misses = int(line_list[7])
            self.caches[cache][block_size].unrealised_hits = int(line_list[9])
            self.caches[cache][block_size].total_hit_rate = float(self.caches[cache][block_size].total_hits/self.caches[cache][block_size].total_accesses)
            self.caches[cache][block_size].unrealised_hit_rate = float(line_list[13])
            self.caches[cache][block_size].total_unrealised_hits = int(line_list[11])
            self.caches[cache][block_size].total_unrealised_hit_rate = float(line_list[15])
        elif "EVICTIONS:" == line_list[2]:
            self.caches[cache][block_size].evictions = int(line_list[3])+1
            self.caches[cache][block_size].total_evictions = int(line_list[5])+1
            self.caches[cache][block_size].cache_line_utilization = 1 - (float(int(line_list[7])/self.caches[cache][block_size].evictions)*(block_size/64))
            self.caches[cache][block_size].total_cache_line_utilization = 1 - (float(int(line_list[9])/self.caches[cache][block_size].total_evictions)*(block_size/64))
        elif "ACCESS:" == line_list[2]:
            self.caches[cache][block_size].accesses = int(line_list[3])
            self.caches[cache][block_size].hits = int(line_list[5])
            self.caches[cache][block_size].misses = int(line_list[7])
            self.caches[cache][block_size].compulsory_misses = int(line_list[9])
            self.caches[cache][block_size].capacity_misses = int(line_list[11])
            self.caches[cache][block_size].conflict_misses = int(line_list[13])
            self.caches[cache][block_size].mshr_merge = int(line_list[15])
        elif "EVICTIONS_BREAKDOWN" == line_list[2]:
            self.setEvictionsBreakdown(self.caches[cache], line, block_size)
        return

    def count_num_files(pattern):
        files = os.listdir(os.path.join(basePath, "results", args.d, "expt_8b", "outputs"))
        count = 0
        for file in files:
            if pattern in file:
                count += 1
        return count
    
    def geomean(values):
        filtered_values = [x for x in values if x > 0]
        geomean = np.exp(np.log(filtered_values).mean())
        return geomean

    def sort_data(benchmarks, yvalues, block_sizes):
        ligra_benchmarks = []
        ligra_yvalues = []
        ligra_block_sizes = []
        spec2006_benchmarks = []
        spec2006_yvalues = []
        spec2006_block_sizes = []
        spec2017_benchmarks = []
        spec2017_yvalues = []
        spec2017_block_sizes = []
        
        for i in range(len(benchmarks)):
            expt_file = benchmarks[i]
            if "ligra" in expt_file:
                pattern = r"^ligra_ligra_(.*?)\."
                ligra_benchmarks.append(re.match(pattern, expt_file).group(1))
                ligra_yvalues.append(yvalues[i])
                ligra_block_sizes.append(str(block_sizes[i]) + 'B')
            elif "spec2006" in expt_file:
                pattern = r"^spec2006_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
                spec2006_benchmarks.append(re.match(pattern, expt_file).group(1))
                spec2006_yvalues.append(yvalues[i])
                spec2006_block_sizes.append(str(block_sizes[i]) + 'B')
            else:
                pattern = r"^spec2017_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
                spec2017_benchmarks.append(re.match(pattern, expt_file).group(1))
                spec2017_yvalues.append(yvalues[i])
                spec2017_block_sizes.append(str(block_sizes[i]) + 'B')                    

        data = {
            'benchmark': spec2006_benchmarks + spec2017_benchmarks + ligra_benchmarks,
            'yvalue': spec2006_yvalues + spec2017_yvalues + ligra_yvalues,
            'block_size': spec2006_block_sizes + spec2017_block_sizes + ligra_block_sizes,
            'suite': ["SPEC 2006"]*len(spec2006_benchmarks) + ["SPEC 2017"]*len(spec2017_benchmarks) + ["LIGRA"]*len(ligra_benchmarks),
        }
        return data
     
    def barplot(title, xlabel, ylabel, min_value, max_value, benchmarks, yvalues, block_sizes):
        data = Stats.sort_data(benchmarks, yvalues, block_sizes)
        df = pd.DataFrame(data)
        if "MPKI" in title:
            block_size_order=['8B','16B','64B']
            benchmarks_with_a_lt_10 = df[(df['block_size'] == "64B") & (df['yvalue'] <= 10)]['benchmark'].unique()
            df_lt_10 = df[df['benchmark'].isin(benchmarks_with_a_lt_10)]
            df_gt_10 = df[~df['benchmark'].isin(benchmarks_with_a_lt_10)]
            suites = ["SPEC 2006", "SPEC 2017", "LIGRA"]
            threshold_dfs = [(df_lt_10, "< 10"), (df_gt_10, "> 10")]
            # Create the plots
            fig, axs = plt.subplots(2, 3, figsize=(16, 9))
            # Plot values < 10
            for row_idx, (threshold_df, row_label) in enumerate(threshold_dfs):
                for col_idx, suite in enumerate(suites):
                    ax = axs[row_idx, col_idx]
                    subset = threshold_df[threshold_df["suite"] == suite]
                    sns.barplot(
                        data=subset,
                        x='benchmark',
                        y='yvalue',
                        hue='block_size',
                        palette='Set1',
                        hue_order=block_size_order,
                        ax=ax
                    )
                    ax.set_title(f"{suite} MPKI {row_label}")
                    ax.set_xlabel("Benchmark")
                    if col_idx == 0:
                        ax.set_ylabel("MPKI")
                    else:
                        ax.set_ylabel("")
                        ax.tick_params(axis='y', labelleft=True)  # show y-ticks even without ylabel
            
                    ax.tick_params(axis='x', rotation=90)
                    if row_label == "< 10":
                        ax.set_ylim(0, 10)
                    ax.legend_.remove()
            # Shared legend
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, 
                       title="Block Size", 
                       loc='lower center', 
                       bbox_to_anchor=(0.5, 0.05), 
                       ncol=len(block_size_order))
        else:
            block_size_order = ['8B', '16B']
            suites = ['SPEC 2006', 'SPEC 2017', 'LIGRA']
            titles = [f"{title} {suite}" for suite in suites]
            dataframes = [df[df["suite"] == suite] for suite in suites]

            fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

            for ax, df_subset, plot_title in zip(axs, dataframes, titles):
                sns.barplot(
                    data=df_subset,
                    x='benchmark',
                    y='yvalue',
                    hue='block_size',
                    palette='Set1',
                    hue_order=block_size_order,
                    ax=ax
                )
                ax.set_title(plot_title)
                ax.set_ylim(min_value, max_value)
                ax.set_xlabel(xlabel)
                ax.tick_params(axis='x', rotation=90)
                ax.tick_params(axis='y', labelleft=True)  # keep y-ticks on all

            # Set ylabel only on first subplot
            axs[0].set_ylabel(ylabel)
            for ax in axs[1:]:
                ax.set_ylabel("")

            # Shared legend
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(
                handles, labels,
                title="Block Size",
                loc='lower center',
                bbox_to_anchor=(0.5, 0.0),
                ncol=len(block_size_order)
            )

            # Remove individual legends
            for ax in axs:
                if ax.get_legend() is not None:
                    ax.legend_.remove()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(basePath, title+'.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        geomeans = df.groupby(['suite','block_size'])['yvalue'].apply(Stats.geomean)
        print(title)
        print(geomeans)
        return

    def plot_mpki(stats, cache, block_sizes):
        num_instructions = 500_000_000
        kilo_instructions = 1_000
        benchmarks = []
        yvalues = []
        sizes = []

        for expt in stats.keys():
           # Store the base MPKI
            mpki = float(stats[expt].caches[cache][8].misses/num_instructions)
            mpki -= (stats[expt].caches[cache][8].mshr_merge/num_instructions)
            yvalues.append(mpki * kilo_instructions)
            sizes.append(64)
            benchmarks.append(expt)


        for block_size in block_sizes:
            for expt in stats.keys():
                benchmarks.append(expt)
                cache_stats = stats[expt].caches[cache][block_size]
                #mpki = float((cache_stats.accesses - cache_stats.hits - cache_stats.unrealised_hits)/num_instructions)
                mpki = ((1 - cache_stats.unrealised_hit_rate)*cache_stats.accesses)/num_instructions
                yvalues.append(mpki * kilo_instructions)
                sizes.append(block_size)    
        Stats.barplot(cache + '_MPKI', "Benchmarks", "MPKI", 0, kilo_instructions/20, benchmarks, yvalues, sizes)
        return
    
    def plot_utilization(stats, cache, block_sizes):
        benchmarks = []
        yvalues = []
        sizes = []
        for block_size in block_sizes:
            for expt in stats.keys():
                benchmarks.append(expt)
                util = stats[expt].caches[cache][block_size].total_cache_line_utilization
                yvalues.append(util*100)
                sizes.append(block_size)
        Stats.barplot(cache + '_Utilization', "Benchmarks", "Average Cache Line Utilization (%)", 0, 100, benchmarks, yvalues, sizes)
        return
    
    def plot_hitrate(stats, cache, block_sizes):
        benchmarks = []
        yvalues = []
        sizes = []

        for expt in stats.keys():
            benchmarks.append(expt)
            hit_rate = float(stats[expt].caches[cache][8].hits/stats[expt].caches[cache][8].accesses)
            yvalues.append(hit_rate)
            sizes.append(64)

        for block_size in block_sizes:
            for expt in stats.keys():
                benchmarks.append(expt)
                hit_rate = float(stats[expt].caches[cache][block_size].unrealised_hit_rate)
                yvalues.append(hit_rate)
                sizes.append(block_size)
        Stats.barplot(cache + '_Hit_Rate', "Benchmarks", "Hit Rate", 0, 1.0, benchmarks, yvalues, sizes)
        return
    
    def plot_miss_breakdown(stats, cache, block_sizes):
        benchmarks = [""] * len(stats.keys())
        compulsory_misses = [0] * len(stats.keys())
        capacity_misses = [0] * len(stats.keys())
        conflict_misses = [0] * len(stats.keys())
        block_size = 8
        spec2006_idx = 0
        spec2017_idx = Stats.count_num_files("spec2006")
        ligra_idx = Stats.count_num_files("spec2017") + spec2017_idx
        suite = [""]* len(stats.keys())
        for expt in stats.keys():
            compulsory_miss = stats[expt].caches[cache][block_size].compulsory_misses
            capacity_miss = stats[expt].caches[cache][block_size].capacity_misses
            conflict_miss = stats[expt].caches[cache][block_size].conflict_misses
            #total_misses = stats[expt].caches[cache][block_size].total_misses
            total_misses = compulsory_miss + capacity_miss + conflict_miss
            if "ligra" in expt:
                pattern = r"^ligra_ligra_(.*?)\."
                idx = ligra_idx
                ligra_idx +=1
                suite[idx] = "LIGRA"
            elif "spec2006" in expt:
                pattern = r"^spec2006_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
                idx = spec2006_idx
                spec2006_idx += 1
                suite[idx] = "SPEC 2006"
            else:
                pattern = r"^spec2017_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
                idx = spec2017_idx
                spec2017_idx += 1
                suite[idx] = "SPEC 2017"
            benchmarks[idx] = re.match(pattern, expt).group(1)
            compulsory_misses[idx] = float(compulsory_miss/total_misses)
            capacity_misses[idx] = float(capacity_miss/total_misses)
            conflict_misses[idx] = float(conflict_miss/total_misses)
        
        data = {
            'benchmark': benchmarks,
            'compulsory': compulsory_misses,
            'capacity': capacity_misses,
            'conflict': conflict_misses,
            'suite': suite,
        }
        df = pd.DataFrame(data)
        sns.set(style='white')

        compulsory_geomean = df.groupby(['suite'])['compulsory'].apply(Stats.geomean)
        capacity_geomean = df.groupby(['suite'])['capacity'].apply(Stats.geomean)
        conflict_geomean = df.groupby(['suite'])['conflict'].apply(Stats.geomean)

        print("Compulsory", compulsory_geomean)
        print("Capacity", capacity_geomean)
        print("Conflict", conflict_geomean)

        #df.set_index('benchmark').plot(kind='bar', stacked=True, color=['steelblue', 'red', 'green'])
        #plt.title(cache + "_Miss_Breakdown")
        #plt.xlabel("Benchmarks")
        #plt.ylabel("Fraction")
        #plt.legend(title = "Miss Type")


        fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        df_ligra = df[df["suite"] == "LIGRA"].set_index("benchmark")
        df_spec2006 = df[df["suite"] == "SPEC 2006"].set_index("benchmark")
        df_spec2017 = df[df["suite"] == "SPEC 2017"].set_index("benchmark")

        dfs = [df_spec2006, df_spec2017, df_ligra]
        titles = ['SPEC 2006', 'SPEC 2017', 'LIGRA']
        colors = ['steelblue', 'red', 'green']  # colors for bar segments (adjust if needed)
        for ax, data, title in zip(axs, dfs, titles):
            # Plot stacked bar plot on each subplot
            data.plot(kind='bar', stacked=True, color=colors, ax=ax)
            ax.set_title(f"{cache}_{title}_Miss_Breakdown")
            ax.set_xlabel("Benchmarks")
            ax.set_ylabel("Fraction")
            ax.legend(title="Miss Type")
            ax.tick_params(axis='x', rotation=90)  # rotate x-axis labels

        plt.tight_layout()
        save_path = os.path.join(basePath, cache + '_Miss_Breakdown.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return
    
    def plot_geomean_utilization(stats, cache, block_sizes):
        benchmarks = []
        yvalues = []
        sizes = []
        for block_size in block_sizes:
            for expt in stats.keys():
                benchmarks.append(expt)
                util = stats[expt].caches[cache][block_size].total_cache_line_utilization
                yvalues.append(util*100)   
                sizes.append(block_size)
        Stats.barplot(cache + '_Utilization', "Benchmarks", "Cache Line Utilization (%)", 0, 100, benchmarks, yvalues, sizes)
        return

    def plot_evictions_breakdown(stats, cache, block_sizes):
        benchmarks = [""] * len(stats.keys())
        evictions_breakdown = [{}] * len(stats.keys())
        block_size = 8
        spec2006_idx = 0
        spec2017_idx = Stats.count_num_files("spec2006")
        ligra_idx = Stats.count_num_files("spec2017") + spec2017_idx
        suite = [""]* len(stats.keys())
        for expt in stats.keys():
            if "ligra" in expt:
                pattern = r"^ligra_ligra_(.*?)\."
                idx = ligra_idx
                ligra_idx +=1
                suite[idx] = "LIGRA"
            elif "spec2006" in expt:
                pattern = r"^spec2006_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
                idx = spec2006_idx
                spec2006_idx += 1
                suite[idx] = "SPEC 2006"
            else:
                pattern = r"^spec2017_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
                idx = spec2017_idx
                spec2017_idx += 1
                suite[idx] = "SPEC 2017"
            benchmarks[idx] = re.match(pattern, expt).group(1)
            print(benchmarks[idx])
            raw_evictions_breakdown = stats[expt].caches[cache][block_size].evictions_breakdown
            total_evictions = stats[expt].caches[cache][block_size].total_evictions
            evictions_breakdown[idx] =  {key: 100*value/total_evictions for key, value in raw_evictions_breakdown.items()}     
        data = {
            'benchmark': benchmarks,
            'evictions_breakdown': evictions_breakdown,
            'suite': suite,
        }
        df = pd.DataFrame(data)
        sns.set(style='white')

        evictions_expanded = df['evictions_breakdown'].apply(pd.Series)
        df_expanded = pd.concat([df.drop(columns=['evictions_breakdown']), evictions_expanded], axis=1)

        fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        df_ligra = df_expanded[df_expanded["suite"] == "LIGRA"].set_index("benchmark")
        df_spec2006 = df_expanded[df_expanded["suite"] == "SPEC 2006"].set_index("benchmark")
        df_spec2017 = df_expanded[df_expanded["suite"] == "SPEC 2017"].set_index("benchmark")

        dfs = [df_spec2006, df_spec2017, df_ligra]
        titles = ['SPEC 2006', 'SPEC 2017', 'LIGRA']
        palette = sns.color_palette('Paired', n_colors=block_size)[::-1]
        for ax, data, title in zip(axs, dfs, titles):
            # Plot stacked bar plot on each subplot
            data.plot(kind='bar', stacked=True, color=palette, ax=ax)
            ax.set_title(f"{cache}_{title}_Cache_Line_Usage")
            ax.set_xlabel("Benchmarks")
            ax.set_ylabel("Percentage of Cache Lines")
            #ax.legend(title="Number of Blocks Used")
            ax.tick_params(axis='x', rotation=90)  # rotate x-axis labels
            ax.set_ylim(0, 100)

        # Shared legend
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(
                handles, labels,
                title="Number of Blocks Used in Cache Line",
                loc='lower center',
                bbox_to_anchor=(0.5, 0.0),
                ncol=block_size
            )

        # Remove individual legends
        for ax in axs:
            if ax.get_legend() is not None:
                ax.legend_.remove()
        plt.tight_layout()
        save_path = os.path.join(basePath, cache + '_Evictions_Breakdown.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return


basePath = os.getcwd()

results = {}

for size in block_size:
    expt_dir = os.path.join(results_dir, "expt_" + str(size) + "b")
    os.chdir(os.path.join(basePath, expt_dir, 'outputs'))
    for expt_file in os.listdir(os.getcwd()):
        f = open(os.path.abspath(expt_file), 'r')
        if expt_file not in results.keys():
            results[expt_file] = Stats(caches)
        for line in f:
            if (len(line.split()) < 2):
                continue
            results[expt_file].setStats(line, int(size))
        f.close()

#Stats.plot_miss_breakdown(results, "L1D", block_size)
Stats.plot_mpki(results, "L1D", block_size)
Stats.plot_utilization(results, "L1D", block_size)
Stats.plot_evictions_breakdown(results, "L1D", block_size)
plt.show()
#os.chdir(basePath)
