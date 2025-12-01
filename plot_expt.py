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

parser.add_argument("--graph",
        type=str,
        default="com-lj",
        help="Ligra input graph")

args = parser.parse_args()


caches = ["L1D"]

results_dir = os.path.join('results', args.d)

num_instructions = 100_000_000
kilo_instructions = 1_000

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
            self.partial_hits = 0
            self.partial_misses = 0
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
            self.caches[cache] = self.CacheStats()
        return
    
    def setEvictionsBreakdown(self, cache, line: str):
        pattern = r'(\d+):\s+(\d+)'
        matches = re.findall(pattern, line)
        for key, value in matches:
            cache.evictions_breakdown[key] = int(value)
        return


    def setStats(self, line: str, num_blocks: int):
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
            self.caches[cache].total_accesses = int(line_list[3])
            self.caches[cache].total_hits = int(line_list[5])
            self.caches[cache].total_misses = int(line_list[7])
            self.caches[cache].unrealised_hits = int(line_list[9])
            self.caches[cache].total_hit_rate = float(self.caches[cache].total_hits/self.caches[cache].total_accesses)
            self.caches[cache].unrealised_hit_rate = float(line_list[13])
            self.caches[cache].total_unrealised_hits = int(line_list[11])
            self.caches[cache].total_unrealised_hit_rate = float(line_list[15])
        elif "EVICTIONS:" == line_list[2]:
            self.caches[cache].evictions = int(line_list[3])
            self.caches[cache].total_evictions = int(line_list[5])
            if self.caches[cache].evictions == 0:
                self.caches[cache].evictions = 1
            if self.caches[cache].total_evictions == 0:
                self.caches[cache].total_evictions = 1
            self.caches[cache].cache_line_utilization = 1 - (float(int(line_list[7])/self.caches[cache].evictions)/8)
            self.caches[cache].total_cache_line_utilization = 1 - (float(int(line_list[9])/self.caches[cache].total_evictions)/8)
        elif "ACCESS:" == line_list[2]:
            self.caches[cache].accesses = int(line_list[3])
            self.caches[cache].hits = int(line_list[5])
            self.caches[cache].misses = int(line_list[7])
            self.caches[cache].compulsory_misses = int(line_list[9])
            self.caches[cache].capacity_misses = int(line_list[11])
            self.caches[cache].conflict_misses = int(line_list[13])
            self.caches[cache].mshr_merge = int(line_list[15])
        elif "EVICTIONS_BREAKDOWN" == line_list[2]:
            self.setEvictionsBreakdown(self.caches[cache], line)
        elif "PARTIAL_HITS:" == line_list[2]:
            self.caches[cache].partial_hits = int(line_list[3])
            self.caches[cache].partial_misses = int(line_list[5])
        return
    
    def geomean(values):
        filtered_values = [x for x in values if x > 0]
        geomean = np.exp(np.log(filtered_values).mean())
        return geomean

    def sort_data(benchmarks, yvalues):
        #benchmarks = []
        #yvalues = []
#
        #
        #for i in range(len(benchmarks)):
        #    expt_file = benchmarks[i]
        #    if "ligra" in expt_file:
        #        pattern = r"^ligra_ligra_(.*?)\."
        #        ligra_benchmarks.append(re.match(pattern, expt_file).group(1))
        #        ligra_yvalues.append(yvalues[i])
        #    elif "spec2006" in expt_file:
        #        pattern = r"^spec2006_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
        #        spec2006_benchmarks.append(re.match(pattern, expt_file).group(1))
        #        spec2006_yvalues.append(yvalues[i])
        #    else:
        #        pattern = r"^spec2017_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
        #        spec2017_benchmarks.append(re.match(pattern, expt_file).group(1))
        #        spec2017_yvalues.append(yvalues[i])
#
        data = {
            'benchmark': benchmarks,
            'yvalue': yvalues,
            #'suite': ["SPEC 2006"]*len(spec2006_benchmarks) + ["SPEC 2017"]*len(spec2017_benchmarks) + ["LIGRA"]*len(ligra_benchmarks),
        }
        return data
     
    def barplot(title, xlabel, ylabel, min_value, max_value, benchmarks, yvalues, blockSize, way, suite, graph):
        data = Stats.sort_data(benchmarks, yvalues)
        df = pd.DataFrame(data)
        if "MPKI" in title:
            benchmarks_with_a_lt_10 = df[(df['yvalue'] <= 10)]['benchmark'].unique()
            df_lt_10 = df[df['benchmark'].isin(benchmarks_with_a_lt_10)]
            df_gt_10 = df[~df['benchmark'].isin(benchmarks_with_a_lt_10)]
            threshold_dfs = [(df_lt_10, "< 10"), (df_gt_10, "> 10")]

            # 1. Get all unique benchmarks from the original dataframe
            all_benchmarks = df['benchmark'].unique()
            all_benchmarks.sort() # Optional: Sort to ensure a deterministic color order

            # 2. Create a palette with enough colors for all benchmarks
            # 'Set2' is a good choice for categorical data
            n_colors = len(all_benchmarks)
            palette = sns.color_palette('Set2', n_colors)

            # 3. Map each benchmark to a specific color
            benchmark_colors = {benchmark: color for benchmark, color in zip(all_benchmarks, palette)}

            # Create the plots
            fig, axs = plt.subplots(2, 1, figsize=(16, 9))
            # Plot values < 10
            for row_idx, (threshold_df, row_label) in enumerate(threshold_dfs):
                    ax = axs[row_idx]
                    sns.barplot(
                        data=threshold_df,
                        x='benchmark',
                        y='yvalue',
                        hue='benchmark',
                        legend=False,
                        palette=benchmark_colors,
                        ax=ax,
                        order=sorted(threshold_df["benchmark"].unique())
                    )
                    if "ligra" in suite:
                        plot_title = f"{suite} MPKI {row_label} (Graph: {graph}) (Cache Block Size: {blockSize}, Num Ways: {way})"
                    else:
                        plot_title = f"{suite} MPKI {row_label} (Cache Block Size: {blockSize}, Num Ways: {way})"
                    ax.set_title(plot_title, fontsize=32)
                    ax.set_xlabel("Benchmark")
                    ax.set_ylabel("MPKI")
            
                    ax.tick_params(axis='x', rotation=90)
                    if row_label == "< 10":
                        ax.set_ylim(0, 10)
                    else:
                        ax.set_ylim(10, max_value)
                    
            # Shared legend
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, 
                       title="Block Size", 
                       loc='lower center', 
                       bbox_to_anchor=(0.5, 0.05))
            for ax in axs:
                for p in ax.patches:
                    ax.annotate(
                        f'{p.get_height():.2f}',  # Format the height to two decimal places
                        (p.get_x() + p.get_width() / 2., p.get_height()),  # Position of the text
                        ha='center',  # Horizontal alignment
                        va='center',  # Vertical alignment
                        xytext=(0, 10),  # Offset from the top of the bar
                        textcoords='offset points' # Use offset points for xytext
                    ) 
        else:
            if suite == "ligra":
                plot_title = f"{suite} {title} (Graph: {graph}) (Cache Block Size: {blockSize}, Num Ways: {way})"
            else:
                plot_title = f"{suite} {title} (Cache Block Size: {blockSize}, Num Ways: {way})"
            plt.figure(figsize=(16, 9))
            ax = sns.barplot(
                    data=df,
                    x='benchmark',
                    y='yvalue',
                    hue='benchmark',
                    legend=False,
                    palette='Set2',
                    order=sorted(df["benchmark"].unique())

            )
            plt.title(plot_title, fontsize=32)
            plt.ylim(min_value, max_value)
            plt.xlabel(xlabel)
            plt.tick_params(axis='x', rotation=90)
            plt.tick_params(axis='y', labelleft=True)  # keep y-ticks on all

            for p in ax.patches:
                ax.annotate(
                    f'{p.get_height():.2f}',  # Format the height to two decimal places
                    (p.get_x() + p.get_width() / 2., p.get_height()),  # Position of the text
                    ha='center',  # Horizontal alignment
                    va='center',  # Vertical alignment
                    xytext=(0, 10),  # Offset from the top of the bar
                    textcoords='offset points' # Use offset points for xytext
                )



            # Set ylabel only on first subplot
            plt.ylabel(ylabel)
        plt.tight_layout()
        config = "cache_blkSize_" + blockSize + "_way_" + way
        if suite == "ligra":
            save_path = os.path.join(plotBaseDir, config + '_' + suite + '_' + title + '_' + graph+'.png')
        else:
            save_path = os.path.join(plotBaseDir, config + '_' + suite + '_' + title+'.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    def plot_mpki(stats, cache, blockSize, way, suite, graph):
        benchmarks = []
        yvalues = []
        sizes = []

        for expt in stats.keys():
           # Store the base MPKI
            mpki = float(stats[expt].caches[cache].misses/num_instructions)
            mpki -= (stats[expt].caches[cache].mshr_merge/num_instructions)
            yvalues.append(mpki * kilo_instructions)
            benchmarks.append(expt)

        #for expt in stats.keys():
        #    benchmarks.append(expt)
        #    cache_stats = stats[expt].caches[cache]
            #mpki = float((cache_stats.accesses - cache_stats.hits - cache_stats.unrealised_hits)/num_instructions)
        #    mpki = ((1 - cache_stats.unrealised_hit_rate)*cache_stats.accesses)/num_instructions
        #    yvalues.append(mpki * kilo_instructions)
        Stats.barplot(cache + '_MPKI', "Benchmarks", "MPKI", 0, 150, benchmarks, yvalues, blockSize, way, suite, graph)
        return
    
    def plot_utilization(stats, cache, blockSize, way, suite, graph):
        benchmarks = []
        yvalues = []
        for expt in stats.keys():
            benchmarks.append(expt)
            util = stats[expt].caches[cache].total_cache_line_utilization
            yvalues.append(util*100)
        Stats.barplot(cache + '_Utilization', "Benchmarks", "Average Cache Line Utilization (%)", 0, 100, benchmarks, yvalues, blockSize, way, suite, graph)
        return
    

    def plot_evictions_breakdown(stats, cache, blockSize, way, suite, graph):
        benchmarks = [""] * len(stats.keys())
        evictions_breakdown = [{}] * len(stats.keys())
        idx = 0
        for expt in stats.keys():
            benchmarks[idx] = expt
            raw_evictions_breakdown = stats[expt].caches[cache].evictions_breakdown
            total_evictions = stats[expt].caches[cache].total_evictions
            evictions_breakdown[idx] =  {key: 100.0*value/total_evictions for key, value in raw_evictions_breakdown.items()}     
            idx += 1

        if suite == "ligra":
            plot_title = f"{suite} Evictions Breakdown (Graph: {graph}) (Cache Block Size: {blockSize}, Num Ways: {way})"
        else:
            plot_title = f"{suite} Evictions Breakdown (Cache Block Size: {blockSize}, Num Ways: {way})"

        df = pd.DataFrame({
            'benchmark': benchmarks,
            'evictions_breakdown': evictions_breakdown,
        })
        plt.figure(figsize=(16, 9))
        sns.set(style='white')
        evictions_expanded = df['evictions_breakdown'].apply(pd.Series)
        df_expanded = pd.concat([df.drop(columns=['evictions_breakdown']), evictions_expanded], axis=1).set_index('benchmark')

        #palette = sns.color_palette('Paired', n_colors=16)
        # Plot stacked bar plot on each subplot
        ax = df_expanded.plot(kind='bar', 
                              stacked=True)
                              #hue='benchmark',
                              #legend=False,
                              #palette='Set2',
                              #order=sorted(df["benchmark"].unique()))
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel("Benchmarks")
        ax.set_ylabel("Percentage of Cache Line Used (%)")
        #plt.legend(title="Number of Blocks Used")
        plt.xticks(np.arange(len(benchmarks)),benchmarks)
        plt.tick_params(axis='x', rotation=90)  # rotate x-axis labels
        ax.set_ylim(0, 100)
        plt.legend(
                title="Number of Blocks Used in Cache Line",
                loc='lower center',
                bbox_to_anchor=(1.0, 0.0),
            )
        plt.tight_layout()
        config = "cache_blkSize_" + blockSize + "_way_" + way
        if suite == "ligra":
            save_path = os.path.join(plotBaseDir, config + '_' + cache + '_' + suite + '_' + graph + '_Evictions_Breakdown.png')
        else:
            save_path = os.path.join(plotBaseDir, config + '_' + cache + "_" + suite + '_Evictions_Breakdown.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    def plot_partials(stats, cache, blockSize, way, suite, graph):
        benchmarks = []
        partial_hits = []
        partial_misses = []
        for expt in stats.keys():
            benchmarks.append(expt)
            partial_hits.append(float(stats[expt].caches[cache].partial_hits + stats[expt].caches[cache].mshr_merge)*(kilo_instructions/num_instructions))
            partial_misses.append(float(stats[expt].caches[cache].partial_misses * (kilo_instructions/num_instructions)))
            
        df = pd.DataFrame({
            'benchmark': benchmarks,
            'partial_hits': partial_hits,
            'partial_misses': partial_misses,
        })
        

        # 2. Melt the DataFrame into a long format
        df_melted = df.melt(
            id_vars=['benchmark'],
            value_vars=['partial_hits', 'partial_misses'],
            var_name='metric',
            value_name='value'
        )

        # 3. Create the grouped bar plot
        plt.figure(figsize=(10, 6))
        sns.set_style('whitegrid')

        ax = sns.barplot(
            x='benchmark',
            y='value',
            hue='metric',  # This is the key argument for side-by-side bars
            data=df_melted,
            palette='viridis' # Choose a color palette
        )

        # 4. Add labels and a title for clarity
        plt.xlabel("Benchmark", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        if suite == "ligra":
            plot_title = f"{suite} Partial Hits and Misses per Kilo Instruction (Graph: {graph}) (Cache Block Size: {blockSize}, Num Ways: {way})"
        else:
            plot_title = f"{suite} Partial Hits and Misses per Kilo Instruction(Cache Block Size: {blockSize}, Num Ways: {way})"
        plt.title(plot_title, fontsize=16)
        plt.legend(title='Metric')
        plt.xticks(rotation=45, ha='right')
        # Adjust layout and display the plot
        plt.tight_layout()
        config = "cache_blkSize_" + blockSize + "_way_" + way
        if suite == "ligra":
            save_path = os.path.join(plotBaseDir, config + '_' + cache + '_' + suite + '_' + graph + '_Partials.png')
        else:
            save_path = os.path.join(plotBaseDir, config + '_' + cache + "_" + suite + '_Partials.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    def plot_miss_breakdown(stats, cache, blockSize, way, suite, graph):
        benchmarks = [""] * len(stats.keys())
        compulsory_misses = [0] * len(stats.keys())
        capacity_misses = [0] * len(stats.keys())
        conflict_misses = [0] * len(stats.keys())
        block_size = int(blockSize)
        idx = 0
        for expt in stats.keys():
            compulsory_miss = stats[expt].caches[cache].compulsory_misses
            capacity_miss = stats[expt].caches[cache].capacity_misses
            conflict_miss = stats[expt].caches[cache].conflict_misses
            #total_misses = stats[expt].caches[cache].total_misses
            total_misses = compulsory_miss + capacity_miss + conflict_miss
            benchmarks[idx] = expt
            compulsory_misses[idx] = float(compulsory_miss*block_size/1024/1024)
            idx += 1
            #capacity_misses[idx] = float(capacity_miss/total_misses)
            #conflict_misses[idx] = float(conflict_miss/total_misses)
        
        data = {
            'benchmark': benchmarks,
            'Cold Misses': compulsory_misses,
        }
        df = pd.DataFrame(data)
        sns.set(style='white')

        print(df)

        #print("Compulsory", compulsory_geomean)
        #print("Capacity", capacity_geomean)
        #print("Conflict", conflict_geomean)

        #df.set_index('benchmark').plot(kind='bar', stacked=True, color=['steelblue', 'red', 'green'])
        #plt.title(cache + "_Miss_Breakdown")
        #plt.xlabel("Benchmarks")
        #plt.ylabel("Fraction")
        #plt.legend(title = "Miss Type")


        plt.figure(figsize=(48, 36))

        colors = ['steelblue', 'red', 'green']  # colors for bar segments (adjust if needed)
        # Plot stacked bar plot on each subplot
        ax = df.plot(
            x='benchmark',
            y='Cold Misses',
            kind='bar'
        )

        if suite == "ligra":
            plot_title = f"{suite} Working Set Size (Graph: {graph}) (Cache Block Size: {blockSize}, Num Ways: {way})"
        else:
            plot_title = f"{suite} Working Set Size (Cache Block Size: {blockSize}, Num Ways: {way})"
        plt.title(plot_title)
        plt.xlabel("Benchmarks")
        plt.ylabel("Data Size (MB)")
        plt.ylim(0, 120)
        #plt.legend(title="Miss Type")
        plt.tick_params(axis='x', rotation=90)  # rotate x-axis labels

        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.2f}',  # Format the height to two decimal places
                (p.get_x() + p.get_width() / 2., p.get_height()),  # Position of the text
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                xytext=(0, 10),  # Offset from the top of the bar
                textcoords='offset points', # Use offset points for xytext
                fontsize=6
            )


        plt.tight_layout()
        config = "cache_blkSize_" + blockSize + "_way_" + way
        if suite == "ligra":
            save_path = os.path.join(plotBaseDir, config + '_' + cache + '_' + suite + '_' + graph + '_Cold_Misses.png')
        else:
            save_path = os.path.join(plotBaseDir, config + '_' + cache + "_" + suite + '_Cold_Misses.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return


def count_num_files(stats, pattern):
        count = 0
        for key in stats.keys():
            if pattern in key:
                count += 1
        return count

def populateStats(results_dir: str, expt_dir: str, suite: str, graph: str, ):
    results = {}
    blockSize = re.findall(r'\d+', expt_dir)[0]
    print(expt_dir, blockSize)
    way = re.findall(r'\d+', expt_dir)[1]  
    expt_dir_path = os.path.join("results", results_dir, "expt", "outputs", expt_dir)
    os.chdir(os.path.join(baseDir, expt_dir_path))
    for expt_file in os.listdir(os.getcwd()):
        f = open(os.path.abspath(expt_file), 'r')
        pattern = r'^[^_]+'
        match = re.match(pattern, expt_file)
        if match is None or match.group(0) != suite:
            continue

        if "ligra" == suite:
            pattern = pattern = r"[^.]+\.([^.]+)"
            match = re.match(pattern, expt_file)
            if match.group(1) != graph:
                continue

            pattern = r"^ligra_ligra_(.*?)\."
        elif "spec2006"  == suite:
            pattern = r"^spec2006_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
        else:
            pattern = r"^spec2017_\d+\.(\w+)-.*\.champsimtrace\.xz\.stdout$"
        expt = (re.match(pattern, expt_file).group(1))

        if expt not in results.keys():
            results[expt] = Stats(caches)
        for line in f:
            if (len(line.split()) < 2):
                continue
            results[expt].setStats(line, int(blockSize)/8)
        f.close()
    
    Stats.plot_mpki(results, "L1D", blockSize, way, suite, graph)
    Stats.plot_utilization(results, "L1D", blockSize, way, suite, graph)
    Stats.plot_evictions_breakdown(results, "L1D", blockSize, way, suite, graph)
    Stats.plot_partials(results, "L1D", blockSize, way, suite, graph)
    #Stats.plot_miss_breakdown(results, "L1D", blockSize, way, suite, graph)



baseDir = os.getcwd()
plotBaseDir = os.path.join(baseDir, "plots", args.d)

if not os.path.exists(plotBaseDir):
    os.makedirs(plotBaseDir)

populateStats(args.d, "blkSize_64_way_16", "spec2006", args.graph)
populateStats(args.d, "blkSize_64_way_16", "spec2017", args.graph)
populateStats(args.d, "blkSize_64_way_16", "ligra", args.graph)
populateStats(args.d, "blkSize_32_way_32", "spec2006", args.graph)
populateStats(args.d, "blkSize_32_way_32", "spec2017", args.graph)
populateStats(args.d, "blkSize_32_way_32", "ligra", args.graph)
populateStats(args.d, "blkSize_16_way_64", "spec2006", args.graph)
populateStats(args.d, "blkSize_16_way_64", "spec2017", args.graph)
populateStats(args.d, "blkSize_16_way_64", "ligra", args.graph)
populateStats(args.d, "blkSize_8_way_128", "spec2006", args.graph)
populateStats(args.d, "blkSize_8_way_128", "spec2017", args.graph)
populateStats(args.d, "blkSize_8_way_128", "ligra", args.graph)
