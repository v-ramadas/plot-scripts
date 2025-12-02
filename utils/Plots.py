#!/opt/homebrew/bin/python3.12
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from fractions import Fraction
import pandas as pd
import numpy as np

plot_base_dir = "plots/"
plot_save_dir = ""

def set_save_path(dir_name):
    global plot_base_dir
    global plot_save_dir
    plot_save_dir = os.path.join(plot_base_dir, dir_name)
    if not os.path.exists(plot_save_dir): 
        os.makedirs(plot_save_dir)
    

def plot_mrc(workload_set, benchmarks, stats_list, graph):
    # Implementation for MRC scatter plot
    for stats in stats_list:
        df_list = [
            pd.DataFrame({
                "Cache Size": stats.mrc[expt].cache_sizes,
                "MPKI": stats.mrc[expt].miss_rate,
                "Benchmark": expt,
            })
            for expt in benchmarks
            if expt in stats.mrc.keys() and len(stats.mrc[expt].cache_sizes) > 0
        ]

        if len(df_list) == 0:
            continue
        df_long = pd.concat(df_list, ignore_index=True)

        df_long = df_long.sort_values(
            by=["Cache Size"],
            ascending=True)
        hue_order = sorted(df_long['Benchmark'].unique())

        ax = sns.scatterplot(
            data=df_long,
            x="Cache Size",      # The shared x-axis variable
            y="MPKI",     # The y-axis variable
            hue="Benchmark", # The categorical variable that defines each line
            hue_order=hue_order,
            palette="Paired",
            marker="_",
            size=15,
            linewidth=2.5,
        )
        plt.xscale('log', base=2)

        formatter = ticker.FuncFormatter(lambda v, pos: f"{Fraction(v).limit_denominator()}")
        #Apply the formatter to the y-axis major ticks
        plt.gca().xaxis.set_major_formatter(formatter)

        plt.ylim(0, 250)
        plt.xlim(1, 1024)
        plt.xlabel("Cache Size (MB)")
        plt.ylabel("MPKI")
        plt.title(f"MRC for {workload_set} (Cache Block Size {stats.block_size}B)")
        save_path = os.path.join(plot_save_dir, f"{workload_set}_mrc_blkSize_{stats.block_size}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    return

def plot_footprint(workload_set, benchmarks, stats_list, graph):
        df_list = [
            pd.DataFrame({
                "Footprint": [stats.mrc[expt].cache_sizes[-1]],
                "Benchmark": [expt],
                "Cache Block Size": [stats.block_size],
            })
            for expt in benchmarks
            for stats in stats_list
            if expt in stats.expts.keys()
        ]

        print(workload_set)
        if len(df_list) == 0:
            print(f'{workload_set} does not have valid data. Skipping')
            return
        df = pd.concat(df_list)

        hue_order = sorted(df['Cache Block Size'].unique(), 
                           reverse=True)
        df = df.sort_values(
            by=["Footprint"],
            ascending=False,
        )

        print(df)

        ax = sns.barplot(
            data=df,
            x="Benchmark",      # The shared x-axis variable
            y="Footprint",     # The y-axis variable
            hue="Cache Block Size", # The categorical variable that defines each line
            palette="pastel",
            hue_order=hue_order,
        )
        
        plt.ylim(1, 1*1024)
        plt.yscale('log', base=2)

        formatter = ticker.FuncFormatter(lambda v, pos: f"{Fraction(v).limit_denominator()} MB")
        #Apply the formatter to the y-axis major ticks
        plt.gca().yaxis.set_major_formatter(formatter)

        vertical_offset = 0.1
        for patch in ax.patches:
            # Get the height (value) and the center x-position of the bar
            height = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2
            # Format the label (e.g., two decimal places)
            label = f'{height:.2f} MB'

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

        plt.title(f"Memory Footprint for {workload_set}")
        plt.xlabel("Benchmark")
        plt.xticks(rotation=45)
        plt.ylabel("Footprint (KB)")
        save_path = os.path.join(plot_save_dir, f"{workload_set}_footprint.png")
        plt.savefig(save_path)#, bbox_inches='tight')
        plt.close()
        return

def plot_utilization(workload_set, benchmarks, stats_list, graph, hue_order):
    df = pd.concat([
        pd.DataFrame({
            "Utilization": [stats.expts[expt].cache_line_utilization * 100],
            "Benchmark": [expt],
            "Cache Block Size": [stats.block_size],
            "Legend": [stats.desc],
        })
        for expt in benchmarks
        for stats in stats_list
        if expt in stats.expts.keys()
    ])
    #hue_order = sorted(df['Cache Block Size'].unique(), 
    #                   reverse=True)

    df = df.sort_values(
        by=["Cache Block Size", "Utilization"],
        ascending=False,
    )

    ax = sns.barplot(
        data=df,
        x="Benchmark",      # The shared x-axis variable
        y="Utilization",     # The y-axis variable
        hue="Legend", # The categorical variable that defines each line
        palette="pastel",
        hue_order=hue_order,
        #marker='o',
    )
    
    plt.ylim(0, 100.0)

    vertical_offset = 0.1
    for patch in ax.patches:
        # Get the height (value) and the center x-position of the bar
        height = patch.get_height()
        x = patch.get_x() + patch.get_width() / 2
        # Format the label (e.g., two decimal places)
        label = f'{height:.2f}%'
        
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

    plt.title(f"Cache Line Utilization for {workload_set}")
    plt.xlabel("Benchmark")
    plt.xticks(rotation=45)
    plt.ylabel("Cache Line Utilization (%)")
    plt.tight_layout()
    save_path = os.path.join(plot_save_dir, f"{workload_set}_utilization.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return

def plot_mpki(workload_set, benchmarks, stats_list, graph, hue_order):
    df = pd.concat([
        pd.DataFrame({
            "MPKI": [(stats.expts[expt].misses - stats.expts[expt].mshr_merge)*1000.0/stats.expts[expt].num_instructions],
            "Benchmark": [expt],
            "Cache Block Size": [stats.block_size],
            "Legend": [stats.desc],
        })
        for expt in benchmarks
        for stats in stats_list
        if expt in stats.expts.keys()
    ])
    #hue_order = sorted(df['Cache Block Size'].unique(), 
    #                   reverse=True)

    df = df.sort_values(
        by=["Cache Block Size", "MPKI"],
        ascending=False
    )         
    ax = sns.barplot(
        data=df,
        x="Benchmark",      # The shared x-axis variable
        y="MPKI",     # The y-axis variable
        hue="Legend", # The categorical variable that defines each line
        palette="pastel",
        hue_order=hue_order,
        #marker='o',
    )
    
    plt.ylim(0, 150.0)
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

    plt.title(f"MPKI for {workload_set}")
    plt.xlabel("Benchmark")
    plt.xticks(rotation=45)
    plt.ylabel("MPKI")
    plt.tight_layout()
    save_path = os.path.join(plot_save_dir, f"{workload_set}_mpki.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return

def plot_evictions_breakdown(workload_set, benchmarks, stats_list, graph):
    for stats in stats_list:
        df = pd.concat([
            pd.DataFrame({
                "Evictions Breakdown": [{key: 100.0*value/stats.expts[expt].total_evictions for key, value in stats.expts[expt].evictions_breakdown.items()}],
                "Benchmark": [expt],
            })
            for expt in benchmarks
            if expt in stats.expts.keys()
        ])

        evictions_expanded = df['Evictions Breakdown'].apply(pd.Series)
        df_expanded = pd.concat([df.drop(columns=['Evictions Breakdown']), evictions_expanded], axis=1).set_index('Benchmark')

        #palette = sns.color_palette('Paired', n_colors=16)
        # Plot stacked bar plot on each subplot
        ax = df_expanded.plot(kind='bar', 
                                  stacked=True,
                                  colormap='Paired',
                                  #order=sorted(df["benchmark"].unique()))
                            )
        ax.set_title(f"Evictions Breakdown (Block Size: {stats.block_size} Bytes)")
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
        save_path = os.path.join(plot_save_dir, f'{workload_set}_evictions_breakdown_blkSize_{stats.block_size}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    return

def plot_ipc(workload_set, benchmarks, stats_list, graph, hue_order):
    df = pd.concat([
        pd.DataFrame({
            "IPC": [stats.expts[expt].ipc],
            "Benchmark": [expt],
            "Cache Block Size": [stats.block_size],
            "Legend": [stats.desc],
        })
        for expt in benchmarks
        for stats in stats_list
        if expt in stats.expts.keys()
    ])
    #hue_order = sorted(df['Cache Block Size'].unique(), 
    #                   reverse=True)

    df = df.sort_values(
        by=["Cache Block Size", "IPC"],
        ascending=False,
    )

    ax = sns.barplot(
        data=df,
        x="Benchmark",      # The shared x-axis variable
        y="IPC",     # The y-axis variable
        hue="Legend", # The categorical variable that defines each line
        palette="pastel",
        hue_order=hue_order,
        #marker='o',
    )
    
    plt.ylim(0, 2.0)

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

    plt.title(f"IPC for {workload_set}")
    plt.xlabel("Benchmark")
    plt.xticks(rotation=45)
    plt.ylabel("IPC")
    plt.tight_layout()
    save_path = os.path.join(plot_save_dir, f"{workload_set}_ipc.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return

