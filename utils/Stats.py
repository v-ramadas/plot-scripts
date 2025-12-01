import os
import re

class CacheStats:
    class ExptStats:
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
            self.cold_misses = 0
            self.capacity_misses = 0
            self.conflict_misses = 0
            self.cache_line_utilization = 0
            self.total_cache_line_utilization = 0
            self.evictions = 0
            self.total_evictions = 0
            self.mshr_merge = 0
            self.evictions_breakdown = {}
            self.simpoints_inst_dropped = ""
            self.num_instructions = 0
            self.ipc = 0
            self.suite = ""
            self.graph = ""
            self.desc = ""
            return

    class MRCStats:
        def __init__(self):
            self.cache_sizes = []
            self.cumulative_hits = []
            self.misses = []
            self.miss_rate = []
            self.partial_misses = {}
            self.suite = ""
            self.graph = ""
            return
        
    def __init__(self, cache_name, cache_size = "2MB", block_size = 64):#, exptList):
        self.expts = {}
        self.mrc = {}
        self.cache_name = cache_name
        self.cache_size = cache_size
        self.block_size = block_size
        self.exptList = []
    
    def setEvictionsBreakdown(self, expt, line: str):
        pattern = r'(\d+):\s+(\d+)'
        matches = re.findall(pattern, line)
        for key, value in matches:
            expt.evictions_breakdown[key] = int(value)
        return


    def setExptStats(self, expt: str, stat_file: str, suite: str, graph: str):
        start_stats = False
        if expt not in self.expts.keys():
            self.expts[expt] = self.ExptStats()
            self.expts[expt].suite = suite
            self.expts[expt].graph = graph
        
        f = open(stat_file, 'r')
        for line in f:
            if not start_stats and "Region of Interest" in line:
                start_stats = True
                #next(f)
                continue
            
            if start_stats:
                line_list = line.split()
                if len(line_list) < 2:
                   # Empty line. Skip it
                   continue

                if "IPC:" in line_list:
                   self.expts[expt].ipc = float(line_list[4])

                if self.expts[expt].num_instructions == 0:
                    match = re.search(r"instructions: (\d+)", line)
                    if match:
                        self.expts[expt].num_instructions = int(match.group(1))
                    continue
                
                match = re.match(r"cpu0->(?:cpu0_)?(.+)", line_list[0])
                if not match:
                    continue
                cache = match.group(1)
                if cache != self.cache_name:
                   continue

                if "TOTAL" != line_list[1]:
                   # Not aggregate stats. Skip it
                   return

                if "TOTAL_ACCESS:" == line_list[2]:
                    self.expts[expt].total_accesses = int(line_list[3])
                    self.expts[expt].total_hits = int(line_list[5])
                    self.expts[expt].total_misses = int(line_list[7])
                    self.expts[expt].unrealised_hits = int(line_list[9])
                    self.expts[expt].total_hit_rate = float(self.expts[expt].total_hits/self.expts[expt].total_accesses)
                    self.expts[expt].unrealised_hit_rate = float(line_list[13])
                    self.expts[expt].total_unrealised_hits = int(line_list[11])
                    self.expts[expt].total_unrealised_hit_rate = float(line_list[15])
                elif "EVICTIONS:" == line_list[2]:
                    self.expts[expt].evictions = int(line_list[3])
                    self.expts[expt].total_evictions = int(line_list[5])
                    if self.expts[expt].evictions == 0:
                        self.expts[expt].evictions = 1
                    if self.expts[expt].total_evictions == 0:
                        self.expts[expt].total_evictions = 1
                    self.expts[expt].cache_line_utilization = 1 - (float(int(line_list[7])/self.expts[expt].evictions)/8)
                    self.expts[expt].total_cache_line_utilization = 1 - (float(int(line_list[9])/self.expts[expt].total_evictions)/8)
                elif "ACCESS:" == line_list[2]:
                    self.expts[expt].accesses = int(line_list[3])
                    self.expts[expt].hits = int(line_list[5])
                    self.expts[expt].misses = int(line_list[7])
                    self.expts[expt].cold_misses = int(line_list[9])
                    self.expts[expt].capacity_misses = int(line_list[11])
                    self.expts[expt].conflict_misses = int(line_list[13])
                    self.expts[expt].mshr_merge = int(line_list[15])
                elif "EVICTIONS_BREAKDOWN" == line_list[2]:
                    self.setEvictionsBreakdown(self.expts[expt], line)
                elif "PARTIAL_HITS:" == line_list[2]:
                    self.expts[expt].partial_hits = int(line_list[3])
                    self.expts[expt].partial_misses = int(line_list[5])
        return

    def setMRCStats(self, expt: str, stat_file: str, suite: str, graph: str):
        start_stats = False
        f = open(stat_file, 'r')
        if expt not in self.mrc.keys():
            self.mrc[expt] = self.MRCStats()
            self.expts[expt].suite = suite
            self.expts[expt].graph = graph
        for line in f:
            if "MPKI Curve (MpkiC)" in line:
                start_stats = True
                next(f)
                next(f)
                continue
            if start_stats:
                line_list = line.split('|')
                if len(line_list) != 4:
                    # End of MRC stats
                    return

                self.mrc[expt].cache_sizes.append(float(line_list[0].strip())*self.block_size/(1024.0 * 1024.0))
                self.mrc[expt].cumulative_hits.append(int(line_list[1].strip()))
                self.mrc[expt].misses.append(int(line_list[2].strip()))
                self.mrc[expt].miss_rate.append(float(line_list[3].strip()))
        return
    
    def populateStats(self, results_dir, graph, block_size, num_ways, multiple_traces=False):        
        stats_dir = os.path.join("results", results_dir, "expt", "outputs", f"blkSize_{block_size}_way_{num_ways}")
        baseDir = os.getcwd()
        os.chdir(os.path.join(baseDir, stats_dir))
        self.block_size = block_size
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
            self.exptList.append(expt)
            self.setExptStats(expt, os.path.abspath(expt_file), suite, graph)
            self.setMRCStats(expt, os.path.abspath(expt_file), suite, graph)
            self.expts[expt].simpoints_inst_dropped = simpoints_inst_dropped

            f.close()
        os.chdir(baseDir)
