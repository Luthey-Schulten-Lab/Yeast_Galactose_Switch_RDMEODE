#!/bin/bash
# ncu --set full \
#     -f \
#     --section ComputeWorkloadAnalysis \
#     --section SourceCounters \
#     --section SpeedOfLight \
#     --section MemoryWorkloadAnalysis \
#     --metrics sm__cycles_elapsed.sum \
#     -o profile_output \
#     python galactose_rdmeode1.12_multi.py -id 99 -t 0.01 -g 11.1 -gpus 1 -tag testcase_gpu1 | tee ./logs/simulation_output_$(date +%Y%m%d_%H%M%S).txt

python galactose_model/main.py -id 1 -t 0.02 -g 11.1 -gpus 1 -tag testcase_gpu1 | tee ./logs/simulation_dynribo_output_$(date +%Y%m%d_%H%M%S).txt