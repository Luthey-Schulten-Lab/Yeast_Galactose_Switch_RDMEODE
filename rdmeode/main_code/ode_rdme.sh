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

# python galactose_rdmeode1.14_multi.py -id 99 -t 0.01 -g 11.1 -gpus 1 -tag testcase_gpu1 -geo "lattice_ribosomes_effective_isolated.pkl.xz"| tee ./logs/simulation_output_$(date +%Y%m%d_%H%M%S).txt

python galactose_rdmeode1.14_multi_noreg.py -id 99 -t 0.01 -g 11.1 -gpus 1 -tag testcase_gpu1 -geo "lattice_ribosomes_effective_isolated.pkl.xz"| tee ./logs/simulation_output_$(date +%Y%m%d_%H%M%S).txt

