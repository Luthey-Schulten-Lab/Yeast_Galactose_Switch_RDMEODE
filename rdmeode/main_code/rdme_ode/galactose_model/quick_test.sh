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

# python galactose_rdmeode1.14_dynribo.py -id 99 -t 0.01 -g 11.1 -gpus 1 -tag testcase_gpu1 | tee ./logs/simulation_output_$(date +%Y%m%d_%H%M%S).txt

# python galactose_rdmeode_combined.py -id 99 -t 0.02 -g 11.1 -gpus 2 -geo "lattice_ER_tunnels_data_Marie.pkl.xz"  --enable-er --enable-chromosome -geloc chromosome
# python galactose_rdmeode_combined.py -id 99 -t 0.02 -g 11.1 -gpus 2 -geo "lattice_ribosomes_ER_345964_isolated.pkl.xz"  --enable-er --enable-effective-ribosome
# python galactose_rdmeode_combined.py -id 99 -t 0.02 -g 11.1 -gpus 2 -geo "lattice_ribosomes_ER_345964_isolated.pkl.xz"  --enable-er 
# python galactose_rdmeode_combined.py -id 99 -t 3 -g 11.1 -gpus 2 -geo "lattice_ribosomes_ER_345964_isolated.pkl.xz"  --enable-chromosome  --enable-er -ckpt "../rdme_ode_results/20251204_ER1121_extension/0_1h/yeast1.17_combined_20251120_104_t60.0min_GAE11.1mM_ER_CHROMOchromo_ER_345k_ERribo_gpu4.lm" 

#lattice_ER_tunnels_data_Marie.pkl.xz -tag test_mRNA_tracking  --enable-chromosome --enable-er --er_num 4

# check effR2 case
python main.py -id 99 -t 0.02 -g 11.1 -gpus 2 -geo "./geometry/lattice_ribosomes_noER_345964_isolated.pkl.xz"  --enable-chromosome -geloc chromosome --enable-ribosome-movement --ribosome-movement-mode hook