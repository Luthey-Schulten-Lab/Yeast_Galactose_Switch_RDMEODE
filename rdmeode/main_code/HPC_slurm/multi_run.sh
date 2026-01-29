#!/usr/bin/env bash
# submit_range.sh — loop over IDs from START to END (inclusive)

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <START_ID> <END_ID>"
  exit 1
fi

START_ID=$1
END_ID=$2

for ((id=START_ID; id<=END_ID; id++)); do
  echo "Submitting job with id=$id"
  
  #sbatch job_submit.slurm "$id" 60 11.1 4 chromo_only yeast-lattice.2.pkl.xz chromosome 1 0 0 
  #sbatch job_submit.slurm "$id" 0.1 11.1 4 speed_test lattice_ER_tunnels_data_Marie.pkl.xz random 0 0 0
  #sbatch job_submit.slurm "$id" 60 11.1 4 chromo_ER_Marie_eff lattice_ER_tunnels_data_Marie.pkl.xz chromosome 1 1 1
  #sbatch job_submit.slurm "$id" 60 11.1 4 chromo_ER_Marie lattice_ER_tunnels_data_Marie.pkl.xz chromosome 1 1 0
  #sbatch job_submit.slurm "$id" 60 11.1 4 chromo_ER_rnatracker lattice_ER_tunnels_data_Marie.pkl.xz chromosome 1 1 0 1
  #sbatch job_submit.slurm "$id" 60 11.1 4 chromo_ER_tb1 lattice_ER_tunnels_data_Marie.pkl.xz chromosome 1 1 0 0 1
  #sbatch job_submit.slurm "$id" 60 11.1 4 chromo_ER_tb2 lattice_ER_tunnels_data_Marie.pkl.xz chromosome 1 1 0 0 2
  sbatch job_submit.slurm "$id" 180 11.1 4 chromoER1_4 lattice_ribosomes_ER_345964_isolated.pkl.xz chromosome 1 1 0 "" "" "./0_1h/yeast1.17_combined_20251120_103_t60.0min_GAE11.1mM_ER_CHROMOchromo_ER_345k_ERribo_gpu4.lm"
done

