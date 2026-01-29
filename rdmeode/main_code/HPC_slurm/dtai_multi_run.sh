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
  
  #sbatch job_submit_dtai.slurm "$id" 120 11.1 4 baseline lattice_ribosomes_noER_345964_isolated.pkl.xz random 0 0 0
  #sbatch job_submit_dtai.slurm "$id" 120 11.1 4 baseline lattice_ribosomes_noER_345964_isolated.pkl.xz random 0 0 0 "" "" "./simulation_results_id_703/yeast1.16_combined_20251111_703_t120.0min_GAE11.1mMbaseline_gpu4.lm" 
  
  #sbatch job_submit_dtai.slurm galactose_rdmeode_combined_maxDG80d.py "$id" 60 11.1 4 baseline lattice_ribosomes_noER_345964_isolated.pkl.xz random 0 0 0
  sbatch job_submit_dtai_effG2.slurm "$id" 60 11.1 4 eff_riboG2 lattice_ribosomes_ER_345964_isolated.pkl.xz chromosome 0 1 0
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 chromo_ER_345k_ERribox2 lattice_ribosomes_ER_345964__ERribo_x2_isolated.pkl.xz chromosome 1 1 0
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 chromo_ER_345k_ERribox2_180K lattice_ribosomes_ER_180000__ERribo_x2_isolated_180k.pkl.xz chromosome 1 1 0  
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 chromo_ER_345k_ERribox2_noncyto_cecNE lattice_ribosomes_ER_345964__ERribo_x2_isolated_nonCECNE.pkl.xz chromosome 1 1 0
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 chromo_ER_345k_ERribo lattice_ribosomes_ER_345964_isolated.pkl.xz chromosome 1 1 0
  #sbatch job_submit_dtai.slurm "$id" 180 11.1 4 chromo1_4 lattice_ribosomes_noER_345964_isolated.pkl.xz chromosome 0 1 0 "" "" "./20251101_chromosome_newcytoribono/yeast1.16_combined_20251031_513_t60.0min_GAE11.1mM_CHROMOchromo_345k_gpu4.lm"
  #sbatch job_submit_dtai.slurm "$id" 180 11.1 4 chromo4_7 lattice_ribosomes_noER_345964_isolated.pkl.xz chromosome 0 1 0 "" "" "./20251101_chromosome_newcytoribono/yeast1.17_combined_20251204_523_t180.0min_GAE11.1mM_CHROMOchromo1_4_gpu4.lm"
  #sbatch job_submit_dtai.slurm "$id" 180 11.1 4 chromoER1_4 lattice_ribosomes_ER_345964_isolated.pkl.xz chromosome 1 1 0 "" "" "./0_1h/yeast1.17_combined_20251120_103_t60.0min_GAE11.1mM_ER_CHROMOchromo_ER_345k_ERribo_gpu4.lm"
  #sbatch job_submit_dtai.slurm "$id" 180 11.1 4 chromoER4_7 lattice_ribosomes_ER_345964_isolated.pkl.xz chromosome 1 1 0 "" "" "./0_1h/yeast1.17_combined_20251212_112_t180.0min_GAE11.1mM_ER_CHROMOchromoER4_5_gpu4.lm"
#sbatch job_submit_dtai.slurm "$id" 180 11.1 4 chromoEFF1_4 lattice_ribosomes_ER_345964_isolated.pkl.xz chromosome 1 1 1 "" "" "./20251204_EFF1121_extension/yeast1.17_combined_20251120_201_t60.0min_GAE11.1mM_ER_CHROMO_EFFRIBOeffecribo345k_gpu4.lm"
  #sbatch job_submit_dtai.slurm "$id" 180 11.1 4 chromoEFF4_7 lattice_ribosomes_ER_345964_isolated.pkl.xz chromosome 1 1 1 "" "" "./20251204_EFF1121_extension/yeast1.17_combined_20251204_213_t180.0min_GAE11.1mM_ER_CHROMO_EFFRIBOchromoEFF1_4_gpu4.lm"
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 effecribo345k lattice_ribosomes_ER_345964_isolated.pkl.xz chromosome 1 1 1
  #sbatch job_submit_dtai.slurm "$id" 0.1 11.1 4 speed_test lattice_ER_tunnels_data_Marie.pkl.xz random 0 0 0
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 random_tracker yeast-lattice.2.pkl.xz random 0 0 0 1
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 chromo_tracker yeast-lattice.2.pkl.xz chromosome 1 0 0 1
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 ribo181_eff lattice_ER_tunnels_data_Marie.pkl.xz chromosome 1 1 1
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 chromo_ER_rnatracker lattice_ER_tunnels_data_Marie.pkl.xz chromosome 1 1 0 1
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 chromo_ER_tb1 lattice_ER_tunnels_data_Marie.pkl.xz chromosome 1 1 0 0 1
  #sbatch job_submit_dtai.slurm "$id" 60 11.1 4 chromo_ER_tb2 lattice_ER_tunnels_data_Marie.pkl.xz chromosome 1 1 0 0 2

done

