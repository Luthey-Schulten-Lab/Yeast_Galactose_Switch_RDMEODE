#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ghx4
#SBATCH --account=bdxm-dtai-gh
#SBATCH --job-name=gs-dtai
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=tianyu16@illinois.edu
#SBATCH --mail-type="BEGIN,FAIL,END"
#SBATCH -e /projects/bddt/twu4/yeast_rdmeode/slurm_logs_dtai/slurm-%N-%x-%j.err
#SBATCH -o /projects/bddt/twu4/yeast_rdmeode/slurm_logs_dtai/slurm-%N-%x-%j.out

module reset
set -euo pipefail

APPTAINER_IMAGE="${APPTAINER_IMAGE:-../LM_build/lm25_arm_20250824.sif}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-/workspace/submit_job_delta_ai_para.sh}"
BIND_MOUNT="${BIND_MOUNT:-$PWD:/workspace}"
MAINTIME_DEFAULT=47
DEFAULT_PY_SCRIPT="galactose_rdmeode_combined.py"

usage() {
  cat <<EOF
Usage:
  sbatch job.slurm [py_script=$DEFAULT_PY_SCRIPT] <id> <t> <g> [gpus=1] [tag=dt_4gpu] [geo=yeast-lattice.2.pkl.xz] [fix_gene=random] [er=0] [chromosome=0] [eff_ribo=0] [rna_tracker=0] [er_num=4] [ckpt=""] [maxtime=$MAINTIME_DEFAULT]

First argument (optional):
  py_script : Python script to run (default: $DEFAULT_PY_SCRIPT)

Then the original positional args follow.
EOF
}

submit_delta_job() {
  local py_script="${DEFAULT_PY_SCRIPT}"
  
  if [[ "$1" == *.py ]]; then
    py_script="$1"
    shift
  fi

  local id="${1:-}"
  local t="${2:-}"
  local g="${3:-}"
  local gpus="${4:-1}"
  local tag="${5:-dt_4gpu}"
  local geo="${6:-yeast-lattice.2.pkl.xz}"
  local fix_gene="${7:-random}"
  local er="${8:-0}"
  local chromosome="${9:-0}"
  local eff_ribo="${10:-0}"
  local rna_tracker="${11:-0}"
  local er_num="${12:-4}"
  local ckpt="${13:-}"
  local maxtime="${14:-$MAINTIME_DEFAULT}"

  if [[ -z "$id" || -z "$t" || -z "$g" ]]; then
    echo "Error: Missing mandatory parameters."
    usage
    return 2
  fi

  echo "Executing job with:"
  echo "  py_script=$py_script"
  echo "  id=$id"
  echo "  t=$t"
  echo "  g=$g"
  echo "  gpus=$gpus"
  echo "  tag=$tag"
  echo "  geo=$geo"
  echo "  fix_gene=$fix_gene"
  echo "  er=$er"
  echo "  chromosome=$chromosome"
  echo "  eff_ribo=$eff_ribo"
  echo "  rna_tracker=$rna_tracker"
  echo "  er_num=$er_num"
  echo "  ckpt=${ckpt:-<none>}"
  echo "  maxtime=$maxtime"

  apptainer exec --nv --bind "$PWD:/workspace" \
    ../LM_build/lm2.5_aarch_restart_v2.sif \
    bash /workspace/submit_job_delta_para.sh \
      "$py_script" \
      "$id" "$t" "$g" "$gpus" "$tag" "$geo" "$fix_gene" \
      "$er" "$chromosome" "$eff_ribo" "$rna_tracker" "$er_num" \
      "$ckpt" "$maxtime"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  submit_delta_job "$@"
  squeue -u "${USER:-$(id -un)}" || true
  echo "Job finished"
fi
