#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4     # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdxm-delta-gpu #bbsv-dtai-gh #bcuj-delta-gpu # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=gs
#SBATCH --time=13:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"

### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest     # <- or closest
#SBATCH --mail-user=tianyu16@illinois.edu
#SBATCH --mail-type="BEGIN,FAIL,END" #See sbatch or srun man pages for more email options
#SBATCH -e /projects/bddt/twu4/yeast_rdmeode/slurm_logs/slurm-%N-%x-%j.err
#SBATCH -o /projects/bddt/twu4/yeast_rdmeode/slurm_logs/slurm-%N-%x-%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set

set -euo pipefail
# -------- Defaults / config --------
APPTAINER_IMAGE="${APPTAINER_IMAGE:-../LM_build/lm2.5_isolated.sif}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-/workspace/submit_job_delta_para.sh}"
BIND_MOUNT="${BIND_MOUNT:-$PWD:/workspace}"
MAINTIME_DEFAULT=47


usage() {
  cat <<EOF
Usage:
  sbatch job.slurm <id> <t> <g> [gpus=1] [tag=dt_4gpu] [geo=yeast-lattice.2.pkl.xz] [fix_gene=random] [er=0] [chromosome=0] [eff_ribo=0] [rna_tracker=0] [er_num=4] [ckpt=""] [maxtime=$MAINTIME_DEFAULT]

Positional args:
  id           : Experiment ID
  t            : Simulation time
  g            : exgal parameter
  gpus         : GPU list or count (default=1)
  tag          : Run tag (default=dt_4gpu)
  geo          : Geometry file (default=yeast-lattice.2.pkl.xz)
  fix_gene     : Fixed gene mode (default=random)
  chromosome   : Chromosome mode (0=no chromosome, 1=with chromosome, default=0)
  er           : ER mode (0=no ER, 1=with ER, default=0)
  eff_ribo     : Effective ribosome (0=off, 1=on, default=0)
  rna_tracker  : RNA tracking (0=off, 1=on default=0)
  er_num       : number of er tunnels added(1,2,3,4, default=4)
  ckpt         : Checkpoint path (optional)
  maxtime      : Max sim time (default=$MAINTIME_DEFAULT)
EOF
}

submit_delta_job() {
  local id="${1:-}"
  local t="${2:-}"
  local g="${3:-}"
  local gpus="${4:-1}"
  local tag="${5:-dt_4gpu}"
  local geo="${6:-yeast-lattice.2.pkl.xz}"
  local fix_gene="${7:-chromosome}"
  local chromosome="${8:-1}"
  local er="${9:-0}"
  local eff_ribo="${10:-0}"
  local rna_tracker="${11:-0}"
  local er_num="${12:-4}"
  local ckpt="${13:-}"
  local maxtime="${14:-$MAINTIME_DEFAULT}"

  if [[ -z "$id" || -z "$t" || -z "$g" ]]; then
    echo "Error: Missing mandatory parameters." >&2
    usage
    return 2
  fi

  echo "Executing Apptainer job with parameters:"
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
  
apptainer exec --nv --bind "$PWD:/workspace" ../LM_build/lm2.5_isolated.sif \
  bash /workspace/submit_job_delta_para.sh \
  "$id" "$t" "$g" "$gpus" "$tag" "$geo" "$fix_gene" "$er" "$chromosome" "$eff_ribo" "$rna_tracker" "$er_num" "$ckpt" "$maxtime"
}

# -------- sbatch entry point --------
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  submit_delta_job "$@"
  squeue -u "${USER:-$(id -un)}" || true
  echo "Job finished"
fi
