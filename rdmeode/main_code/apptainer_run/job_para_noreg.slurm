#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4     # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdxm-delta-gpu #bbsv-dtai-gh #bcuj-delta-gpu # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=gw_ng
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"

### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest     # <- or closest


module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set

echo "Job is starting on `hostname`"

# Parameters passed to this script
id=$1
t=$2
g=$3
gpus=${4:-1}  # Default value for gpus is 4 if not provided
tag=${5:-"dt_4gpu"}  # Default value for tag if not provided
geo=${6:-"yeast-lattice.2.pkl.xz"}  # Default value for geo if not provided
ckpt=${7:-""}
maxtime=47
# Check if the required number of parameters are passed
if [ "$#" -lt 3 ]; then
    echo "Error: Missing mandatory parameters."
    echo "Usage: sbatch job.slurm <id> <t> <g> [gpus] [tag] [geo] [checkpiont.lm]"
    echo "Example: sbatch job.slurm 0 60 11.1 4 delta_4gpu yeast-lattice.2.pkl.xz checkpoint.lm"
    exit 1
fi

# Test Apptainer by running a command inside the container
echo "Executing command: apptainer exec --nv --bind $PWD:/workspace ../LM_build/lm2.5_dev_isolated1.sif bash /workspace/submit_job_delta_para.sh \"$id\" \"$t\" \"$g\" \"$gpus\" \"$tag\" \"$geo\" \"$ckpt\""
apptainer exec --nv --bind $PWD:/workspace ../LM_build/lm2.5_isolated0929.sif bash /workspace/submit_job_delta_para_noreg.sh "$id" "$t" "$g" "$gpus" "$tag" "$geo" "$maxtime" "$ckpt"

squeue -u $USER
# Optionally, you can run other commands inside the container
echo "Job finished"

