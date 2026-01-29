#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    # <- match to OMP_NUM_THREADS

#SBATCH --partition=ghx4     # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdxm-dtai-gh # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=galactose4_dtai
#SBATCH --time=8:00:00      # hh:mm:ss for the job

### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest     # <- or closest

#SBATCH --mail-user=hli106@illinois.edu
#SBATCH --mail-type="BEGIN,FAIL,END" #See sbatch or srun man pages for more email options
#SBATCH -e testing_dtai.err
#SBATCH -o testing_dtai.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set

echo "Job is starting on `hostname`"

# Parameters passed to this script
id=$1
t=$2
g=$3
gpus=${4:-1}  # Default value for gpus is 4 if not provided
tag=${5:-"dtgrace_4gpu"}  # Default value for tag if not provided
geo=${6:-"yeast-lattice.2.pkl.xz"}  # Default value for geo if not provided
maxtime=8
# Check if the required number of parameters are passed
if [ "$#" -lt 3 ]; then
    echo "Error: Missing mandatory parameters."
    echo "Usage: sbatch job.slurm <id> <t> <g> [gpus] [tag] [geo]"
    echo "Example: sbatch job.slurm 0 60 11.1 4 delta_4gpu yeast-lattice.2.pkl.xz"
    exit 1
fi

# Test Apptainer by running a command inside the container
echo "Executing command: apptainer exec --nv --bind $PWD:/workspace LM25.sif bash /workspace/submit_job_delta_81.sh \"$id\" \"$t\" \"$g\" \"$gpus\" \"$tag\" \"$geo\""
apptainer exec --nv --bind $PWD:/workspace ../LM_build/lm25_arm_20250824.sif bash submit_job_delta_81.sh "$id" "$t" "$g" "$gpus" "$tag" "$geo" "$maxtime"

squeue -u $USER
# Optionally, you can run other commands inside the container
echo "Job finished"

