#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4     # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdxm-delta-gpu #bbsv-dtai-gh #bcuj-delta-gpu # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=yeastER
#SBATCH --time=13:00:00      # hh:mm:ss for the job
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
gpus=${4:-4}  # Default value for gpus is 4 if not provided
tag=${5:-"dt_4gpu"}  # Default value for tag if not provided
geo=${6:-"lattice_ER_tunnels_data.pkl.xz"}  # Default value for geo if not provided
fix_gene=${7:-"random"}  #default value for fixed gene or not.
maxtime=47
# Check if the required number of parameters are passed
if [ "$#" -lt 3 ]; then
    echo "Error: Missing mandatory parameters."
    echo "Usage: sbatch job.slurm <id> <t> <g> [gpus] [tag] [geo] [fix_gene]"
    echo "Example: sbatch job.slurm 0 60 11.1 4 delta_4gpu yeast-lattice.2.pkl.xz random"
    exit 1
fi

# Test Apptainer by running a command inside the container
echo "Executing command: apptainer exec --nv --bind $PWD:/workspace ../LM_build/lm2.5_dev_isolated1.sif bash /workspace/submit_job_delta_para.sh \"$id\" \"$t\" \"$g\" \"$gpus\" \"$tag\" \"$geo\" \"$fix_gene\""
apptainer exec --nv --bind $PWD:/workspace ../LM_build/lm2.5_isolated.sif bash /workspace/submit_job_delta_para_er.sh "$id" "$t" "$g" "$gpus" "$tag" "$geo" "$maxtime" "$fix_gene"

squeue -u $USER
# Optionally, you can run other commands inside the container
echo "Job finished"

