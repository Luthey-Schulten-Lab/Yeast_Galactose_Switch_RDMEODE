#!/bin/bash

# Accepting inputs from command line arguments to match slurm script order
id=$1
t=$2
g=$3
gpus=$4
tag=$5
geo=$6
fix_gene=${7}
er=${8}
chromosome=${9}
eff_ribo=${10}
ckpt=${11}
maxtime=${12}




# Source the conda environment
# Get the system architecture
ARCH=$(uname -m)

# Check if the architecture is aarch64 (ARM)
if [[ "$ARCH" == "aarch64" ]]; then
    # If ARM architecture, source miniforge3 conda
    source /root/miniforge3/etc/profile.d/conda.sh
elif [[ "$ARCH" == "x86_64" ]]; then
    # If Intel-based architecture, source miniconda3 conda
    source /root/miniconda3/etc/profile.d/conda.sh
else
    echo "Unsupported architecture: $ARCH"
fi

echo "env sourced"

# Activate the environment
conda activate lm_2.5_dev

# Print today's date
echo "Today's date: $(date)"

# Build the Python command with arguments
python_cmd="python galactose_rdmeode_combined.py -id \"$id\" -t \"$t\" -g \"$g\" -gpus \"$gpus\" -tag \"$tag\" -geo \"$geo\" -mt \"$maxtime\" -geloc \"$fix_gene\""

# Add checkpoint if provided
if [[ -n "$ckpt" && "$ckpt" != "" ]]; then
    python_cmd="$python_cmd -ckpt \"$ckpt\""
fi

# Add feature flags based on numeric arguments from slurm
if [[ "$chromosome" == "1" ]]; then
    python_cmd="$python_cmd --enable-chromosome"
fi

if [[ "$er" == "1" ]]; then
    python_cmd="$python_cmd --enable-er"
fi

if [[ "$eff_ribo" == "1" ]]; then
    python_cmd="$python_cmd --enable-effective-ribosome"
fi

# Print the Python command that will be executed
echo "Executing Python command: $python_cmd"

# Run your Python script with the input parameters
eval $python_cmd

echo "simulation job completed."