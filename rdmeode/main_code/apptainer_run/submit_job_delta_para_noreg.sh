#!/bin/bash

# Accepting inputs from command line arguments
id=$1
t=$2
g=$3
gpus=$4
tag=$5
geo=$6
maxtime=$7
ckpt=$8
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
#source /root/miniconda3/etc/profile.d/conda.sh

echo "env sourced"

# Check Conda version
#conda --version

# Activate the environment
conda activate lm_2.5_dev

# Print today's date
echo "Today's date: $(date)"

# Print the Python command that will be executed
echo "Executing Python command: python /workspace/galactose_rdmeode1.14_multi_noreg.py -id \"$id\" -t \"$t\" -g \"$g\" -gpus \"$gpus\" -tag \"$tag\" -geo \"$geo\" -mt \"$maxtime\" -ckpt \"$ckpt\" "

# Run your Python script with the input parameters
python /workspace/galactose_rdmeode1.14_multi_noreg.py   -id "$id" -t "$t" -g "$g" -gpus "$gpus" -tag "$tag" -geo "$geo" -mt "$maxtime" -ckpt "$ckpt"

echo "simulation job completed."

