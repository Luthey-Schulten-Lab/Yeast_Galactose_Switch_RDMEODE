#!/bin/bash

# Optional Python script name (default: galactose_rdmeode_combined.py)
py_script=${1:-"galactose_rdmeode_combined.py"}

# Shift so remaining arguments keep their original order
shift
# Accepting inputs from command line arguments
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
rna_tracker=${11:-0}
er_num=${12:-4}
ckpt=${13}
maxtime=${14}
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
# Activate the environment
conda activate lm_2.5_dev

# Print today's date
echo "Today's date: $(date)"

# Build the Python command with arguments
python_cmd="python $py_script -id \"$id\" -t \"$t\" -g \"$g\" -gpus \"$gpus\" -tag \"$tag\" -geo \"$geo\" -mt \"$maxtime\" -geloc \"$fix_gene\""

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

if [[ "$rna_tracker" == "1" ]]; then
    python_cmd="$python_cmd --enable_rna_tracking"
fi

if [[ "$er_num" != "4" ]]; then
    python_cmd="$python_cmd -ernum $er_num"
fi


eval $python_cmd

echo "simulation job completed."

