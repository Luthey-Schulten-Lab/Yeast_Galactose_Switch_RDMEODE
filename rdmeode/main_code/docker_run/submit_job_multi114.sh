#!/bin/bash



i=$1
exgal=$2 # mM
time=$3 #mins
gpus=$4 #number of gpus allocated
tag=$5 #tags for output file
geo=$6 #geometry file to use 
fixgene=$7 # whether gene is fixed or not
echo "SLURM GPU Configuration:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Running NVIDIA-Docker"
NV_GPU=$CUDA_VISIBLE_DEVICES nvidia-docker run --name "yeastmt14_"$i --rm -v $(pwd):/workspace lm2.5_dev:1005v2  /workspace/galactose_rdmeode1.14_multi.py -id $i -t $time -g $exgal -gpus $gpus -tag $tag -geo $geo -geloc $fixgene

# Additional check: Run nvidia-smi inside the container
#docker exec -it "twu_yeast_multi_$i" nvidia-smi
