#!bin/bash
#Written by Tianyu Wu, tianyu16@illinois.edu

#echo "Total Submission Count: " $rep_cnt 
# Check if the correct number of parameters is provided
if [ "$#" -ne 8 ]; then
    echo "Error: Missing parameters."
    echo "Usage: $0 <start_idx> <stop_idx> <exgal (mM)> <time (mins)> <gpus> <tags> <geometry> <gene_fixed>"
    echo "Example: $0 1 10 50 30 2 mytags lattice_ribosomes_105000.pkl.xz random."
    echo " anything other than random for gene_fixed would fix the gene location."
    exit 1
fi

start_idx=$1
stop_idx=$2
exgal=$3 #mM
time=$4 #mins
gpus=$5 #number of gpus to use
tags=$6 #tags for final output file
geo=$7
fix=$8
while [ $start_idx -le $stop_idx ]
do	
	sbatch --output ./logfile1.14_mt_noreg_$(date +%Y-%m-%d)_$start_idx.$exgal.$time.log \
	       --job-name="yemn14_"$start_idx \
	       --gres=gpu:$gpus \
	       --mail-type=BEGIN,FAIL,END \

	       submit_job_multi114_noreg.sh $start_idx $exgal $time $gpus $tags $geo $fix
	((start_idx++))
done


squeue
