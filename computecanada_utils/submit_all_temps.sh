#!/bin/bash


temps=('6' '65' '7' '8' '9')

for T in ${temps[@]}; do
	jobname="tdot$T"
	sbatch --job-name="$jobname" --output="${jobname}/slurm-%t.out" --error="${jobname}/slurm-%t.err"  submit_template.sh $T
done
