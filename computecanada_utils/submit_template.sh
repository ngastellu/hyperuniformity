#!/bin/bash

#SBATCH --account=ctb-simine
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-00:30

module load mpi4py

pyscript='ising_nfs_mpi.py'
temp=$1


echo -e "~~~ Starting run at: $(date) ~~~\n\n"
srun python "$pyscript" "$temp"
echo -e "\n\n ~~~ Ending run at: $(date) ~~~"
