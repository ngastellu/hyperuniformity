#!/bin/bash

#SBATCH --account=ctb-simine
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=0-00:10
#SBATCH --array=0-131
#SBATCH --output=slurm-%a.out
#SBATCH --error=slurm-%a.err

module load mpi4py

nn=$SLURM_ARRAY_TASK_ID

if [[ ! -d job-${nn} ]]; then
	mkdir job-${nn}
fi

cp dfs_mpi_realspace.py job-${nn}
cd job-${nn}

srun python dfs_mpi_realspace.py ${nn} 'tempdot6' 20
