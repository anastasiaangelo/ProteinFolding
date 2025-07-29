#!/bin/sh
#BSUB -q scafellpikeSKL
#BSUB -W 36:00
#BSUB -o A%J.%I.o
#BSUB -e A%J.%I.e
#BSUB -n 1
#BSUB -J test[1-768]


# declare source dir

TOTAL_JOBS=768

PYTHONPATH="/lustre/scafellpike/local/HT05604/bxs12/dxm15-bxs12/proteinfolding:${PYTHONPATH}"



# create run directory

HOMEDIR=$(pwd)
RUNDIR=$HOMEDIR/A$LSB_REMOTEJID/run_$LSB_REMOTEINDEX

mkdir -p $RUNDIR


wait $!

# set STDOUT/STDERR

LSB_STDOUT_DIRECT=$RUNDIR/

wait $!

# copy input files

cp $HOMEDIR/simulations_array_job.py $RUNDIR/   # copied from repository on LUSTRE

wait $!

# move to run dir

cd $RUNDIR/

wait $!


# create job_data.dat


# load modules

#module load python3/anaconda
module load python3/anaconda-2022.05
wait $!
source $condaDotFile
wait $!
#conda init bash
wait $!
conda info --envs
wait $!
conda deactivate
wait $!
conda activate protein_q1
wait $!
echo $CONDA_DEFAULT_ENV
wait $!

# Get the git commit hash
COMMIT=$(git rev-parse HEAD)

#python vary_bond_length.py &
# python vary_params_tenpy.py &
wait $!
python simulations_array_job.py $COMMIT $LSB_REMOTEINDEX &
wait $!

