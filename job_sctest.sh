#!/bin/bash
#PBS -l nodes=2:ppn=8
#PBS -q batch
#PBS -r n
#PBS -m abe
#PBS -l walltime=00:15:00
#PBS -N scalapack_test1

module load gcc/4.4.0 python/2.7.1 intel/intel-v12.0.0.084 openmpi/1.4.3-gcc-v4.4.0-ofed

cd /home/jrs65/scalapack/pyscalapack
export OMP_NUM_THREADS=1
mpirun -np 16 -npernode 8 python pyevtest2.py &> /scratch/jrs65/scalapack/test_2n1t.log