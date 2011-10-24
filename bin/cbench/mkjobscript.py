
import sys
import os
import os.path

gsize = int(sys.argv[1])

bsize = int(sys.argv[2])

pside = int(sys.argv[3])

nthread = int(sys.argv[4])

stem = sys.argv[5]

nodes = pside**2
ppn = 8
name = "%s_%in_%ib_%ip_%it" % (stem, gsize, bsize, pside, nthread)
nomp = nthread
numproc = pside**2
pernode = 1


script="""
#!/bin/bash
#PBS -l nodes=%(nodes)i:ib:ppn=%(ppn)i
#PBS -q batch
#PBS -r n
#PBS -m abe
#PBS -l walltime=00:30:00
#PBS -N pdgemm_%(name)s

module load gcc/4.4.0 python/2.7.1 intel/intel-v12.0.0.084 intelmpi

cd $HOME/code/PyScalapack/bin/cbench/

export OMP_NUM_THREADS=%(nomp)i
mpirun --mca btl self,sm,openib -np %(numproc)i -npernode %(pernode)i ./pdgemm_bench %(gsize)i %(pside)i %(bsize)i %(nomp)i $SCRATCH/cbench/jt_%(name)s.dat &> $SCRATCH/cbench/jo_%(name)s.log
#mpirun -ppn %(pernode)i -genv I_MPI_FABRICS shm:dapl -np %(numproc)i ./pdgemm_bench %(gsize)i %(pside)i %(bsize)i %(nomp)i $SCRATCH/cbench/jt_%(name)s.dat &> $SCRATCH/cbench/jo_%(name)s.log
"""

script = script % { 'nodes':nodes, 'ppn':ppn, 'name':name, 'nomp':nomp,
                    'numproc': numproc, 'pernode':pernode, 'gsize':gsize,
                    'bsize': bsize, 'pside':pside}

scriptname = "jobscript_%s.sh" % name

with open(scriptname, 'w') as f:
    f.write(script)

os.system('cd ~; pwd; qsub %s' % os.path.abspath(scriptname))
