
import sys
import os
import os.path

gsize = int(sys.argv[1])

bsize = int(sys.argv[2])

pside = int(sys.argv[3])

nthread = int(sys.argv[4])

nodes = pside**2
ppn = 8
name = "T2_%in_%ib_%ip_%it" % (gsize, bsize, pside, nthread)
nomp = nthread
numproc = pside**2
pernode = 1


script="""
#!/bin/bash
#PBS -l nodes=%(nodes)i:qdr:ppn=%(ppn)i
#PBS -q batch
#PBS -r n
#PBS -m abe
#PBS -l walltime=04:00:00
#PBS -N evtest_%(name)s

#module load gcc/4.4.0 python/2.7.1 intel/intel-v12.0.0.084 intelmpi

cd /home/p/pen/jrs65/code/PyScalapack/bin
export OMP_NUM_THREADS=%(nomp)i
#mpirun -np %(numproc)i -npernode %(pernode)i python benchmark.py %(gsize)i %(bsize)i %(pside)i %(pside)i %(nomp)i /scratch/jrs65/scalapack/jobtime_%(name)s.dat &> /scratch/jrs65/scalapack/jobout_%(name)s.log
mpirun --mca btl self,sm,openib -np %(numproc)i -npernode %(pernode)i python benchmark.py %(gsize)i %(bsize)i %(pside)i %(pside)i %(nomp)i /scratch/p/pen/jrs65/scalapack2/jobtime_%(name)s.dat &> /scratch/p/pen/jrs65/scalapack2/jobout_%(name)s.log
#mpirun -ppn %(pernode)i -genv I_MPI_FABRICS shm:dapl -np %(numproc)i python benchmark.py %(gsize)i %(bsize)i %(pside)i %(pside)i %(nomp)i /scratch/jrs65/scalapack/jobtime_%(name)s.dat &> /scratch/jrs65/scalapack/jobout_%(name)s.log
"""

script = script % { 'nodes':nodes, 'ppn':ppn, 'name':name, 'nomp':nomp,
                    'numproc': numproc, 'pernode':pernode, 'gsize':gsize,
                    'bsize': bsize, 'pside':pside}

scriptname = "jobscript_%s.sh" % name

with open(scriptname, 'w') as f:
    f.write(script)

os.system('cd ~; pwd; qsub %s' % os.path.abspath(scriptname))
