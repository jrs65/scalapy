from __future__ import print_function, division, absolute_import

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


script = """
#!/bin/bash
#PBS -l nodes=%(nodes)i:ppn=%(ppn)i
#PBS -q batch
#PBS -r n
#PBS -m abe
#PBS -l walltime=04:00:00
#PBS -N evtest_%(name)s

source /home/k/krs/jrs65/setupscalapy.sh

cd /home/k/krs/jrs65/code/scalapy/bin
export OMP_NUM_THREADS=%(nomp)i
mpirun --mca io ompio -np %(numproc)i -npernode %(pernode)i python benchmark.py %(gsize)i %(bsize)i %(pside)i %(pside)i %(nomp)i /scratch/k/krs/jrs65/scbench/jobtime_%(name)s.dat &> /scratch/k/krs/jrs65/scbench/jobout_%(name)s.log
"""

script = script % { 'nodes': nodes, 'ppn': ppn, 'name': name, 'nomp': nomp,
                    'numproc': numproc, 'pernode': pernode, 'gsize': gsize,
                    'bsize': bsize, 'pside': pside}

scriptname = "jobscript_%s.sh" % name

with open(scriptname, 'w') as f:
    f.write(script)

os.system('cd ~; pwd; qsub %s' % os.path.abspath(scriptname))
