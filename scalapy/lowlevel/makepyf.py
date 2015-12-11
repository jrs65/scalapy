#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import
import os
import shutil
import glob
import subprocess

from termcolor import colored

from . import scalapack2pyf as s2p

# Script for generating pyf file.

SC2PYF = "python scalapack2pyf.py"
F2PY = "f2py2.7"

PBLAS_SRCDIR = "scsource/PBLAS/SRC"
PBLAS_OUTDIR = "tmp.pblas"

SCL_SRCDIR = "scsource/SRC"
SCL_OUTDIR = "tmp.scalapack"

# Read in blacklist of routines to ignore
blacklist = []
with open('blacklist.txt') as f:
    blacklist = f.read().split('\n')

# Clear out existing directory
shutil.rmtree(PBLAS_OUTDIR)
os.makedirs(PBLAS_OUTDIR)


# Files in PBLAS to process
pblas_files = glob.glob(PBLAS_SRCDIR + '/*_.c')

print("Scanning PBLAS - processing %i files" % len(pblas_files))

# Iterate over files in PBLAS and create a pyf signature file.
for pbfile in pblas_files:
    basefile = os.path.splitext(os.path.basename(pbfile))[0]
    #echo "Processing $basefile..."
    outfile = PBLAS_OUTDIR + '/' + basefile + '.pyf'
    try:
        s2p.scalapack2pyf(pbfile, outfile)
    except s2p.ParseException as p:
        print(colored('FAILURE', 'red') + ': processing %s' % basefile)

# Construct list of signature files that are not blacklisted.
pblas_sigfiles = []
for sigfile in glob.glob(PBLAS_OUTDIR + '/*.pyf'):
    fname = os.path.splitext(os.path.basename(sigfile))[0]
    if fname in blacklist:
        print("Not processing blacklisted: %s" % fname)
        continue

    pblas_sigfiles.append(sigfile)

# Run f2py to create a master signature file for PBLAS
try:
    output = subprocess.check_output([F2PY, '-h', 'pblas.pyf'] + pblas_sigfiles + ['-m', 'pblas', '--overwrite-signature'], stderr=subprocess.STDOUT)
except Exception as e:
    print(colored('ERROR'))
    print(e.output)



# Clear out existing directory
shutil.rmtree(SCL_OUTDIR)
os.makedirs(SCL_OUTDIR)

# Files in Scalapack to process
scl_files = glob.glob(SCL_SRCDIR + '/*.f')

print("Scanning Scalapack - processing %i files" % len(scl_files))

# Iterate over files in Scalapack and create a pyf signature file.
for pbfile in scl_files:
    basefile = os.path.splitext(os.path.basename(pbfile))[0]
    #echo "Processing $basefile..."
    outfile = SCL_OUTDIR + '/' + basefile + '.pyf'
    try:
        s2p.scalapack2pyf(pbfile, outfile)
    except s2p.ParseException as p:
        print(colored('FAILURE', 'red') + ': processing %s' % basefile)

# Construct list of signature files that are not blacklisted.
scl_sigfiles = []
for sigfile in glob.glob(SCL_OUTDIR + '/*.pyf'):
    fname = os.path.splitext(os.path.basename(sigfile))[0]
    if fname in blacklist:
        print("Not processing blacklisted: %s" % fname)
        continue

    scl_sigfiles.append(sigfile)

# Run f2py to create a master signature file for Scalapack
try:
    output = subprocess.check_output([F2PY, '-h', 'scalapack.pyf'] + scl_sigfiles + ['-m', 'scalapack', '--overwrite-signature'], stderr=subprocess.STDOUT)
except Exception as e:
    print(colored('ERROR'))
    print(e.output)
