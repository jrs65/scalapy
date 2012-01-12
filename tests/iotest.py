
import unittest
import glob
import os

import numpy as np
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#nproc = comm.Get_size()
#rank = comm.Get_rank()

# If not using the mpi bits yet.
#nproc = 1
#rank = 0
#class Comm(object):

#    def barrier(self):
#        pass
#comm = Comm()

#import pyscalapack as pysc
#from pyscalapack import npyutils  
import npyutils as npyutils

class TestIO(unittest.TestCase):

    def setUp(self):
        n = 2500
        self.n = n
        # Make a matrix that to use as test data.
        # Non semetric and easy to check: A[i, j] = i + 6j + 5
        self.mat = (np.arange(n, dtype=np.float64)[:,None]
                    + 6 * np.arange(n, dtype=np.float64) + 5)
        self.mat.shape = (n, n)

    def test_read_header(self):
        # Write one copy to disk.
        np.save("tmp_test_origional.npy", self.mat)

        shape, fortran_order, dtype, offset = npyutils.read_header_data(
                "tmp_test_origional.npy")
        self.assertEqual(shape, (self.n, self.n))
        self.assertFalse(fortran_order)
        self.assertEqual(dtype, '<f8')
        self.assertEqual(offset % 16, 0)

    def test_write_header(self):
        fname = "tmp_test_new_hdr.npy"
        
        # Find out how much space is needed for the header.
        header_data = npyutils.pack_header_data((self.n, self.n), False,
                                                float)
        header_len =  npyutils.get_header_length(header_data)
        self.assertEqual(header_len % 4096, 0)
       
        # Make an empty file, big enough to hold the header only.
        fp = open(fname, 'w')
        fp.seek(40000 - 1)
        fp.write("\0")
        fp.close()
        
        # Write it and read it and make sure the data is right.
        npyutils.write_header_data(fname, header_data)
        shape, fortran_order, dtype, offset = npyutils.read_header_data(fname)
        self.assertEqual(shape, (self.n, self.n))
        self.assertFalse(fortran_order)
        self.assertEqual(dtype, '<f8')
        self.assertEqual(offset, header_len)

        # Make sure the file is the same size as before.
        fp = open(fname, 'r')
        fp.seek(0, 2)
        self.assertEqual(fp.tell(), 40000)



    def tearDown(self):
        files = glob.glob("tmp_test_*")
        for f in files:
            os.remove(f)


if __name__ == '__main__' :
    unittest.main()

