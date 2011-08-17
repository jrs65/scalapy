
import numpy as np

import scipy.linalg as la

n = 1000
h = n/2
a = np.empty((n,n))


def f1(x):
    return np.exp(-x**2 / h**2)
    #return np.where(x < 3, x, np.zeros_like(x))
    #return np.abs(x)

def f2(x):
    #return np.exp(-x**2 / h**2)
    return np.where(x < 10, np.ones_like(x), np.zeros_like(x))
    #return np.abs(x)
    

x = np.arange(n, dtype=np.float64)

a = np.abs(x[:,np.newaxis] - x[np.newaxis,:])
na = n - a
a = np.where(a < na, a, na)

#fa = f(a)

fq1 = np.fft.fftn(f1(a))
fq1[1:,:] = fq1[-1:0:-1,:].copy()


fa2 = f2(a)
fq2 = np.fft.fftn(fa2)
fq2[1:,:] = fq2[-1:0:-1,:].copy()


fd2 = np.sort(fq2.diagonal()) / n

fe2 = la.eigvalsh(fa2)

px = np.where(x < n-x, x, n-x)

ff2 = np.sort(np.fft.fft(f2(px)).real)

print "2d FFT max: ", np.abs((fd2 - fe2) / fe2).max()
print "1d FFT max: ", np.abs((ff2 - fe2) / fe2).max()
