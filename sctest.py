
import numpy as np

import scarray


v1 = np.arange(10000, dtype=np.float64)

v2 = scarray.vector_pagealign(v1, 1000)

v3 = scarray.vector_from_pagealign(v2, 10000, 1000)


m1 = np.arange(10000*1000, dtype=np.float64).reshape((10000,1000))

m2 = scarray.matrix_pagealign(m1, [1000, 100])

m3 = scarray.matrix_from_pagealign(m2, [10000, 1000], [1000, 100])


if (v1 == v3).all():
    print "Vectors correct."
else:
    print "Eeek."

if (m1 == m3).all():
    print "Matrices correct."
else:
    print "Eeek."
