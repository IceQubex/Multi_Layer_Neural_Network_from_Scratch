from numba import jit, cuda
import numpy as np
import math
# to measure exec time
from timeit import default_timer as timer


# normal function to run on cpu
def func(a):
    for i in range(10000000):
        a[i] += 1


@cuda.jit
def func2(a):
    pos = cuda.grid(1)
    a[pos] += 1
    # for i in range(10000000):
    #     a[i] += 1

if __name__ == "__main__":
    n = 100000000
    a = np.ones(n, dtype=np.float64)
    # b = np.ones(n, dtype=np.float32)
    threadsperblock = 1000

    blockspergrid = math.ceil(a.shape[0]/threadsperblock)


    start = timer()
    func(a)
    print("without GPU:", timer() - start)

    start = timer()
    func2[blockspergrid,threadsperblock](a)
    print("with GPU:", timer() - start)
