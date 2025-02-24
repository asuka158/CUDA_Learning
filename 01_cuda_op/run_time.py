# run_time.py
import time   
import numpy as np
import torch
import add2

# c = a + b (shape: [n * n])
n = 1024
a = torch.rand((n,n), device="cuda:0")
b = torch.rand((n,n), device="cuda:0")
cuda_c = torch.rand((n,n), device="cuda:0")

ntest = 100

def show_time(func):
    times = list()
    res = None
    for _ in range(10):
        res = func()
    for _ in range(ntest):
        
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()   
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
    return times, res

def run_cuda():
    add2.torch_launch_add2(cuda_c, a, b, n)
    return cuda_c

def run_torch():
    c = a + b
    return c.contiguous() 

if __name__ == "__main__":
        
    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time, torch_res = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    torch.allclose(cuda_res, torch_res)
    print("Kernel test passed.")

    add2.fun()