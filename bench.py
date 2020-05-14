import sys
import json
import glob
import os
import subprocess
from datetime import datetime

# w,h,c,n (elem / batch),k,f_w,f_h,pad_w,pad_h,stride_w,stride_h,precision,fwd_time (usec),fwd_algo,Ops (mill),TOPS,Max TFLOPS,Roof Model:,Model TOPS,Mem Read (GB),Mem Time (usec),Mem TOPS,,Model Error,Abs Error

# int Nx = atoi(argv[1]);
# int Ny = atoi(argv[2]);
# int Kx = atoi(argv[3]);
# int Ky = atoi(argv[4]);
# int Ni = atoi(argv[5]);
# int Nn = atoi(argv[6]);
# int B = atoi(argv[7]);
# int Tn = atoi(argv[8]);

Nx = [ 224 ]
Ny = [ 224 ]
Kx = [ 3 ]
Ky = [ 3 ]
Ni = [ 64 ]
Nn = [ 64 ]
B = [ 1, 2, 3, 4, 8, 12, 14]
T = [ 1, 2, 4, 8, 16, 32]

params = ''
for nx in Nx:
  for ny in Ny:
    for kx in Kx:
      for ky in Ky:
        for ni in Ni:
          for nn in Nn:
            for b in B:
              for t in T:
                # /usr/local/cuda-10.1/bin/nvprof ./convolution 224 224 3 3 64 64 1 $i
                # | egrep GFlops;
                # to run in parallel uses ' & '
                params += './convolution %s %s %s %s %s %s %s %s & ' % (nx, ny, kx, ky, ni, nn, b, t)
                print(params)

# make sure you do not leave a final & sign on a command
# causes difficult behavior
os.system(params[:-2])
print("Run success...")
