import sys
import json
import glob
import os
import subprocess
from datetime import datetime

# import the consolidated parameter 
# space across codebase
from param_space import *

# w,h,c,n (elem / batch),k,f_w,f_h,pad_w,pad_h,stride_w,stride_h,precision,fwd_time (usec),fwd_algo,Ops (mill),TOPS,Max TFLOPS,Roof Model:,Model TOPS,Mem Read (GB),Mem Time (usec),Mem TOPS,,Model Error,Abs Error

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
