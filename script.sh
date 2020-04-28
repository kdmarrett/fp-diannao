for i in 1 8 16 32 64 128 
do
  echo "batch size $i"; 
  ./convolution 224 224 3 3 64 64 $i 16 | grep speedup;
done;

for i in 1 8 16 32 
do
  echo "tile size $i"; 
  # ./convolution 224 224 3 3 64 64 128 $i | grep speedup;
  /usr/local/cuda-10.1/bin/nvprof ./convolution 224 224 3 3 64 64 128 $i 
done;
