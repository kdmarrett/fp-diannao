for i in 1 2 3 4 8 12 14
do
  echo "batch size $i"; 
  for j in 1 2 4 8 16 32 
  do
    echo "tile size $j"; 
    ./convolution 224 224 3 3 64 64 $i $j | egrep GFlops;
    # /usr/local/cuda-10.1/bin/nvprof ./convolution 224 224 3 3 64 64 1 $i 
  done;
  echo;
done;
