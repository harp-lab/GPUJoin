## Run instructions
```shell
nvcc vec_add.cu -o vec_add.o
time ./vec_add.o 
SM: 108
Grid X Thread: 2097152 X 512
Max error: 0

real	0m15.332s
user	0m12.550s
sys	0m2.729s



nvcc vec_add.cu -o vec_add.o
time ./vec_add.o 
SM: 108
Grid X Thread: 19419 X 55296
Max error: 1

real	0m12.930s
user	0m11.158s
sys	0m1.748s

```