### Using UAB's Cheaha interactive jobs

- Navigate to [UAB Research Computing](https://rc.uab.edu/) and log in. 
- Go to the "My Interactive Sessions" link at the top.
- Select "HPC Desktop", choose settings, click Launch, and then refresh and select "Launch desktop in new tab."

Now you have an interactive desktop on Cheaha with the nodes and cores selected.

### Running MPI code on Cheaha

- Once running a terminal on a Cheaha node, load OpenMPI:
```commandline
module load OpenMPI
```

- Navigate to the MPI code you want to run, and then compile and run:
```commandline
mpicc CODE.c -o CODE
mpirun -np <number of cores to run with> ./CODE
```

### Running CUDA code on Cheaha

- Make sure when requesting an interactive job to select "pascalnodes" as the partition.
- Load CUDA module:
```commandline
module load cuda11.2/toolkit/11.2.2
```
- Compile and run/profile CUDA code:
```commandline
nvcc CODE.cu -o CODE
nvprof ./CODE
```