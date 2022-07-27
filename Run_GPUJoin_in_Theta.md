### Running CUDA code in Theta GPU

- Login to Theta account:
```commandline
ssh USERNAME@theta.alcf.anl.gov
```
- Change directory to code directory:
```commandline
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin
```
- Check available modules and load CUDA toolkit:
```commandline
module avail
module load cudatoolkit/10.2.89_3.28-7.0.2.0_2.32__g52c0314
module list
```
- Check `nvcc` version:
```commandline
nvcc --version
```
- Sync from Git or upload `.cu` file:
```commandline
git fetch
git reset --hard origin/main
```
- Compile the `natural_join.cu` program from login node:
```
nvcc natural_join.cu -o gpu_join
```
- Connect to ThetaGPU head node and change directory to project directory:
```
ssh thetagpusn1
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin
# exit # return to login nodes
```
- Example command to submit a job
```commandline
qsub -A YourProject -n 256  -t 30  \
--env MYVAR=value1 -i inputdata -O Project1_out \
–attrs mcdram=flat:numa=quad program.exe progarg1
```
- Create a job to `single-gpu` queue with single node:
```commandline
qsub -A dist_relational_alg -n 1 -t 5 -q single-gpu \
--attrs mcdram=flat:filesystems=home -O gpu_join_out gpu_join natural_join

qsub -A dist_relational_alg -n 1 -t 5 -q single-gpu \
--attrs mcdram=flat:filesystems=home -O nested_loop_out join nested_unified

Job routed to queue "single-gpu".
```
- Show all jobs from user:
```commandline
qstat -u $USER
```
- Delete a queued job:
```commandline
# qdel <JOB_ID>
qdel 10076390
```
- Check output:
```commandline
cat gpu_join_out.output 
cat nested_loop_out.output
```

- Interactive job on theta
```shell
ssh USERNAME@theta.alcf.anl.gov
ssh thetagpusn1
qsub -I -n 1 -t 10 -q single-gpu -A dist_relational_alg
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin
nvcc nested_loop_join_dynamic_size.cu -o join -run
nsys profile --stats=true ./join
```
- Running hashgraph on theta
```shell
ssh USERNAME@theta.alcf.anl.gov
ssh thetagpusn1
qsub -I -n 1 -t 10 -q single-gpu -A dist_relational_alg
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph
git reset --hard origin/master
rm -rf build
mkdir build
cd build
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/nvvm/lib64:/usr/local/lib32:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
rm -rf ../build/*; cmake ..; make -j

arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/build$ nvidia-smi
Tue Jul 26 13:33:22 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.129.06   Driver Version: 470.129.06   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:BD:00.0 Off |                    0 |
| N/A   26C    P0    51W / 400W |      0MiB / 40536MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/build$ 

arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/build$ rm -rf ../build/*; cmake ..; make -j
-- The CXX compiler identification is GNU 9.4.0
-- The CUDA compiler identification is NVIDIA 11.4.152
-- Check for working CXX compiler: /lus/theta-fs0/software/thetagpu/openmpi-4.0.5/bin/mpicxx
-- Check for working CXX compiler: /lus/theta-fs0/software/thetagpu/openmpi-4.0.5/bin/mpicxx -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc -- works
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Found OpenMP_CXX: -fopenmp (found version "4.5") 
-- Found OpenMP: TRUE (found version "4.5")  
-- Using Nvidia Tools Extension
-- Building with optimization flags
-- Configuring done
-- Generating done
-- Build files have been written to: /lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/build
Scanning dependencies of target alg
[ 28%] Building CUDA object CMakeFiles/alg.dir/src/MultiHashGraph.cu.o
[ 28%] Building CUDA object CMakeFiles/alg.dir/src/SingleHashGraph.cu.o
In file included from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/system/cuda/detail/execution_policy.h:33,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/iterator/detail/device_system_tag.h:23,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/iterator/detail/iterator_facade_category.h:22,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/iterator/iterator_facade.h:37,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/iterator/iterator_adaptor.h:36,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/iterator/detail/normal_iterator.h:25,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/detail/vector_base.h:25,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/device_vector.h:26,
                 from /lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/include/MultiHashGraph.cuh:33,
                 from /lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/src/MultiHashGraph.cu:16:
/usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/system/cuda/config.h:78:2: error: #error The version of CUB in your include path is not compatible with this release of Thrust. CUB is now included in the CUDA Toolkit, so you no longer need to use your own checkout of CUB. Define THRUST_IGNORE_CUB_VERSION_CHECK to ignore this.
   78 | #error The version of CUB in your include path is not compatible with this release of Thrust. CUB is now included in the CUDA Toolkit, so you no longer need to use your own checkout of CUB. Define THRUST_IGNORE_CUB_VERSION_CHECK to ignore this.
      |  ^~~~~
In file included from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/system/cuda/detail/execution_policy.h:33,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/iterator/detail/device_system_tag.h:23,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/iterator/detail/iterator_facade_category.h:22,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/iterator/iterator_facade.h:37,
                 from /lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/externals/cub-1.8.0/cub/device/../iterator/arg_index_input_iterator.cuh:48,
                 from /lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/externals/cub-1.8.0/cub/device/device_reduce.cuh:41,
                 from /lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/externals/cub-1.8.0/cub/cub.cuh:53,
                 from /lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/include/SingleHashGraph.cuh:38,
                 from /lus/theta-fs0/projects/dist_relational_alg/shovon/hashgraph/src/SingleHashGraph.cu:16:
/usr/local/cuda/bin/../targets/x86_64-linux/include/thrust/system/cuda/config.h:78:2: error: #error The version of CUB in your include path is not compatible with this release of Thrust. CUB is now included in the CUDA Toolkit, so you no longer need to use your own checkout of CUB. Define THRUST_IGNORE_CUB_VERSION_CHECK to ignore this.
   78 | #error The version of CUB in your include path is not compatible with this release of Thrust. CUB is now included in the CUDA Toolkit, so you no longer need to use your own checkout of CUB. Define THRUST_IGNORE_CUB_VERSION_CHECK to ignore this.
      |  ^~~~~
make[2]: *** [CMakeFiles/alg.dir/build.make:63: CMakeFiles/alg.dir/src/MultiHashGraph.cu.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[2]: *** [CMakeFiles/alg.dir/build.make:76: CMakeFiles/alg.dir/src/SingleHashGraph.cu.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:134: CMakeFiles/alg.dir/all] Error 2
make: *** [Makefile:84: all] Error 2
```

### (Bonus) C++ run script example for theta
```shell
#!/bin/bash
module load intel

aprun -n 64 -N 64 ./ata &
sleep 1

aprun -n 128 -N 64 ./ata &
sleep 1

aprun -n 256 -N 64 ./ata &
wait

aprun -n 512 -N 64 ./ata
```

### Running rapids in theta
- Login to theta
```shell
ssh USERNAME@theta.alcf.anl.gov
ssh thetagpusn1
qsub -I -n 1 -t 10 -q single-gpu -A dist_relational_alg
module load conda/2022-07-01
conda create -p $HOME/gpujoinenv
conda activate /home/arsho/gpujoinenv
conda install -c rapidsai -c nvidia -c numba -c conda-forge     cudf=22.06 python=3.9 cudatoolkit=11.2
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/rapids_implementation/
mkdir output
python data_merge.py
```
- Running hashgraph (python) on theta
```shell
ssh USERNAME@theta.alcf.anl.gov
ssh thetagpusn1
qsub -I -n 1 -t 10 -q single-gpu -A dist_relational_alg
exit
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/rapids_implementation/
module load conda/2022-07-01
conda activate /home/arsho/gpujoinenv
python hg.py 
```
```
Time taken = 2.805170e+00 seconds
Rate = 9.56931e+07 keys/sec

Time taken = 1.044507e-01 seconds
Rate = 2.56997e+09 keys/sec

Time taken = 8.341455e-02 seconds
Rate = 3.21809e+09 keys/sec

Time taken = 8.331728e-02 seconds
Rate = 3.22185e+09 keys/sec
```


#### Basic conda commands:
- List the environments:
```shell
conda env list
```
- Activate an environment:
```shell
conda activate /home/arsho/gpujoinenv
```
- List installed packages in the current environment:
```shell
conda list
```

- Loaded `miniconda` and tried to create an environment
```
module load miniconda-3
conda create -p /envs/gpujoin_env_2 --clone $CONDA_PREFIX
```
Got the following error:
```shell
(miniconda-3/latest/base) arsho@thetalogin4:~> conda create -p /envs/gpujoin_env --clone $CONDA_PREFIX
Source:      /soft/datascience/conda/miniconda3/latest
Destination: /envs/gpujoin_env
The following packages cannot be cloned out of the root environment:
 - defaults/linux-64::conda-4.8.3-py37_0
Packages: 175
Files: 117276
```

### Misc commands
- See full gpu name:
```shell
nvidia-smi -L
```

### References
- [Short CUDA tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)
- [nVidia CUDA C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Theta GPU nodes](https://www.alcf.anl.gov/support-center/theta-gpu-nodes)
- [Getting started video](https://www.alcf.anl.gov/support-center/theta-and-thetagpu/submit-job-theta)
- [Getting Started on ThetaGPU](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/getting-started-thetagpu)
- [Submit a Job on ThetaGPU](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/submit-job-thetagpu)
- [Running jobs at Theta](https://www.alcf.anl.gov/support-center/theta/running-jobs-and-submission-scripts)
- [Interactive job on theta](https://www.alcf.anl.gov/support-center/theta/running-jobs-and-submission-scripts)