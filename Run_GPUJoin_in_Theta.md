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
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin
git fetch; git reset --hard origin/main
ssh thetagpusn1
qsub -I -n 1 -t 10 -q single-gpu -A dist_relational_alg --attrs filesystems=home,grand,theta-fs0
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin
nvcc transitive_closure.cu -run -o join -run-args benchmark -run-args 23874 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0 -run-args TG.cedge
nvcc triangle_counting.cu -run -o join -run-args benchmark -run-args 23874 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0 -run-args TG.cedge
nvcc nested_loop_join_dynamic_size.cu -o join -run
nvcc hashtable_gpu.cu -run -o join -run-args data/link.facts_412148.txt -run-args 150000 -run-args 2 -run-args 0.3 -run-args 1 -run-args 30
nvcc hashtable_gpu.cu -run -o join -run-args random -run-args 150000 -run-args 2 -run-args 0.3 -run-args 1 -run-args 30
nsys profile --stats=true ./join
```

- `NCCL` on theta
```shell
module load nccl/nccl-v2.12.12-1_CUDA11.4
module avail
```

### C++ run script example for theta
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
### Submitting a job for rapids:
- Login to theta gpu node:
```shell
ssh USERNAME@theta.alcf.anl.gov
ssh thetagpusn1
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/rapids_implementation/
```
- Create a job script `submit.sh`:
```shell
#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -A dist_relational_alg
module load conda/2022-07-01
conda activate /home/arsho/gpujoinenv
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/rapids_implementation/
/home/arsho/gpujoinenv/bin/python transitive_closure.py
```
- Change the permission
```shell
chmod u+x cudf_submit.sh
chmod u+x pandas_submit.sh
```
- Submit the job:
```shell
qsub -O cudf_submit -e cudf_submit.error cudf_submit.sh
qsub -O pandas_submit -e pandas_submit.error pandas_submit.sh
```
- Show queue:
```shell
qstat -u $USER
```
- Delete a job from queue:
```shell
qdel JOB_ID
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

### Running NCCL code on theta:
- Use interactive job on theta (single node NCCL)
```shell
ssh USERNAME@theta.alcf.anl.gov
ssh thetagpusn1
qsub -I -n 1 -t 10 -q full-node -A dist_relational_alg --attrs filesystems=home,grand,theta-fs0
module load openmpi
module load nccl
nvcc -o nccl_ex nccl_test.cu -L/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.12.12-1_gcc-9.4.0-1ubuntu1-20.04/lib -lnccl
./nccl_ex
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

### Misc commands
- See full gpu name:
```shell
nvidia-smi -L
```
- Remove untracked files:
```shell
git clean -f
```
- Project allocations:
```shell
sbank allocations
```

## Using spack in thetagpu

```commandline
arsho@thetalogin4:~> module load cobalt/cobalt-gpu
qsub -I -n 1 -t 60 -q single-gpu -A dist_relational_alg
arsho@thetagpu06:~$ mkdir -p .spack/linux
git clone https://github.com/spack/spack.git
cd spack/bin/
./spack install cmake@3.23.1
./spack module tcl find cmake@3.23.1
module use ~/spack/share/spack/modules/linux-ubuntu20.04-zen2
arsho@thetagpu06:~/spack/bin$ module avail cmake
-------------------- /home/arsho/spack/share/spack/modules/linux-ubuntu20.04-zen2 --------------------
   cmake-3.23.1-gcc-9.4.0-byiudpe
module load cmake-3.23.1-gcc-9.4.0-byiudpe
arsho@thetagpu06:~/spack/bin$ cmake --version
cmake version 3.23.1
/home/arsho/spack/bin
/lus/theta-fs0/projects/dist_relational_alg/shovon/cu_hashmap

```

### Unsuccessful attempt
- Creating directory for Spack:
```commandline
mkdir -p .spack/linux
```
- Creating configuration:
```
vim ~/.spack/linux/upstreams.yaml
```
`upstreams.yaml`:
```commandline
upstreams:
alcf-spack:
install_tree: /lus/theta-fs0/software/thetagpu/spack/root/opt/spack
modules:
tcl: /lus/theta-fs0/software/spack/share/spack/modules/linux-ubuntu20.04-zen2
```
- Cloning spack from Github:
```
git clone https://github.com/spack/spack.git
cd spack/bin/
```
- Installing a package:
```
./spack install cmake@3.23.1
./spack find cmake
```
- Loading and using a package:
```
spack load cmake@3.23.1
```
- This may give an error:
```commandline
Error: `spack load` requires Spack's shell support.  
  To set up shell support, run the command below for your shell.
```
- Add spack's shell support:
```commandline
. /lus/swift/home/arsho/spack/share/spack/setup-env.sh
```
- Load and use an installed package:
```commandline
spack load cmake@3.23.1
which cmake
/lus/swift/home/arsho/spack/opt/spack/cray-cnl7-haswell/gcc-11.2.0/cmake-3.23.1-l4orfxc3d2vw463cddpguzm25aaiicv2/bin/cmake
```
As it can be seen, "spack" command can be used in the login node.

- Then I started an interactive session on ThetaGPU and it does not know spack command:
```
arsho@thetagpu06:~$ spack --version
Command 'spack' not found, did you mean:
```
- I tried to use ./spack from that folder. It finds the version number but not the installed cmake package:
```
arsho@thetagpu06:~$ cd spack/bin/
arsho@thetagpu06:~/spack/bin$ ./spack --version
0.20.0.dev0 (f66ec00fa9378cff3e97616f97e4bc676a0999ba)
arsho@thetagpu06:~/spack/bin$ ./spack find cmake
==> Error: /home/arsho/.spack/linux/upstreams.yaml:1: Additional properties are not allowed ('install_tree', 'alcf-spack', 'tcl', 'modules' were unexpected)
```
- I tried to install the cmake version again from the interactive session but it returns the same error:
```
arsho@thetagpu06:~/spack/bin$ ./spack install cmake@3.23.1
==> Error: /home/arsho/.spack/linux/upstreams.yaml:1: Additional properties are not allowed ('install_tree', 'tcl', 'modules', 'alcf-spack' were unexpected)
arsho@thetagpu06:~/spack/bin$ cat /home/arsho/.spack/linux/upstreams.yaml
upstreams:
alcf-spack:
install_tree: /lus/theta-fs0/software/thetagpu/spack/root/opt/spack
modules:
tcl: /lus/theta-fs0/software/spack/share/spack/modules/linux-ubuntu20.04-x86_64
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
- [Running jobs at theta](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/running-jobs-thetagpu)
- [Spack documentation](https://spack.readthedocs.io/en/latest/index.html)