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
vim 
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
chmod u+x submit.sh
```
- Submit the job:
```shell
qsub -q single-gpu -o transitive_closure.output -e transitive_closure.error submit.sh
qsub -q single-gpu -o p.output -e p.error submit.sh
```
- Show queue:
```shell
qstat -u $USER
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
- [Running jobs at theta](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/running-jobs-thetagpu)