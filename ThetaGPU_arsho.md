# How to use Theta GPU
This documentation shows step by step procedure to use Theta GPU with CUDA. It uses Theta username `arsho`. Replace it with your own Theta username. 

## Project Information
- Project Name: dist_relational_alg
- Project Title: Distributed relational algebra at scale
- URL:
- Project Status: Active
- Principal Investigator: Sidharth Kumar (Sidharth Kumar)

## Login
- URL: https://accounts.alcf.anl.gov/home
- Username: arsho
- Password: OTP from Mobilepass+ mobile app


## SSH
- `ssh arsho@theta.alcf.anl.gov`
- Password: OTP from Mobilepass+ mobile app

## Environment
```
arsho@thetalogin6:~> lsb_release -a
Description:	SUSE Linux Enterprise Server 15 SP1
Release:	15.1

arsho@thetalogin6:~> lscpu
Architecture:        x86_64
CPU(s):              32
Model name:          Intel(R) Xeon(R) CPU E5-2698 v3 @ 2.30GHz
```

## Using Theta GPU
- Steps to use Theta GPU:
    - SSH to account
    - Change to project directory 
  ```
  cd /lus/theta-fs0/projects/dist_relational_alg/shovon/
  ```
    - Update code and cmake files
    - Load modules
    - Compile at the login node
    - Submit job using `qsub`
    - Check the folder for `.output` file

### Checking quota and project list in Theta GPU
- Login to Theta account:
```commandline
ssh arsho@theta.alcf.anl.gov
```
- Active projects for the logged-in user from login node:
```commandline
arsho@thetalogin4:~> sbank allocations
 Allocation  Suballocation  Start       End         Resource  Project              Jobs  Charged  Available Balance 
 ----------  -------------  ----------  ----------  --------  -------------------  ----  -------  ----------------- 
 7328        7196           2021-06-24  2022-07-01  theta     dist_relational_alg   210  3,188.5            4,811.5 
 8321        8189           2022-03-15  2022-10-01  thetagpu  dist_relational_alg     0      0.0            1,000.0 

Totals:
  Rows: 2
  Theta:
    Available Balance: 4,811.5 node hours
    Charged          : 3,188.5 node hours
    Jobs             : 210 
  Thetagpu:
    Available Balance: 1,000.0 node hours
    Charged          : 0.0 node hours
    Jobs             : 0 

Effective Monday, August 3, 2020, ALCF compute allocation units have changed from core-hours to node-hours
```
- Check user and project quota:
```commandline
arsho@thetalogin4:~> myquota
Name                           Type     Filesystem        GB_Used       GB_Quota          Grace
===============================================================================================
arsho                          User     mira-home         0.00        100.00             none      
arsho@thetalogin4:~> myprojectquotas

Lustre : Current Project Quota information for projects you're a member of:

Name                           Type     Filesystem          Used          Quota           Grace
===============================================================================================
dist_relational_alg         Project     theta-fs0         623.5G            10T               -

```

### Running CUDA code in Theta GPU

- Login to Theta account:
```commandline
ssh arsho@theta.alcf.anl.gov
```
- Change directory to code directory:
```commandline
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/CUDA_PRACTICE
```
- Check available modules and load CUDA toolkit:
```commandline
module avail
module load cudatoolkit/10.2.89_3.28-7.0.2.0_2.32__g52c0314
module list
```
- Check `nvcc` version:
```commandline
arsho@thetalogin4:/lus/theta-fs0/projects/dist_relational_alg/shovon> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:24:38_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```

- Upload `vector_add.cu` file and `CMakeLists.txt` file.
- Compile the `vector_add.cu` program from login node:

```
nvcc vector_add.cu -o gpu_add
```
- Connect to ThetaGPU head node and change directory to project directory:

```
ssh thetagpusn1
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/CUDA_PRACTICE
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
qsub -A dist_relational_alg -n 1 -t 5 -q single-gpu --attrs mcdram=flat:filesystems=home -O gpu_add_out gpu_add gpu_vector_add

arsho@thetagpusn1:/lus/theta-fs0/projects/dist_relational_alg/shovon/CUDA_PRACTICE$ qsub -A dist_relational_alg -n 1 -t 5 -q single-gpu --attrs mcdram=flat:filesystems=home -O gpu_add_out gpu_add gpu_vector_add
Job routed to queue "single-gpu".
10076391
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
```
arsho@thetagpusn1:/lus/theta-fs0/projects/dist_relational_alg/shovon/CUDA_PRACTICE$ qstat -u $USER
JobID     User   WallTime  Nodes  State   Location  
====================================================
10076391  arsho  00:05:00  1      queued  None      
arsho@thetagpusn1:/lus/theta-fs0/projects/dist_relational_alg/shovon/CUDA_PRACTICE$ qstat -u $USER
JobID     User   WallTime  Nodes  State     Location         
=============================================================
10076391  arsho  00:05:00  1      starting  thetagpu16-gpu1  
arsho@thetagpusn1:/lus/theta-fs0/projects/dist_relational_alg/shovon/CUDA_PRACTICE$ ls -l
total 660
-rw-r--r-- 1 arsho dist_relational_alg    217 May 13 05:31 CMakeLists.txt
-rwxr-xr-x 1 arsho dist_relational_alg 656168 May 13 05:34 gpu_add
-rw-r--r-- 1 arsho dist_relational_alg   2373 May 13 07:02 gpu_add_out.cobaltlog
-rw-r--r-- 1 arsho dist_relational_alg      0 May 13 07:02 gpu_add_out.error
-rw-r--r-- 1 arsho dist_relational_alg    124 May 13 07:02 gpu_add_out.output
-rw-r--r-- 1 arsho dist_relational_alg   2416 May 13 05:34 vector_add.cu
arsho@thetagpusn1:/lus/theta-fs0/projects/dist_relational_alg/shovon/CUDA_PRACTICE$ cat gpu_add_out.output 
Vector sum: 9799999.999995
Result vector sum: 9799999.999995
Passed the test
Grid size: 62501 , Block size: 16 , N: 1000000
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
### References
- [Short CUDA tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)
- [nVidia CUDA C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Theta GPU nodes](https://www.alcf.anl.gov/support-center/theta-gpu-nodes)
- [Getting started video](https://www.alcf.anl.gov/support-center/theta-and-thetagpu/submit-job-theta)
- [Getting Started on ThetaGPU](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/getting-started-thetagpu)
- [Submit a Job on ThetaGPU](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/submit-job-thetagpu)
- [Running jobs at Theta](https://www.alcf.anl.gov/support-center/theta/running-jobs-and-submission-scripts)