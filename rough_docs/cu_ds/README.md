## Using cuCollections in Interactive node in ThetaGPU

### Install cmake using Spack (once only)
- From login node load cobalt and start and interactive session:
```commandline
arsho@thetalogin4:~> module load cobalt/cobalt-gpu
qsub -I -n 1 -t 20 -q single-gpu -A dist_relational_alg
```
- Create directory for spack, clone spack and install cmake:
```commandline
mkdir -p .spack/linux
git clone https://github.com/spack/spack.git
cd spack/bin/
./spack install cmake@3.23.1
# this will take several minutes to install cmake
```
### Use cuCollections and run sample_map.cu
- From login node load cobalt and start and interactive session: 
```commandline
arsho@thetalogin4:~> module load cobalt/cobalt-gpu
qsub -I -n 1 -t 20 -q single-gpu -A dist_relational_alg
```
- Go to spack folder and load cmake
```commandline
cd spack/bin/
./spack module tcl find cmake@3.23.1
module use ~/spack/share/spack/modules/linux-ubuntu20.04-zen2
module load cmake-3.23.1-gcc-9.4.0-byiudpe
cmake --version
# cmake version 3.23.1
```
- Run the `sample_map.cu` file:
```commandline
cd cu_ds
mkdir build
cd build
cmake ..
make
./main
```