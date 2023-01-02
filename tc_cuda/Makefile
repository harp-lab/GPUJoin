TARGET = tc_cuda
TARGET_EXP = tc_cuda_exp
COMMON = common
NVCC_FLAGS += -dlto -lineinfo
LDFLAGS = -lmpi -lnccl
COMPILER_FLAGS = -O3 -w
DEBUG_FLAGS = -DDEBUG
THETAFLAGS = -L/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.12.12-1_gcc-9.4.0-1ubuntu1-20.04/lib -L/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0/lib
SRC = $(TARGET).cu
SRC_EXP = $(TARGET_EXP).cu

build:
	nvcc $(SRC) -o $(TARGET).out $(COMPILER_FLAGS)

test:
	./$(TARGET).out

run: build test

exp:
	nvcc $(SRC_EXP) -o $(TARGET_EXP).out $(COMPILER_FLAGS)
	./$(TARGET_EXP).out

debug:
	nvcc $(SRC) -o $(TARGET).out $(COMPILER_FLAGS) $(DEBUG_FLAGS)
	./$(TARGET).out

clean:
	rm -f $(TARGET).out

.PHONY.: build