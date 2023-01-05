TARGET = tc_cuda
COMMON = common
NVCC_FLAGS += -dlto -lineinfo
LDFLAGS = -lmpi -lnccl
COMPILER_FLAGS = -O3 -w
DEBUG_FLAGS = -DDEBUG
THETA_FLAGS = -L/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0/lib
SRC = $(TARGET).cu

build:
	nvcc $(SRC) -o $(TARGET).out $(COMPILER_FLAGS) $(THETA_FLAGS)

test:
	./$(TARGET).out

run: build test

debug:
	nvcc $(SRC) -o $(TARGET).out $(COMPILER_FLAGS) $(DEBUG_FLAGS) $(THETA_FLAGS)
	./$(TARGET).out

clean:
	rm -f $(TARGET).out

.PHONY.: build