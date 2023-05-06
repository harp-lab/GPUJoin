#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// String hashing function
static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

// Use unistd to access system host name, then change period
// to null to make string
static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


int main(int argc, char* argv[])
{
  // Test size, could be anything
  int size = 32*1024*1024;

  // CPU rank, total CPU ranks, local rank is for keeping track when 
  // overprescribing threads per cpu
  int myRank, nRanks, localRank = 0;


  // Initialize MPI variables
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


  // Find hash for each CPU host and use allgather to send array of all hashes to each thread
  // If multiple threads on one CPU device, use localRank to keep track
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  /* int MPI_Allgather(const void *sendbuf, int  sendcount,
        MPI_Datatype sendtype, void *recvbuf, int recvcount,
        MPI_Datatype recvtype, MPI_Comm comm) */
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }


  // Use two GPUs for each MPI CPU thread
  int nDev = 2;

  // Create buffers and CUDA streams for each GPU
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  // Creating and filling buffers and streams on GPUs, have to use localrank
  // if a CPU has multiple MPI threads
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(localRank*nDev + i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  // Create NCCL communicator for the GPUs on this CPU
  ncclUniqueId id;
  ncclComm_t comms[nDev];

  // Generate NCCL unique ID at root process and broadcast it to all
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  // Initialize NCCL, group API is required around ncclCommInitRank as it is
  // called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++) {
    CUDACHECK(cudaSetDevice(localRank*nDev + i));
    // ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)
    // The rank of each GPU device is MPI rank * GPUs per MPI thread + i
    NCCLCHECK(ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i));
  }
  NCCLCHECK(ncclGroupEnd());


  // Use NCCL group API to call allReduce on every GPU
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++)
    // ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
           comms[i], s[i]));
   NCCLCHECK(ncclGroupEnd());


  // Synchronize on CUDA stream to complete NCCL communication
  for (int i=0; i<nDev; i++)
      CUDACHECK(cudaStreamSynchronize(s[i]));


  // Free device memory
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
  }


  // Finalize NCCL
  for (int i=0; i<nDev; i++) {
     ncclCommDestroy(comms[i]);
  }


  // Finalize MPI
  MPICHECK(MPI_Finalize());

  // Print success
  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}