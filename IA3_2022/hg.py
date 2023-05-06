#!/usr/bin/env python


# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Details of HashGraph can be found in additional detail in:
# [1] Oded Green,
# 	"HashGraph - Scalable Hash Tables Using A Sparse Graph Data Structure"
#  https://arxiv.org/pdf/1907.02900.pdf



import numpy as np
import numba
from numba import cuda
from numba import jit
import cupy as cp
import time
import ctypes
# from numpy import random
from cupy import random


import cudf as gd
from cudf.core.column import column
from cudf.core.dataframe import DataFrame, Series
# from cudf.tests import utils

from packaging import version
if version.parse(cp.__version__) >= version.parse("10.0.0"):
    cupy_from_dlpack = cp.from_dlpack
else:
    cupy_from_dlpack = cp.fromDlpack


@cuda.jit
def normailizeHashArray(inputArray, hashRange):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = inputArray.shape[0]
    for idx in range(start, size, stride):
        inputArray[idx] = inputArray[idx]%hashRange

@cuda.jit
def resetCounter(counterArray):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = counterArray.shape[0]
    for idx in range(start, size, stride):
        counterArray[idx]=0

@cuda.jit
def countHashBins(hashA, counterArray, binSize):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = hashA.shape[0]
    for idx in range(start, size, stride):
        mybin = int(hashA[idx]//binSize)+1
        cuda.atomic.add(counterArray,mybin,1)


@cuda.jit
def reOrderInput(inputA,hashA, d_InputAReOrdered, binSize, counterArray, prefixArray, indices, needIndices):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = inputA.shape[0]
    for idx in range(start, size, stride):
        mybin = int(hashA[idx]//binSize)
        pos = cuda.atomic.add(counterArray,mybin,1) + prefixArray[mybin]
        d_InputAReOrdered[pos]=inputA[idx] # Todo store index
        if(needIndices==True):
            indices[pos]=idx

@cuda.jit
def reOrderHash(hashAReordered, counterArray):
    idx  = cuda.grid(1)
    size   = hashAReordered.shape[0]
    if(idx<size):
        cuda.atomic.add(counterArray,hashAReordered[idx]+1,1)

@cuda.jit
def fillTable(table,hashAReordered,d_InputAReOrdered, needIndices,d_indicesReorderd,d_indices, counterArray,prefixArray):
    idx  = cuda.grid(1)
    size   = hashAReordered.shape[0]
    if(idx<size):
        pos = cuda.atomic.add(counterArray,hashAReordered[idx],1)+prefixArray[hashAReordered[idx]]
        table[pos] = d_InputAReOrdered[idx]
        if(needIndices==True):
            d_indices[pos] = d_indicesReorderd[idx]


@cuda.jit
def fromReOrderedToOriginalOrder(inputReordered,prefixArray,inputOriginalOrder):
    idx  = cuda.grid(1)
    size   = inputReordered.shape[0]
    if(idx<size):
        inputOriginalOrder[prefixArray[idx]] = inputReordered[idx]

@cuda.jit(device=True)
def compAndFindFirst(val1, val2, array,index):
    if (val1==val2):
        array[index] = 1
        return False


#queryType = 0 // Check existance
#queryType = 1 // Count number of instances
#queryType = 2 //


@cuda.jit
def queryKernel(tableA,prefixArrayA,tableB,prefixArrayB,returnArray, queryType,indicesA):
    idx  = cuda.grid(1)
    size   = prefixArrayA.shape[0]-1
    if(idx<size):
        sizeAList=prefixArrayA[idx+1]-prefixArrayA[idx]
        sizeBList=prefixArrayB[idx+1]-prefixArrayB[idx]

        # queryFunc = GetFuncByFlag(0)

        if(sizeBList==0 or sizeAList==0):
            return

        for b in range(sizeBList):
            bVal = tableB[prefixArrayB[idx]+b]

            for a in range(sizeAList):
                aVal = tableA[prefixArrayA[idx]+a]
                if(queryType==0):
                    if(aVal==bVal):
                        returnArray[prefixArrayB[idx]+b]=1
                        break
                    continue
                if(queryType==1):
                    if(aVal==bVal):
                        returnArray[prefixArrayB[idx]+b]=returnArray[prefixArrayB[idx]+b]+1
                    continue
                if(queryType==2):
                    if(aVal==bVal):
                        # returnArray[prefixArrayB[idx]+b]=prefixArrayA[idx]+a
                        returnArray[prefixArrayB[idx]+b]=indicesA[prefixArrayA[idx]+a]
                        # returnArray[prefixArrayB[idx]+b]=indicesA[0]
                        break
                    continue



class HashGraph:

    threads_per_block = 256
    blocks_per_grid   = 256


    def __init__(self, d_inputA, need_indices=False, numBins=1<<14, seedType="deterministic", seed=0):

        self.numBins = numBins
        self.hashRange = (d_inputA.shape[0]) >> 1
        self.binSize = (self.hashRange+numBins-1)//numBins
        self.inputSize = d_inputA.shape[0]
        self.indices_flag = need_indices;
        self.seedType = seedType
        if(seedType=="deterministic"):
            seedType=="deterministic"
        elif (seedType=="random"):
            random.seed(time.time_ns())
            self.seedArray=random.randint(low=0, high=min(d_inputA.shape[0]*10,2**30), size=d_inputA.shape[0],dtype=np.int32)
            # print(self.seedArray)
        elif (seedType=="user-defined"):
            random.seed(seed)
            self.seedArray=random.randint(low=0, high=min(d_inputA.shape[0]*10,2**30), size=d_inputA.shape[0],dtype=np.int32)
            # print(self.seedArray)
        else:
            print("Unsupported seedType. Defaulting to determinstic seed type.")

        # self.d_table           = cuda.device_array(d_inputA.shape, dtype=inputA.dtype)
        # self.d_PrefixSum       = cuda.device_array(self.hashRange+1, dtype=np.uintc)
        if (need_indices==False):
            self.d_table, self.d_PrefixSum = self.build(d_inputA,False)
        else:
            self.d_table, self.d_PrefixSum, self.d_indices = self.build(d_inputA,True)


    def build(self, d_input, need_indices=False):
        # d_InputAReOrdered  = cuda.device_array(d_input.shape, dtype=inputA.dtype)
        d_InputAReOrdered  = cp.empty(d_input.shape, dtype=d_input.dtype)

        # d_HashA            = cuda.device_array(d_input.shape, dtype=np.uintc)
        # print(type(d_HashA))

        # Arrays used for HashGraph proceess (size of the hash-range)
        # d_CounterArray    = cuda.device_array(self.hashRange+1, dtype=np.uintc)
        d_CounterArray    = cp.empty(self.hashRange+1, dtype=np.uintc)



        # d_BinCounterArray = cuda.device_array(self.numBins+1, dtype=np.uintc)
        # d_BinPrefixSum    = cuda.device_array(self.numBins+1, dtype=np.uintc)
        d_BinCounterArray = cp.empty(self.numBins+1, dtype=np.uintc)
        # d_BinPrefixSum    = cuda.device_array(self.numBins+1, dtype=np.uintc)


        # d_table           = cuda.device_array(d_input.shape, dtype=inputA.dtype)
        d_table           = cp.empty(d_input.shape, dtype=d_input.dtype)
        # d_PrefixSum       = cuda.device_array(self.hashRange+1, dtype=np.uintc)

        # d_indices         = cuda.device_array(d_input.shape, dtype=np.uintc)
        # d_indicesReorderd = cuda.device_array(d_input.shape, dtype=np.uintc)
        d_indices         = cp.empty(d_input.shape, dtype=np.uintc)
        d_indicesReorderd = cp.empty(d_input.shape, dtype=np.uintc)


        # start = time.time()

        gdf = DataFrame()

        if(self.seedType!="deterministic"):
            d_input = d_input+self.seedArray
            # print(d_input)



        gdf["a"] = d_input
        gdf["b"] = d_InputAReOrdered

        # d_HashA = gdf.hash_columns(["a"])
        # d_HashA = cupy_from_dlpack(gdf["a"].hash_values()
        d_HashA = gdf["a"].hash_values()


        normailizeHashArray[HashGraph.blocks_per_grid,HashGraph.threads_per_block](d_HashA,self.hashRange)

        resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_BinCounterArray)
        countHashBins[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_HashA,d_BinCounterArray,self.binSize)

        d_BinPrefixSum = numba.cuda.to_device(cp.cumsum(d_BinCounterArray,dtype=np.uintc))




        resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_BinCounterArray)
        reOrderInput[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_input,d_HashA,d_InputAReOrdered, self.binSize, d_BinCounterArray, d_BinPrefixSum, d_indicesReorderd, need_indices)

        # continue;
        # d_hashA = gdf.hash_columns(["b"])
        d_hashA = gdf["b"].hash_values()
        normailizeHashArray[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_hashA,self.hashRange)



        resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_CounterArray)
        bp2 = (self.inputSize + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block
        reOrderHash[bp2, HashGraph.threads_per_block](d_hashA, d_CounterArray)


        # d_PrefixSum = numba.cuda.to_device(cp.cumsum(d_CounterArray,dtype=np.uintc))
        d_PrefixSum = cp.cumsum(d_CounterArray,dtype=np.uintc)

        resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_CounterArray)
        fillTable[bp2, HashGraph.threads_per_block] (d_table,d_hashA,d_InputAReOrdered, need_indices,d_indicesReorderd, d_indices, d_CounterArray,d_PrefixSum)

        cuda.synchronize()



        # thisTime = time.time()-start
        # print('Internal Time taken = %2.6e seconds'%(thisTime))
        # print('Internal Rate = %.5e keys/sec'%((self.inputSize)/thisTime))
        if(need_indices==False):
            return d_table, d_PrefixSum
        else:
            return d_table, d_PrefixSum, d_indices


    def queryExistance(self, d_inputB):
        d_tableB, d_PrefixSumB, d_indicesB = self.build(d_inputB,True)
        # print(self.d_table.copy_to_host())
        # print(d_tableB.copy_to_host())

        # d_flagArray = cuda.device_array(d_inputB.shape, dtype=np.int8)
        d_flagArray = cp.zeros(d_inputB.shape, dtype=np.int8)
        d_flagArrayFinal = cp.zeros(d_inputB.shape, dtype=np.int8)
        d_dummy = cp.zeros(1, dtype=np.int32)

        bp2 = (self.hashRange + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block
        queryKernel[bp2, HashGraph.threads_per_block](self.d_table, self.d_PrefixSum, d_tableB, d_PrefixSumB, d_flagArray, 0,d_dummy)

        bp2 = (d_inputB.shape[0] + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block
        fromReOrderedToOriginalOrder[bp2, HashGraph.threads_per_block](d_flagArray,d_indicesB,d_flagArrayFinal)
        cuda.synchronize()
        return d_flagArrayFinal

    def queryCount(self, d_inputB):
        d_tableB, d_PrefixSumB, d_indicesB = self.build(d_inputB,True)

        d_countArray = cp.zeros(d_inputB.shape, dtype=np.int32)
        d_countArrayFinal = cp.zeros(d_inputB.shape, dtype=np.int32)
        # d_countArrayFinal = cuda.device_array(d_inputB.shape, dtype=np.int32)
        d_dummy = cp.zeros(1, dtype=np.int32)

        bp2 = (self.hashRange + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block

        queryKernel[bp2, HashGraph.threads_per_block](self.d_table, self.d_PrefixSum, d_tableB, d_PrefixSumB, d_countArray, 1, d_dummy)

        bp2 = (d_inputB.shape[0] + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block
        fromReOrderedToOriginalOrder[bp2, HashGraph.threads_per_block](d_countArray,d_indicesB,d_countArrayFinal)
        cuda.synchronize()
        return d_countArrayFinal

    def queryFindFirst(self, d_inputB):

        if(self.indices_flag==False):
            print("queryFindFirst requires that the HashGraph be built with indices=True in the constructor")
        d_tableB, d_PrefixSumB, d_indicesB = self.build(d_inputB,True)

        d_FirstArray = cp.zeros(d_inputB.shape, dtype=np.int32)
        d_FirstArrayFinal = cp.zeros(d_inputB.shape, dtype=np.int32)

        bp2 = (self.hashRange + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block
        queryKernel[bp2, HashGraph.threads_per_block](self.d_table, self.d_PrefixSum, d_tableB, d_PrefixSumB, d_FirstArray, 2, self.d_indices)

        print(self.d_indices.shape)

        bp2 = (d_inputB.shape[0] + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block
        fromReOrderedToOriginalOrder[bp2, HashGraph.threads_per_block](d_FirstArray,d_indicesB,d_FirstArrayFinal)
        cuda.synchronize()
        return d_FirstArrayFinal




def hgMain():
    inputSize = 1<<28
    low = 0
    high = inputSize
    # hashRange = inputSize >> 2
    # numBins = 1<<14
    # binSize = (hashRange+numBins-1)//numBins

    # np.random.seed(123)


    for i in range(4):

        # inputA = np.random.randint(low,high,inputSize,dtype=np.int32)
        inputA = np.arange(low,inputSize,step=1,dtype=np.int32)  # //2
        # inputB = np.random.randint(low,high,inputSize,dtype=inputA.dtype)
        inputB = inputA

        d_inputA           = cuda.device_array(inputA.shape, dtype=inputA.dtype)
        d_inputA           = cuda.to_device(inputA)

        d_inputB           = cuda.device_array(inputB.shape, dtype=inputA.dtype)
        d_inputB           = cuda.to_device(inputB)


        cuda.synchronize()
        start = time.time()
        hg = HashGraph(d_inputA,True)
        thisTime = time.time()-start
        print('Time taken = %2.6e seconds'%(thisTime))
        print('Rate = %.5e keys/sec'%((inputSize)/thisTime))


        start = time.time()
        # flagArray   = hg.queryExistance(d_inputB)
        flagTime = time.time()-start
        start = time.time()
        # countArray = hg.queryCount(d_inputB)
        countTime = time.time()-start
        start = time.time()
        # firstArray = hg.queryFindFirst(d_inputB)
        firstTime = time.time()-start
        # print(flagArray)
        # print(flagArray.sum())
        # print(countArray)
        # print(firstArray[0:100])

        # print('Time taken = %2.6e seconds'%(flagTime))
        # print('Time taken = %2.6e seconds'%(countTime))
        # print('Time taken = %2.6e seconds'%(firstTime))
        # print('Time taken = %2.6e seconds'%(thisTime))
        # print('Rate = %.5e keys/sec'%((inputSize)/thisTime))

        print("")

if __name__ == "__main__":

    import rmm
    pool = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(),
        initial_pool_size=2**33,
        maximum_pool_size=2**35
    )
    rmm.mr.set_current_device_resource(pool)


    hgMain()