/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_kmeans.cu  (CUDA version)                            */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#define BLOCKSIZE2 1024

#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"

static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;

    // original code
    // for (i = 0; i < numCoords; i++) {
    //     ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
    //            (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    // }

    // 6-way unrolling
    for (i = 0; i < numCoords - 5; i += 6) {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
        ans += (objects[numObjs * (i+1) + objectId] - clusters[numClusters * (i+1) + clusterId]) *
               (objects[numObjs * (i+1) + objectId] - clusters[numClusters * (i+1) + clusterId]);
        ans += (objects[numObjs * (i+2) + objectId] - clusters[numClusters * (i+2) + clusterId]) *
               (objects[numObjs * (i+2) + objectId] - clusters[numClusters * (i+2) + clusterId]);
        ans += (objects[numObjs * (i+3) + objectId] - clusters[numClusters * (i+3) + clusterId]) *
               (objects[numObjs * (i+3) + objectId] - clusters[numClusters * (i+3) + clusterId]);
        ans += (objects[numObjs * (i+4) + objectId] - clusters[numClusters * (i+4) + clusterId]) *
               (objects[numObjs * (i+4) + objectId] - clusters[numClusters * (i+4) + clusterId]);
        ans += (objects[numObjs * (i+5) + objectId] - clusters[numClusters * (i+5) + clusterId]) *
               (objects[numObjs * (i+5) + objectId] - clusters[numClusters * (i+5) + clusterId]);
    }
    // boundary condition
    for (int j = i; j < numCoords; j++) {
        ans += (objects[numObjs * j + objectId] - clusters[numClusters * j + clusterId]) *
               (objects[numObjs * j + objectId] - clusters[numClusters * j + clusterId]);
    }

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];
    unsigned int tid = threadIdx.x;

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block.
    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
    float *clusters = (float *)(sharedMemory + blockDim.x);

    membershipChanged[tid] = 0;

    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates!

    // using CUDA unroll
    #pragma unroll
    for (int i = tid; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();

    int objectId = blockDim.x * blockIdx.x + tid;

    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[tid] = 1;
        }

        /* assign the membership to object objectId */
        membership[objectId] = index;

        __syncthreads();    //  For membershipChanged[]

        //blockDim.x *must* be a power of two!
        /*for (unsigned int s = blockDim.x / 2; s > 64; s >>= 1) {
            if (tid < s) {
                membershipChanged[tid] +=
                    membershipChanged[tid + s];
            }
            __syncthreads();
        }*/

        // if (blockDim.x >= 1024) {
        //     if (tid < 512) { 
        //         membershipChanged[tid] += membershipChanged[tid + 512]; 
        //     } 
        //     __syncthreads(); 
        // }
        // if (blockDim.x >= 512) {
        //     if (tid < 256) { 
        //         membershipChanged[tid] += membershipChanged[tid + 256]; 
        //     }    
        //     __syncthreads(); 
        // }
        // if (blockDim.x >= 256) {
        //     if (tid < 128) { 
        //         membershipChanged[tid] += membershipChanged[tid + 128]; 
        //     } 
        //     __syncthreads(); 
        // }
        if (blockDim.x >= 128) {
            if (tid < 64) { 
                membershipChanged[tid] += membershipChanged[tid + 64]; 
            }    
            __syncthreads(); 
        }

        // Unrolling warp
        if(tid < 32){
            volatile unsigned char* vmem = membershipChanged;
            if (blockDim.x >= 64) vmem[tid] += vmem[tid+32];
            if (blockDim.x >= 32) vmem[tid] += vmem[tid+16];
            if (blockDim.x >= 16) vmem[tid] += vmem[tid+8];
            if (blockDim.x >= 8) vmem[tid] += vmem[tid+4];
            if (blockDim.x >= 4) vmem[tid] += vmem[tid+2];
            if (blockDim.x >= 2) vmem[tid] += vmem[tid+1];
        }

        // only first thread in the grid executes this statement
        if (tid == 0) {
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}

__global__ static
void compute_delta(int *deviceIntermediates,
                   int numIntermediates,    //  The actual number of intermediates
                   int numIntermediates2)   //  The next power of two
{
    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];

    unsigned int tid = threadIdx.x;

    //  Copy global intermediate values into shared memory.
    intermediates[tid] =
        (tid < numIntermediates) ? deviceIntermediates[tid] : 0;

    __syncthreads();

    //  numIntermediates2 *must* be a power of two!
    for (unsigned int s = numIntermediates2 / 2; s > 32; s >>= 1) {
        if (tid < s) {
            intermediates[tid] += intermediates[tid + s];
        }
        __syncthreads();
    }

    // for (unsigned int s = numIntermediates2 / 2; s > 32; s >>= 1) {
    //     if (tid < s) {
    //         intermediates[tid] += intermediates[tid + s];
    //     }
    //     __syncthreads();
    // }

     // Unrolling warp
    if (tid < 32){
        volatile unsigned int* vmem = intermediates;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }

    if (tid == 0) {
        deviceIntermediates[0] = intermediates[0];
    }
}

__global__ static
void compute_delta2(int *deviceIntermediates,
                   int numIntermediates,    //  The actual number of intermediates
                   int numIntermediates2)   //  The next power of two
{
    // numIntermediates is 3817

    // limit is shared memory size
    int limit = BLOCKSIZE2;
    unsigned int tid = threadIdx.x;
    /*
    while ((tid + limit) < numIntermediates) {
        deviceIntermediates[tid] += deviceIntermediates[tid + limit];
        limit += BLOCKSIZE2;
    }*/
    //printf("%d", numIntermediates);

    // If BLOCKSIZE2 == 1024, HARD CODE HERE
    deviceIntermediates[tid] += deviceIntermediates[tid + 1024]; 
    deviceIntermediates[tid] += deviceIntermediates[tid + 2048];
    if(tid + 3072 < numIntermediates){
        deviceIntermediates[tid] += deviceIntermediates[tid + 3072];
    }

    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];

    //  Copy global intermediate values into shared memory.
    intermediates[tid] = deviceIntermediates[tid];

    __syncthreads();

    //  numIntermediates2 *must* be a power of two!
    // for (unsigned int s = numIntermediates2 / 2; s > 32; s >>= 1) {
    //     if (tid < s) {
    //         intermediates[tid] += intermediates[tid + s];
    //     }
    //     __syncthreads();
    // }

    // try complete unrolling
    // Since we know numIntermediates2==1024, no need for judgments
    //if (numIntermediates2 >= 1024) {
        if (tid < 512) { 
            intermediates[tid] += intermediates[tid + 512]; 
        } 
        __syncthreads(); 
    //}
    //if (numIntermediates2 >= 512) {
        if (tid < 256) { 
            intermediates[tid] += intermediates[tid + 256]; 
        } 
        __syncthreads(); 
    //}
    //if (numIntermediates2 >= 256) {
        if (tid < 128) { 
            intermediates[tid] += intermediates[tid + 128]; 
        } 
        __syncthreads(); 
    //}
    //if (numIntermediates2 >= 128) {
        if (tid < 64) { 
            intermediates[tid] += intermediates[tid + 64]; 
        } 
        __syncthreads(); 
    //}

     // Unrolling warp
    if (tid < 32){
        volatile unsigned int* vmem = intermediates;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }

    if (tid == 0) {
        deviceIntermediates[0] = intermediates[0];
    }
    
    // for (unsigned int s = 1; s < numIntermediates; s++) 
    //     deviceIntermediates[0]+=deviceIntermediates[s];
    
}

__global__ static
void compute_delta3(int *deviceIntermediates,
                   int numIntermediates,    //  The actual number of intermediates
                   int numIntermediates2)   //  The next power of two
{
    // limit is shared memory size
    int limit = BLOCKSIZE2;
    unsigned int tid = threadIdx.x;
    // divergence at the end!!!!!
    while ((tid + limit) < numIntermediates) {
        deviceIntermediates[tid] += deviceIntermediates[tid + limit];
        limit += BLOCKSIZE2;
    }
    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];

    //  Copy global intermediate values into shared memory.
    intermediates[tid] = deviceIntermediates[tid];

    __syncthreads();

    //numIntermediates2 *must* be a power of two!
    for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            intermediates[tid] += intermediates[tid + s];
        }
        __syncthreads();
    }
        if (tid == 0) {
        deviceIntermediates[0] = intermediates[0];
    }
   
    
}


/*----< cuda_kmeans() >-------------------------------------------------------*/
//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
float** cuda_kmeans(float **objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **dimObjects;
    float  **clusters;       /* out: [numClusters][numCoords] */
    float  **dimClusters;
    float  **newClusters;    /* [numCoords][numClusters] */

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;
    int *deviceIntermediates;

    //  Copy objects given in [numObjs][numCoords] layout to new
    //  [numCoords][numObjs] layout
    malloc2D(dimObjects, numCoords, numObjs, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    malloc2D(dimClusters, numCoords, numClusters, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    malloc2D(newClusters, numCoords, numClusters, float);
    memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));

    //  To support reduction, numThreadsPerClusterBlock *must* be a power of
    //  two, and it *must* be no larger than the number of bits that will
    //  fit into an unsigned char, the type used to keep track of membership
    //  changes in the kernel.
    const unsigned int numThreadsPerClusterBlock = 128;
    const unsigned int numClusterBlocks =
        (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char) +
        numClusters * numCoords * sizeof(float);

    const unsigned int numReductionThreads =
        nextPowerOfTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);

    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));

    do {
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));

        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);

        cudaThreadSynchronize(); checkLastCudaError();

        if (numReductionThreads==4096) {
            // rewrite the kernel dimenstion for reduction
            //blockDim must equal limit in compute_delta2!
            dim3 blockDim = BLOCKSIZE2;
            dim3 gridDim = numClusterBlocks/blockDim.x + 1;
            compute_delta2 <<< gridDim, blockDim, BLOCKSIZE2 * sizeof(unsigned int) >>>
                (deviceIntermediates, numClusterBlocks, BLOCKSIZE2);
           //compute_delta2 <<< 1,  numReductionThreads/4, reductionBlockSharedDataSize >>>
              //  (deviceIntermediates, numClusterBlocks, numReductionThreads/4);
            }
        else if (numReductionThreads < 1024) {
            dim3 blockDim = numReductionThreads;
            dim3 gridDim = numClusterBlocks/blockDim.x + 1;
            compute_delta <<< gridDim, blockDim, reductionBlockSharedDataSize >>>
                (deviceIntermediates, numClusterBlocks, numReductionThreads);
            }
        else{
            dim3 blockDim = BLOCKSIZE2;
            dim3 gridDim = numClusterBlocks/blockDim.x + 1;
            compute_delta3 <<< gridDim, blockDim, BLOCKSIZE2 * sizeof(unsigned int) >>>
                (deviceIntermediates, numClusterBlocks, BLOCKSIZE2);
        }

        cudaThreadSynchronize(); checkLastCudaError();

        int d;
        checkCuda(cudaMemcpy(&d, deviceIntermediates,
                  sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d;

        checkCuda(cudaMemcpy(membership, deviceMembership,
                  numObjs*sizeof(int), cudaMemcpyDeviceToHost));

        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = membership[i];

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[j][index] += objects[i][j];
        }

        //  TODO: Flip the nesting order
        //  TODO: Change layout of newClusters to [numClusters][numCoords]
        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        //printf("\ndelta:%f\n",delta);
        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;

    /* allocate a 2D space for returning variable clusters[] (coordinates
       of cluster centers) */
    malloc2D(clusters, numClusters, numCoords, float);
    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < numCoords; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));
    checkCuda(cudaFree(deviceIntermediates));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}

