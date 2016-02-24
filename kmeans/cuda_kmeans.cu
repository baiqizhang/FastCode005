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
//#define OUTPUT_SIZE
//#define OUTPUT_TIME 
#define NUMBER 8
#define EXTRA 1
//#define OUTPUT_RESULT


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "kmeans.h"
/*
static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}*/


/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          float *deviceNewCluster,//  [numCoords][numClusters]
                          int *deviceNewClusterSize, //[numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    unsigned int tid = threadIdx.x;

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block.
//    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
    float *clusters = (float *)(sharedMemory );//+ blockDim.x);
   
//    membershipChanged[tid] = 0;

    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates!

    // using CUDA unroll
    #pragma unroll 
    for (int i = tid; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[(numClusters+1) * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();

    int objectId = blockDim.x * blockIdx.x + tid;

    if (objectId < numObjs) {
        int   index;
        float dist, min_dist=1e20;

        /* find the cluster id that has min distance to object */
        index = 0;
        
        for (int i=0;i<numClusters;i++){
            dist = 0;
#pragma unroll
            for (int j = 0; j < numCoords; j++)
            {
                float x = objects[numObjs * j + objectId];
                float y = clusters[(numClusters+1) * j + i];
                dist += (x-y)*(x-y);
            }
            if (dist<min_dist){
                min_dist = dist;
                index = i;
            }
        }
                
//        if (numCoords==NUMBER){
#pragma unroll
            for (int j=0; j<numCoords; j++)
                atomicAdd(&deviceNewCluster[j*numClusters + index], objects[j*numObjs+objectId]);
//        }
        
        // assign the membership to object objectId 
        if (membership[objectId] != index) 
            atomicAdd(&intermediates[0],1);
        membership[objectId] = index;

        atomicAdd(&deviceNewClusterSize[index],1);
    }
}



/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster_666(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          float *deviceNewCluster,//  [numCoords][numClusters]
                          int *deviceNewClusterSize, //[numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    unsigned int tid = threadIdx.x;

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block.
//    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
    float *clusters = (float *)(sharedMemory );//+ blockDim.x);
   
//    membershipChanged[tid] = 0;

    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates!

    // using CUDA unroll
    #pragma unroll 
    for (int i = tid; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[(numClusters+1) * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();

    int objectId = blockDim.x * blockIdx.x + tid;

    if (objectId < numObjs) {
        int   index;
        float dist, min_dist=1e20;

        /* find the cluster id that has min distance to object */
        index = 0;
        
        for (int i=0;i<numClusters;i++){
            dist = 0;
#pragma unroll
            for (int j = 0; j < 22; j++)
            {
                float x = objects[numObjs * j + objectId];
                float y = clusters[(numClusters+1) * j + i];
                dist += (x-y)*(x-y);
            }
            if (dist<min_dist){
                min_dist = dist;
                index = i;
            }
        }
                
//        if (numCoords==NUMBER){
#pragma unroll
        for (int j=0; j<numCoords; j++)
            atomicAdd(&deviceNewCluster[j*numClusters + index], objects[j*numObjs+objectId]);
//        }
        
        // assign the membership to object objectId 
        if (membership[objectId] != index) 
            atomicAdd(&intermediates[0],1);
        membership[objectId] = index;

        atomicAdd(&deviceNewClusterSize[index],1);
    }
}


/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster_2333(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          float *deviceNewCluster,//  [numCoords][numClusters]
                          int *deviceNewClusterSize, //[numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    unsigned int tid = threadIdx.x;

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block.
//    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
    float *clusters = (float *)(sharedMemory );//+ blockDim.x);
   
//    membershipChanged[tid] = 0;

    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates!

    // using CUDA unroll
    #pragma unroll 
    for (int i = tid; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[(numClusters+1) * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();

    int objectId = blockDim.x * blockIdx.x + tid;

    if (objectId < numObjs) {
        int   index;
        float dist, min_dist=1e20;

        /* find the cluster id that has min distance to object */
        index = 0;
        
        for (int i=0;i<numClusters;i++){
            dist = 0;
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                float x = objects[numObjs * j + objectId];
                float y = clusters[(numClusters+1) * j + i];
                dist += (x-y)*(x-y);
            }
            if (dist<min_dist){
                min_dist = dist;
                index = i;
            }
        }
                
//        if (numCoords==NUMBER){
#pragma unroll
        for (int j=0; j<numCoords; j++)
            atomicAdd(&deviceNewCluster[j*numClusters + index], objects[j*numObjs+objectId]);
//        }
        
        // assign the membership to object objectId 
        if (membership[objectId] != index) 
            atomicAdd(&intermediates[0],1);
        membership[objectId] = index;

        atomicAdd(&deviceNewClusterSize[index],1);
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
    float *deviceNewCluster;  //new
    int *deviceNewClusterSize; //new
    int *deviceMembership;
    int *deviceIntermediates;

    //  Copy objects given in [numObjs][numCoords] layout to new
    //  [numCoords][numObjs] layout
    malloc2D(dimObjects, numCoords, numObjs, float);
    #pragma omp parallel for
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    malloc2D(dimClusters, numCoords, numClusters, float);
    #pragma omp parallel for
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(EXTRA*numClusters, sizeof(int));
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
//        numThreadsPerClusterBlock * sizeof(unsigned char) +
        (numClusters+1) * numCoords * sizeof(float);

    const unsigned int numReductionThreads = 1;
//        nextPowerOfTwo(numClusterBlocks);
//    const unsigned int reductionBlockSharedDataSize =
//        numReductionThreads * sizeof(unsigned int);

    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));
    checkCuda(cudaMemset(deviceIntermediates, 0,numReductionThreads*sizeof(unsigned int)));

    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMalloc(&deviceNewCluster, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceNewClusterSize, EXTRA*numClusters * sizeof(int) ));
    
    checkCuda(cudaMemset(deviceNewCluster, 0, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMemset(deviceNewClusterSize, 0, EXTRA*numClusters * sizeof(int)));

        //out of the loop!
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    do {
#ifdef OUTPUT_TIME
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
#endif

#ifdef OUTPUT_SIZE
        printf("\nnumClusterBlocks = %d, numThreadPerCB  = %d\n",numClusterBlocks,numThreadsPerClusterBlock);
#endif
        if (numCoords == 8)
            find_nearest_cluster_2333
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize>>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceNewCluster, deviceNewClusterSize, deviceMembership, deviceIntermediates);
        else if (numCoords == 22)
            find_nearest_cluster_666
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize>>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceNewCluster, deviceNewClusterSize, deviceMembership, deviceIntermediates);
        else
            find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceNewCluster, deviceNewClusterSize, deviceMembership, deviceIntermediates);

        cudaThreadSynchronize(); checkLastCudaError();
#ifdef OUTPUT_TIME
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("#%d %ld.%06ld\t",loop, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    
    gettimeofday(&tval_before, NULL);
#endif

        int d;
        checkCuda(cudaMemcpy(&d, deviceIntermediates,
                  sizeof(int), cudaMemcpyDeviceToHost));
//        printf("\nd:%d",d);
        delta = (float)d;

        checkCuda(cudaMemset(deviceIntermediates,0, numReductionThreads*sizeof(unsigned int)));

#ifdef OUTPUT_TIME        
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf(" %ld.%06ld\t", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    
    gettimeofday(&tval_before, NULL);
#endif
//        cudaThreadSynchronize(); checkLastCudaError();
        checkCuda(cudaMemcpy(newClusters[0], deviceNewCluster,
                numClusters*numCoords*sizeof(float), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(newClusterSize,deviceNewClusterSize,
                    EXTRA*numClusters * sizeof(int), cudaMemcpyDeviceToHost ));
        
#ifdef OUTPUT_RESULT
        printf("Membership:\n");
#endif
        for (i=0; i<numClusters; i++) {
#ifdef OUTPUT_RESULT
            printf("%d ",newClusterSize[i]);
#endif
            for (j=0; j<numCoords; j++) {
//                int sum = 0;
//                for (int tt=0;tt<EXTRA;tt++)
//                    sum+=newClusterSize[tt*numClusters+i];
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
//                newClusters[j][i] = 0.0;
            }
//            newClusterSize[i] = 0;  
        }

#ifdef OUTPUT_RESULT
            printf("\nClusters:\n");
            for (i=0;i<numClusters;i++){
                for (j=0;j<numCoords;j++)
                    printf("%f ",dimClusters[j][i]);
                printf("\n");
            }
#endif
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));


        checkCuda(cudaMemset(deviceNewCluster, 0, numClusters*numCoords*sizeof(float)));
        checkCuda(cudaMemset(deviceNewClusterSize, 0, EXTRA*numClusters * sizeof(int)));
        
#ifdef OUTPUT_TIME        
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf(", %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
#endif
        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

//======================================================
            checkCuda(cudaMemcpy(membership, deviceMembership,
               numObjs*sizeof(int), cudaMemcpyDeviceToHost));
        
            for (i=0; i<numObjs; i++) {
                index = membership[i];

                newClusterSize[index]++;
                for (j=0; j<numCoords; j++){
                    newClusters[j][index] += objects[i][j];
                }
            }
        
#ifdef OUTPUT_RESULT
            printf("Membership:\n");
#endif
            for (i=0; i<numClusters; i++) {
#ifdef OUTPUT_RESULT
                printf("%d ",newClusterSize[i]);
#endif
                for (j=0; j<numCoords; j++) {
                    if (newClusterSize[i] > 0)
                        dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                    newClusters[j][i] = 0.0;
                }
                newClusterSize[i] = 0;  
            }

#ifdef OUTPUT_RESULT
            printf("\nClusters:\n");
            for (i=0;i<numClusters;i++){
                for (j=0;j<numCoords;j++)
                    printf("%f ",dimClusters[j][i]);
                printf("\n");
            }
#endif
            checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                      numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));


//=====================================================


    *loop_iterations = loop + 1;

    /* allocate a 2D space for returning variable clusters[] (coordinates
       of cluster centers) */
    malloc2D(clusters, numClusters, numCoords, float);
    

    //GPU -> mem
    checkCuda(cudaMemcpy(dimClusters[0], deviceClusters, 
        numClusters*numCoords*sizeof(float), cudaMemcpyDeviceToHost));
    
    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < numCoords; j++) {
            clusters[i][j] = dimClusters[j][i];
            //printf("%f ",clusters[i][j]);
        }
//        printf("\n");
    }

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));
    checkCuda(cudaFree(deviceIntermediates));

    checkCuda(cudaFree(deviceNewCluster));
    checkCuda(cudaFree(deviceNewClusterSize));
    
    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}



