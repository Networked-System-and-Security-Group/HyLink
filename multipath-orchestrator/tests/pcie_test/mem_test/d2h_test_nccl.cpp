#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <cuda_runtime.h>

#define NUM_GPUS 4
#define DATA_SIZE (1024 * 1024 * 256)  // 256MB per block

typedef struct {
    int device_id;
} ThreadArgs;

void* gpu_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int dev = args->device_id;
    
    // 1. Set current device
    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        printf("GPU %d: failed to set device\n", dev);
        return NULL;
    }

    // 2. Allocate pinned host memory
    float *h_data;
    cudaMallocHost((void**)&h_data, DATA_SIZE * sizeof(float));

    // 3. Allocate device memory
    float *d_data;
    cudaMalloc((void**)&d_data, DATA_SIZE * sizeof(float));

    // 4. Create async stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    while (1) {
        // Async D2H copy
        cudaMemcpyAsync(h_data, d_data, DATA_SIZE * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);
        
        // Sync stream
        cudaStreamSynchronize(stream);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaFreeHost(h_data);
    return NULL;
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount < NUM_GPUS) {
        printf("Error: only %d GPU(s) available, %d requested.\n", deviceCount, NUM_GPUS);
        return -1;
    }

    pthread_t threads[NUM_GPUS];
    ThreadArgs args[NUM_GPUS];

    // Create thread per GPU
    for (int i = 0; i < NUM_GPUS; i++) {
        args[i].device_id = i;
        pthread_create(&threads[i], NULL, gpu_worker, &args[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
