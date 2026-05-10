#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <cuda_runtime.h>

#define NUM_GPUS 2
#define DATA_SIZE (1024 * 1024 * 16) // 256MB 每块数据

// 线程参数结构体
typedef struct {
    int device_id;
} ThreadArgs;

// GPU 任务函数
void* gpu_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int dev = args->device_id;
    
    // 1. 设置当前设备
    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        printf("GPU %d: 无法设置设备\n", dev);
        return NULL;
    }

    // 2. 分配主机固定内存 (Pinned Memory)
    float *h_data;
    cudaMallocHost((void**)&h_data, DATA_SIZE * sizeof(float));

    // 3. 分配设备内存
    float *d_data;
    cudaMalloc((void**)&d_data, DATA_SIZE * sizeof(float));

    // 4. 创建异步流
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    while (1) {
        // 执行异步 D2H 拷贝
        cudaMemcpyAsync(h_data, d_data, DATA_SIZE * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);
        
        // 显式同步流（可选，取决于你是否要测量单次循环或压满带宽）
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
        printf("错误：系统中只有 %d 个 GPU，但请求使用 %d 个。\n", deviceCount, NUM_GPUS);
        return -1;
    }

    pthread_t threads[NUM_GPUS];
    ThreadArgs args[NUM_GPUS];

    // 为每个 GPU 创建线程
    for (int i = 0; i < NUM_GPUS; i++) {
        args[i].device_id = i;
        pthread_create(&threads[i], NULL, gpu_worker, &args[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
