#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thread>

__global__ void findUniqueNumbers
(
    const int* arrayPtr,
    int* uniqueNumbersPtr,
    int* uniqueCountersPtr
)
{
    const int ARRAY_SIZE = 10000000;
    const int BLOCK_ARRAY_SIZE = 10000;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    extern __shared__ int sharedSet[];

    for (int i = tid; i < ARRAY_SIZE; i += stride)
    {
        int value = arrayPtr[i];

        bool isPrevSame = (0 < i && value == arrayPtr[i - 1]);
        bool isNextSame = (i < ARRAY_SIZE - 1 && value == arrayPtr[i + 1]);

        bool isUnique = !(isPrevSame || isNextSame);

        int uniqueIndex = atomicAdd(&uniqueCountersPtr[blockIdx.x], isUnique);

        sharedSet[isUnique * uniqueIndex + !isUnique * BLOCK_ARRAY_SIZE] = value;
    }

    __syncthreads();

    if (threadIdx.x != 0) return;

    for (int i = 0; i < uniqueCountersPtr[blockIdx.x]; ++i)
    {
        uniqueNumbersPtr[blockIdx.x * BLOCK_ARRAY_SIZE + i] = sharedSet[i];
    }
}

void createArray
(
    std::vector<int>* finalValues, 
    int* uniqueValuesAmount, 
    const int arraySize, 
    const int uniqueMaxAmount, 
    const int uniqueValuesRange
)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> uniqueValuesAmountDis(1, uniqueMaxAmount);
    std::uniform_int_distribution<int> valuesDis(0, uniqueValuesRange);

    *uniqueValuesAmount = uniqueValuesAmountDis(gen);

    std::vector<int> uniqueSet;
    for (int i = 0; i < *uniqueValuesAmount; ++i) {
        int value;
        do
        {
            value = valuesDis(gen);
        } while (std::find(uniqueSet.begin(), uniqueSet.end(), value) != uniqueSet.end());
        uniqueSet.push_back(value);
    }
    std::cout << "Unique Amount: " << uniqueSet.size() << " ";
    std::cout << "GeneratedUnique: ";
    std::sort(uniqueSet.begin(), uniqueSet.end());
    for (int i : uniqueSet) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    int randomValuesSize = arraySize - *uniqueValuesAmount;
    std::vector<int> randomSet;
    for (int i = 0; i < arraySize - *uniqueValuesAmount; ++i) {
        int value;
        do
        {
            value = valuesDis(gen);
        } while (std::find(uniqueSet.begin(), uniqueSet.end(), value) != uniqueSet.end());
        randomSet.push_back(value);
    }

    finalValues->reserve(arraySize);
    finalValues->insert(finalValues->end(), uniqueSet.begin(), uniqueSet.end());
    finalValues->insert(finalValues->end(), randomSet.begin(), randomSet.end());
}

int main() {
    const int arraySize = 10000000;
    const int uniqueMaxAmount = 1000;
    const int uniqueValuesRange = 10000;

    // CPU

    int uniqueValuesAmount = 0;
    std::vector<int> finalValues;
    createArray(&finalValues, &uniqueValuesAmount, arraySize, uniqueMaxAmount, uniqueValuesRange);

    size_t streamsAmount = finalValues.size() / 10000; // 1000

    // GPU

    thrust::sort(thrust::host, finalValues.begin(), finalValues.end());

    int* gpuArrayPtr;
    int* gpuUniqueNumbersPtr;
    int* gpuUniqueCountersPtr;

    cudaMalloc((void**)&gpuArrayPtr, arraySize * sizeof(int));
    cudaMalloc((void**)&gpuUniqueNumbersPtr, arraySize * sizeof(int));
    cudaMalloc((void**)&gpuUniqueCountersPtr, streamsAmount * sizeof(int));

    cudaMemcpy(gpuArrayPtr, &finalValues.at(0), arraySize * sizeof(int), cudaMemcpyHostToDevice);

    int numBlock = streamsAmount;
    int numThreads = 512;

    findUniqueNumbers
        << <
        numBlock,
        numThreads,
        (arraySize / streamsAmount + 1) * sizeof(int) // Shared memory per block
        >> >
        (
            gpuArrayPtr,
            gpuUniqueNumbersPtr,
            gpuUniqueCountersPtr
            );

    cudaDeviceSynchronize();

    std::vector<int> finalVl;

    for (int blockIndex = 0; blockIndex < streamsAmount; ++blockIndex)
    {
        int hostUniqueCount = 0;
        cudaMemcpy(&hostUniqueCount, (gpuUniqueCountersPtr + blockIndex), sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<int> hostUniqueNumbers(hostUniqueCount);

        cudaMemcpy(hostUniqueNumbers.data(), (gpuUniqueNumbersPtr + 10000 * blockIndex), hostUniqueCount * sizeof(int), cudaMemcpyDeviceToHost);

        finalVl.insert(finalVl.end(), hostUniqueNumbers.begin(), hostUniqueNumbers.end());
    }

    cudaFree(gpuArrayPtr);
    cudaFree(gpuUniqueNumbersPtr);
    cudaFree(gpuUniqueCountersPtr);

    std::cout << "Unique Amount: " << finalVl.size() << " ";
    std::cout << "   Found Unique: ";
    std::sort(finalVl.begin(), finalVl.end());

    for (auto o : finalVl)
    {
        std::cout << o << " ";
    }

    return 0;
}