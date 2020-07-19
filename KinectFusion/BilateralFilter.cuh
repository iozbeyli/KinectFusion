#pragma once
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __forceinline__
float getGaussian(float input, float sigmaPart,float expDenominator) {
	float expPart = expf(-input / expDenominator);
	return sigmaPart * expPart;
}

__global__
void applyBilateralFilter(float* outputMap, float* inputMap, int imageWidth, int imageHeight,float sigmaSpatial, float sigmaRange, int filterHalfSize) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= imageWidth) || (y >= imageHeight)) {
		return;
	}
	
	float result = 0.0f;
	float weightSum = 0.0f;
	float middle = inputMap[y * imageWidth + x];
	float sigmaPartSpatial = 1 / (2 * M_PI * sigmaSpatial * sigmaSpatial);
	float sigmaPartRange = 1 / (2 * M_PI * sigmaRange * sigmaRange);
	float expDenominatorSpatial = 2 * sigmaSpatial * sigmaSpatial;
	float expDenominatorRange = 2 * sigmaRange * sigmaRange;

	for (int i = -filterHalfSize; i <= filterHalfSize; ++i) {
		for (int j = -filterHalfSize; j <= filterHalfSize; ++j) {
			int row = y + i;
			int column = x + j;

			if ((row >= 0) && (column >= 0) && (row < imageHeight) && (column < imageWidth)) {
				float value = inputMap[row * imageWidth + column];
				if (isfinite(value)) {
					float weight = getGaussian(j*j + i*i, sigmaPartSpatial, expDenominatorSpatial)
						* getGaussian((middle - value) * (middle - value), sigmaPartRange, expDenominatorRange);
					result += weight * value;
					weightSum += weight;
				}
			}
		}
	}
	if(weightSum>0){
		outputMap[y * imageWidth + x] = result / weightSum;
	}
	else {
		outputMap[y * imageWidth + x] = -INFINITY;
	}
	
}

__global__
void subSample(float* output, float* input, int inputWidth, int inputHeight, float sigmaRange)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (((2 * x) >= inputWidth) || ((2 * y) >= inputHeight)) {
		return;
	}
	int filterHalfSize = ceilf(3 * sigmaRange);
	int validCount = 0;
	float result = 0;
	for (int i = -filterHalfSize; i <= filterHalfSize; ++i) {
		for (int j = -filterHalfSize; j <= filterHalfSize; ++j) {
			int inputRow = (2 * y) + i;
			int inputColumn = (2 * x) + j;

			if ((inputRow >= 0) && (inputColumn >= 0) && (inputRow < inputHeight) && (inputColumn < inputWidth)) {
				float value = input[(inputRow * inputWidth) + inputColumn];
				if (isfinite(value)) {
					result += value;
					validCount += 1;
				}
			}
		}
	}
	if (validCount > 0)
	{
		output[(y * (inputWidth / 2)) + x] = result/validCount;
	}
	else {
		output[y * (inputWidth / 2) + x] = -INFINITY;
	}
}

class Filterer {
public: 
	Filterer(int width, int height) 
	{
		m_width = width;
		m_height = height;
		m_size = width * height * sizeof(float);
		m_cudaStatusInput = cudaMalloc((void**)&m_inputMap, m_size);
		m_cudaStatusOutput = cudaMalloc((void**)&m_outputMap, m_size);
		m_cudaStatusOutputFirstLevel = cudaMalloc((void**)&m_outputFirstLevelMap, m_size/4);
		m_cudaStatusOutputSecondLevel = cudaMalloc((void**)&m_outputSecondLevelMap, m_size/16);
		m_outputMapCPU = (float*)malloc(m_size);
		m_outputFirstLevelMapCPU = (float*)malloc(m_size/4);
		m_outputSecondLevelMapCPU = (float*)malloc(m_size/16);
	}
	~Filterer() {
		cudaFree(m_inputMap);
		cudaFree(m_outputMap);
		cudaFree(m_outputFirstLevelMap);
		cudaFree(m_outputSecondLevelMap);
		free(m_outputMapCPU);
		free(m_outputFirstLevelMapCPU);
		free(m_outputSecondLevelMapCPU);
	}
	
	bool isOK() 
	{
		if (m_cudaStatusInput != cudaSuccess && m_cudaStatusOutput != cudaSuccess && m_cudaStatusOutputFirstLevel != cudaSuccess && m_cudaStatusOutputSecondLevel != cudaSuccess)
		{
			return false;
		}
		return true;
	}

	bool applyFilter(float* input) 
	{
		cudaError_t cudaStatus = cudaMemcpy(m_inputMap, input, m_size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) 
		{
			return false;
		}
		dim3 gridSize(m_width / 16, m_height / 16);
		dim3 blockSize(16, 16);

		int filterHalfSize = 3;
		float sigmaSpatial = 1.0f;
		float sigmaRange = 1.0f;

		applyBilateralFilter<<<gridSize, blockSize>>>(m_outputMap, m_inputMap, m_width, m_height,sigmaSpatial,sigmaRange,filterHalfSize);
		gridSize = dim3(m_width / 32, m_height / 32);
		subSample <<<gridSize, blockSize>>> (m_outputFirstLevelMap, m_outputMap, m_width, m_height, sigmaRange);
		gridSize = dim3(m_width / 32, m_height / 32);
		blockSize = dim3(8, 8);
		subSample <<<gridSize, blockSize>>> (m_outputSecondLevelMap, m_outputFirstLevelMap, m_width / 2, m_height / 2, sigmaRange);
		return true;
	}

	bool copyToCPU() 
	{
		cudaError_t cudaStatus = cudaMemcpy(m_outputMapCPU, m_outputMap, m_size, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusFirstLevel = cudaMemcpy(m_outputFirstLevelMapCPU, m_outputFirstLevelMap, m_size/4, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusSecondLevel = cudaMemcpy(m_outputSecondLevelMapCPU, m_outputSecondLevelMap, m_size/16, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess || cudaStatusFirstLevel != cudaSuccess || cudaStatusSecondLevel != cudaSuccess)
		{
			return false;
		}
		return true;
	}

	float* getOutputCPU(int level) {
		if (level == 1) 
		{
			return m_outputFirstLevelMapCPU;
		} 
		if (level == 2)
		{
			return m_outputSecondLevelMapCPU;
		}
		return m_outputMapCPU;
	}

	float* getInputGPU()
	{
		return m_inputMap;
	}

	float* getOutputGPU(int level)
	{
		if (level == 1)
		{
			return m_outputFirstLevelMap;
		}
		if (level == 2)
		{
			return m_outputSecondLevelMap;
		}
		return m_outputMap;
	}

private:
	float* m_inputMap;
	float* m_outputMap;
	float* m_outputFirstLevelMap;
	float* m_outputSecondLevelMap;
	cudaError_t m_cudaStatusInput;
	cudaError_t m_cudaStatusOutput;
	cudaError_t m_cudaStatusOutputFirstLevel;
	cudaError_t m_cudaStatusOutputSecondLevel;
	size_t m_size;
	int m_width;
	int m_height;
	float* m_outputMapCPU;
	float* m_outputFirstLevelMapCPU;
	float* m_outputSecondLevelMapCPU;
};
