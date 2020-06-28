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
void applyBilateralFilter(float* outputMap, float* inputMap, int imageWidth, int imageHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= imageWidth) || (y >= imageHeight)) {
		return;
	}
	int filterHalfSize = 3;
	float sigmaSpatial = 1.6f;
	float sigmaRange = 1.0f;
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

class Filterer {
public: 
	Filterer(int width, int height) 
	{
		m_width = width;
		m_height = height;
		m_size = width * height * sizeof(float);
		m_cudaStatusInput = cudaMalloc((void**)&m_inputMap, m_size);
		m_cudaStatusOutput = cudaMalloc((void**)&m_outputMap, m_size);
		m_outputMapCPU = (float*)malloc(m_size);
	}
	~Filterer() {
		cudaFree(m_inputMap);
		cudaFree(m_outputMap);
		free(m_outputMapCPU);
	}
	
	bool isOK() 
	{
		if (m_cudaStatusInput != cudaSuccess && m_cudaStatusOutput != cudaSuccess)
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
		applyBilateralFilter<<<gridSize, blockSize>>>(m_outputMap, m_inputMap, m_width, m_height);
		return true;
	}

	bool copyToCPU() 
	{
		cudaError_t cudaStatus = cudaMemcpy(m_outputMapCPU, m_outputMap, m_size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			return false;
		}
		return true;
	}

	float* getOutput() {
		return m_outputMapCPU;
	}

private:
	float* m_inputMap;
	float* m_outputMap;
	cudaError_t m_cudaStatusInput;
	cudaError_t m_cudaStatusOutput;
	size_t m_size;
	int m_width;
	int m_height;
	float* m_outputMapCPU;
};
