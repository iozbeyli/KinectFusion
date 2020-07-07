#pragma once
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "malloc.h"

__global__
void applyBackProjection(float* output, float* depthInput, int depthInputWidth, int depthInputHeight, float f_X, float f_Y, float c_X, float c_Y)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((x >= depthInputWidth) || (y >= depthInputHeight)) {
		return;
	}
	int depthIndex = depthInputWidth * y + x;
	float depth = depthInput[depthIndex];
	int outputIndex = 3 * depthIndex;
	if (isinf(depth)) 
	{
		output[outputIndex] = depth;
		output[outputIndex + 1] = depth;
		output[outputIndex + 2] = depth;
	}
	else 
	{
		output[outputIndex] = ((x - c_X) / f_X) * depth;
		output[outputIndex + 1] = ((y - c_Y) / f_Y) * depth;
		output[outputIndex + 2] = depth;
	}
}

class BackProjector {
public:
	BackProjector(int width, int height)
	{
		m_width = width;
		m_height = height;
		m_size = 3 * width * height * sizeof(float);
		

		m_cudaStatusOutput = cudaMalloc((void**)&m_outputBackProjected, m_size);
		m_cudaStatusOutputFirstLevel = cudaMalloc((void**)&m_outputBackProjectedFirstLevel, m_size / 4);
		m_cudaStatusOutputSecondLevel = cudaMalloc((void**)&m_outputBackProjectedSecondLevel, m_size / 16);
		m_outputBackProjectedCPU = (float*)malloc(m_size);
		m_outputBackProjectedFirstLevelCPU = (float*)malloc(m_size / 4);
		m_outputBackProjectedSecondLevelCPU = (float*)malloc(m_size / 16);
		if (m_cudaStatusInput != cudaSuccess && m_cudaStatusOutput != cudaSuccess && m_cudaStatusOutputFirstLevel != cudaSuccess && m_cudaStatusOutputSecondLevel != cudaSuccess)
		{
			m_OK = false;
		}
		else {
			m_OK = true;
		}
	}
	~BackProjector() {
		cudaFree(m_outputBackProjected);
		cudaFree(m_outputBackProjectedFirstLevel);
		cudaFree(m_outputBackProjectedSecondLevel);
		free(m_outputBackProjectedCPU);
		free(m_outputBackProjectedFirstLevelCPU);
		free(m_outputBackProjectedSecondLevelCPU);
	}

	bool isOK()
	{
		return m_OK;
	}

	bool apply(float* input, float* inputFirstLevel, float* inputSecondLevel)
	{
		dim3 gridSize(m_width / 16, m_height / 16);
		dim3 blockSize(16, 16);

		int filterHalfSize = 3;
		float sigmaSpatial = 1.0f;
		float sigmaRange = 1.0f;

		applyBackProjection<<<gridSize, blockSize>>> (m_outputBackProjected, input, m_width, m_height, m_f_X, m_f_Y, m_c_X, m_c_Y);
		gridSize = dim3(m_width / 32, m_height / 32);
		applyBackProjection<<<gridSize, blockSize>>> (m_outputBackProjectedFirstLevel, inputFirstLevel, m_width/2, m_height/2, m_f_X, m_f_Y, m_c_X, m_c_Y);
		gridSize = dim3(m_width / 32, m_height / 32);
		blockSize = dim3(8, 8);
		applyBackProjection<<<gridSize, blockSize>>> (m_outputBackProjectedSecondLevel, inputSecondLevel, m_width/4, m_height/4, m_f_X, m_f_Y, m_c_X, m_c_Y);
		return true;
	}

	bool copyToCPU()
	{
		cudaError_t cudaStatus = cudaMemcpy(m_outputBackProjectedCPU, m_outputBackProjected, m_size, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusFirstLevel = cudaMemcpy(m_outputBackProjectedFirstLevelCPU, m_outputBackProjectedFirstLevel, m_size / 4, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusSecondLevel = cudaMemcpy(m_outputBackProjectedSecondLevelCPU, m_outputBackProjectedSecondLevel, m_size / 16, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess || cudaStatusFirstLevel != cudaSuccess || cudaStatusSecondLevel != cudaSuccess)
		{
			return false;
		}
		return true;
	}

	float* getOutputCPU(int level) {
		if (level == 1)
		{
			return m_outputBackProjectedFirstLevelCPU;
		}
		if (level == 2)
		{
			return m_outputBackProjectedSecondLevelCPU;
		}
		return m_outputBackProjectedCPU;
	}

	float* getOutputGPU(int level) {
		if (level == 1)
		{
			return m_outputBackProjectedFirstLevel;
		}
		if (level == 2)
		{
			return m_outputBackProjectedSecondLevel;
		}
		return m_outputBackProjected;
	}

	void setIntrinsics(float f_X, float f_Y, float c_X, float c_Y)
	{
		m_f_X = f_X;
		m_f_Y = f_Y;
		m_c_X = c_X;
		m_c_Y = c_Y;
	}

private:
	float* m_input;
	float* m_inputFirstLevel;
	float* m_inputSecondLevel;
	
	float* m_outputBackProjected;
	float* m_outputBackProjectedFirstLevel;
	float* m_outputBackProjectedSecondLevel;
	
	float* m_outputBackProjectedCPU;
	float* m_outputBackProjectedFirstLevelCPU;
	float* m_outputBackProjectedSecondLevelCPU;
	
	float* m_outputNormal;
	float* m_outputNormalFirstLevel;
	float* m_outputNormalSecondLevel;

	float* m_outputNormalCPU;
	float* m_outputNormalFirstLevelCPU;
	float* m_outputNormalSecondLevelCPU;

	cudaError_t m_cudaStatusInput;
	cudaError_t m_cudaStatusOutput;
	cudaError_t m_cudaStatusOutputFirstLevel;
	cudaError_t m_cudaStatusOutputSecondLevel;

	bool m_OK;

	size_t m_size;
	int m_width;
	int m_height;
	float m_f_X; 
	float m_f_Y; 
	float m_c_X; 
	float m_c_Y;
};