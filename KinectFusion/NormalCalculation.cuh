#pragma once
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "malloc.h"


__global__
void findNormalVector(float* output, bool* validMask, float* backProjectedInput, int backProjectedInputWidth, int backProjectedInputHeight)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int indexWithoutChannel = backProjectedInputWidth * y + x;

	if ((x >= (backProjectedInputWidth - 1)) || (y >= (backProjectedInputHeight - 1)))
	{
		if ((x == (backProjectedInputWidth - 1)) || (y == (backProjectedInputHeight - 1)))
		{
			validMask[indexWithoutChannel] = false;
		}
		return;
	}

	int index = 3 * indexWithoutChannel;
	int indexRight = 3 * (indexWithoutChannel + 1);
	int indexDown = 3 * (indexWithoutChannel + backProjectedInputWidth);

	float values[3] = { backProjectedInput[index],backProjectedInput[index + 1],backProjectedInput[index + 2] };
	float valuesDown[3] = { backProjectedInput[indexDown],backProjectedInput[indexDown + 1],backProjectedInput[indexDown + 2] };
	float valuesRight[3] = { backProjectedInput[indexRight],backProjectedInput[indexRight + 1],backProjectedInput[indexRight + 2] };

	if (isinf(values[0]) || isinf(valuesDown[0]) || isinf(valuesRight[0]))
	{
		validMask[indexWithoutChannel] = false;
	}
	else
	{
		validMask[indexWithoutChannel] = true;
		float vectorDown[3] = { valuesDown[0] - values[0],valuesDown[1] - values[1],valuesDown[2] - values[2] };
		float vectorRight[3] = { valuesRight[0] - values[0],valuesRight[1] - values[1],valuesRight[2] - values[2] };

		float x = vectorRight[1] * vectorDown[2] - vectorRight[2] * vectorDown[1];
		float y = vectorRight[2] * vectorDown[0] - vectorRight[0] * vectorDown[2];
		float z = vectorRight[0] * vectorDown[1] - vectorRight[1] * vectorDown[0];

		float norm = sqrtf(x * x + y * y + z * z);
		output[index] = x / norm;
		output[index + 1] = y / norm;
		output[index + 2] = z / norm;
	}
}

class NormalCalculator {
public:
	NormalCalculator(int width, int height)
	{
		m_width = width;
		m_height = height;
		m_size = 3 * width * height * sizeof(float);
		m_sizeValidMask = width * height * sizeof(bool);

		cudaError_t cudaStatusOutputUnfiltered = cudaMalloc((void**)&m_outputNormalUnfiltered, m_size);
		cudaError_t cudaStatusOutput = cudaMalloc((void**)&m_outputNormal, m_size);
		cudaError_t cudaStatusOutputFirstLevel = cudaMalloc((void**)&m_outputNormalFirstLevel, m_size / 4);
		cudaError_t cudaStatusOutputSecondLevel = cudaMalloc((void**)&m_outputNormalSecondLevel, m_size / 16);
		m_outputNormalUnfilteredCPU = (float*)malloc(m_size);
		m_outputNormalCPU = (float*)malloc(m_size);
		m_outputNormalFirstLevelCPU = (float*)malloc(m_size / 4);
		m_outputNormalSecondLevelCPU = (float*)malloc(m_size / 16);

		cudaError_t cudaStatusValidMaskUnfiltered = cudaMalloc((void**)&m_validMaskUnfiltered, m_sizeValidMask);
		cudaError_t cudaStatusValidMask = cudaMalloc((void**)&m_validMask, m_sizeValidMask);
		cudaError_t cudaStatusValidMaskFirstLevel = cudaMalloc((void**)&m_validMaskFirstLevel, m_sizeValidMask / 4);
		cudaError_t cudaStatusValidMaskSecondLevel = cudaMalloc((void**)&m_validMaskSecondLevel, m_sizeValidMask / 16);
		m_validMaskUnfilteredCPU = (bool*)malloc(m_sizeValidMask);
		m_validMaskCPU = (bool*)malloc(m_sizeValidMask);
		m_validMaskFirstLevelCPU = (bool*)malloc(m_sizeValidMask / 4);
		m_validMaskSecondLevelCPU = (bool*)malloc(m_sizeValidMask / 16);
		if (cudaStatusOutputUnfiltered != cudaSuccess && 
			cudaStatusOutput != cudaSuccess &&
			cudaStatusOutputFirstLevel != cudaSuccess && 
			cudaStatusOutputSecondLevel != cudaSuccess &&
			cudaStatusValidMaskUnfiltered != cudaSuccess &&
			cudaStatusValidMask != cudaSuccess &&
			cudaStatusValidMaskFirstLevel != cudaSuccess &&
			cudaStatusValidMaskSecondLevel != cudaSuccess
			)
		{
			m_OK = false;
		}
		else {
			m_OK = true;
		}
	}
	~NormalCalculator() {
		cudaFree(m_outputNormalUnfiltered);
		cudaFree(m_outputNormal);
		cudaFree(m_outputNormalFirstLevel);
		cudaFree(m_outputNormalSecondLevel);
		free(m_outputNormalUnfilteredCPU);
		free(m_outputNormalCPU);
		free(m_outputNormalFirstLevelCPU);
		free(m_outputNormalSecondLevelCPU);

		cudaFree(m_validMaskUnfiltered);
		cudaFree(m_validMask);
		cudaFree(m_validMaskFirstLevel);
		cudaFree(m_validMaskSecondLevel);
		free(m_validMaskUnfilteredCPU);
		free(m_validMaskCPU);
		free(m_validMaskFirstLevelCPU);
		free(m_validMaskSecondLevelCPU);
	}

	bool isOK()
	{
		return m_OK;
	}

	bool apply(float* inputUnfiltered, float* input, float* inputFirstLevel, float* inputSecondLevel)
	{
		dim3 gridSize(m_width / 16, m_height / 16);
		dim3 blockSize(16, 16);

		int filterHalfSize = 3;
		float sigmaSpatial = 1.0f;
		float sigmaRange = 1.0f;

		findNormalVector<<<gridSize, blockSize>>> (m_outputNormalUnfiltered, m_validMaskUnfiltered, inputUnfiltered, m_width, m_height);
		findNormalVector<<<gridSize, blockSize>>>(m_outputNormal, m_validMask, input, m_width, m_height);
		gridSize = dim3(m_width / 32, m_height / 32);
		findNormalVector<<<gridSize, blockSize>>>(m_outputNormalFirstLevel, m_validMaskFirstLevel, inputFirstLevel, m_width / 2, m_height / 2);
		gridSize = dim3(m_width / 32, m_height / 32);
		blockSize = dim3(8, 8);
		findNormalVector<<<gridSize, blockSize>>>(m_outputNormalSecondLevel, m_validMaskSecondLevel, inputSecondLevel, m_width / 4, m_height / 4);
		return true;
	}

	bool copyToCPU()
	{
		cudaError_t cudaStatusUnfiltered = cudaMemcpy(m_outputNormalUnfilteredCPU, m_outputNormalUnfiltered, m_size, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatus = cudaMemcpy(m_outputNormalCPU, m_outputNormal, m_size, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusFirstLevel = cudaMemcpy(m_outputNormalFirstLevelCPU, m_outputNormalFirstLevel, m_size / 4, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusSecondLevel = cudaMemcpy(m_outputNormalSecondLevelCPU, m_outputNormalSecondLevel, m_size / 16, cudaMemcpyDeviceToHost);
		if (cudaStatusUnfiltered != cudaSuccess || cudaStatus != cudaSuccess || cudaStatusFirstLevel != cudaSuccess || cudaStatusSecondLevel != cudaSuccess)
		{
			return false;
		}
		return true;
	}

	float* getOutputCPU(int level) {
		if (level == -1)
		{
			return m_outputNormalUnfilteredCPU;
		}
		if (level == 1)
		{
			return m_outputNormalFirstLevelCPU;
		}
		if (level == 2)
		{
			return m_outputNormalSecondLevelCPU;
		}
		return m_outputNormalCPU;
	}

	float* getOutputGPU(int level) {
		if (level == -1)
		{
			return m_outputNormalUnfiltered;
		}
		if (level == 1)
		{
			return m_outputNormalFirstLevel;
		}
		if (level == 2)
		{
			return m_outputNormalSecondLevel;
		}
		return m_outputNormal;
	}

	bool copyValidMaskToCPU()
	{
		cudaError_t cudaStatusUnfiltered = cudaMemcpy(m_validMaskUnfilteredCPU, m_validMaskUnfiltered, m_sizeValidMask, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatus = cudaMemcpy(m_validMaskCPU, m_validMask, m_sizeValidMask, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusFirstLevel = cudaMemcpy(m_validMaskFirstLevelCPU, m_validMaskFirstLevel, m_sizeValidMask / 4, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusSecondLevel = cudaMemcpy(m_validMaskSecondLevelCPU, m_validMaskSecondLevel, m_sizeValidMask / 16, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess || cudaStatusFirstLevel != cudaSuccess || cudaStatusSecondLevel != cudaSuccess)
		{
			return false;
		}
		return true;
	}

	bool* getValidMaskCPU(int level) {
		if (level == -1)
		{
			return m_validMaskUnfilteredCPU;
		}
		if (level == 1)
		{
			return m_validMaskFirstLevelCPU;
		}
		if (level == 2)
		{
			return m_validMaskSecondLevelCPU;
		}
		return m_validMaskCPU;
	}

	bool* getValidMaskGPU(int level) {
		if (level == -1)
		{
			return m_validMaskUnfiltered;
		}
		if (level == 1)
		{
			return m_validMaskFirstLevel;
		}
		if (level == 2)
		{
			return m_validMaskSecondLevel;
		}
		return m_validMask;
	}

	void copy(NormalCalculator* currentNormalCalculator)
	{
		cudaError_t statusUnfiltered = cudaMemcpy(m_outputNormalUnfiltered, currentNormalCalculator->m_outputNormalUnfiltered, m_size, cudaMemcpyDeviceToDevice);
		cudaError_t status = cudaMemcpy(m_outputNormal, currentNormalCalculator->m_outputNormal, m_size, cudaMemcpyDeviceToDevice);
		cudaError_t statusFirst = cudaMemcpy(m_outputNormalFirstLevel, currentNormalCalculator->m_outputNormalFirstLevel, m_size/4, cudaMemcpyDeviceToDevice);
		cudaError_t statusSecond = cudaMemcpy(m_outputNormalSecondLevel, currentNormalCalculator->m_outputNormalSecondLevel, m_size/16, cudaMemcpyDeviceToDevice);
		if (statusUnfiltered != cudaSuccess || status != cudaSuccess || statusFirst != cudaSuccess || statusSecond != cudaSuccess)
		{
			std::cout << "Normal copy fails" << std::endl;
		}
		
		statusUnfiltered = cudaMemcpy(m_validMaskUnfiltered, currentNormalCalculator->m_validMaskUnfiltered, m_sizeValidMask, cudaMemcpyDeviceToDevice);
		status = cudaMemcpy(m_validMask, currentNormalCalculator->m_validMask, m_sizeValidMask, cudaMemcpyDeviceToDevice);
		statusFirst = cudaMemcpy(m_validMaskFirstLevel, currentNormalCalculator->m_validMaskFirstLevel, m_sizeValidMask / 4, cudaMemcpyDeviceToDevice);
		statusSecond = cudaMemcpy(m_validMaskSecondLevel, currentNormalCalculator->m_validMaskSecondLevel, m_sizeValidMask / 16, cudaMemcpyDeviceToDevice);
	
		if (statusUnfiltered != cudaSuccess || status != cudaSuccess || statusFirst != cudaSuccess || statusSecond != cudaSuccess)
		{
			std::cout << "Valid copy fails" << std::endl;
		}
	}

private:
	float* m_outputNormalUnfiltered;
	float* m_outputNormal;
	float* m_outputNormalFirstLevel;
	float* m_outputNormalSecondLevel;

	float* m_outputNormalUnfilteredCPU;
	float* m_outputNormalCPU;
	float* m_outputNormalFirstLevelCPU;
	float* m_outputNormalSecondLevelCPU;

	bool* m_validMaskUnfiltered;
	bool* m_validMask;
	bool* m_validMaskFirstLevel;
	bool* m_validMaskSecondLevel;

	bool* m_validMaskUnfilteredCPU;
	bool* m_validMaskCPU;
	bool* m_validMaskFirstLevelCPU;
	bool* m_validMaskSecondLevelCPU;

	bool m_OK;

	size_t m_size;
	size_t m_sizeValidMask;
	int m_width;
	int m_height;
};