#pragma once

#include <minwindef.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class GlobalModel
{
public:
	BYTE* colorMapGPU;
	BYTE* colorMapCPU;
	const UINT FRAME_WIDTH;
	const UINT FRAME_HEIGHT;

	GlobalModel(const UINT frameWidth, const UINT frameHeight)
		: FRAME_WIDTH{ frameWidth }
		, FRAME_HEIGHT{ frameHeight }
	{
		const size_t MODEL_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 4 * sizeof(BYTE);
		cudaStatus = cudaMalloc((void**)&colorMapGPU, MODEL_SIZE);

		if (cudaStatus != cudaSuccess)
			return;

		cudaStatus = cudaMemset((void*)colorMapGPU, 0, MODEL_SIZE);

		colorMapCPU = (BYTE*)malloc(MODEL_SIZE);
	}

	~GlobalModel()
	{
		cudaFree(colorMapGPU);
	}

	bool reset()
	{
		const size_t MODEL_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 4 * sizeof(BYTE);
		cudaStatus = cudaMemset((void*)colorMapGPU, 0, MODEL_SIZE);

		return cudaStatus == cudaSuccess;
	}

	bool isOk()
	{
		return this->cudaStatus == cudaSuccess;
	}

	bool copyToCPU()
	{
		const size_t MODEL_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 4 * sizeof(BYTE);
		cudaStatus = cudaMemcpy(colorMapCPU, colorMapGPU, MODEL_SIZE, cudaMemcpyDeviceToHost);

		return cudaStatus == cudaSuccess;
	}

	BYTE* getColorMapCPU()
	{
		return colorMapCPU;
	}

	cudaError_t status()
	{
		return cudaStatus;
	}

private:
	cudaError_t cudaStatus;

};