#pragma once

#include <minwindef.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class TsdfVolume
{
public:
	float* sdf;
	float* weights;
	BYTE* colors;
	
	float* cpuSdf;
	float* cpuWeights;
	BYTE* cpuColors;

	const UINT frameWidth;
	const UINT frameHeight;
	const float truncation;
	const UINT voxelCountX;
	const UINT voxelCountY;
	const UINT voxelCountZ;
	const float voxelSize;
	const float maxVoxelWeight;
	float fovX;
	float fovY;
	float fovCenterX;
	float fovCenterY;

	TsdfVolume
	(
		const UINT frameWidth, 
		const UINT frameHeight, 
		const float truncation, 
		const UINT voxelCount,
		const float voxelSize, 
		const float voxelMaxWeight
	)
		: frameWidth{ frameWidth }
		, frameHeight{ frameHeight }
		, truncation{ truncation }
		, voxelCountX{ voxelCount }
		, voxelCountY{ voxelCount }
		, voxelCountZ{ voxelCount }
		, voxelSize{ voxelSize }
		, maxVoxelWeight{ voxelMaxWeight }
		, fovX{ 0.0f }
		, fovY{ 0.0f }
		, fovCenterX{ 0.0f }
		, fovCenterY{ 0.0f }
	{
		const size_t VOLUME_SIZE = voxelCountX * voxelCountY * voxelCountZ * sizeof(float);
		const size_t VOLUME_COLOR_SIZE = voxelCountX * voxelCountY * voxelCountZ * 4 * sizeof(BYTE);

		cudaStatus = cudaMalloc((void**)&sdf, VOLUME_SIZE);

		if (cudaStatus != cudaSuccess)
			return;

		cudaStatus = cudaMemset((void*)sdf, 0, VOLUME_SIZE);

		if (cudaStatus != cudaSuccess)
			return;

		cudaStatus = cudaMalloc((void**)&weights, VOLUME_SIZE);

		if (cudaStatus != cudaSuccess)
			return;

		cudaStatus = cudaMemset((void*)weights, 0, VOLUME_SIZE);

		if (cudaStatus != cudaSuccess)
			return;

		cudaStatus = cudaMalloc((void**)&colors, VOLUME_COLOR_SIZE);

		if (cudaStatus != cudaSuccess)
			return;

		cudaStatus = cudaMemset((void*)colors, 0, VOLUME_COLOR_SIZE);

		cpuSdf = (float*)malloc(VOLUME_SIZE);
		cpuWeights = (float*)malloc(VOLUME_SIZE);
		cpuColors = (BYTE*)malloc(VOLUME_COLOR_SIZE);
	}

	bool isOk()
	{
		return this->cudaStatus == cudaSuccess;
	}

	void setIntrinsics(float f_X, float f_Y, float c_X, float c_Y)
	{
		fovX = f_X;
		fovY = f_Y;
		fovCenterX = c_X;
		fovCenterY = c_Y;
	}

	bool copyToCPU()
	{
		const size_t VOLUME_SIZE = voxelCountX * voxelCountY * voxelCountZ * sizeof(float);
		const size_t VOLUME_COLOR_SIZE = voxelCountX * voxelCountY * voxelCountZ * 4 * sizeof(BYTE);

		cudaStatus = cudaMemcpy(cpuSdf, sdf, VOLUME_SIZE, cudaMemcpyDeviceToHost);

		if (cudaStatus != cudaSuccess)
			return false;

		cudaStatus = cudaMemcpy(cpuWeights, weights, VOLUME_SIZE, cudaMemcpyDeviceToHost);

		if (cudaStatus != cudaSuccess)
			return false;

		cudaStatus = cudaMemcpy(cpuColors, colors, VOLUME_COLOR_SIZE, cudaMemcpyDeviceToHost);

		return cudaStatus == cudaSuccess;
	}

	inline const unsigned int idx(const int x, const int y, const int z) noexcept
	{
		return (voxelCountX * voxelCountY * z) + (voxelCountX * y) + x;
	}

	cudaError_t status()
	{
		return cudaStatus;
	}

	~TsdfVolume()
	{
		cudaFree(sdf);
		cudaFree(weights);
		cudaFree(colors);
	}

private:
	cudaError_t cudaStatus;
};

