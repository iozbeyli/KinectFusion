#pragma once

#include <minwindef.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Volume
{
public:
	float* sdf;
	float* weights;
	BYTE* colors;

	const UINT FRAME_WIDTH;
	const UINT FRAME_HEIGHT;
	const float TRUNCATION;
	const UINT VOXEL_COUNT_X;
	const UINT VOXEL_COUNT_Y;
	const UINT VOXEL_COUNT_Z;
	const UINT VOXEL_SIZE;
	const float VOXEL_MAX_WEIGHT;
	float FOV_X;
	float FOV_Y;
	float CENTER_FOV_X;
	float CENTER_FOV_Y;

	Volume(const UINT frameWidth, const UINT frameHeight, const float truncation, const UINT voxelCount,
		const UINT voxelSize, const float voxelMaxWeight)
		: FRAME_WIDTH{ frameWidth }
		, FRAME_HEIGHT{ frameHeight }
		, TRUNCATION{ truncation }
		, VOXEL_COUNT_X{ voxelCount }
		, VOXEL_COUNT_Y{ voxelCount }
		, VOXEL_COUNT_Z{ voxelCount }
		, VOXEL_SIZE{ voxelSize }
		, VOXEL_MAX_WEIGHT{ voxelMaxWeight }
		, FOV_X{ 0.0f }
		, FOV_Y{ 0.0f }
		, CENTER_FOV_X{ 0.0f }
		, CENTER_FOV_Y{ 0.0f }
	{
		const size_t VOLUME_SIZE = VOXEL_COUNT_X* VOXEL_COUNT_Y* VOXEL_COUNT_Z * sizeof(float);
		const size_t VOLUME_COLOR_SIZE = VOXEL_COUNT_X * VOXEL_COUNT_Y * VOXEL_COUNT_Z * 4 * sizeof(BYTE);

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
	}

	bool isOk()
	{
		return this->cudaStatus == cudaSuccess;
	}

	void setIntrinsics(float f_X, float f_Y, float c_X, float c_Y)
	{
		FOV_X = f_X;
		FOV_Y = f_Y;
		CENTER_FOV_X = c_X;
		CENTER_FOV_Y = c_Y;
	}

	bool copyToCPU()
	{

		return true;
	}

	cudaError_t status()
	{
		return cudaStatus;
	}

	~Volume()
	{
		cudaFree(sdf);
		cudaFree(weights);
		cudaFree(colors);
	}

private:
	cudaError_t cudaStatus;
};

