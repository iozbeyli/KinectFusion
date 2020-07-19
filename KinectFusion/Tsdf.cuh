#pragma once

#include <minwindef.h>
#include "cuda_runtime.h"
#include "Eigen.h"
#include "Volume.cuh"


__device__ __forceinline__ 
float l2norm(const float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ 
float3 mul(const float* m4f, const float3* v)
{
	//float m01 = m4f[0];
	//float m02 = m4f[4];
	//float m03 = m4f[8];
	//float m04 = m4f[12];
	//float x = v->x;
	//float y = v->y;
	//float z = v->z;

	return make_float3
	(
		v->x * m4f[0] + v->y * m4f[4] + v->z *  m4f[8] + m4f[12],
		v->x * m4f[1] + v->y * m4f[5] + v->z *  m4f[9] + m4f[13],
		v->x * m4f[2] + v->y * m4f[6] + v->z * m4f[10] + m4f[14]
	);
}

__global__
void applyTsdf
(
	float* gpuFrameDepthMap,
	BYTE* const gpuFrameColorMap,
	const float* worldToCamera,
	float* sdfs,
	float* weights,
	BYTE* colors,
	const int frameWidth,
	const int frameHeight,
	const int voxelCountX,
	const int voxelCountY,
	const int voxelCountZ,
	const int voxelSize,
	const float fovX,
	const float fovY,
	const float fovCenterX,
	const float fovCenterY,
	const float truncation,
	const float maxWeight)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// SJ: Idle gpu thread, if we are outside the bounds of our volume.
	if ((x >= voxelCountX) || (y >= voxelCountY))
		return;

	// SJ: Each GPU thread handles a voxel row in z-direction.
	for (int z = 0; z < voxelCountZ; ++z)
	{
		// SJ: x,y,z are voxel coordinates, so we need to convert them to real world coordinates first.
		// SJ: Calculate the currents voxel center position in the world coordinate system. 
		float3 voxelCenterWorldCoord = make_float3((x + .5f) * voxelSize, (y + .5f) * voxelSize, (z + .5f) * voxelSize);	
		
		// SJ: Transform to camera coordinate system
		float3 voxelCenterCameraCoord = mul(worldToCamera, &voxelCenterWorldCoord);

		// SJ: If voxel is behind the camera ignore and continue.
		if (voxelCenterCameraCoord.z <= 0)
			continue;

		// SJ: Project voxel position to camera image plane.
		const int u = (int)(voxelCenterCameraCoord.x / voxelCenterCameraCoord.z * fovX + fovCenterX);
		const int v = (int)(voxelCenterCameraCoord.y / voxelCenterCameraCoord.z * fovY + fovCenterY);

		// SJ: If voxel position is not on the image plane continue.
		if (u < 0 || u > frameWidth || v < 0 || v > frameHeight)
			continue;

		const int FRAME_IDX = v * frameWidth + u;
		const float depth = gpuFrameDepthMap[FRAME_IDX];

		// SJ: Check if we have a valid depth for that pixel coordinate.
		if (isinf(depth) || depth <= 0)
			continue;

		// SJ: Paper formula (7)
		const float lambda = l2norm(make_float3((u - fovCenterX) / fovX, (v - fovCenterY) / fovY, 1.0f));

		// SJ: Paper formula (6), calculate how far away the voxel center is from the actual depth of the backprojected pixel (our surface)
		const float voxelCenterDistanceToCamera = l2norm(voxelCenterCameraCoord);
		const float signedDistance = (voxelCenterDistanceToCamera / lambda) - depth;

		// SJ: If we are outside the truncation range continue.
		if (fabsf(signedDistance) > truncation)
			continue;

		// SJ: Paper formula (9), normalizing the signed distance to [-1,1]
		const float normalizedSignedDistance = fminf(1.f, signedDistance / truncation) * (signedDistance > 0.0f ? 1.0f : -1.0f);

		// SJ: Formula (11-13) from the paper (running weighted average as explained in lecture 5. page 51).
		// SJ: As explained in the paper on page 5 "...simply letting W_Rk(p) = 1, resulting in a simple average, provides good results ..."
		// SJ: Setting new signed distance, colors and weight.
		const int VOXEL_IDX = (voxelCountX * voxelCountY * z) + (voxelCountX * y) + x;
		const int VOXEL_COLOR_IDX = VOXEL_IDX * 4;
		const int FRAME_COLOR_IDX = FRAME_IDX * 4;
		const float W_R = 1.0f;

		//sdfs[VOXEL_IDX] = (weights[VOXEL_IDX] * sdfs[VOXEL_IDX]) + (W_R * normalizedSignedDistance) / (weights[VOXEL_IDX] + W_R);
		//colors[VOXEL_COLOR_IDX + 0] = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 0]) + (W_R * gpuFrameColorMap[FRAME_COLOR_IDX + 0]) / (weights[VOXEL_IDX] + W_R));
		//colors[VOXEL_COLOR_IDX + 1] = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 1]) + (W_R * gpuFrameColorMap[FRAME_COLOR_IDX + 1]) / (weights[VOXEL_IDX] + W_R));
		//colors[VOXEL_COLOR_IDX + 2] = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 2]) + (W_R * gpuFrameColorMap[FRAME_COLOR_IDX + 2]) / (weights[VOXEL_IDX] + W_R));
		//colors[VOXEL_COLOR_IDX + 3] = 1;
		//weights[VOXEL_IDX] = fminf(weights[VOXEL_IDX] + W_R, maxWeight);

		float s = (weights[VOXEL_IDX] * sdfs[VOXEL_IDX]) + (W_R * normalizedSignedDistance) / (weights[VOXEL_IDX] + W_R);
		sdfs[VOXEL_IDX] = s;

		BYTE r = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 0]) + (W_R * gpuFrameColorMap[FRAME_COLOR_IDX + 0]) / (weights[VOXEL_IDX] + W_R));
		colors[VOXEL_COLOR_IDX + 0] = r;
		
		BYTE g = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 1]) + (W_R * gpuFrameColorMap[FRAME_COLOR_IDX + 1]) / (weights[VOXEL_IDX] + W_R));
		colors[VOXEL_COLOR_IDX + 1] = g; 

		BYTE b = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 2]) + (W_R * gpuFrameColorMap[FRAME_COLOR_IDX + 2]) / (weights[VOXEL_IDX] + W_R));
		colors[VOXEL_COLOR_IDX + 2] = b;

		colors[VOXEL_COLOR_IDX + 3] = 1;

		float w = fminf(weights[VOXEL_IDX] + W_R, maxWeight);
		weights[VOXEL_IDX] = w; 
	}
}

class Tsdf
{
public:
	bool apply(Volume& volume, float* gpuFrameDepthMap, const BYTE* cpuFrameColorMap, const Eigen::Matrix4f& frameCameraToWorld)
	{
		// SJ: color map is not in GPU memory yet, so let's put it there.
		const size_t COLOR_MAP_SIZE = volume.FRAME_WIDTH * volume.FRAME_HEIGHT * 4 * sizeof(BYTE);

		cudaStatus = cudaMalloc((void**)&gpuFrameColorMap, COLOR_MAP_SIZE);

		if (cudaStatus != cudaSuccess)
			return false;

		cudaStatus = cudaMemcpy(gpuFrameColorMap, cpuFrameColorMap, COLOR_MAP_SIZE, cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess)
			return false;

		cudaStatus = cudaMalloc((void**)&gpuMatrix, 16 * sizeof(float));

		if (cudaStatus != cudaSuccess)
			return false;

		Eigen::Matrix4f worldToCamera = frameCameraToWorld.inverse();

		cudaStatus = cudaMemcpy(gpuMatrix, worldToCamera.data(), 16*sizeof(float), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess)
			return false;

		// SJ: Max numbers of threads per block is limited to 1024, so lets make full use of it.
		dim3 threads(32, 32);

		// SJ: (N + x - 1) / x gives us the smallest multiple of x greater or equal to N.
		// SJ: So we have one thread per row in z-direction to calculate the voxel values.
		dim3 blocks((volume.VOXEL_COUNT_X + threads.x - 1) / threads.x, 
					(volume.VOXEL_COUNT_Y + threads.y - 1) / threads.y);

		applyTsdf <<<threads, blocks>>>(
			gpuFrameDepthMap, 
			gpuFrameColorMap,
			gpuMatrix,
			volume.sdf,
			volume.weights,
			volume.colors,
			volume.FRAME_WIDTH,
			volume.FRAME_HEIGHT,
			volume.VOXEL_COUNT_X, 
			volume.VOXEL_COUNT_Y,
			volume.VOXEL_COUNT_Z,
			volume.VOXEL_SIZE,
			volume.FOV_X,
			volume.FOV_Y,
			volume.CENTER_FOV_X,
			volume.CENTER_FOV_Y,
			volume.TRUNCATION,
			volume.VOXEL_MAX_WEIGHT
			);

		cudaThreadSynchronize();

		cudaFree(gpuFrameColorMap);
		cudaFree(gpuMatrix);

		return true;
	}

	bool isOk()
	{
		return cudaStatus == cudaSuccess;
	}

	cudaError_t status()
	{
		return cudaStatus;
	}


private:
	BYTE* gpuFrameColorMap;
	float* gpuMatrix;
	cudaError_t cudaStatus;
};