#pragma once

#include <minwindef.h>
#include "cuda_runtime.h"
#include "Eigen.h"
#include "TsdfVolume.cuh"


__global__
void applyTsdf_v1
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
	const float voxelSize,
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
		const float signedDistance = (-1.0f) * (voxelCenterDistanceToCamera / lambda) - depth;

		// SJ: If we are outside the truncation range continue.
		if (fabsf(signedDistance) > truncation)
			continue;

		// SJ: Paper formula (9), normalizing the signed distance to [-1,1]
		const float normalizedSignedDistance = fminf(1.f, signedDistance / truncation);

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

__global__
void applyTsdf_v2
(
	float* gpuFrameDepthMap,
	BYTE* const gpuFrameColorMap,
	const float* worldToCamera,
	const float* cameraToWorld,
	float* sdfs,
	float* weights,
	BYTE* colors,
	const int frameWidth,
	const int frameHeight,
	const int voxelCountX,
	const int voxelCountY,
	const int voxelCountZ,
	const float voxelSize,
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
		const int vidx = (voxelCountX * voxelCountY * z) + (voxelCountX * y) + x;
		// sdfs[vidx] = INFINITY;

		// x = 5, y = 5, z = 5
		// (5.5, 5.5, 5.5) * 0.01 = (0.055, 0.055, 0.055) 
		float3 voxelCenterWorldCoord = make_float3((x + .5f) * voxelSize, (y + .5f) * voxelSize, (z + .5f) * voxelSize);
		voxelCenterWorldCoord.x -= 3.5f;
		voxelCenterWorldCoord.y -= 3.5f;
		voxelCenterWorldCoord.z -= 3.5f;

		// SJ: Transform to camera coordinate system
		float3 voxelCenterCameraCoord = mul(worldToCamera, &voxelCenterWorldCoord);

		// SJ: If voxel is behind the camera ignore and continue.
		if (voxelCenterCameraCoord.z <= 0)
			continue;

		// SJ: Project voxel position to camera image plane.
		const int u = (int)(voxelCenterCameraCoord.x / voxelCenterCameraCoord.z * fovX + fovCenterX);
		const int v = (int)(voxelCenterCameraCoord.y / voxelCenterCameraCoord.z * fovY + fovCenterY);

		// SJ: If voxel position is not on the image plane continue.
		if (u < 0 || u >= frameWidth || v < 0 || v >= frameHeight)
			continue;

		const int FRAME_IDX = v * frameWidth + u;
		const float depth = gpuFrameDepthMap[FRAME_IDX];

		// SJ: Check if we have a valid depth for that pixel coordinate.
		if (isinf(depth) || depth <= 0)
			continue;

		// SJ: Paper formula (7)
		// const float lambda = l2norm(make_float3((u - fovCenterX) / fovX, (v - fovCenterY) / fovY, 1.0f));

		// SJ: Paper formula (6), calculate how far away the voxel center is from the actual depth of the backprojected pixel (our surface)
		// const float voxelCenterDistanceToCamera = l2norm(voxelCenterCameraCoord);
		// const float signedDistance = (-1.0f) * (voxelCenterDistanceToCamera / lambda) - depth;
		const float signedDistance = depth - voxelCenterCameraCoord.z;

		// SJ: If we are outside the truncation range continue.
		if (fabsf(signedDistance) > truncation)
			continue;

		// SJ: Paper formula (9), normalizing the signed distance to [-1,1]
		// const float normalizedSignedDistance = fminf(1.f, signedDistance / truncation);
		const float tsdf = signedDistance;

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

		float w = fminf(weights[VOXEL_IDX] + W_R, maxWeight);
		weights[VOXEL_IDX] = w;

		float s = (weights[VOXEL_IDX] * sdfs[VOXEL_IDX]) + (w * tsdf) / (weights[VOXEL_IDX] + w);
		sdfs[VOXEL_IDX] = s;

		BYTE r = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 0]) + (w * gpuFrameColorMap[FRAME_COLOR_IDX + 0]) / (weights[VOXEL_IDX] + w));
		BYTE g = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 1]) + (w * gpuFrameColorMap[FRAME_COLOR_IDX + 1]) / (weights[VOXEL_IDX] + w));
		BYTE b = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 2]) + (w * gpuFrameColorMap[FRAME_COLOR_IDX + 2]) / (weights[VOXEL_IDX] + w));
		colors[VOXEL_COLOR_IDX + 0] = r;
		colors[VOXEL_COLOR_IDX + 1] = g;
		colors[VOXEL_COLOR_IDX + 2] = b;
		colors[VOXEL_COLOR_IDX + 3] = 255;

	}
}

__global__
void applyTsdf_v3
(
	float* gpuFrameDepthMap,
	BYTE* const gpuFrameColorMap,
	const float* worldToCamera,
	const float* cameraToWorld,
	float* sdfs,
	float* weights,
	BYTE* colors,
	const int frameWidth,
	const int frameHeight,
	const int voxelCountX,
	const int voxelCountY,
	const int voxelCountZ,
	const float voxelSize,
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
		voxelCenterWorldCoord.x -= 3.5f;
		voxelCenterWorldCoord.y -= 3.5f;
		voxelCenterWorldCoord.z -= 3.5f;

		// SJ: Transform to camera coordinate system
		float3 voxelCenterCameraCoord = mul(worldToCamera, &voxelCenterWorldCoord);

		// SJ: If voxel is behind the camera ignore and continue.
		if (voxelCenterCameraCoord.z <= 0)
			continue;

		// SJ: Project voxel position to camera image plane.
		const int u = (int)(voxelCenterCameraCoord.x / voxelCenterCameraCoord.z * fovX + fovCenterX);
		const int v = (int)(voxelCenterCameraCoord.y / voxelCenterCameraCoord.z * fovY + fovCenterY);

		// SJ: If voxel position is not on the image plane continue.
		if (u < 0 || u >= frameWidth || v < 0 || v >= frameHeight)
			continue;

		const int FRAME_IDX = v * frameWidth + u;
		const float depth = gpuFrameDepthMap[FRAME_IDX];

		// SJ: Check if we have a valid depth for that pixel coordinate.
		if (isinf(depth) || depth <= 0)
			continue;

		float3 surfacePointCameraCoord = make_float3((x - fovCenterX) / fovX, (y - fovCenterY) / fovY, 1.0f);
		surfacePointCameraCoord.x *= depth;
		surfacePointCameraCoord.y *= depth;
		surfacePointCameraCoord.z *= depth;

		float3 surfacePointWorldCoord = mul(cameraToWorld, &surfacePointCameraCoord);

		// const float3 cameraTranslation = make_float3(worldToCamera[12], worldToCamera[13], worldToCamera[14]);
		const float3 cameraTranslation = make_float3(cameraToWorld[12], cameraToWorld[13], cameraToWorld[14]);
		const float distanceCameraToVoxel = l2norm(make_float3
		(
			cameraTranslation.x - voxelCenterWorldCoord.x,
			cameraTranslation.y - voxelCenterWorldCoord.y,
			cameraTranslation.z - voxelCenterWorldCoord.z
		));

		const float distanceCameraToSurface = l2norm(make_float3
		(
			cameraTranslation.x - surfacePointWorldCoord.x,
			cameraTranslation.y - surfacePointWorldCoord.y,
			cameraTranslation.z - surfacePointWorldCoord.z
		));

		const float signedDistance = (-1) * (distanceCameraToVoxel - distanceCameraToSurface);
		// const float signedDistance = (-1)*(distanceCameraToVoxel - depth);

		// SJ: If we are outside the truncation range continue.
		if (fabsf(signedDistance) > truncation)
			continue;

		float tsdf = fminf(1.0f, signedDistance / truncation);

		const int VOXEL_IDX = (voxelCountX * voxelCountY * z) + (voxelCountX * y) + x;
		const int VOXEL_COLOR_IDX = VOXEL_IDX * 4;
		const int FRAME_COLOR_IDX = FRAME_IDX * 4;
		const float W_R = 1.0f;

		float w = fminf(weights[VOXEL_IDX] + W_R, maxWeight);
		weights[VOXEL_IDX] = w;

		float s = (weights[VOXEL_IDX] * sdfs[VOXEL_IDX]) + (w * tsdf) / (weights[VOXEL_IDX] + w);
		sdfs[VOXEL_IDX] = s;

		/*BYTE r = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 0]) + (w * gpuFrameColorMap[FRAME_COLOR_IDX + 0]) / (weights[VOXEL_IDX] + w));
		BYTE g = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 1]) + (w * gpuFrameColorMap[FRAME_COLOR_IDX + 1]) / (weights[VOXEL_IDX] + w));
		BYTE b = (BYTE)((weights[VOXEL_IDX] * colors[VOXEL_COLOR_IDX + 2]) + (w * gpuFrameColorMap[FRAME_COLOR_IDX + 2]) / (weights[VOXEL_IDX] + w));*/

		colors[VOXEL_COLOR_IDX + 0] = 255;
		colors[VOXEL_COLOR_IDX + 1] = 0;
		colors[VOXEL_COLOR_IDX + 2] = 0;
		colors[VOXEL_COLOR_IDX + 3] = 255;

	}
}