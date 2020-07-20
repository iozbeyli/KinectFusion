#pragma once

#include <minwindef.h>
#include "cuda_runtime.h"
#include "Eigen.h"
#include "Volume.cuh"
#include "GlobalModel.cuh"


__device__
float getRayToVolumeIntersectionScale(float3* maxRayPos, float3* rayOrigin, float3* rayDirection)
{
	// Raycast AABB (Axis Aligned Bounding Box)
	// Source[1]: http://people.csail.mit.edu/amy/papers/box-jgt.pdf
	// Source[2]: (Gabor Szauer) Game Physics Cookbook p. 204-209 

	float txmin = ((rayDirection->x >= 0.0f ? 0.0f : maxRayPos->x) - rayOrigin->x) / rayDirection->x;
	float txmax = ((rayDirection->x >= 0.0f ? maxRayPos->x : 0.0f) - rayOrigin->x) / rayDirection->x;

	float tymin = ((rayDirection->x >= 0.0f ? 0.0f : maxRayPos->y) - rayOrigin->y) / rayDirection->y;
	float tymax = ((rayDirection->y >= 0.0f ? maxRayPos->y : 0.0f) - rayOrigin->y) / rayDirection->y;

	float tzmin = ((rayDirection->z >= 0.0f ? 0.0f : maxRayPos->z) - rayOrigin->z) / rayDirection->z;
	float tzmax = ((rayDirection->z >= 0.0f ? maxRayPos->z : 0.0f) - rayOrigin->z) / rayDirection->z;

	float tmin = fmaxf(fmaxf(fminf(txmin, txmax), fminf(tymin, tymax)), fminf(tzmin, tzmax));
	float tmax = fminf(fminf(fmaxf(txmin, txmax), fmaxf(tymin, tymax)), fmaxf(tzmin, tzmax));

	// SJ: If tmax < 0, the volume is behind the origin of the ray.
	if (tmax < 0)
		return -1;

	// SJ: If tmin > tmax, the ray does not intersect AABB.
	if (tmin > tmax)
		return -1;

	// SJ: If tmin < 0, origin of ray is inside AABB.
	if (tmin < 0.0f)
		return tmax;

	//SJ: First ray intersection scale.
	return tmin;
}

__device__ __forceinline__
float3 normalize(const float3 v)
{
	float l = l2norm(v);	
	return make_float3(v.x / l, v.y / l, v.z / l);
}

__device__ __forceinline__
float3 matrot(const float* m4f, const float3* v)
{
	return make_float3
	(
		v->x * m4f[0] + v->y * m4f[4] + v->z * m4f[8],
		v->x * m4f[1] + v->y * m4f[5] + v->z * m4f[9],
		v->x * m4f[2] + v->y * m4f[6] + v->z * m4f[10]
	);
}


__global__
void applyRaycaster
(
	BYTE* gpuColorMap,
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
	const float truncation
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// SJ: Idle GPU thread, if we are outside the bounds of our volume.
	if ((x >= frameWidth) || (y >= frameHeight))
		return;

	// SJ: Max ray expansion in each direction.
	float3 rayMaxPos = make_float3(voxelCountX * voxelSize, voxelCountY * voxelSize, voxelCountZ * voxelSize);

	// SJ: Back project current pixel onto a world plane with depth 1.0, so we can calculate a direction.
	// SJ: Calculate ray direction
	float3 pixelWorldPlaneCoord = make_float3((x - fovCenterX) / fovX, (y - fovCenterY) / fovY, 1.0f);
	float3 rotatedPixelWorldPlaneCoord = matrot(cameraToWorld, &pixelWorldPlaneCoord);
	float3 rayDirection = normalize(rotatedPixelWorldPlaneCoord);

	if (isinf(rayDirection.x) || isinf(rayDirection.y) || isinf(rayDirection.z))
		return;

	// SJ: Calculate ray direction and scale it by half the truncation so we have a ray step that we can add to the lates ray position
	const float rayStepScale = truncation * 0.5f;
	float3 rayStep = make_float3(rayDirection.x * rayStepScale, rayDirection.y * rayStepScale, rayDirection.z * rayStepScale);

	// SJ: We calculate the max number of steps by first calculating the max volume diagonal for each volume direction (diagonal is about sqrt(2.0) of a cubes side).
	// SJ: The diagonal gives us the longest distance to traverse the volume cube, which we then divide by the RAY_STEP for each dimension.
	float3 maxRayStepsPerAxis
	{
		rayMaxPos.x * sqrtf(2.0) / rayStep.x,
		rayMaxPos.y * sqrtf(2.0) / rayStep.y,
		rayMaxPos.z * sqrtf(2.0) / rayStep.z
	};

	// SJ: Check for the largest amount of ray steps in each dimension and use it as the max ray steps.
	UINT maxRaySteps = static_cast<UINT>(fmaxf(fmaxf(maxRayStepsPerAxis.x, maxRayStepsPerAxis.y), maxRayStepsPerAxis.z));

	// SJ: Determine how much we have to scale the ray from its current origin so that it intersects the volume bounds.
	float3 rayOrigin = make_float3(cameraToWorld[12], cameraToWorld[13], cameraToWorld[14]);
	float rayDistanceToVolume = getRayToVolumeIntersectionScale(&rayMaxPos, &rayOrigin, &rayDirection);
	rayDistanceToVolume += voxelSize;

	// SJ: If ray does not intersect with the volume
	if (rayDistanceToVolume <= 0.0f || isinf(rayDistanceToVolume))
		return;

	// SJ: Ray start position
	float3 rayPos = make_float3
	(
		rayOrigin.x + rayDirection.x * rayDistanceToVolume, 
		rayOrigin.y + rayDirection.y * rayDistanceToVolume, 
		rayOrigin.z + rayDirection.z * rayDistanceToVolume
	);

	// SJ: Find Voxel coordinates
	int3 voxelCoords = make_int3(rayPos.x / voxelSize, rayPos.y / voxelSize, rayPos.z / voxelSize);

	if (voxelCoords.x >= voxelCountX || voxelCoords.x < 0 ||
		voxelCoords.y >= voxelCountY || voxelCoords.y < 0 ||
		voxelCoords.z >= voxelCountZ || voxelCoords.z < 0)
		return;

	const UINT VOXEL_IDX = (voxelCountX * voxelCountY * voxelCoords.z) + (voxelCountX * voxelCoords.y) + voxelCoords.x;

	float sdf = sdfs[VOXEL_IDX];

	UINT stepCount = maxRaySteps;

	// SJ: Do max steps until we reach a zero crossing
	while (stepCount > 0)
	{
		// SJ: Decrease numbers of ray steps
		--stepCount;

		// SJ: Step into ray direction
		rayPos.x += rayStep.x;
		rayPos.y += rayStep.y;
		rayPos.z += rayStep.z;

		// SJ: We are ouside the volume
		if (rayPos.x > rayMaxPos.x || rayPos.x < 0 ||
			rayPos.y > rayMaxPos.y || rayPos.y < 0 ||
			rayPos.z > rayMaxPos.z || rayPos.z < 0)
			break;

		// SJ: Get Voxel coordinates from current ray position.
		int3 voxelCoords = make_int3(rayPos.x / voxelSize, rayPos.y / voxelSize, rayPos.z / voxelSize);

		// SJ: Double check if the coordiantes are valid.
		if (voxelCoords.x >= voxelCountX || voxelCoords.x < 0 ||
			voxelCoords.y >= voxelCountY || voxelCoords.y < 0 ||
			voxelCoords.z >= voxelCountZ || voxelCoords.z < 0)
			continue;

		// SJ: Get the voxel sdf from the voxel coordiantes, and check if we had a zero crosssing.
		const UINT VOXEL_IDX = (voxelCountX * voxelCountY * voxelCoords.z) + (voxelCountX * voxelCoords.y) + voxelCoords.x;
		const float newSdf = sdfs[VOXEL_IDX];

		// SJ: Ooops! from inside to outside 
		if (newSdf > 0.0f && sdf < 0.0f)
			break;

		// SJ: Zero crossing, set color, normals and vertices to global model.
		if (newSdf < 0.0f && sdf > 0.0f)
		{
			// SJ: Set volume color to global model and return
			const UINT FRAME_COLOR_IDX = (y * frameWidth + x) * 4;
			const UINT VOXEL_COLOR_IDX = VOXEL_IDX * 4;
			gpuColorMap[FRAME_COLOR_IDX + 0] = colors[VOXEL_COLOR_IDX + 0];
			gpuColorMap[FRAME_COLOR_IDX + 1] = colors[VOXEL_COLOR_IDX + 1];
			gpuColorMap[FRAME_COLOR_IDX + 2] = colors[VOXEL_COLOR_IDX + 2];
			gpuColorMap[FRAME_COLOR_IDX + 3] = 255;

			// SJ: TODO: set vertices and normals.
			return;
		}

		sdf = newSdf;
	}
}


class Raycaster
{
public:
	bool apply(Volume& volume, GlobalModel& model, Eigen::Matrix4f frameCameraToWorld)
	{
		if (!model.reset())
		{
			cudaStatus = model.status();
			return false;
		}

		cudaStatus = cudaMalloc((void**)&gpuMatrix, 16 * sizeof(float));

		if (cudaStatus != cudaSuccess)
			return false;

		cudaStatus = cudaMemcpy(gpuMatrix, frameCameraToWorld.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess)
			return false;

		dim3 threads(32, 32);

		dim3 blocks((model.FRAME_WIDTH + threads.x - 1) / threads.x,
					(model.FRAME_HEIGHT + threads.y - 1) / threads.y);

		applyRaycaster<<<threads, blocks>>>(
				model.colorMapGPU,
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
				volume.TRUNCATION
			);

		cudaThreadSynchronize();

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
	float* gpuMatrix;
	cudaError_t cudaStatus;
};