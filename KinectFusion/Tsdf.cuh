#pragma once

#include <minwindef.h>
#include "cuda_runtime.h"
#include "Eigen.h"
#include "TsdfVolume.cuh"

#define X_OFFSET 4
#define Y_OFFSET 3.5
#define Z_OFFSET 1.5 
#define NORMAL_WEIGHT 0


__device__ __forceinline__ 
float l2norm(const float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ 
float3 mul(const float* m4f, const float3* v)
{
	return make_float3
	(
		v->x * m4f[0] + v->y * m4f[4] + v->z *  m4f[8] + m4f[12],
		v->x * m4f[1] + v->y * m4f[5] + v->z *  m4f[9] + m4f[13],
		v->x * m4f[2] + v->y * m4f[6] + v->z * m4f[10] + m4f[14]
	);
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

__device__ __forceinline__
float3 voxelToWorld(int3 index, float voxelSize)
{
	float3 coord = make_float3((index.x + .5f) * voxelSize, (index.y + .5f) * voxelSize, (index.z + .5f) * voxelSize);
	coord.x -= X_OFFSET;
	coord.y -= Y_OFFSET;
	coord.z -= Z_OFFSET;
	return coord;
}

__global__
void applyTsdf_v4
(
	float* gpuFrameDepthMap,
	bool* gpuFrameValidMask,
	float* gpuFrameNormals,
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
		float3 voxelCenterWorldCoord = voxelToWorld(make_int3(x, y, z), voxelSize);
		/*make_float3((x + .5f) * voxelSize, (y + .5f) * voxelSize, (z + .5f) * voxelSize);
		voxelCenterWorldCoord.x -= X_OFFSET;
		voxelCenterWorldCoord.y -= Y_OFFSET;
		voxelCenterWorldCoord.z -= Z_OFFSET;*/

		// SJ: Transform to camera coordinate system
		float3 voxelCenterCameraCoord = mul(worldToCamera, &voxelCenterWorldCoord);
		//float3 voxelCenterCameraCoord = matrot(worldToCamera, &voxelCenterWorldCoord);
		//voxelCenterCameraCoord.x += worldToCamera[12];
		//voxelCenterCameraCoord.y += worldToCamera[13];
		//voxelCenterCameraCoord.z += worldToCamera[14];

		// SJ: If voxel is behind the camera ignore and continue.
		if (voxelCenterCameraCoord.z <= 0)
			continue;

		// SJ: Project voxel position to camera image plane.
		const int u = static_cast<int>(round(voxelCenterCameraCoord.x / voxelCenterCameraCoord.z * fovX + fovCenterX));
		const int v = static_cast<int>(round(voxelCenterCameraCoord.y / voxelCenterCameraCoord.z * fovY + fovCenterY));

		// SJ: If voxel position is not on the image plane continue.
		if (u < 0 || u >= frameWidth || v < 0 || v >= frameHeight)
			continue;

		const int FRAME_IDX = v * frameWidth + u;
		if (!gpuFrameValidMask[FRAME_IDX])
			continue;

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

		const float lambda = l2norm(make_float3((u - fovCenterX) / fovX, (v - fovCenterY) / fovY, 1.0f));
		const float signedDistance = (-1.f) * ((1.f / lambda) * l2norm(voxelCenterCameraCoord) - depth);
		// const float signedDistance = (-1) * (distanceCameraToVoxel - distanceCameraToSurface);
		// const float signedDistance = (surfacePointCameraCoord.z - voxelCenterCameraCoord.z);
		// const float signedDistance = (-1)*(distanceCameraToVoxel - depth);

		// SJ: If we are outside the truncation range continue.
		if (fabsf(signedDistance) > truncation)
			continue;

		float tsdf = fminf(1.0f, signedDistance / truncation);

		//float tsdf = signedDistance > 0.0f
		//	? fminf(1.0f, signedDistance / truncation)
		//	: signedDistance;

		const int VOXEL_IDX = (voxelCountX * voxelCountY * z) + (voxelCountX * y) + x;
		const int VOXEL_COLOR_IDX = VOXEL_IDX * 4;
		const int FRAME_COLOR_IDX = FRAME_IDX * 4;
		const int FRAME_NORMAL_IDX = FRAME_IDX * 3;
		float W_R = 1.0f;
		
#if NORMAL_WEIGHT
		float voxelNorm =  l2norm(voxelCenterCameraCoord);
		if (voxelNorm >= 0.001)
		{
			//weight accoring to normals
			float3 normal = make_float3(
				gpuFrameNormals[FRAME_NORMAL_IDX],
				gpuFrameNormals[FRAME_NORMAL_IDX + 1],
				gpuFrameNormals[FRAME_NORMAL_IDX + 2]);
			float3 ray = make_float3(
				voxelCenterCameraCoord.x / voxelNorm, 
				voxelCenterCameraCoord.y / voxelNorm, 
				voxelCenterCameraCoord.z / voxelNorm);
			// TODO: rename dot, mul etc. so we can use cuda functons with float3
			W_R = fmaxf(
				fminf(dot(ray.x, ray.y, ray.z, normal.x, normal.y, normal.z), 0.1),
				fminf(dot(ray.x, ray.y, ray.z, -normal.x, -normal.y, -normal.z), 0.1)
			);
		}
#endif

		float s = (weights[VOXEL_IDX] * sdfs[VOXEL_IDX]) + (W_R * tsdf) / (weights[VOXEL_IDX] + W_R);
		if (isinf(s)) {
			continue;
		}
		sdfs[VOXEL_IDX] = s;

		float r = ((weights[VOXEL_IDX] * (float)colors[VOXEL_COLOR_IDX + 0]) + (W_R * (float)gpuFrameColorMap[FRAME_COLOR_IDX + 0])) / (weights[VOXEL_IDX] + W_R);
		float g = ((weights[VOXEL_IDX] * (float)colors[VOXEL_COLOR_IDX + 1]) + (W_R * (float)gpuFrameColorMap[FRAME_COLOR_IDX + 1])) / (weights[VOXEL_IDX] + W_R);
		float b = ((weights[VOXEL_IDX] * (float)colors[VOXEL_COLOR_IDX + 2]) + (W_R * (float)gpuFrameColorMap[FRAME_COLOR_IDX + 2])) / (weights[VOXEL_IDX] + W_R);
		colors[VOXEL_COLOR_IDX + 0] = r; // gpuFrameColorMap[FRAME_COLOR_IDX + 0];
		colors[VOXEL_COLOR_IDX + 1] = g; // gpuFrameColorMap[FRAME_COLOR_IDX + 1];
		colors[VOXEL_COLOR_IDX + 2] = b; // gpuFrameColorMap[FRAME_COLOR_IDX + 2];
		colors[VOXEL_COLOR_IDX + 3] = 255;

		float w = fminf(weights[VOXEL_IDX] + W_R, maxWeight);
		weights[VOXEL_IDX] = w;

	}
}
	
class Tsdf
{
public:
	const size_t COLOR_MAP_SIZE;

	Tsdf(int imageWidth, int imageHeight) : COLOR_MAP_SIZE{ imageWidth * imageHeight * 4 * sizeof(BYTE) }
	{

		cudaStatus = cudaMalloc((void**)&gpuFrameColorMap, COLOR_MAP_SIZE);
		if (cudaStatus != cudaSuccess)
		{
			return;
		}

		cudaStatus = cudaMalloc((void**)&gpuCameraToWorld, 16 * sizeof(float));
		if (cudaStatus != cudaSuccess)
		{
			return;
		}

		cudaStatus = cudaMalloc((void**)&gpuWorldToCamera, 16 * sizeof(float));

		if (cudaStatus != cudaSuccess)
		{
			return;
		}
	}

	bool apply(TsdfVolume& volume, float* gpuFrameDepthMap, bool* gpuFrameValidMask, float* gpuFrameNormals, const BYTE* cpuFrameColorMap, Eigen::Matrix4f& frameCameraToWorld)
	{
		// SJ: color map is not in GPU memory yet, so let's put it there.


		//std::cout << frameCameraToWorld << std::endl;
		//std::cout << frameCameraToWorld.block(0, 3, 3, 1) << std::endl;
		//frameCameraToWorld.block(0, 3, 3, 1) = frameCameraToWorld.block(0, 3, 3, 1) * (-5);
		//std::cout << frameCameraToWorld << std::endl;
		cudaStatus = cudaMemcpy(gpuFrameColorMap, cpuFrameColorMap, COLOR_MAP_SIZE, cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess)
			return false;
		cudaStatus = cudaMemcpy(gpuCameraToWorld, frameCameraToWorld.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess)
			return false;

		Eigen::Matrix4f worldToCamera = frameCameraToWorld.inverse();

		// std::cout << worldToCamera << std::endl;

		cudaStatus = cudaMemcpy(gpuWorldToCamera, worldToCamera.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess)
			return false;

		// SJ: Max numbers of threads per block is limited to 1024, so lets make full use of it.
		dim3 threads(32, 32);

		// SJ: (N + x - 1) / x gives us the smallest multiple of x greater or equal to N.
		// SJ: So we have one thread per row in z-direction to calculate the voxel values.
		dim3 blocks((volume.VOXEL_COUNT_X + threads.x - 1) / threads.x,
			(volume.VOXEL_COUNT_Y + threads.y - 1) / threads.y);

		applyTsdf_v4 << <threads, blocks >> > (
			gpuFrameDepthMap,
			gpuFrameValidMask,
			gpuFrameNormals,
			gpuFrameColorMap,
			gpuWorldToCamera,
			gpuCameraToWorld,
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

		//cudaThreadSynchronize();

		//cudaFree(gpuFrameColorMap);
		//cudaFree(gpuWorldToCamera);
		//cudaFree(gpuCameraToWorld);

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

	~Tsdf()
	{
		cudaFree(gpuFrameColorMap);
		cudaFree(gpuWorldToCamera);
		cudaFree(gpuCameraToWorld);
	}

private:
	BYTE* gpuFrameColorMap=nullptr;
	float* gpuWorldToCamera=nullptr;
	float* gpuCameraToWorld=nullptr;
	bool ok = true;
	cudaError_t cudaStatus;
};