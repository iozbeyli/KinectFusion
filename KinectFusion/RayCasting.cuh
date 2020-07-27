#pragma once
#include <malloc.h>

#include "Eigen.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Tsdf.cuh" // TODO: refactor helpers

__device__ __forceinline__
float3 rot(const float* m4f, const float3* v)
{
	return make_float3
	(
		v->x * m4f[0] + v->y * m4f[4] + v->z * m4f[8],
		v->x * m4f[1] + v->y * m4f[5] + v->z * m4f[9],
		v->x * m4f[2] + v->y * m4f[6] + v->z * m4f[10]
	);
}

__device__ __forceinline__
float3 scaled(const float3 vec, float scale)
{
	return make_float3(vec.x * scale, vec.y * scale, vec.z * scale);
}

__device__ __forceinline__
void takeStep(float3* vec, const float3 step)
{
	vec->x += step.x;
	vec->y += step.y;
	vec->z += step.z;
}

__device__ __forceinline__
int3 voxelIndex(const float3 point, float voxelSize)
{
	int3 index;
	index.x = (point.x + X_OFFSET) / voxelSize - .5;
	index.y = (point.y + Y_OFFSET) / voxelSize - .5;
	index.z = (point.z + Z_OFFSET) / voxelSize - .5;
	return index;
}

__device__ __forceinline__
float3 voxelPoint(const float3 point, float voxelSize)
{
	float3 index;
	index.x = (point.x + X_OFFSET) / voxelSize - .5;
	index.y = (point.y + Y_OFFSET) / voxelSize - .5;
	index.z = (point.z + Z_OFFSET) / voxelSize - .5;
	return index;
}

__device__ __forceinline__
int voxelIndex1d(int3 index, int3 voxelDims)
{
	return (voxelDims.x * voxelDims.y * index.z) + (voxelDims.y * index.y) + index.x;
}

__device__ __forceinline__
int3 fitIndex(int3 index, int3 voxelDims)
{
	if (index.x < 0) index.x = 0;
	if (index.y < 0) index.y = 0;
	if (index.z < 0) index.z = 0;

	if (index.x >= voxelDims.x) index.x = voxelDims.x - 1;
	if (index.y >= voxelDims.y) index.y = voxelDims.y - 1;
	if (index.z >= voxelDims.z) index.z = voxelDims.z - 1;

	return index;
}


// See https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/trilinear-interpolation
// Assumes the provided index is inside the grid boundaries
__device__ __forceinline__
float interpolate3(float *grid, float* weights, float3 point, int3 index, int3 voxelDims, float voxelSize)
{
	// Get the indices
	int index000 = voxelIndex1d(index, voxelDims);
	int index100 = voxelIndex1d(fitIndex(make_int3(index.x + 1, index.y, index.z), voxelDims), voxelDims);
	int index010 = voxelIndex1d(fitIndex(make_int3(index.x, index.y + 1, index.z), voxelDims), voxelDims);
	int index001 = voxelIndex1d(fitIndex(make_int3(index.x, index.y, index.z + 1), voxelDims), voxelDims);
	int index110 = voxelIndex1d(fitIndex(make_int3(index.x + 1, index.y + 1, index.z), voxelDims), voxelDims);
	int index101 = voxelIndex1d(fitIndex(make_int3(index.x + 1, index.y, index.z + 1), voxelDims), voxelDims);
	int index011 = voxelIndex1d(fitIndex(make_int3(index.x, index.y + 1, index.z + 1), voxelDims), voxelDims);
	int index111 = voxelIndex1d(fitIndex(make_int3(index.x + 1, index.y + 1, index.z + 1), voxelDims), voxelDims);
	
	// Set the interpolation params
	float3 pointIndex = voxelPoint(point, voxelSize);
	float tx = pointIndex.x - index.x;
	float ty = pointIndex.y - index.y;
	float tz = pointIndex.z - index.z;

	// Return the value (Again, assume 000 is inside boundaries)
	float value = grid[index000];
	// This should not happen with the current call
	if (weights[index000] < 0.1)
	{
		return -INFINITY;
	}

#define SAFE_GET(indexNear) ((weights[indexNear] >= 0.1) ? grid[indexNear] : value)
	return 
		(1 - tx) * (1 - ty) * (1 - tz) * value +
		tx * (1 - ty) * (1 - tz) * SAFE_GET(index100) +
		(1 - tx) * ty * (1 - tz) * SAFE_GET(index010) +
		(1 - tx) * ty * (1 - tz) * SAFE_GET(index001) +
		tx * ty * (1 - tz) * SAFE_GET(index110) +
		tx * (1 - ty) * tz * SAFE_GET(index101) +
		(1 - tx) * ty * tz * SAFE_GET(index011) +
		tx * ty * tz * SAFE_GET(index111);
}

__global__ void rayCast(
	float* depth,
	float* normal,
	BYTE* color,
	float* sdf,
	float* weights,
	BYTE* sdfColor,
	float* c2w,
	float* w2c,
	float truncation,
	float voxelSize,
	int voxelX, int voxelY, int voxelZ,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// SJ: Idle gpu thread, if we are outside the bounds of our volume.
	if ((x >= width) || (y >= height))
		return;

	
	// calculate ray in camera space
	float3 ray;
	ray.x = x - width / 2;
	ray.y = y - height / 2; //ascending to bottom
	const float tanfovd2 = 0.5 * (float)height / 525.0f;
	ray.z = (0.5 * (float)height) / tanfovd2; // ascending outward
	const float l = l2norm(ray);
	ray.x /= l;
	ray.y /= l;
	ray.z /= l;
	
	// Transform ray to world space
	const float3 step = rot(c2w, &ray);

	// Set ray iteration parameters
	float3 point = make_float3(c2w[12], c2w[13], c2w[14]);
	float stepSize = 0.4;
	int maxIteration = sqrtf(voxelX * voxelX + voxelY * voxelY + voxelZ * voxelZ);
	bool success = false;
	bool negative = false;
	int3 index;
	float sdfValue;
	int voxelId;
	// Send the ray
	for (int i = 0; i < maxIteration; ++i)
	{
		// Move the ray
		takeStep(&point, scaled(step, stepSize));
		// Calculate the voxel index
		index = voxelIndex(point, voxelSize);
		// Exit if goes out or on boundary for normals
		if (index.x >= (voxelX-1) || index.x < 0 ||
			index.y >= (voxelY-1) || index.y < 0 ||
			index.z >= (voxelZ-1) || index.z < 0)
		{
			break;
		}
		// Take the last sdf value
		// if weight is zero or inf, set step to truncation distance
		// otherwise, set step to distance
		voxelId = (voxelX * voxelY * index.z) + (voxelX * index.y) + index.x;
		if (weights[voxelId] < 0.1)
		{
			stepSize = truncation;
			continue;
		}
		sdfValue = sdf[voxelId];
		
		if (i > 0 && ((sdfValue > 0 && negative) || (sdfValue < 0 && !negative)))
		{
			stepSize = - stepSize * 0.6;
		}

		negative = sdfValue < 0;

		if (fabsf(stepSize) < voxelSize)
		{
			success = true;
			break;
		}
	}
	int indexImage = y * width + x;

	int3 voxelDims = make_int3(voxelX, voxelY, voxelZ);
	
	int3 indexRight = make_int3(index.x + 1, index.y, index.z);
	int voxelIdRight = voxelIndex1d(indexRight, voxelDims);
	
	int3 indexTop = make_int3(index.x, index.y + 1, index.z);
	int voxelIdTop = voxelIndex1d(indexTop, voxelDims);

	int3 indexBehind = make_int3(index.x, index.y, index.z + 1);
	int voxelIdBehind = voxelIndex1d(indexBehind, voxelDims);

	if (success && weights[voxelIdRight] > 0.1 && weights[voxelIdTop] > 0.1 && weights[voxelIdBehind] > 0.1)
	{
		float current = interpolate3(sdf, weights, point, index, voxelDims, voxelSize);
		float right = interpolate3(sdf, weights, voxelToWorld(indexRight, voxelSize), indexRight, voxelDims, voxelSize);
		float top = interpolate3(sdf, weights, voxelToWorld(indexTop, voxelSize), indexTop, voxelDims, voxelSize);
		float behind = interpolate3(sdf, weights, voxelToWorld(indexBehind, voxelSize), indexBehind, voxelDims, voxelSize);

		float3 beforeNormalization = make_float3(current-right, current-top, current-behind);
		float3 normalizedCamera = rot(w2c, &beforeNormalization);
		float magnitude = l2norm(normalizedCamera);
		float3 normalized = scaled(normalizedCamera, 1.0f / magnitude);
		normal[3 * indexImage] = -normalized.x;
		normal[3 * indexImage + 1] = -normalized.y;
		normal[3 * indexImage + 2] = -normalized.z;
	}
	
	// Fill depth and color even without normals
	if (success)
	{
		depth[indexImage] = mul(w2c, &point).z;
		color[3 * indexImage] = sdfColor[4 * voxelId];
		color[3 * indexImage + 1] = sdfColor[4 * voxelId + 1];
		color[3 * indexImage + 2] = sdfColor[4 * voxelId + 2];
	}
	else 
	{
		depth[indexImage] = -INFINITY; // 10.0f;
		normal[3 * indexImage] = -INFINITY; //1.0f;
		normal[3 * indexImage + 1] = -INFINITY; //-1.0f;
		normal[3 * indexImage + 2] = -INFINITY; //1.0f;
		color[3 * indexImage] = 0;
		color[3 * indexImage + 1] = 0;
		color[3 * indexImage + 2] = 0;
	}

}

class RayCaster
{
public:
	RayCaster(int width, int height, const float truncation, const UINT voxelCount, const float voxelSize)
	{
		m_width = width;
		m_height = height;
		m_truncation = truncation;
		m_voxelCount = voxelCount;
		m_voxelSize = voxelSize;
		m_size = width * height * sizeof(float);
		auto status = cudaMalloc((void**)&m_normal, 3*m_size);
		m_normalCpu = (float*)malloc(3*m_size);
		if (status != cudaSuccess)
		{
			m_ok = false;
			return;
		}
		status = cudaMalloc((void**)&m_depth, m_size);
		m_depthCpu = (float*)malloc(m_size);
		if (status != cudaSuccess)
		{
			m_ok = false;
			return;
		}
		status = cudaMalloc((void**)&m_c2w, 16 * sizeof(float));
		if (status != cudaSuccess)
		{
			m_ok = false;
			return;
		}
		status = cudaMalloc((void**)&m_w2c, 16 * sizeof(float));
		if (status != cudaSuccess)
		{
			m_ok = false;
			return;
		}

		m_colorCpu = (BYTE*)malloc(width * height * 3);
		status = cudaMalloc((void**)&m_color, width * height * 3);
		if (status != cudaSuccess)
		{
			m_ok = false;
			return;
		}
		
	}

	bool isOk() { return m_ok; }

	bool apply(float* sdf, float *weights, BYTE *sdfColor, Matrix4f cameraToWorld)
	{
		auto status = cudaMemcpy(m_c2w, cameraToWorld.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
		if (status != cudaSuccess) return false;
		
		Matrix4f worldToCamera = cameraToWorld.inverse();
		status = cudaMemcpy(m_w2c, worldToCamera.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
		if (status != cudaSuccess) return false;

		dim3 gridSize(m_width / 16, m_height / 16);
		dim3 blockSize(16, 16);

		rayCast<<<gridSize, blockSize>>>(m_depth,m_normal,m_color, sdf, weights, sdfColor, m_c2w, m_w2c, m_truncation, m_voxelSize, m_voxelCount, m_voxelCount, m_voxelCount, m_width, m_height);
		
		return true;
	}
	
	bool copyToCPU()
	{
		cudaError_t cudaStatus = cudaMemcpy(m_depthCpu, m_depth, m_size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			return false;
		}
		cudaStatus = cudaMemcpy(m_normalCpu, m_normal, 3*m_size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			return false;
		}
		cudaStatus = cudaMemcpy(m_colorCpu, m_color, 3 * m_width * m_height, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			return false;
		}
		return true;
	}

	float* getOutputDepthCPU() 
	{
		return m_depthCpu;
	}

	float* getOutputDepthGPU() 
	{
		return m_depth;
	}

	float* getOutputNormalCPU()
	{
		return m_normalCpu;
	}

	float* getOutputNormalGPU()
	{
		return m_normal;
	}

	BYTE* getOutputColorGPU()
	{
		return m_color;
	}

	BYTE* getOutputColorCPU()
	{
		return m_colorCpu;
	}

	~RayCaster()
	{
		free(m_depthCpu);
		cudaFree(m_depth);
		free(m_normalCpu);
		cudaFree(m_normal);
		free(m_colorCpu);
		cudaFree(m_color);
	}

private:
	int m_width;
	int m_height;

	float* m_depth;
	float* m_depthCpu;
	
	float* m_normal;
	float* m_normalCpu;

	BYTE* m_color;
	BYTE* m_colorCpu;

	float* m_c2w;
	float* m_w2c;
	bool m_ok = true;
	float m_truncation;
	UINT m_voxelCount;
	float m_voxelSize;

	size_t m_size;

};