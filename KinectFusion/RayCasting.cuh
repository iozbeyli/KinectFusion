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
	index.x = (point.x + 3.8) / voxelSize - .5;
	index.y = (point.y + 3.5) / voxelSize - .5;
	index.z = point.z / voxelSize - .5;
	return index;
}

__global__ void rayCast(
	float* depth, // TODO: add normals
	float* normal,
	float* sdf,
	float* weights,
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
	// Send the ray
	for (int i = 0; i < maxIteration; ++i)
	{
		// Move the ray
		takeStep(&point, scaled(step, stepSize));
		// Calculate the voxel index
		index=voxelIndex(point, voxelSize);
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
		const int voxelId = (voxelX * voxelY * index.z) + (voxelX * index.y) + index.x;
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

		if (fabsf(stepSize) < 0.05f)
		{
			success = true;
			break;
		}
	}
	int indexImage = y * width + x;
	int voxelIdRight = (voxelX * voxelY * index.z) + (voxelX * index.y) + index.x + 1;
	int voxelIdTop = (voxelX * voxelY * index.z) + (voxelX * (index.y + 1)) + index.x;
	int voxelIdBehind = (voxelX * voxelY * (index.z + 1)) + (voxelX * index.y) + index.x;
	if (success && weights[voxelIdRight] > 0.1 && weights[voxelIdTop] > 0.1 && weights[voxelIdBehind] > 0.1)
	{
		float right = sdf[voxelIdRight] - sdfValue;
		float top = sdf[voxelIdTop] - sdfValue;
		float behind = sdf[voxelIdBehind] - sdfValue;
		float3 beforeNormalization = make_float3(right, top, behind);
		float3 normalizedCamera = rot(w2c, &beforeNormalization);
		float magnitude = l2norm(normalizedCamera);
		float3 normalized = scaled(normalizedCamera, 1.0f / magnitude);
		depth[indexImage] = mul(w2c, &point).z;
		normal[3 * indexImage] = normalized.x;
		normal[3 * indexImage + 1] = normalized.y;
		normal[3 * indexImage + 2] = normalized.z;
	}
	else {
		depth[indexImage] = 10.0f;
		normal[3 * indexImage] = 1.0f;
		normal[3 * indexImage + 1] = 1.0f;
		normal[3 * indexImage + 2] = 1.0f;
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
	}

	bool isOk() { return m_ok; }

	bool apply(float* sdf, float *weights, Matrix4f cameraToWorld)
	{
		auto status = cudaMemcpy(m_c2w, cameraToWorld.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
		if (status != cudaSuccess) return false;
		
		Matrix4f worldToCamera = cameraToWorld.inverse();
		status = cudaMemcpy(m_w2c, worldToCamera.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
		if (status != cudaSuccess) return false;

		dim3 gridSize(m_width / 16, m_height / 16);
		dim3 blockSize(16, 16);

		rayCast<<<gridSize, blockSize>>>(m_depth,m_normal, sdf, weights, m_c2w, m_w2c, m_truncation, m_voxelSize, m_voxelCount, m_voxelCount, m_voxelCount, m_width, m_height);
		
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


	~RayCaster()
	{
		free(m_depthCpu);
		cudaFree(m_depth);
		free(m_normalCpu);
		cudaFree(m_normal);
	}

private:
	int m_width;
	int m_height;

	float* m_depth;
	float* m_depthCpu;
	
	float* m_normal;
	float* m_normalCpu;

	float* m_c2w;
	float* m_w2c;
	bool m_ok = true;
	float m_truncation;
	UINT m_voxelCount;
	float m_voxelSize;

	size_t m_size;

};
