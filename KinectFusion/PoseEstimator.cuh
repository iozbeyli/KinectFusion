#pragma once
#include <math.h>
#include "Eigen.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "malloc.h"

// TODO: Refactor
#define FX 525.0f
#define FY 525.0f
#define CX 319.5f
#define CY 239.5f

__device__ __forceinline__ void setZero(float* outputA, float* outputB, int indexA, int indexB)
{
	outputB[indexB] = 0;
#pragma unroll
	for (int i = 0; i < 6; ++i)
	{
		outputA[indexA + i] = 0;
	}
	return;
}

__device__ __forceinline__ float l2Diff(float x1, float y1, float z1, 
										float x2, float y2, float z2)
{
	float d1 = (x1 - x2);
	float d2 = (y1 - y2);
	float d3 = (z1 - z2);
	return sqrtf(d1 * d1 + d2 * d2 + d3 * d3);
}

__device__ __forceinline__ float dot(float x1, float y1, float z1,
									 float x2, float y2, float z2)
{
	return x1 * x2 + z1 * z2 + y1 * y2;
}

__global__
void fillMatrix(float* outputA, float* outputB,
				float* sourcePoints, float* targetPoints,
				float* sourceNormals, float* targetNormals,
				bool* validMask,
				float rX, float rY, float rZ,
				float tX, float tY, float tZ,
				int width, int height,
				float distanceThreshold, float normalThreshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= width) || (y >= height))
	{
		return;
	}

	// Define indices
	int indexWithoutChannel = width * y + x;
	int indexA = 6 * indexWithoutChannel;
	int indexB = indexWithoutChannel;
	int indexPoints = 3 * indexWithoutChannel;
	int indexNormals = indexPoints;

	// Set 0-0 for invalids
	if (!validMask[indexWithoutChannel])
	{
		setZero(outputA, outputB, indexA, indexB);
		return;
	}

	// Transform source to target
	float sX = sourcePoints[indexPoints];
	float sY = sourcePoints[indexPoints + 1];
	float sZ = sourcePoints[indexPoints + 2];
	float pX = sX - rZ * sY + rY * sZ + tX;
	float pY = rZ * sX + sY - rZ * sZ + tY;
	float pZ = -rY * sX + rX * sY + sZ + tZ;
	if (pZ < 0.1)
	{
		setZero(outputA, outputB, indexA, indexB);
		return;
	}

	// Project source to target's image plane
	int col = roundf(FX * pX / pZ + CX);
	int row = roundf(FY * pY / pZ + CY);
	if (col < 0 && col >= width && row < 0 && row >= height)
	{
		setZero(outputA, outputB, indexA, indexB);
		return;
	}

	// Get the destination
	int indexTarget = 3 * (row * width + col);
	float dX = targetPoints[indexTarget];
	float dY = targetPoints[indexTarget + 1];
	float dZ = targetPoints[indexTarget + 2];

	// Perform distance threshold
	if (l2Diff(pX, pY, pZ, dX, dY, dZ) > distanceThreshold)
	{
		setZero(outputA, outputB, indexA, indexB);
		return;
	}

	// Get the normals
	float nX = targetNormals[indexNormals];
	float nY = targetNormals[indexNormals + 1];
	float nZ = targetNormals[indexNormals + 2];

	// Perform normal threshold (TODO)
	
	// Write values to A and B
	outputA[indexA] = nZ * sY - nY * sZ;
	outputA[indexA + 1] = nX * sZ - nZ * sX;
	outputA[indexA + 2] = nY * sX - nX * sY;
	outputA[indexA + 3] = nX;
	outputA[indexA + 4] = nY;
	outputA[indexA + 5] = nZ;

	outputB[indexB] = dot(nX, nY, nZ, dX, dY, dZ) - dot(nX, nY, nZ, sX, sY, sZ);
}

#define CUCHECK					\
	if (status != cudaSuccess)  \
	{							\
		m_ok = false;			\
		return;					\
	}

class PoseEstimator
{
public:

	PoseEstimator(int width, int height, int niter)
	{
		m_width = width;
		m_height = height;
		m_iter = niter;
		auto size = m_height * m_width * sizeof(float);

		cudaError_t status;
		status = cudaMalloc(&m_A, 6 * size);
		CUCHECK(status);
		status = cudaMalloc(&m_B, size);
		CUCHECK(status);
		status = cudaMalloc(&m_AtA, 36 * sizeof(float));
		CUCHECK(status);
		status = cudaMalloc(&m_AtB, 6 * sizeof(float));
		CUCHECK(status);

		s_instances++;
		if (s_instances == 1)
		{
			if (cublasCreate_v2(&s_handle) == CUBLAS_STATUS_SUCCESS)
			{
				m_ok = false;
				return;
			}
		}
	}

	void setParams(Vector<float, 6> params)
	{
		m_rX = params(0);
		m_rY = params(1);
		m_rZ = params(2);
		m_tX = params(3);
		m_tY = params(4);
		m_tZ = params(5);
	}

	auto getParamVector()
	{
		return Vector<float, 6>(m_rX, m_rY, m_rZ, m_tX, m_tY, m_tZ);
	}

	void apply(float* depthPoints, float* normals)
	{
		for (int i = 0; i < m_iter; ++i)
		{

		}
	}

	bool isOk()
	{
		return m_ok;
	}

	~PoseEstimator()
	{
		cudaFree(m_A);
		cudaFree(m_B);
		cudaFree(m_AtA);
		cudaFree(m_AtB);
		s_instances--;
		if (s_instances == 0)
		{
			cublasDestroy(s_handle);
		}
	}

private:
	static int s_instances;
	static cublasHandle_t s_handle;

	bool matMul()
	{
		cudaError memsetStatus;
		memsetStatus = cudaMemset(&m_AtA, 0, 36 * sizeof(float));
		if (memsetStatus != cudaSuccess)
		{
			return false;
		}
		memsetStatus = cudaMemset(&m_AtB, 0, 6 * sizeof(float));
		if (memsetStatus != cudaSuccess)
		{
			return false;
		}

		cublasStatus_t status;
		int m = 6;
		int n = m_width * m_height;
		int k = 6;
		int lda = m, ldb = k, ldc = m;
		const float alf = 1;
		const float bet = 0;
		const float* alpha = &alf;
		const float* beta = &bet;
		status = cublasSgemm(
			s_handle, CUBLAS_OP_T, CUBLAS_OP_N,
			m, n, k,
			alpha, m_A, lda, m_A, ldb, beta, m_AtA, ldc);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			return false;
		}

		k = 1;
		ldb = k;
		status = cublasSgemm(
			s_handle, CUBLAS_OP_T, CUBLAS_OP_N,
			m, n, k,
			alpha, m_A, lda, m_B, ldb, beta, m_AtB, ldc);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			return false;
		}

		return true;
	}

	bool solve()
	{
		Matrix<float, 6, 6> Gmat;
		Vector<float, 6> bvec;
		auto stat0 = cudaMemcpy(Gmat.data(), m_AtA, 36 * sizeof(float), cudaMemcpyDeviceToHost);
		auto stat1 = cudaMemcpy(bvec.data(), m_AtB, 6 * sizeof(float), cudaMemcpyDeviceToHost);
		if (stat0 != cudaSuccess || stat1 != cudaSuccess)
		{
			return false;
		}
		// TODO: solve
		return true;
	}

	int m_width;
	int m_height;
	bool m_ok = true;
	int m_iter;

	float* m_A;
	float* m_B;
	float* m_AtA;
	float* m_AtB;

	float m_rX = 0;
	float m_rY = 0;
	float m_rZ = 0;
	float m_tX = 0;
	float m_tY = 0;
	float m_tZ = 0;
};


int PoseEstimator::s_instances = 0;