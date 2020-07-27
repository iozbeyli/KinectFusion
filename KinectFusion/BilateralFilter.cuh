#pragma once
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __forceinline__
float getGaussian(float input, float sigmaPart,float expDenominator) {
	float expPart = expf(-input / expDenominator);
	return sigmaPart * expPart;
}

__global__
void applyBilateralFilter(
	float* outputMap, float* inputMap, int imageWidth, int imageHeight, float sigmaSpatial, float sigmaRange, int filterHalfSize,
	bool filterNormals, float* inputNormals, float* outputNormals, bool* validMask)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= imageWidth) || (y >= imageHeight)) {
		return;
	}

	float result = 0.0f;
	float weightSum = 0.0f;
	float middle = inputMap[y * imageWidth + x];
	float sigmaPartSpatial = 1 / (2 * M_PI * sigmaSpatial * sigmaSpatial);
	float sigmaPartRange = 1 / (2 * M_PI * sigmaRange * sigmaRange);
	float expDenominatorSpatial = 2 * sigmaSpatial * sigmaSpatial;
	float expDenominatorRange = 2 * sigmaRange * sigmaRange;

	// Normal related stuff
	float3 resultNormal = make_float3(0, 0, 0);
	float weightSumNormal = 0; // make_float3(0, 0, 0);
	int indexNormal = 3 * (y * imageWidth + x);
	float3 centerNormal;
	//filterNormals = false;
	if (filterNormals) {
		centerNormal = make_float3(
			inputNormals[indexNormal],
			inputNormals[indexNormal + 1],
			inputNormals[indexNormal + 2]
		);
	}

	for (int i = -filterHalfSize; i <= filterHalfSize; ++i) {
		for (int j = -filterHalfSize; j <= filterHalfSize; ++j) {
			int row = y + i;
			int column = x + j;

			if ((row >= 0) && (column >= 0) && (row < imageHeight) && (column < imageWidth)) {
				float value = inputMap[row * imageWidth + column];
				if (isfinite(value)) {
					float weight = getGaussian(j*j + i*i, sigmaPartSpatial, expDenominatorSpatial)
						* getGaussian((middle - value) * (middle - value), sigmaPartRange, expDenominatorRange);
					result += weight * value;
					weightSum += weight;

					if (filterNormals) {
						int index = 3 * (row * imageWidth + column);
						float3 valueNormal = make_float3(
							inputNormals[index], inputNormals[index+1], inputNormals[index + 2]
						);
						if (isfinite(valueNormal.x) && isfinite(valueNormal.y) && isfinite(valueNormal.z)) {
							resultNormal.x += weight * valueNormal.x;
							resultNormal.y += weight * valueNormal.y;
							resultNormal.z += weight * valueNormal.z;
							weightSumNormal += weight;
						}
					}
				}
			}
		}
	}
	if(weightSum>0){
		outputMap[y * imageWidth + x] = result / weightSum;
	}
	else {
		outputMap[y * imageWidth + x] = -INFINITY;
	}

	if (filterNormals) {
		if (weightSumNormal > 0) {
			outputNormals[indexNormal] = resultNormal.x / weightSumNormal;
			outputNormals[indexNormal + 1] = resultNormal.y / weightSumNormal;
			outputNormals[indexNormal + 2] = resultNormal.z / weightSumNormal;
			validMask[indexNormal / 3] = true;
		}
		else {
			outputNormals[indexNormal] = -INFINITY;
			outputNormals[indexNormal + 1] = -INFINITY;
			outputNormals[indexNormal + 2] = -INFINITY;
			validMask[indexNormal / 3] = false;
		}
	}
	
}

__global__
void subSample(
	float* output, float* input, int inputWidth, int inputHeight, float sigmaRange,
	bool doNormals, float* outputNormals, float* inputNormals, bool *validMask )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (((2 * x) >= inputWidth) || ((2 * y) >= inputHeight)) {
		return;
	}
	int filterHalfSize = ceilf(3 * sigmaRange);
	int validCount = 0;
	float result = 0;

	float3 resultNormal = make_float3(0, 0, 0);
	int validCountNormal = 0;
	//doNormals = false;

	for (int i = -filterHalfSize; i <= filterHalfSize; ++i) {
		for (int j = -filterHalfSize; j <= filterHalfSize; ++j) {
			int inputRow = (2 * y) + i;
			int inputColumn = (2 * x) + j;

			if ((inputRow >= 0) && (inputColumn >= 0) && (inputRow < inputHeight) && (inputColumn < inputWidth)) {
				float value = input[(inputRow * inputWidth) + inputColumn];
				if (isfinite(value)) {
					result += value;
					validCount += 1;
				}

				if (doNormals) {
					int index = 3 * (inputRow * inputWidth + inputColumn);
					float3 valueNormal = make_float3(
						inputNormals[index], inputNormals[index + 1], inputNormals[index + 2]
					);
					if (isfinite(valueNormal.x) && isfinite(valueNormal.y) && isfinite(valueNormal.z)) {
						resultNormal.x += valueNormal.x;
						resultNormal.y += valueNormal.y;
						resultNormal.z += valueNormal.z;
						validCountNormal += 1;
					}
				}
			}
		}
	}

	if (validCount > 0) {
		output[(y * (inputWidth / 2)) + x] = result/validCount;
	}
	else {
		output[y * (inputWidth / 2) + x] = -INFINITY;
	}

	int indexNormal = 3 * (y * (inputWidth / 2) + x);
	if (doNormals) {
		if (validCountNormal > 0) {
			outputNormals[indexNormal] = resultNormal.x / validCountNormal;
			outputNormals[indexNormal + 1] = resultNormal.y / validCountNormal;
			outputNormals[indexNormal + 2] = resultNormal.z / validCountNormal;
			validMask[indexNormal / 3] = true;
		}
		else {
			outputNormals[indexNormal] = -INFINITY;
			outputNormals[indexNormal + 1] = -INFINITY;
			outputNormals[indexNormal + 2] = -INFINITY;
			validMask[indexNormal / 3] = false;
		}
	}
}

class Filterer 
{
public: 
	Filterer(int width, int height, bool allocateInput, bool useNormals=false) 
	{
		m_width = width;
		m_height = height;
		m_size = width * height * sizeof(float);
		if (allocateInput)
		{
			m_cudaStatusInput = cudaMalloc((void**)&m_inputMap, m_size);
		}
		else 
		{
			m_cudaStatusInput = cudaSuccess;
		}
		m_cudaStatusOutput = cudaMalloc((void**)&m_outputMap, m_size);
		m_cudaStatusOutputFirstLevel = cudaMalloc((void**)&m_outputFirstLevelMap, m_size/4);
		m_cudaStatusOutputSecondLevel = cudaMalloc((void**)&m_outputSecondLevelMap, m_size/16);
		m_outputMapCPU = (float*)malloc(m_size);
		m_outputFirstLevelMapCPU = (float*)malloc(m_size/4);
		m_outputSecondLevelMapCPU = (float*)malloc(m_size/16);

		m_filterNormals = useNormals;
		createNormals();
	}

	~Filterer() 
	{
		if (m_ownInput) cudaFree(m_inputMap);
		cudaFree(m_outputMap);
		cudaFree(m_outputFirstLevelMap);
		cudaFree(m_outputSecondLevelMap);
		free(m_outputMapCPU);
		free(m_outputFirstLevelMapCPU);
		free(m_outputSecondLevelMapCPU);
		deleteNormals();
	}
	
	bool isOK() 
	{

		if (m_normalsOk && m_cudaStatusInput != cudaSuccess && m_cudaStatusOutput != cudaSuccess && m_cudaStatusOutputFirstLevel != cudaSuccess && m_cudaStatusOutputSecondLevel != cudaSuccess)
		{
			return false;
		}
		return true;
	}

	bool applyFilter(float* input, float* normals=nullptr) 
	{
		cudaError_t cudaStatus = cudaMemcpy(m_inputMap, input, m_size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) 
		{
			return false;
		}
		dim3 gridSize(m_width / 16, m_height / 16);
		dim3 blockSize(16, 16);

		int filterHalfSize = 5;
		float sigmaSpatial = 1.6f;
		float sigmaRange = 1.6f;

		bool useNormals = m_filterNormals && (normals != nullptr);

		applyBilateralFilter<<<gridSize, blockSize>>>(
			m_outputMap, m_inputMap, m_width, m_height,sigmaSpatial,sigmaRange,filterHalfSize,
			useNormals, m_normals, normals, m_validMask
		);

		gridSize = dim3(m_width / 32, m_height / 32);
		subSample <<<gridSize, blockSize>>> (
			m_outputFirstLevelMap, m_outputMap, m_width, m_height, sigmaRange,
			useNormals, m_normals, m_normalsFirstLevel, m_validMaskFirstLevel
		);

		gridSize = dim3(m_width / 32, m_height / 32);
		blockSize = dim3(8, 8);
		subSample <<<gridSize, blockSize>>> (
			m_outputSecondLevelMap, m_outputFirstLevelMap, m_width / 2, m_height / 2, sigmaRange,
			useNormals, m_normalsFirstLevel, m_normalsSecondLevel, m_validMaskSecondLevel
		);
		return true;
	}

	bool applyFilterGPU(float* input, float* normals=nullptr)
	{
		m_ownInput = false;
		dim3 gridSize(m_width / 16, m_height / 16);
		dim3 blockSize(16, 16);

		int filterHalfSize = 5;
		float sigmaSpatial = 1.6f;
		float sigmaRange = 1.6f;

		bool useNormals = m_filterNormals && (normals != nullptr);
		applyBilateralFilter <<<gridSize, blockSize>>> (
			m_outputMap, input, m_width, m_height, sigmaSpatial, sigmaRange, filterHalfSize,
			useNormals, m_normals, normals, m_validMask
		);
		//m_outputMap = input;
		gridSize = dim3(m_width / 32, m_height / 32);
		subSample<<<gridSize, blockSize>>> (
			m_outputFirstLevelMap, m_outputMap, m_width, m_height, sigmaRange,
			useNormals, m_normalsFirstLevel, m_normals, m_validMaskFirstLevel
		);

		gridSize = dim3(m_width / 32, m_height / 32);
		blockSize = dim3(8, 8);
		subSample<<<gridSize, blockSize >>> (
			m_outputSecondLevelMap, m_outputFirstLevelMap, m_width / 2, m_height / 2, sigmaRange,
			useNormals, m_normalsSecondLevel, m_normalsFirstLevel, m_validMaskSecondLevel
		);

		return true;
	}

	bool copyToCPU() 
	{
		cudaError_t cudaStatus = cudaMemcpy(m_outputMapCPU, m_outputMap, m_size, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusFirstLevel = cudaMemcpy(m_outputFirstLevelMapCPU, m_outputFirstLevelMap, m_size/4, cudaMemcpyDeviceToHost);
		cudaError_t cudaStatusSecondLevel = cudaMemcpy(m_outputSecondLevelMapCPU, m_outputSecondLevelMap, m_size/16, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess || cudaStatusFirstLevel != cudaSuccess || cudaStatusSecondLevel != cudaSuccess)
		{
			return false;
		}
		return true;
	}

	float* getOutputCPU(int level) {
		if (level == 1) 
		{
			return m_outputFirstLevelMapCPU;
		} 
		if (level == 2)
		{
			return m_outputSecondLevelMapCPU;
		}
		return m_outputMapCPU;
	}

	float* getOutputGPU(int level)
	{
		if (level == 1)
		{
			return m_outputFirstLevelMap;
		}
		if (level == 2)
		{
			return m_outputSecondLevelMap;
		}
		return m_outputMap;
	}

	float* getInputGPU()
	{
		return m_inputMap;
	}

	float* getNormalsGPU(int level)
	{
		if (level == 1)
		{
			return m_normalsFirstLevel;
		}
		if (level == 2)
		{
			return m_normalsSecondLevel;
		}
		return m_normals;
	}

	bool* getValidMaskGPU(int level)
	{
		if (level == 1)
		{
			return m_validMaskFirstLevel;
		}
		if (level == 2)
		{
			return m_validMaskSecondLevel;
		}
		return m_validMask;
	}

private:
	float* m_inputMap = nullptr;
	float* m_outputMap;
	float* m_outputFirstLevelMap;
	float* m_outputSecondLevelMap;
	cudaError_t m_cudaStatusInput;
	cudaError_t m_cudaStatusOutput;
	cudaError_t m_cudaStatusOutputFirstLevel;
	cudaError_t m_cudaStatusOutputSecondLevel;
	size_t m_size;
	int m_width;
	int m_height;
	float* m_outputMapCPU;
	float* m_outputFirstLevelMapCPU;
	float* m_outputSecondLevelMapCPU;

	bool m_ownInput = true;

	// Normal filtering is necessary for model-to-frame tracking
	bool m_filterNormals = false;
	float* m_normals = nullptr;
	float* m_normalsFirstLevel = nullptr;
	float* m_normalsSecondLevel = nullptr;

	bool* m_validMask = nullptr;
	bool* m_validMaskFirstLevel = nullptr;
	bool* m_validMaskSecondLevel = nullptr;

	bool m_normalsOk = true;

#define CHECK_NORMALS_FILTER if (status != cudaSuccess) { m_normalsOk = false; return; }

	void createNormals()
	{
		if (!m_filterNormals) return;

		const auto size = m_size * 3;
		const auto boolSize = sizeof(bool) * (m_size / sizeof(float));
		auto status = cudaMalloc((void**)&m_normals, size);
		CHECK_NORMALS_FILTER
		status = cudaMalloc((void**)&m_normalsFirstLevel, size / 4);
		CHECK_NORMALS_FILTER
		status = cudaMalloc((void**)&m_normalsSecondLevel, size / 16);
		CHECK_NORMALS_FILTER

		status = cudaMalloc((void**)&m_validMask, boolSize);
		CHECK_NORMALS_FILTER
		status = cudaMalloc((void**)&m_validMaskFirstLevel, boolSize / 4);
		CHECK_NORMALS_FILTER
		status = cudaMalloc((void**)&m_validMaskSecondLevel, boolSize / 16);
		CHECK_NORMALS_FILTER
	}

	void deleteNormals()
	{
		if (!m_filterNormals) return;
		cudaFree(m_normals);
		cudaFree(m_normalsFirstLevel);
		cudaFree(m_normalsSecondLevel);
		cudaFree(m_validMask);
		cudaFree(m_validMaskFirstLevel);
		cudaFree(m_validMaskSecondLevel);
	}
};
