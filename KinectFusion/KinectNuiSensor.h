#pragma once

// #define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// #define _WIN32_WINNT 0x0601             // Compile against Windows 7 headers

#include <iostream>
#include <MsHTML.h>
#include <NuiApi.h>

#include <memory>
// #include <windows.h>
// #include <Shlobj.h>

#include "Sensor.h"

class KinectNuiSensor: public Sensor
{
public:

	const UINT m_frameWidth;
	const UINT m_frameHeight;
	const bool m_isNearMode;

	KinectNuiSensor(const UINT frameWidth, const UINT frameHeight, const bool isNearMode = false)
		: m_frameWidth{frameWidth}
		, m_frameHeight{frameHeight}
		, m_isNearMode {isNearMode}
	{}

	~KinectNuiSensor()
	{
		Destroy();
	}

	bool Init()
	{
		int numKinects = 0;
		HRESULT hr = NuiGetSensorCount(&numKinects);

		if (FAILED(hr) || numKinects <= 0)
			return false;

		hr = NuiCreateSensorByIndex(0, &context);

		if (FAILED(hr))
			return false;

		DWORD nuiFlags = NUI_INITIALIZE_FLAG_USES_DEPTH |
			NUI_INITIALIZE_FLAG_USES_COLOR;

		hr = context->NuiInitialize(nuiFlags);

		if (FAILED(hr))
			return false;

		hr = context->NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH, NUI_IMAGE_RESOLUTION_640x480, 0, 2, NULL, &depthStreamHandle);

		if (FAILED(hr))
			return false;

		hr = context->NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, NULL, &colorStreamHandle);

		if (FAILED(hr))
			return false;

		hr = context->NuiImageStreamSetImageFrameFlags(depthStreamHandle, m_isNearMode ? NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE : 0);

		if (FAILED(hr))
			return false;

		const size_t FRAME_DEPTH_SIZE = m_frameWidth * m_frameHeight * sizeof(float);
		const size_t FRAME_COLOR_SIZE = m_frameWidth * m_frameHeight * sizeof(BYTE) * 4;
		const size_t FRAME_MAP_SIZE = m_frameWidth * m_frameHeight * sizeof(long) * 2;

		//m_depthFrame = (float*)malloc(FRAME_DEPTH_SIZE);
		//m_colorFrame = (BYTE*)malloc(FRAME_COLOR_SIZE);

		m_depthFrame.reset(new float[FRAME_DEPTH_SIZE]);
		m_colorFrame.reset(new BYTE[FRAME_COLOR_SIZE]);
		m_depthToRGBMap.reset(new long[FRAME_MAP_SIZE]);

		return true;
	}

	bool Init(const std::string& datasetDir)
	{
		throw "Not Implemented";
	}

	bool ProcessNextFrame_v1()
	{
		HRESULT hr;

		NUI_IMAGE_FRAME depthFrame;
		hr = context->NuiImageStreamGetNextFrame(depthStreamHandle, 33, &depthFrame);

		if (SUCCEEDED(hr))
		{
			updateDepthFrame(depthFrame);
			context->NuiImageStreamReleaseFrame(depthStreamHandle, &depthFrame);
		}

		NUI_IMAGE_FRAME colorFrame;
		hr = context->NuiImageStreamGetNextFrame(colorStreamHandle, 33, &colorFrame);

		if (SUCCEEDED(hr))
		{
			updateImageFrame(colorFrame);
			context->NuiImageStreamReleaseFrame(colorStreamHandle, &colorFrame);
		}

		return SUCCEEDED(hr);
	}

	bool ProcessNextFrame_v2()
	{
		HRESULT hr;
		bool hasDepth = false;
		bool hasRgb = false;

		std::cout << "[SENSOR]: Start grabing Frame." << std::endl;

		while (!hasDepth || !hasRgb)
		{
			NUI_IMAGE_FRAME depthFrame;

			if (!hasDepth)
			{
				hr = context->NuiImageStreamGetNextFrame(depthStreamHandle, 20, &depthFrame);

				if (SUCCEEDED(hr))
				{
					updateDepthFrame(depthFrame);
					context->NuiImageStreamReleaseFrame(depthStreamHandle, &depthFrame);
					hasDepth = true;
				}
			}

			if (!hasRgb)
			{
				NUI_IMAGE_FRAME colorFrame;
				hr = context->NuiImageStreamGetNextFrame(colorStreamHandle, 20, &colorFrame);

				if (SUCCEEDED(hr))
				{
					updateImageFrame(colorFrame);
					context->NuiImageStreamReleaseFrame(colorStreamHandle, &colorFrame);
					hasRgb = true;
				}
			}
		}

		std::cout << "[SENSOR]: Done grabing Frame." << std::endl;

		return true;
	}

	bool ProcessNextFrame()
	{
		if (!context)
			return false;

		HRESULT hr;

		NUI_IMAGE_FRAME depthFrame;
		hr = context->NuiImageStreamGetNextFrame(depthStreamHandle, 0, &depthFrame);

		if (!SUCCEEDED(hr))
		{
			std::cout << "[SENSOR]: No depth frame." << std::endl;
			return false;
		}

		updateDepthFrame(depthFrame);
		context->NuiImageStreamReleaseFrame(depthStreamHandle, &depthFrame);

		NUI_IMAGE_FRAME colorFrame;
		hr = context->NuiImageStreamGetNextFrame(colorStreamHandle, 0, &colorFrame);

		if (!SUCCEEDED(hr))
		{
			std::cout << "[SENSOR]: No color frame." << std::endl;
			return false;
		}

		updateImageFrame(colorFrame);
		context->NuiImageStreamReleaseFrame(colorStreamHandle, &colorFrame);

		return true;
	}

	float* GetDepth()
	{	
		return m_depthFrame.get();
	}

	BYTE* GetColorRGBX()
	{
		return m_colorFrame.get();
	}

	void Destroy()
	{
		std::cout << "[SENSOR] Destroying" << std::endl;

		if (context)
		{
			context->NuiShutdown();
			context->Release();
			context = nullptr;
		}
	}

	float GetFX()
	{
		return 525.0f;
	}

	float GetFY()
	{
		return 525.0f;
	}

	float GetCX()
	{
		return 319.5f;
	}

	float GetCY()
	{
		return 239.5f;
	}

	//~KinectNuiSensor()
	//{
	//	delete[] m_depthFrame;
	//	delete[] m_colorFrame;
	//}

private:
	void updateImageFrame(NUI_IMAGE_FRAME& imageFrame)
	{
		BYTE* dst = m_colorFrame.get();
		long* depthToRGB = m_depthToRGBMap.get();

		INuiFrameTexture* nuiTexture = imageFrame.pFrameTexture;
		NUI_LOCKED_RECT lockedRect;

		nuiTexture->LockRect(0, &lockedRect, NULL, 0);

		if (lockedRect.Pitch != NULL)
		{
			const BYTE* start = (const BYTE*)lockedRect.pBits;

			for (int j = 0; j < m_frameHeight; ++j)
			{
				for (int i = 0; i < m_frameWidth; ++i)
				{
					// Determine aligned rgb color for each depth pixel
					long x = *depthToRGB++;
					long y = *depthToRGB++;

					// If out of bounds, then don't color it at all
					if (x < 0 || y < 0 || x > m_frameWidth || y > m_frameHeight)
					{
						*(dst++) = 0;
						*(dst++) = 0;
						*(dst++) = 0;
						*(dst++) = 0;
					}
					else
					{
						const BYTE* src = start + (x + m_frameWidth * y) * 4;

						// dst = rgba, src = bgaa
						*(dst++) = *(src + 2);
						*(dst++) = *(src + 1);
						*(dst++) = *(src + 0);
						*(dst++) = 255;
					}
				}
			}
		}

		nuiTexture->UnlockRect(0);
	}

	void updateDepthFrame(NUI_IMAGE_FRAME& depthFrame)
	{
		float* dst = m_depthFrame.get();
		long* depthToRGB = m_depthToRGBMap.get();

		INuiFrameTexture* nuiTexture = depthFrame.pFrameTexture;
		NUI_LOCKED_RECT lockedRect;

		nuiTexture->LockRect(0, &lockedRect, NULL, 0);

		if (lockedRect.Pitch != NULL)
		{
			const USHORT* curr = (const USHORT*)lockedRect.pBits;
			//USHORT maxDepth = 0;
			//USHORT minDepth = 5000;

			for (int j = 0; j < m_frameHeight; ++j)
			{
				for (int i = 0; i < m_frameWidth; ++i)
				{
					// Get depth of pixel in millimeters
					const size_t IDX = i + m_frameWidth * j;
					USHORT depth = NuiDepthPixelToDepth(curr[IDX]);

					//if (depth > maxDepth)
					//	maxDepth = depth;

					//if (depth < minDepth)
					//	minDepth = depth;

					if (depth == 0)
					{
						dst[IDX] = -std::numeric_limits<float>::infinity();
					}
					else
					{
						dst[IDX] = depth / 1000.0f;
					}

					// Store the index into the color array corresponding to this pixel
					NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(
						NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, NULL,
						i, j, depth << 3, depthToRGB, depthToRGB + 1);

					depthToRGB += 2;
				}
			}

			// std::cout << "maxDepth: " << maxDepth << " minDepth: " << minDepth << std::endl;
		}

		nuiTexture->UnlockRect(0);
	}


	HANDLE colorStreamHandle = nullptr;
	HANDLE depthStreamHandle = nullptr;
	//float* m_depthFrame;
	//BYTE* m_colorFrame;
	std::unique_ptr<float[]> m_depthFrame;
	std::unique_ptr<BYTE[]> m_colorFrame;
	std::unique_ptr<long[]> m_depthToRGBMap;

	INuiSensor* context = nullptr;
};