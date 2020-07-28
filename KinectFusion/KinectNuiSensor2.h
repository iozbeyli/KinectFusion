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

class KinectNuiSensor2: public Sensor
{
public:

	const UINT m_frameWidth;
	const UINT m_frameHeight;
	const bool m_isNearMode;

	KinectNuiSensor2(const UINT frameWidth, const UINT frameHeight, const bool isNearMode = false)
		: m_frameWidth{frameWidth}
		, m_frameHeight{frameHeight}
		, m_isNearMode {isNearMode}
	{
		m_hNextDepthFrameEvent = CreateEvent(
			nullptr,
			TRUE,	/* bManualReset - KinectSDK will reset this internally */
			FALSE,	/* bInitialState */
			nullptr);

		m_hNextColorFrameEvent = CreateEvent(
			nullptr,
			TRUE,	/* bManualReset - KinectSDK will reset this internally */
			FALSE,	/* bInitialState */
			nullptr);
	}

	~KinectNuiSensor2()
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

		hr = context->NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH, NUI_IMAGE_RESOLUTION_640x480, 0, 2, m_hNextDepthFrameEvent, &depthStreamHandle);

		if (FAILED(hr))
			return false;

		hr = context->NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, m_hNextColorFrameEvent, &colorStreamHandle);

		if (FAILED(hr))
			return false;

		if (m_isNearMode)
		{
			hr = context->NuiImageStreamSetImageFrameFlags(depthStreamHandle, NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE);

			if (FAILED(hr))
				return false;
		}

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

	bool ProcessNextFrame()
	{
		if (WaitForSingleObject(m_hNextDepthFrameEvent, 0) == WAIT_OBJECT_0)
		{
			processDepth();
			return true;
		}

		return false;
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

		if (m_hNextDepthFrameEvent != INVALID_HANDLE_VALUE)
			CloseHandle(m_hNextDepthFrameEvent);

		if (m_hNextColorFrameEvent != INVALID_HANDLE_VALUE)
			CloseHandle(m_hNextColorFrameEvent);

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

private:

	void copyColor(NUI_IMAGE_FRAME& imageFrame)
	{
		BYTE* dst = m_colorFrame.get();
		long* depthToRGB = m_depthToRGBMap.get();

		INuiFrameTexture* nuiTexture = imageFrame.pFrameTexture;
		NUI_LOCKED_RECT lockedRect;

		nuiTexture->LockRect(0, &lockedRect, NULL, 0);

		if (lockedRect.Pitch != NULL)
		{
			const BYTE* src = (const BYTE*)lockedRect.pBits;

			for (int j = 0; j < m_frameHeight; ++j)
			{
				for (int i = 0; i < m_frameWidth; ++i)
				{
					// Determine aligned rgb color for each depth pixel
					long x = *depthToRGB++;
					long y = *depthToRGB++;

					const size_t IDX = (y * m_frameWidth + x) * 4;
					const size_t IDX_REV = (j * m_frameWidth + (m_frameWidth - i)) * 4;

					// If out of bounds, then don't color it at all
					if (x < 0 || y < 0 || x > m_frameWidth || y > m_frameHeight)
					{
						dst[IDX_REV + 0] = 0;
						dst[IDX_REV + 1] = 0;
						dst[IDX_REV + 2] = 0;
						dst[IDX_REV + 3] = 0;
					}
					else
					{
						// dst = rgba, src = bgaa
						dst[IDX_REV + 0] = src[IDX + 2];
						dst[IDX_REV + 1] = src[IDX + 1];
						dst[IDX_REV + 2] = src[IDX + 0];
						dst[IDX_REV + 3] = 255;
					}
				}
			}
		}

		nuiTexture->UnlockRect(0);
	}

	void copyDepth(NUI_IMAGE_FRAME& depthFrame)
	{
		float* dst = m_depthFrame.get();
		long* depthToRGB = m_depthToRGBMap.get();

		INuiFrameTexture* nuiTexture = depthFrame.pFrameTexture;
		NUI_LOCKED_RECT lockedRect;

		nuiTexture->LockRect(0, &lockedRect, NULL, 0);

		if (lockedRect.Pitch != NULL)
		{
			const USHORT* curr = (const USHORT*)lockedRect.pBits;

			for (int j = 0; j < m_frameHeight; ++j)
			{
				for (int i = 0; i < m_frameWidth; ++i)
				{
					const size_t IDX = j * m_frameWidth + i;
					const size_t IDX_REV = j * m_frameWidth + (m_frameWidth - i);

					USHORT depth = NuiDepthPixelToDepth(curr[IDX]);

					if (depth == 0)
					{
						dst[IDX_REV] = -std::numeric_limits<float>::infinity();
					}
					else
					{
						dst[IDX_REV] = depth / 1000.0f;
					}

					// Store the index into the color array corresponding to this pixel
					NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(
						NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, NULL,
						i, j, depth << 3, depthToRGB, depthToRGB + 1);

					depthToRGB += 2;
				}
			}
		}

		nuiTexture->UnlockRect(0);
	}

	void processDepth()
	{
		if (!context)
			return;

		HRESULT hr;
		NUI_IMAGE_FRAME imageFrame;
		bool integrateColor = true;
		hr = context->NuiImageStreamGetNextFrame(depthStreamHandle, 0, &imageFrame);

		if (FAILED(hr))
			return;

		copyDepth(imageFrame);

		LONGLONG currentDepthFrameTime = imageFrame.liTimeStamp.QuadPart;

		hr = context->NuiImageStreamReleaseFrame(depthStreamHandle, &imageFrame);

		if (FAILED(hr))
			return;

		LONGLONG currentColorFrameTime = m_cLastColorFrameTimeStamp;

		hr = context->NuiImageStreamGetNextFrame(colorStreamHandle, 0, &imageFrame);

		if (FAILED(hr))
		{
			integrateColor = false;
		}
		else
		{
			copyColor(imageFrame);

			currentColorFrameTime = imageFrame.liTimeStamp.QuadPart;

			// Release the Kinect camera frame
			context->NuiImageStreamReleaseFrame(colorStreamHandle, &imageFrame);

			if (FAILED(hr))
				return;
		}

		int timestampDiff = static_cast<int>(abs(currentColorFrameTime - currentDepthFrameTime));

		if (integrateColor && timestampDiff >= cMinTimestampDifferenceForFrameReSync)
		{
			// Get another frame to try and re-sync
			if (currentColorFrameTime - currentDepthFrameTime >= cMinTimestampDifferenceForFrameReSync)
			{
				// Get another depth frame to try and re-sync as color ahead of depth
				hr = context->NuiImageStreamGetNextFrame(depthStreamHandle, timestampDiff, &imageFrame);
				if (FAILED(hr))
					return;

				copyDepth(imageFrame);

				currentDepthFrameTime = imageFrame.liTimeStamp.QuadPart;

				// Release the Kinect camera frame
				context->NuiImageStreamReleaseFrame(depthStreamHandle, &imageFrame);

				if (FAILED(hr))
					return;
			}
			else if (currentDepthFrameTime - currentColorFrameTime >= cMinTimestampDifferenceForFrameReSync && WaitForSingleObject(colorStreamHandle, 0) != WAIT_TIMEOUT)
			{
				// Get another color frame to try and re-sync as depth ahead of color
				hr = context->NuiImageStreamGetNextFrame(colorStreamHandle, 0, &imageFrame);
				if (FAILED(hr))
				{
					integrateColor = false;
				}
				else
				{
					copyColor(imageFrame);

					currentColorFrameTime = imageFrame.liTimeStamp.QuadPart;

					// Release the Kinect camera frame
					context->NuiImageStreamReleaseFrame(colorStreamHandle, &imageFrame);

					if (FAILED(hr))
					{
						integrateColor = false;
					}
				}
			}

			timestampDiff = static_cast<int>(abs(currentColorFrameTime - currentDepthFrameTime));

			// If the difference is still too large, we do not want to integrate color
			if (timestampDiff > cMinTimestampDifferenceForFrameReSync)
			{
				integrateColor = false;
			}
		}

		m_cLastDepthFrameTimeStamp = currentDepthFrameTime;
		m_cLastColorFrameTimeStamp = currentColorFrameTime;
	}

	HANDLE colorStreamHandle = nullptr;
	HANDLE depthStreamHandle = nullptr;
	std::unique_ptr<float[]> m_depthFrame;
	std::unique_ptr<BYTE[]> m_colorFrame;
	std::unique_ptr<long[]> m_depthToRGBMap;

	INuiSensor* context = nullptr;

	static const int cMinTimestampDifferenceForFrameReSync = 17;
	HANDLE m_hNextDepthFrameEvent;
	HANDLE m_hNextColorFrameEvent;
	LONGLONG m_cLastDepthFrameTimeStamp;
	LONGLONG m_cLastColorFrameTimeStamp;
};