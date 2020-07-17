#pragma once
#include "Sensor.h"

class KinectSensor: public Sensor
{
	public:
		bool Init()
		{
			int numKinects = 0;
			HRESULT hr = NuiGetSensorCount(&numKinects);

			if (FAILED(hr) || numKinects <= 0)
				return false;

			hr = NuiCreateSensorByIndex(0, &context);

			if (FAILED(hr))
				return false;

			DWORD nuiFlags = NUI_INITIALIZE_FLAG_USES_COLOR |
				NUI_INITIALIZE_FLAG_USES_DEPTH;

			hr = context->NuiInitialize(nuiFlags);

			if (FAILED(hr))
				return false;

			hr = context->NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, NULL, &colorStreamHandle);

			if (FAILED(hr))
				return false;

			hr = context->NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH, NUI_IMAGE_RESOLUTION_640x480, 0, 2, NULL, &depthStreamHandle);

			if (FAILED(hr))
				return false;
			// EXPLANATION: This init function needs to return if the init is successful, main function checks in this on constructor
			return true;
		}

		bool Init(const std::string& datasetDir)
		{
			return Init();
		}

		bool ProcessNextFrame()
		{
			// EXPLANATION: This should get the next frame

			HRESULT hr;

			/*NUI_IMAGE_FRAME colorFrame;
			hr = Instance->context->NuiImageStreamGetNextFrame(Instance->colorStreamHandle, 0, &colorFrame);

			if (SUCCEEDED(hr))
			{
				updateImageFrame(colorFrame, false);
				Instance->context->NuiImageStreamReleaseFrame(Instance->colorStreamHandle, &colorFrame);
			}*/

			NUI_IMAGE_FRAME depthFrame;
			hr = context->NuiImageStreamGetNextFrame(depthStreamHandle, 0, &depthFrame);

			if (SUCCEEDED(hr))
			{
				updateImageFrame(depthFrame, true);
				context->NuiImageStreamReleaseFrame(depthStreamHandle, &depthFrame);
			}

			// This should set m_depthFrame with float* value in metric space

			return SUCCEEDED(hr);
		}

		float* GetDepth()
		{
			
			return m_depthFrame;
		}

		void updateImageFrame(NUI_IMAGE_FRAME& imageFrame, bool isDepthFrame)
		{
			INuiFrameTexture* nuiTexture = imageFrame.pFrameTexture;
			NUI_LOCKED_RECT lockedRect;

			nuiTexture->LockRect(0, &lockedRect, NULL, 0);

			if (lockedRect.Pitch != NULL)
			{
				const BYTE* buffer = (const BYTE*)lockedRect.pBits;

				// TODO: This for loop must convert to float in metric space instead of texture because the main loop
				// uses float to filter than uses the resulting image as texture
				for (int y = 0; y < 480; ++y)
				{
					const BYTE* line = buffer + y * lockedRect.Pitch;
					const USHORT* bufferWord = (const USHORT*)line;

					for (int x = 0; x < 640; ++x)
					{
						if (!isDepthFrame)
						{
							// Reading bytewise ...
							unsigned char* ptr = colorTexture->bits + 3 * (y * 640 + x);
							*(ptr + 0) = line[4 * x + 2];
							*(ptr + 1) = line[4 * x + 1];
							*(ptr + 2) = line[4 * x + 0];
						}
						else
						{
							// Reading wordwise ... (Depth-Value = 16 Bit)
							unsigned char* ptr = depthTexture->bits + (y * 640 + x);
							*ptr = (unsigned char)NuiDepthPixelToDepth(bufferWord[x]);
						}
					}
				}
				TextureObject* tobj = (isDepthFrame ? Instance->depthTexture : Instance->colorTexture);
				glBindTexture(GL_TEXTURE_2D, tobj->id);
				glTexImage2D(GL_TEXTURE_2D, 0, tobj->internalFormat, tobj->width, tobj->height, 0, tobj->imageFormat, GL_UNSIGNED_BYTE, tobj->bits);
			}
			nuiTexture->UnlockRect(0);
		}

		void Destroy()
		{
			if (context)
			{
				context->NuiShutdown();
			}
		}

		// get current color data
		BYTE* GetColorRGBX()
		{
			return nullptr;
		}

	private:
		HANDLE colorStreamHandle = nullptr;
		HANDLE depthStreamHandle = nullptr;
		float* m_depthFrame;
		INuiSensor* context = nullptr;
};