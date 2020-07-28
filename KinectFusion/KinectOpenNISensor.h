#pragma once

#include <iostream>
#include <memory>
#include "Sensor.h"
#include "OpenNI.h"

class KinectOpenNISensor: public Sensor
{
public:

	const UINT frameWidth;
	const UINT frameHeight;

	KinectOpenNISensor(const UINT frameWidth, const UINT frameHeight)
		: frameWidth{frameWidth}
		, frameHeight{frameHeight}
		, device{}
		, depthStream{}
		, colorStream{}
	{}

	bool Init()
	{
		openni::Status status;

		// Initialize Device
		status = openni::OpenNI::initialize();
		if (!HandleStatus(status, "Failed to initialize OpenNI."))
			return false;

		// Opening Device
		status = device.open(openni::ANY_DEVICE);
		if (!HandleStatus(status, "Failed to open device."))
			return false;

		// Creating Depth VideoStream
		status = depthStream.create(device, openni::SENSOR_DEPTH);
		if (!HandleStatus(status, "Failed to create depth stream."))
			return false;

		// Set DepthStream Video Mode
		openni::VideoMode vm;
		vm.setFps(30);
		vm.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
		vm.setResolution(this->frameWidth, this->frameHeight);

		status = depthStream.setVideoMode(vm);
		if (!HandleStatus(status, "Failed to set depth video mode."))
			return false;

		// Start DepthStream
		status = depthStream.start();
		if (!HandleStatus(status, "Failed to start depth stream."))
		{
			depthStream.destroy();
			return false;
		}

		// Creating Color VideoStream
		status = colorStream.create(device, openni::SENSOR_COLOR);
		if (!HandleStatus(status, "Failed to create color stream."))
			return false;

		// Set DepthStream Video Mode
		vm.setFps(30);
		vm.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
		vm.setResolution(this->frameWidth, this->frameHeight);

		status = colorStream.setVideoMode(vm);
		if (!HandleStatus(status, "Failed to set color video mode."))
			return false;

		// Start ColorStream
		status = colorStream.start();
		if (!HandleStatus(status, "Failed to start color stream."))
		{
			colorStream.destroy();
			return false;
		}

		// Enable depth to image mapping
		status = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
		if (!HandleStatus(status, "Failed to set image registration mode."))
			return false;

		// Enable depth/color sync mode
		//status = device.setDepthColorSyncEnabled(true);
		//if (!HandleStatus(status, "Failed to enable depth/color sync mode."))
		//	return false;



		const size_t FRAME_DEPTH_SIZE = frameWidth * frameHeight * sizeof(float);
		const size_t FRAME_COLOR_SIZE = frameWidth * frameHeight * sizeof(BYTE) * 4;

		depthFrame.reset(new float[FRAME_DEPTH_SIZE]);
		colorFrame.reset(new BYTE[FRAME_COLOR_SIZE]);

		return true;
	}

	bool Init(const std::string& datasetDir)
	{
		throw "Not Implemented";
	}

	bool ProcessNextFrame()
	{
		openni::Status status;
		openni::VideoStream* vs = &depthStream;
		int streamIdx;

		status = openni::OpenNI::waitForAnyStream(&vs, 1, &streamIdx, 33);

		if (!HandleStatus(status, "Timeout waiting for depth stream."))
			return false;

		status = depthStream.readFrame(&depthFrameRef);
		if (!HandleStatus(status, "Failed reading depth frame."))
			return false;

		status = colorStream.readFrame(&colorFrameRef);
		if (!HandleStatus(status, "Failed reading color frame."))
			return false;

		if(!depthFrameRef.isValid() 
			|| !colorFrameRef.isValid() 
			|| (depthFrameRef.getHeight() != colorFrameRef.getHeight())
			|| (depthFrameRef.getWidth() != colorFrameRef.getWidth()))
			return false;


		openni::DepthPixel* pDepth = (openni::DepthPixel*)depthFrameRef.getData();
		openni::RGB888Pixel* pColor = (openni::RGB888Pixel*)colorFrameRef.getData();
	
		UINT idx = 0;
		for (UINT x = 0; x < frameWidth; ++x)
		{
			for (UINT y = 0; y < frameHeight; ++y)
			{
				idx = y * frameWidth + x;
				
				float depth = (float)pDepth[idx];

				int i = 0;
				if (depth > 0)
					i = 1;

				depthFrame[idx] = depth / 1000.0f;
				colorFrame[idx * 4 + 0] = pColor[idx].r;
				colorFrame[idx * 4 + 1] = pColor[idx].g;
				colorFrame[idx * 4 + 2] = pColor[idx].b;
				colorFrame[idx * 4 + 3] = 255;
			}
		}
	
		return true;
	}

	float* GetDepth()
	{	
		return depthFrame.get();
	}

	BYTE* GetColorRGBX()
	{
		return colorFrame.get();
	}

	void Destroy()
	{
		std::cout << "[SENSOR] Destroying" << std::endl;
		depthStream.destroy();
		colorStream.destroy();
		depthFrameRef.release();
		colorFrameRef.release();
		openni::OpenNI::shutdown();
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

	bool HandleStatus(openni::Status& status, const char* msg)
	{
		if (status == openni::STATUS_OK)
			return true;

		std::cerr << "[ERROR] Sensor: " << msg << std::endl;
		std::cerr << openni::OpenNI::getExtendedError() << std::endl;
		Destroy();

		return false;
	}

	std::unique_ptr<float[]> depthFrame;
	std::unique_ptr<BYTE[]> colorFrame;

	openni::Device device;
	openni::VideoStream colorStream;
	openni::VideoStream depthStream;
	openni::VideoFrameRef depthFrameRef;
	openni::VideoFrameRef colorFrameRef;
};