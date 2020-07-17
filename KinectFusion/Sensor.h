#pragma once
#include <string>
#include "GLUtilities.h"

class Sensor {
	public:
		virtual bool Init() = 0;
		virtual bool Init(const std::string &filename) = 0;
		virtual bool ProcessNextFrame() = 0;
		virtual float* GetDepth() = 0;
		virtual void Destroy() = 0;
		virtual float GetFX() = 0;
		virtual float GetFY() = 0;
		virtual float GetCX() = 0;
		virtual float GetCY() = 0;
		virtual BYTE* GetColorRGBX() = 0;
};