#pragma once

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <array>
#include <GL/glew.h>
#include <freeglut.h>

#include "TextureManager.h"
#include "GLUtilities.h"
#include "Eigen.h"
#include "VirtualSensor.h"
#include "BilateralFilter.cuh"

#define KINECT 0;

#if KINECT
	#include "KinectSensor.h"
#endif

class Visualizer
{
public:

	Visualizer(int skip = 1) : sensor(skip), filterer(640,480)
	{
		std::string filenameIn = "../data/rgbd_dataset_freiburg1_xyz/";
		
		if (!sensor.Init(filenameIn))
		{
			std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
			exit(1);
		}

		if (!filterer.isOK()) {
			std::cout << "Failed to initialize the filterer!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

	}

	static Visualizer* getInstance() {
		if (Instance == nullptr) {
			Instance = new Visualizer();
		}
		return Instance;
	}

	static void deleteInstance() {
		if (Instance != nullptr) {
			delete Instance;
			Instance = nullptr;
		}
	}

	static void updateImageFrame() {
		float* image;
		Instance->filterer.applyFilter(Instance->depthImage);
		if (Instance->filterer.copyToCPU()) 
		{
			image=Instance->filterer.getOutput();
			for (int y = 0; y < 480; ++y)
			{
				for (int x = 0; x < 640; ++x)
				{
					int index = (y * 640 + x);
					unsigned char* ptr = Instance->depthTexture->bits + index;
					unsigned char* ptrUnfiltered = Instance->depthTextureUnfiltered->bits + index;
					float current = (std::fmaxf(image[index], 0) / 5) * 255;
					float currentUnfiltered = (std::fmaxf(Instance->depthImage[index], 0) / 5) * 255;
					//float currentUnfiltered = std::fabsf(((std::fmaxf(Instance->depthImage[index], 0) / 5) * 255)- current );
					*ptr = (unsigned char)std::fminf(current, 255);
					*ptrUnfiltered = (unsigned char)std::fminf(currentUnfiltered, 255);
				}
			}

			TextureObject* tobj = Instance->depthTexture;
			glBindTexture(GL_TEXTURE_2D, tobj->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobj->internalFormat, tobj->width, tobj->height, 0, tobj->imageFormat, GL_UNSIGNED_BYTE, tobj->bits);

			TextureObject* tobjUnfiltered = Instance->depthTextureUnfiltered;
			glBindTexture(GL_TEXTURE_2D, tobjUnfiltered->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjUnfiltered->internalFormat, tobjUnfiltered->width, tobjUnfiltered->height, 0, tobjUnfiltered->imageFormat, GL_UNSIGNED_BYTE, tobjUnfiltered->bits);
		}
		
	}

	static void update()
	{
		if (Instance->sensor.ProcessNextFrame())
		{
			Instance->depthImage = Instance->sensor.GetDepth();
			updateImageFrame();
		}
		glutPostRedisplay();
	}

	static void render()
	{
		glClearColor(.0f, .0f, .0f, .0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_TEXTURE_2D);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		GLfloat vertices[][3] =
		{
			{0.0f, 0.0f, 0.0f}, {0.5f, 0.0f, 0.0f},
			{0.5f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}
		};

		GLfloat textcoords[][2] =
		{
			{0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}
		};

		VertexData meshData = { &(vertices[0][0]), NULL, NULL, &(textcoords[0][0]) };

		/*glBindTexture(GL_TEXTURE_2D, Instance->colorTexture->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData, GL_QUADS);*/

		glBindTexture(GL_TEXTURE_2D, Instance->depthTextureUnfiltered->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData, GL_QUADS);

		glTranslatef(0.5f, 0.0f, 0.0f);

		glBindTexture(GL_TEXTURE_2D, Instance->depthTexture->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData, GL_QUADS);

		glutSwapBuffers();
	}

	static void reshape(int w, int h)
	{
		glViewport(0, 0, w, h);
	}

	static void keyEvents(unsigned char key, int x, int y)
	{
		switch (key)
		{
		case 27:
		case 'Q':
		case 'q':
			glutLeaveMainLoop();
			return;
		}

		glutPostRedisplay();
	}

	

	void destroyKinect()
	{
	
	}

	void run()
	{
		int argc = 0;
		char* arg0 = const_cast<char*>("Sensor Reading");
		char* argv[] = { arg0 };

		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
		glutInitWindowSize(1280, 480);
		glutCreateWindow("Sensor Reading");
		// glutFullScreen();

		glutIdleFunc(update);
		glutDisplayFunc(render);
		glutReshapeFunc(reshape);
		glutKeyboardFunc(keyEvents);

		colorTexture = createTexture(640, 480, GL_RGB, 3);
		depthTextureUnfiltered = createTexture(640, 480, GL_LUMINANCE, 1);
		depthTexture = createTexture(640, 480, GL_LUMINANCE, 1);

		glutMainLoop();

		destroyKinect();

		destroyTexture(colorTexture);
		destroyTexture(depthTexture);
		destroyTexture(depthTextureUnfiltered);

	}

	void setDepthImage(float* depth) {
		depthImage = depth;
		if (glutGet(GLUT_INIT_STATE) == 1) 
		{
			glutPostRedisplay();
		}
	}

private:
	static Visualizer* Instance;
	
	//std::string hudText;
	TextureObject* colorTexture = nullptr;
	TextureObject* depthTexture = nullptr;
	TextureObject* depthTextureUnfiltered = nullptr;
	float* depthImage;
	#if KINECT
		KinectSensor sensor;
	#else
		VirtualSensor sensor;
	#endif 
	Filterer filterer;
};

Visualizer* Visualizer::Instance = nullptr;