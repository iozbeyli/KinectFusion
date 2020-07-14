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
#include "BackProjection.cuh"
#include "NormalCalculation.cuh"
#include "NormalCalculationEigen.h"
#include "PoseEstimator.cuh"


#define KINECT 0;

#if KINECT
	#include "KinectSensor.h"
#endif

class Visualizer
{
public:

	Visualizer(int skip = 1) :	sensor(skip), 
								filterer(640,480), 
								backProjector(640, 480), 
								normalCalculator(640, 480), 
								poseEstimator(640, 480, 10),
								prevBackProjector(640, 480), 
								prevNormalCalculator(640, 480)
	{
		std::string filenameIn = "../data/rgbd_dataset_freiburg1_xyz/";
		
		if (!sensor.Init(filenameIn))
		{
			std::cerr << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
			exit(1);
		}

		if (!filterer.isOK()) 
		{
			std::cerr << "Failed to initialize the filterer!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		if (!backProjector.isOK()) 
		{
			std::cerr << "Failed to initialize the back projector!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		if (!normalCalculator.isOK())
		{
			std::cerr << "Failed to initialize the back projector!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		if (!poseEstimator.isOk())
		{
			std::cerr << "Failed to initialize the pose estimator!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		backProjector.setIntrinsics(sensor.GetFX(), sensor.GetFY(), sensor.GetCX(), sensor.GetCY());

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
		float* imageFirstLevel;
		float* imageSecondLevel;
		float* vertices;
		float* verticesFirstLevel;
		float* verticesSecondLevel;
		float* normals;
		float* normalsFirstLevel;
		float* normalsSecondLevel;
		Instance->frameNumber++;
		Instance->filterer.applyFilter(Instance->depthImage);
		Instance->backProjector.apply(Instance->filterer.getOutputGPU(0), Instance->filterer.getOutputGPU(1), Instance->filterer.getOutputGPU(2));
		Instance->normalCalculator.apply(Instance->backProjector.getOutputGPU(0), Instance->backProjector.getOutputGPU(1), Instance->backProjector.getOutputGPU(2));
		
		if (Instance->frameNumber == 2)
		{
			if(Instance->poseEstimator.apply(	Instance->backProjector.getOutputGPU(0),
											Instance->prevBackProjector.getOutputGPU(0),
											Instance->normalCalculator.getOutputGPU(0),
											Instance->prevNormalCalculator.getOutputGPU(0),
											Instance->normalCalculator.getValidMaskGPU(0)))
			{
				std::cout << Instance->poseEstimator.getTransform() << std::endl;
			}
			else {
				std::cout << "After solve fails" << std::endl;
			}
		}
		
		//Instance->poseEstimator.apply(
		//Normals CPU Calculation Comparison
		/*if (Instance->backProjector.copyToCPU() && Instance->normalCalculator.copyToCPU()) {
			vertices = Instance->backProjector.getOutputCPU(0);
			verticesFirstLevel = Instance->backProjector.getOutputCPU(1);
			verticesSecondLevel = Instance->backProjector.getOutputCPU(2);
			normals = Instance->normalCalculator.getOutputCPU(0);
			normalsFirstLevel = Instance->normalCalculator.getOutputCPU(1);
			normalsSecondLevel = Instance->normalCalculator.getOutputCPU(2);
			std::cout << "Error: " << calculateNormalError(normals, vertices, 640, 480) << std::endl;
		}*/
		
		/* Writing Point Cloud to .off
		if (!Instance->isWritten)
		{
			std::cout << Instance->isWritten << std::endl;
			Instance->isWritten = true;
			if (Instance->backProjector.copyToCPU()) {
				std::ofstream outFile("./vertices.off");
				if (!outFile.is_open()) return exit(1);
				outFile << "COFF" << std::endl;
				outFile << 480 * 640 << " " << 0 << " 0" << std::endl;
				vertices = Instance->backProjector.getOutputCPU(0);
				verticesFirstLevel = Instance->backProjector.getOutputCPU(1);
				verticesSecondLevel = Instance->backProjector.getOutputCPU(2);
				for (int i = 0; i < 480 * 640; ++i)
				{
					int index = 3 * i;
					if (isinf(vertices[index])) 
					{
						outFile << 0 << " " << 0 << " " << 0 << std::endl;
					}
					else
					{
						int color = std::fmin(((std::fmax(vertices[index + 2], 0) / 5) * 255),255);
						outFile << vertices[index] << " " << vertices[index + 1] << " " << vertices[index + 2] << " "
							<< color << " " << color << " " << color << " " << 255 << std::endl;
					}
				}
				outFile.close();
			}
		}*/
		
		
		
		if (Instance->filterer.copyToCPU()) 
		{
			image=Instance->filterer.getOutputCPU(0);
			imageFirstLevel = Instance->filterer.getOutputCPU(1);
			imageSecondLevel = Instance->filterer.getOutputCPU(2);	
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

					if (x < 320 && y < 240) 
					{
						int indexFirstLevel = (y * 320 + x);
						unsigned char* ptrFirstLevel = Instance->depthTextureFirstLevel->bits + indexFirstLevel;
						float currentFirstLevel = (std::fmaxf(imageFirstLevel[indexFirstLevel], 0) / 5) * 255;
						*ptrFirstLevel = (unsigned char)std::fminf(currentFirstLevel, 255);
					}
					if (x < 160 && y < 120) 
					{
						int indexSecondLevel = (y * 160 + x);
						unsigned char* ptrSecondLevel = Instance->depthTextureSecondLevel->bits + indexSecondLevel;
						float currentSecondLevel = (std::fmaxf(imageSecondLevel[indexSecondLevel], 0) / 5) * 255;
						*ptrSecondLevel = (unsigned char)std::fminf(currentSecondLevel, 255);
					}
				}
			}

			TextureObject* tobjUnfiltered = Instance->depthTextureUnfiltered;
			glBindTexture(GL_TEXTURE_2D, tobjUnfiltered->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjUnfiltered->internalFormat, tobjUnfiltered->width, tobjUnfiltered->height, 0, tobjUnfiltered->imageFormat, GL_UNSIGNED_BYTE, tobjUnfiltered->bits);

			TextureObject* tobj = Instance->depthTexture;
			glBindTexture(GL_TEXTURE_2D, tobj->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobj->internalFormat, tobj->width, tobj->height, 0, tobj->imageFormat, GL_UNSIGNED_BYTE, tobj->bits);

			TextureObject* tobjFirstLevel = Instance->depthTextureFirstLevel;
			glBindTexture(GL_TEXTURE_2D, tobjFirstLevel->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjFirstLevel->internalFormat, tobjFirstLevel->width, tobjFirstLevel->height, 0, tobjFirstLevel->imageFormat, GL_UNSIGNED_BYTE, tobjFirstLevel->bits);

			TextureObject* tobjSecondLevel = Instance->depthTextureSecondLevel;
			glBindTexture(GL_TEXTURE_2D, tobjSecondLevel->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjSecondLevel->internalFormat, tobjSecondLevel->width, tobjSecondLevel->height, 0, tobjSecondLevel->imageFormat, GL_UNSIGNED_BYTE, tobjSecondLevel->bits);

			
		}

		Instance->backProjector.swap(&Instance->prevBackProjector);
		Instance->normalCalculator.swap(&Instance->prevNormalCalculator); 
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
			{0.0f, 0.0f, 0.0f}, {0.25f, 0.0f, 0.0f},
			{0.25f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}
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

		glTranslatef(0.25f, 0.0f, 0.0f);

		glBindTexture(GL_TEXTURE_2D, Instance->depthTexture->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData, GL_QUADS);

		glTranslatef(0.25f, 0.0f, 0.0f);

		glBindTexture(GL_TEXTURE_2D, Instance->depthTextureFirstLevel->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData, GL_QUADS);

		glTranslatef(0.25f, 0.0f, 0.0f);

		glBindTexture(GL_TEXTURE_2D, Instance->depthTextureSecondLevel->id);
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
		glutInitWindowSize(1280, 240);
		glutCreateWindow("Sensor Reading");
		// glutFullScreen();

		glutIdleFunc(update);
		glutDisplayFunc(render);
		glutReshapeFunc(reshape);
		glutKeyboardFunc(keyEvents);

		colorTexture = createTexture(640, 480, GL_RGB, 3);
		depthTextureUnfiltered = createTexture(640, 480, GL_LUMINANCE, 1);
		depthTexture = createTexture(640, 480, GL_LUMINANCE, 1);
		depthTextureFirstLevel = createTexture(320, 240, GL_LUMINANCE, 1);
		depthTextureSecondLevel = createTexture(160, 120, GL_LUMINANCE, 1);

		glutMainLoop();

		destroyKinect();

		destroyTexture(colorTexture);
		destroyTexture(depthTexture);
		destroyTexture(depthTextureFirstLevel);
		destroyTexture(depthTextureSecondLevel);
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
	TextureObject* depthTextureFirstLevel = nullptr;
	TextureObject* depthTextureSecondLevel = nullptr;
	TextureObject* depthTextureUnfiltered = nullptr;
	float* depthImage;
	#if KINECT
		KinectSensor sensor;
	#else
		VirtualSensor sensor;
	#endif 
	Filterer filterer;
	BackProjector backProjector;
	NormalCalculator normalCalculator;
	BackProjector prevBackProjector;
	NormalCalculator prevNormalCalculator;
	PoseEstimator poseEstimator;
	bool isWritten = false;
	int frameNumber = 0;
};

Visualizer* Visualizer::Instance = nullptr;