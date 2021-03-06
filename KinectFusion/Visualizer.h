#pragma once

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <array>
#include <GL/glew.h>
#include <freeglut.h>
#include "Tsdf.cuh"

#if KINECT
#include "KinectNuiSensor2.h"
#endif

#include "TextureManager.h"
#include "GLUtilities.h"
#include "Eigen.h"
#include "VirtualSensor.h"
#include "BilateralFilter.cuh"
#include "BackProjection.cuh"
#include "NormalCalculation.cuh"
#include "NormalCalculationEigen.h"
#include "PoseEstimator.cuh"

#include "TsdfVolume.cuh"

#include "Volume.h"
#include "SimpleMesh.h"
#include "MeshExporter.h"

#include "RayCasting.cuh"

//Set this macro to 1 if you use model-to-frame alignment
#define MODEL_TO_FRAME 1



class Visualizer
{
public:

	Visualizer(int skip = 1) :  
#if KINECT 
		sensor{ 640, 480, false },
#else 
		sensor(skip),
#endif
								filterer(640,480,true),
								filtererModel(640, 480, false),
								backProjector(640, 480), 
								normalCalculator(640, 480), 
								backProjectorModel(640, 480),
								normalCalculatorModel(640, 480),
								poseEstimator(640, 480, 4, 1.0f), // poseEstimator(640, 480, 10, 1.0f),
								prevBackProjector(640, 480), 
								prevNormalCalculator(640, 480),
								poseEstimatorFirstLevel(320, 240, 5, 0.5f),
								poseEstimatorSecondLevel(160, 120, 10, 0.25f), //poseEstimatorSecondLevel(160, 120, 4, 0.25f),
								volume(640, 480, 0.06, 500, 0.01, std::numeric_limits<float>().max()), //volume(640, 480, 0.06, 500, 0.01, 1),
								raycaster(640, 480, 0.06, 500, 0.01), // raycaster(640, 480, 0.06, 500, 0.01),
								tsdf{}
	{
		
		
		#if KINECT

		if (!sensor.Init())
		{
			std::cerr << "Failed to initialize the sensor!" << std::endl;
			exit(1);
		}

		#else

		// std::string filenameIn = R"(c:\tmp\rgbd_dataset_freiburg1_xyz\)";
		//std::string filenameIn = R"(C:\Tmp\rgbd_dataset_freiburg1_rpy\)";
		std::string filenameIn = "../data/rgbd_dataset_freiburg1_xyz/";

		if (!sensor.Init(filenameIn))
		{
			std::cerr << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
			exit(1);
		}

		#endif 

		std::cout << "fx: " << sensor.GetFX() << " fy: " << sensor.GetFY() << " cx: " << sensor.GetCX() << " cy: " << sensor.GetCY() << std::endl;

		if (!filterer.isOK()) 
		{
			std::cerr << "Failed to initialize the filterer!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		if (!filtererModel.isOK())
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
			std::cerr << "Failed to initialize the normal calculator!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		if (!backProjectorModel.isOK())
		{
			std::cerr << "Failed to initialize the back projector model!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		if (!normalCalculatorModel.isOK())
		{
			std::cerr << "Failed to initialize the normal calculator model!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		if (!poseEstimator.isOk())
		{
			std::cerr << "Failed to initialize the pose estimator!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		if (!poseEstimatorFirstLevel.isOk())
		{
			std::cerr << "Failed to initialize the first level pose estimator!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		if (!poseEstimatorSecondLevel.isOk())
		{
			std::cerr << "Failed to initialize the second level pose estimator!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}

		backProjector.setIntrinsics(sensor.GetFX(), sensor.GetFY(), sensor.GetCX(), sensor.GetCY());
		prevBackProjector.setIntrinsics(sensor.GetFX(), sensor.GetFY(), sensor.GetCX(), sensor.GetCY());
		currentTransform = Matrix4f::Identity();

		volume.setIntrinsics(sensor.GetFX(), sensor.GetFY(), sensor.GetCX(), sensor.GetCY());

		if (!volume.isOk())
		{
			std::cerr << "Failed to initialize the volume for calculating the truncated signed distance field!\nCheck your gpu memory!" << std::endl;
			std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(volume.status()) << std::endl;
			exit(1);
		}

		if (!raycaster.isOk())
		{
			std::cerr << "Failed to initialize the raycaster!\nCheck your gpu memory!" << std::endl;
			exit(1);
		}
	}

	static Visualizer* getInstance() 
	{
		if (Instance == nullptr) {
			Instance = new Visualizer();
		}
		return Instance;
	}

	static void deleteInstance() 
	{
		if (Instance != nullptr) 
		{
			delete Instance;
			Instance = nullptr;
		}
	}

	static void updateImageFrame() 
	{
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

		// Input pipeline
		Instance->filterer.applyFilter(Instance->depthImage);
		Instance->backProjector.apply(Instance->filterer.getInputGPU(), Instance->filterer.getOutputGPU(0), Instance->filterer.getOutputGPU(1), Instance->filterer.getOutputGPU(2));
		Instance->normalCalculator.apply(Instance->backProjector.getOutputGPU(-1), Instance->backProjector.getOutputGPU(0), Instance->backProjector.getOutputGPU(1), Instance->backProjector.getOutputGPU(2));

		// Pose estimation
		if (Instance->frameNumber > 0)
		{
			 std::cout << Instance->frameNumber << std::endl;

			Instance->poseEstimatorSecondLevel.resetParams();
			if (Instance->poseEstimatorSecondLevel.apply(Instance->backProjector.getOutputGPU(2),
				Instance->prevBackProjector.getOutputGPU(2),
				Instance->normalCalculator.getOutputGPU(2),
				Instance->prevNormalCalculator.getOutputGPU(2),
				Instance->normalCalculator.getValidMaskGPU(2),
				Instance->prevNormalCalculator.getValidMaskGPU(2)))
			{
				Instance->poseEstimatorFirstLevel.setParams(Instance->poseEstimatorSecondLevel.getParamVector());
				if (Instance->poseEstimatorFirstLevel.apply(Instance->backProjector.getOutputGPU(1),
					Instance->prevBackProjector.getOutputGPU(1),
					Instance->normalCalculator.getOutputGPU(1),
					Instance->prevNormalCalculator.getOutputGPU(1),
					Instance->normalCalculator.getValidMaskGPU(1),
					Instance->prevNormalCalculator.getValidMaskGPU(1)))
				{
					Instance->poseEstimator.setParams(Instance->poseEstimatorFirstLevel.getParamVector());
					if (Instance->poseEstimator.apply(Instance->backProjector.getOutputGPU(0),
						Instance->prevBackProjector.getOutputGPU(0),
						Instance->normalCalculator.getOutputGPU(0),
						Instance->prevNormalCalculator.getOutputGPU(0),
						Instance->normalCalculator.getValidMaskGPU(0),
						Instance->prevNormalCalculator.getValidMaskGPU(0)))
					{
						//std::cout << Instance->poseEstimator.getTransform() << std::endl;
						Instance->currentTransform = Instance->currentTransform * Instance->poseEstimator.getTransform();
						//std::cout << Instance->sensor.GetTrajectory() << std::endl;
						//std::cout << Instance->sensor.GetTrajectory().inverse() << std::endl;
					}
				}
			}
		}

		// Export Mesh
		if (Instance->userRequestedExport)
		{
			exportMesh();
			std::cout << "After exportMesh()" << std::endl;
		}

		if (false && (Instance->frameNumber % 50 == 0) && Instance->backProjector.copyToCPU())
		{
			//std::cout << Instance->frameNumber << std::endl;
			std::ofstream outFile("./vertices" + std::to_string(Instance->frameNumber) + ".off");
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
					outFile << 0 << " " << 0 << " " << 0 << " " << 255 << " " << 255 << " " << 255 << " " << 255 << std::endl;
				}
				else
				{
					int color = std::fmin(((std::fmax(vertices[index + 2], 0) / 5) * 255), 255);
					Vector4f vertex = Instance->currentTransform * Vector4f(vertices[index], vertices[index + 1], vertices[index + 2], 1.0f);
					outFile << vertex[0] << " " << vertex[1] << " " << vertex[2] << " "
						<< (int)Instance->colorMap[i * 4] << " " << (int)Instance->colorMap[i * 4 + 1] << " " << (int)Instance->colorMap[i * 4 + 2] << " " << (int)Instance->colorMap[i * 4 + 3] << std::endl;
				}
			}
			outFile.close();

		}

		// Fuse surface into model
		Instance->tsdf.apply(
			Instance->volume, Instance->filterer.getInputGPU(), Instance->normalCalculator.getValidMaskGPU(-1), 
			Instance->sensor.GetColorRGBX(), Instance->currentTransform
		);

		if (!Instance->tsdf.isOk())
		{
			std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(Instance->tsdf.status()) << std::endl;
			exit(1);
		}

		// Raycast the available measurement
		if (Instance->raycaster.apply(Instance->volume.sdf, Instance->volume.weights, Instance->volume.colors, Instance->currentTransform))
		{
			// std::cout << "Raycasted" << std::endl;
		}
		else {
			std::cout << ":( Cannot raycast" << std::endl;
		}

#if KINECT
		// Visualize the current input
		auto& filtererInUse = Instance->filterer;
		if (Instance->raycaster.copyToCPU())
		{
			for (int y = 0; y < 480; ++y)
			{
				for (int x = 0; x < 640; ++x)
				{
					int index = (y * 640 + x);
					unsigned char* ptr = Instance->normalTexture->bits + 3 * index;
					*ptr = Instance->raycaster.getOutputColorCPU()[3 * index];
					*(ptr + 1) = Instance->raycaster.getOutputColorCPU()[3 * index + 1];
					*(ptr + 2) = Instance->raycaster.getOutputColorCPU()[3 * index + 2];
				}
			}
			TextureObject* tobj = Instance->normalTexture;
			glBindTexture(GL_TEXTURE_2D, tobj->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobj->internalFormat, tobj->width, tobj->height, 0, tobj->imageFormat, GL_UNSIGNED_BYTE, tobj->bits);
		}
#else
		auto& filtererInUse = Instance->filterer;
		if (filtererInUse.copyToCPU() && Instance->raycaster.copyToCPU() && Instance->prevNormalCalculator.copyToCPU() && Instance->normalCalculator.copyToCPU())
		{
			image = filtererInUse.getOutputCPU(0);
			imageFirstLevel = filtererInUse.getOutputCPU(1);
			imageSecondLevel = filtererInUse.getOutputCPU(2);
			for (int y = 0; y < 480; ++y)
			{
				for (int x = 0; x < 640; ++x)
				{
					int index = (y * 640 + x);
					unsigned char* ptrNormal = Instance->normalTexture->bits + 3 * index;
					unsigned char* ptrNormalRaycast = Instance->normalTextureRaycast->bits + 3 * index;
					unsigned char* ptrColor = Instance->colorTexture->bits + 3 * index;
					unsigned char* ptr = Instance->depthTexture->bits + index;
					unsigned char* ptrRaycast = Instance->depthTextureRaycast->bits + index;
					unsigned char* ptrUnfiltered = Instance->depthTextureUnfiltered->bits + index;
					float currentX = (std::fmaxf(Instance->normalCalculator.getOutputCPU(-1)[3 * index], -1) + 1.0f) * 127.5f;
					float currentY = (std::fmaxf(Instance->normalCalculator.getOutputCPU(-1)[(3 * index) + 1], -1) + 1.0f) * 127.5f;
					float currentZ = (std::fmaxf(Instance->normalCalculator.getOutputCPU(-1)[(3 * index) + 2], -1) + 1.0f) * 127.5f;
					float currentRaycastX = (std::fmaxf(Instance->prevNormalCalculator.getOutputCPU(-1)[3 * index], -1) + 1.0f) * 127.5f;
					float currentRaycastY = (std::fmaxf(Instance->prevNormalCalculator.getOutputCPU(-1)[(3 * index) + 1], -1) + 1.0f) * 127.5f;
					float currentRaycastZ = (std::fmaxf(Instance->prevNormalCalculator.getOutputCPU(-1)[(3 * index) + 2], -1) + 1.0f) * 127.5f;
					float current = (std::fmaxf(image[index], 0) / 5) * 255;
					float currentRaycast = (std::fmaxf(Instance->raycaster.getOutputDepthCPU()[index], 0) / 5) * 255;
					float currentUnfiltered = ((std::fmaxf(Instance->depthImage[index], 0) / 5) * 255);
					*ptrNormal = (unsigned char)std::fminf(currentX, 255);
					*(ptrNormal + 1) = (unsigned char)std::fminf(currentY, 255);
					*(ptrNormal + 2) = (unsigned char)std::fminf(currentZ, 255);
					*ptrNormalRaycast = (unsigned char)std::fminf(currentRaycastX, 255);
					*(ptrNormalRaycast + 1) = (unsigned char)std::fminf(currentRaycastY, 255);
					*(ptrNormalRaycast + 2) = (unsigned char)std::fminf(currentRaycastZ, 255);
					*ptrColor = Instance->raycaster.getOutputColorCPU()[3 * index];
					*(ptrColor + 1) = Instance->raycaster.getOutputColorCPU()[3 * index + 1];
					*(ptrColor + 2) = Instance->raycaster.getOutputColorCPU()[3 * index + 2];
					//*ptr = (unsigned char)std::fminf(current, 255);
					*ptrUnfiltered = (unsigned char)std::fminf(currentUnfiltered, 255);
					*ptr = (unsigned char)std::fminf(current, 255);
					*ptrRaycast = (unsigned char)std::fminf(currentRaycast, 255);

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

			TextureObject* tobjNormal = Instance->normalTexture;
			glBindTexture(GL_TEXTURE_2D, tobjNormal->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjNormal->internalFormat, tobjNormal->width, tobjNormal->height, 0, tobjNormal->imageFormat, GL_UNSIGNED_BYTE, tobjNormal->bits);

			TextureObject* tobjNormalRaycast = Instance->normalTextureRaycast;
			glBindTexture(GL_TEXTURE_2D, tobjNormalRaycast->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjNormalRaycast->internalFormat, tobjNormalRaycast->width, tobjNormalRaycast->height, 0, tobjNormalRaycast->imageFormat, GL_UNSIGNED_BYTE, tobjNormalRaycast->bits);

			TextureObject* tobjColor = Instance->colorTexture;
			glBindTexture(GL_TEXTURE_2D, tobjColor->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjColor->internalFormat, tobjColor->width, tobjColor->height, 0, tobjColor->imageFormat, GL_UNSIGNED_BYTE, tobjColor->bits);

			TextureObject* tobj = Instance->depthTexture;
			glBindTexture(GL_TEXTURE_2D, tobj->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobj->internalFormat, tobj->width, tobj->height, 0, tobj->imageFormat, GL_UNSIGNED_BYTE, tobj->bits);

			TextureObject* tobjRaycast = Instance->depthTextureRaycast;
			glBindTexture(GL_TEXTURE_2D, tobjRaycast->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjRaycast->internalFormat, tobjRaycast->width, tobjRaycast->height, 0, tobjRaycast->imageFormat, GL_UNSIGNED_BYTE, tobjRaycast->bits);

			TextureObject* tobjFirstLevel = Instance->depthTextureFirstLevel;
			glBindTexture(GL_TEXTURE_2D, tobjFirstLevel->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjFirstLevel->internalFormat, tobjFirstLevel->width, tobjFirstLevel->height, 0, tobjFirstLevel->imageFormat, GL_UNSIGNED_BYTE, tobjFirstLevel->bits);

			TextureObject* tobjSecondLevel = Instance->depthTextureSecondLevel;
			glBindTexture(GL_TEXTURE_2D, tobjSecondLevel->id);
			glTexImage2D(GL_TEXTURE_2D, 0, tobjSecondLevel->internalFormat, tobjSecondLevel->width, tobjSecondLevel->height, 0, tobjSecondLevel->imageFormat, GL_UNSIGNED_BYTE, tobjSecondLevel->bits);
		}
#endif

		// Set target points for tracking
#if MODEL_TO_FRAME
		// fill prev objects with model rendering
		Instance->filtererModel.applyFilterGPU(Instance->raycaster.getOutputDepthGPU());
		Instance->prevBackProjector.apply(
			Instance->raycaster.getOutputDepthGPU(),
			Instance->filtererModel.getOutputGPU(0),
			Instance->filtererModel.getOutputGPU(1),
			Instance->filtererModel.getOutputGPU(2)
		);
		Instance->prevNormalCalculator.apply(
			Instance->prevBackProjector.getOutputGPU(-1),
			Instance->prevBackProjector.getOutputGPU(0),
			Instance->prevBackProjector.getOutputGPU(1),
			Instance->prevBackProjector.getOutputGPU(2)
		);
#else
		Instance->prevBackProjector.copy(&Instance->backProjector);
		Instance->prevNormalCalculator.copy(&Instance->normalCalculator);
#endif

	} // Update image frame

	static void exportMesh()
	{
		std::cout << "Exporting mesh ..." << std::endl;

		if (!Instance->volume.copyToCPU())
			return;

		MeshExporter me{ 0.0f };
		me.exportMesh("mesh.off", Instance->volume);
		Instance->userRequestedExport = false;

		std::cout << "Finished exporting mesh!" << std::endl;

		Instance->sensor.Destroy();
		glutLeaveMainLoop();
		// exit(0);
	}


	static void update()
	{
		if (Instance->sensor.ProcessNextFrame())
		{
			Instance->depthImage = Instance->sensor.GetDepth();
			Instance->colorMap = Instance->sensor.GetColorRGBX();
			updateImageFrame();
		}
		else
		{
#if KINECT
#else
			std::cout << "Exporting mesh..." << std::endl;
			exportMesh();
			std::cout << "Done" << std::endl;
			exit(0);
#endif
			
		}

		glutPostRedisplay();
	}

#if KINECT
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

		GLfloat vertices2[][3] =
		{
			{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
			{1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}
		};

		GLfloat textcoords[][2] =
		{
			{0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}
		};

		VertexData meshData2 = { &(vertices2[0][0]), NULL, NULL, &(textcoords[0][0]) };

		glBindTexture(GL_TEXTURE_2D, Instance->normalTexture->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData2, GL_QUADS);

		glutSwapBuffers();
	}
#else
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
			{0.0f, 0.5f, 0.0f}, {0.25f, 0.5f, 0.0f},
			{0.25f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}
		};

		GLfloat textcoords[][2] =
		{
			{0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}
		};

		VertexData meshData = { &(vertices[0][0]), NULL, NULL, &(textcoords[0][0]) };


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

		glTranslatef(-0.75f, -0.5f, 0.0f);

		glBindTexture(GL_TEXTURE_2D, Instance->normalTexture->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData, GL_QUADS);

		glTranslatef(0.25f, 0.0f, 0.0f);

		glBindTexture(GL_TEXTURE_2D, Instance->depthTextureRaycast->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData, GL_QUADS);

		glTranslatef(0.25f, 0.0f, 0.0f);

		glBindTexture(GL_TEXTURE_2D, Instance->normalTextureRaycast->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData, GL_QUADS);

		glTranslatef(0.25f, 0.0f, 0.0f);

		glBindTexture(GL_TEXTURE_2D, Instance->colorTexture->id);
		drawSimpleMesh(WITH_POSITION | WITH_TEXCOORD, 4, meshData, GL_QUADS);


		glutSwapBuffers();
	}
#endif
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
			Instance->sensor.Destroy();
			break;
		case 'E':
		case 'e':
			Instance->userRequestedExport = true;
			break;
		}

		glutPostRedisplay();
	}

	void run()
	{

		int argc = 0;
		char* arg0 = const_cast<char*>("Sensor Reading");
		char* argv[] = { arg0 };

		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
		// glutInitWindowSize(1280, 240);
#if KINECT
		glutInitWindowSize(640, 480);
#else
		glutInitWindowSize(1280, 480);
#endif
		glutCreateWindow("Sensor Reading");
		// glutFullScreen();

		glutIdleFunc(update);
		glutDisplayFunc(render);
		glutReshapeFunc(reshape);
		glutKeyboardFunc(keyEvents);

		colorTexture = createTexture(640, 480, GL_RGB, 3);
		normalTexture = createTexture(640, 480, GL_RGB, 3);
		normalTextureRaycast = createTexture(640, 480, GL_RGB, 3);
		depthTextureUnfiltered = createTexture(640, 480, GL_LUMINANCE, 1);
		depthTexture = createTexture(640, 480, GL_LUMINANCE, 1);
		depthTextureFirstLevel = createTexture(320, 240, GL_LUMINANCE, 1);
		depthTextureSecondLevel = createTexture(160, 120, GL_LUMINANCE, 1);
		depthTextureRaycast = createTexture(640, 480, GL_LUMINANCE, 1);

		glutMainLoop();

		destroyTexture(colorTexture);
		destroyTexture(normalTexture);
		destroyTexture(normalTextureRaycast);
		destroyTexture(depthTexture);
		destroyTexture(depthTextureRaycast);
		destroyTexture(depthTextureFirstLevel);
		destroyTexture(depthTextureSecondLevel);
		destroyTexture(depthTextureUnfiltered);

	}

	void setDepthImage(float* depth) 
	{
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
	TextureObject* normalTexture = nullptr;
	TextureObject* normalTextureRaycast = nullptr;
	TextureObject* depthTexture = nullptr;
	TextureObject* depthTextureFirstLevel = nullptr;
	TextureObject* depthTextureSecondLevel = nullptr;
	TextureObject* depthTextureUnfiltered = nullptr;
	TextureObject* depthTextureRaycast = nullptr;
	float* depthImage;
	float* depthImageRaw;
	BYTE* colorMap;
	#if KINECT
	// KinectOpenNISensor sensor;
	KinectNuiSensor2 sensor;
	#else
		VirtualSensor sensor;
	#endif 
	Filterer filterer;
	Filterer filtererModel;
	BackProjector backProjector;
	BackProjector backProjectorModel;
	NormalCalculator normalCalculator;
	NormalCalculator normalCalculatorModel;
	BackProjector prevBackProjector;
	NormalCalculator prevNormalCalculator;
	PoseEstimator poseEstimator;
	PoseEstimator poseEstimatorFirstLevel;
	PoseEstimator poseEstimatorSecondLevel;
	TsdfVolume volume;
	Tsdf tsdf;
	bool isWritten = false;
	int frameNumber = 0;
	Matrix4f currentTransform;
	RayCaster raycaster;
	bool userRequestedExport = false;
};

Visualizer* Visualizer::Instance = nullptr;
