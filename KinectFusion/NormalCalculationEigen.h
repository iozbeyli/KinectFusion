#pragma once
#include <iostream>
#include <stdio.h>
#include "Eigen.h"

double calculateNormalError(float* normalsCalculated, float* backProjected, int imageWidth, int imageHeight)
{
	bool* validMask = new bool[imageHeight * imageWidth];
	double result = 0.0;
	for (int i = 0; i < imageWidth; ++i)
	{
		for (int j = 0; j < imageHeight; ++j)
		{
			int indexWithoutChannel = imageWidth * j + i;
			if ((i >= (imageWidth - 1)) || (j >= (imageHeight - 1)))
			{
				if ((i == (imageWidth - 1)) || (j == (imageHeight - 1)))
				{
					validMask[indexWithoutChannel] = false;
				}
				continue;
			}

			int index = 3 * indexWithoutChannel;
			int indexRight = 3 * (indexWithoutChannel + 1);
			int indexDown = 3 * (indexWithoutChannel + imageWidth);

			Eigen::Vector3f values(backProjected[index],backProjected[index + 1],backProjected[index + 2]);
			Eigen::Vector3f valuesDown(backProjected[indexDown],backProjected[indexDown + 1],backProjected[indexDown + 2]);
			Eigen::Vector3f valuesRight(backProjected[indexRight],backProjected[indexRight + 1],backProjected[indexRight + 2]);

			if (isinf(values[0]) || isinf(valuesDown[0]) || isinf(valuesRight[0]))
			{
				validMask[indexWithoutChannel] = false;
			}
			else
			{
				Eigen::Vector3f normal = (valuesRight - values).cross(valuesDown - values);
				normal.normalize();
				Eigen::Vector3f normalCalculated(normalsCalculated[index], normalsCalculated[index + 1], normalsCalculated[index + 2]);
				result += (normalCalculated - normal).norm();
			}

		}
	}
	delete[] validMask;
	return result/(imageWidth*imageHeight);
}