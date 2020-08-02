#pragma once

#include <string>
#include "TsdfVolume.cuh"
#include "SimpleMesh.h"
#include "MarchingCubes.h"
#include "Volume.h"

class MeshExporter
{
public:
	//MeshExporter(const TsdfVolume& tsdfVolume, float isolevel)
	//	: tsdf {tsdfVolume}
	//	, isolevel {isolevel}
	//{}

	MeshExporter( float isolevel)
		: isolevel{ isolevel }
	{}

	void exportMesh(const std::string& filePath, TsdfVolume& tsdf)
	{
		Volume vol(Vector3d(0, 0, 0), Vector3d(1, 1, 1), tsdf.voxelCountX, tsdf.voxelCountY, tsdf.voxelCountZ, 1);
		UINT infcount = 0;
		UINT nancount = 0;
		UINT weightCount = 0;
		UINT valueCount = 0;
		float maxSdf = std::numeric_limits<float>().min();
		float minSdf = std::numeric_limits<float>().max();
		float maxWeight = std::numeric_limits<float>().min();
		float minWeight = std::numeric_limits<float>().max();

		for (unsigned int x = 0; x < tsdf.voxelCountX; ++x)
		{
			for (unsigned int y = 0; y < tsdf.voxelCountY; ++y)
			{
				for (unsigned int z = 0; z < tsdf.voxelCountZ; ++z)
				{
					double val = tsdf.cpuSdf[tsdf.idx(x,y,z)];
					BYTE* c = tsdf.cpuColors + (tsdf.idx(x, y, z) * 4);

					if (isinf(val))
					{
						vol.set(x, y, z, 0);
						infcount++;
						continue;
					}

					if (isnan(val))
					{
						vol.set(x, y, z, 0);
						nancount++;
						continue;
					}

					valueCount++;

					if (val > maxSdf)
						maxSdf = val;

					if (val < minSdf)
						minSdf = val;

					vol.set(x, y, z, val);
					vol.setColor(x, y, z, c);
				}
			}
		}

		std::cout << "Infinity sdf: " << infcount << " NAN sdf: " << nancount << " sdf count: " << valueCount << std::endl;
		std::cout << "Min sdf: " << minSdf << " Max sdf: " << maxSdf << std::endl;

		SimpleMesh mesh{ true };
		for (unsigned int x = 0; x < tsdf.voxelCountX - 1; ++x)
		{
			for (unsigned int y = 0; y < tsdf.voxelCountY - 1; ++y)
			{
				for (unsigned int z = 0; z < tsdf.voxelCountZ - 1; ++z)
				{
					float weight = tsdf.cpuWeights[tsdf.idx(x,y,z)];

					if (weight > 0)
					{
						if (weight > maxWeight)
							maxWeight = weight;

						if (weight < minWeight)
							minWeight = weight;

						weightCount++;
						ProcessVolumeCell(&vol, x, y, z, isolevel, &mesh);
					}
				}
			}
		}

		std::cout << "Weights count: " << weightCount << " Min weight: " << minWeight << " Max weight: " << maxWeight << std::endl;

		// write mesh to file
		if (!mesh.WriteMesh(filePath))
		{
			std::cout << "ERROR: unable to write output file!" << std::endl;
		}
	}

private:
	float isolevel;
};