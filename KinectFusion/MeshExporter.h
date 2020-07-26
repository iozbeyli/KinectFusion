#pragma once

#include <string>
#include "TsdfVolume.cuh"
#include "SimpleMesh.h"
#include "MarchingCubes.h"
#include "Volume.h"

class MeshExporter
{
public:
	MeshExporter(const TsdfVolume& tsdfVolume, float isolevel)
		: tsdf {tsdfVolume}
		, isolevel {isolevel}
	{}

	void exportMesh(const std::string& filePath)
	{
		if (!tsdf.copyToCPU())
			return;

		Volume vol(Vector3d(0, 0, 0), Vector3d(1, 1, 1), tsdf.voxelCountX, tsdf.voxelCountY, tsdf.voxelCountZ, 1);
		UINT infcount = 0;
		UINT nancount = 0;
		UINT weightscount = 0;
		UINT valuescount = 0;

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

					if (val > 0)
						valuescount++;

					vol.set(x, y, z, val);
					vol.setColor(x, y, z, c);
				}
			}
		}

		std::cout << "Infinity values: " << infcount << " NAN values: " << nancount << " Aktual values: " << valuescount << std::endl;

		SimpleMesh mesh;
		for (unsigned int x = 0; x < tsdf.voxelCountX - 1; ++x)
		{
			for (unsigned int y = 0; y < tsdf.voxelCountY - 1; ++y)
			{
				for (unsigned int z = 0; z < tsdf.voxelCountZ - 1; ++z)
				{
					float weight = tsdf.cpuWeights[tsdf.idx(x,y,z)];

					if (weight > 0)
					{
						weightscount++;
						ProcessVolumeCell(&vol, x, y, z, isolevel, &mesh);
					}
				}
			}
		}

		std::cout << "Weights count: " << weightscount << std::endl;

		// write mesh to file
		if (!mesh.WriteMesh(filePath))
		{
			std::cout << "ERROR: unable to write output file!" << std::endl;
		}
	}

private:
	TsdfVolume tsdf;
	float isolevel;
};