// Define the patch window size here (2 * patchWindowRadius - 1)
#define patchWindowSize 5
// Constant memory cannot be dynamically allocated like shared memory using extern
__device__ __constant__ float kernel[patchWindowSize * patchWindowSize * patchWindowSize];
#include "stdio.h"
__global__ void nonLocalMeansConstMem(const float *inputDataWithNoise, const int imageWidth, const int imageHeight,
                                        const int slicesCount, const int searchWindowRadius,
                                            const int patchWindowRadius, const float stdDev, float *outputDenoisedData)

{
    // Normal indexing scheme in x, y, z since both block and grid dimensions are specified
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int	z = threadIdx.z + blockIdx.z * blockDim.z;
    // Threads with image pixels should only be denoising
    if (x >= imageWidth || y >= imageHeight || z >= slicesCount)
        return;
    // Initialize all voxels of the output data to zeros
    outputDenoisedData[z * imageHeight * imageWidth + y * imageWidth + x] = 0;
    // Initialize the sum of weights and max weight to zero
    float sumWeights = 0;
    float maxWeight = 0;

    // Calculate start and end points for search window in all three dimensions
    int startSearchZ = (patchWindowRadius > (z + patchWindowRadius - searchWindowRadius) ? patchWindowRadius :
                                            (z + patchWindowRadius - searchWindowRadius));
    int endSearchZ = ((slicesCount + patchWindowRadius - 1) < (z + patchWindowRadius + searchWindowRadius) ?
                        (slicesCount + patchWindowRadius - 1) : (z + patchWindowRadius + searchWindowRadius));
    int startSearchY = (patchWindowRadius > (y + patchWindowRadius - searchWindowRadius) ?
                                            patchWindowRadius : (y + patchWindowRadius - searchWindowRadius));
    int endSearchY = ((imageHeight + patchWindowRadius - 1) < (y + patchWindowRadius + searchWindowRadius) ?
                        (imageHeight + patchWindowRadius - 1) : (y + patchWindowRadius + searchWindowRadius));
    int startSearchX = (patchWindowRadius > (x + patchWindowRadius - searchWindowRadius) ?
                                            patchWindowRadius : (x + patchWindowRadius - searchWindowRadius));
    int endSearchX = ((imageWidth + patchWindowRadius - 1) < (x + patchWindowRadius + searchWindowRadius) ?
                        (imageWidth + patchWindowRadius - 1) : (x + patchWindowRadius + searchWindowRadius));

    // Run over the search window of the voxel
    for (int searchPosZ = startSearchZ; searchPosZ <= endSearchZ; searchPosZ++)
    {
        for (int searchPosY = startSearchY; searchPosY <= endSearchY; searchPosY++)
        {
            for (int searchPosX = startSearchX; searchPosX <= endSearchX; searchPosX++)
            {
                // If the search window position has reached the boundary in all 3 dimensions then don't consider this window
                if (searchPosX == (x + patchWindowRadius) && searchPosY == (y + patchWindowRadius) && searchPosZ == (z + patchWindowRadius))
                    continue;

                // Initialize the distance and weight metrics to zero
                float distance = 0;
                float weight = 0;

                // Run over the patch window of the voxel
                for (int patchPosZ = -patchWindowRadius; patchPosZ <= patchWindowRadius; patchPosZ++)
                {
                    for (int patchPosY = -patchWindowRadius; patchPosY <= patchWindowRadius; patchPosY++)
                    {
                        for (int patchPosX = -patchWindowRadius; patchPosX <= patchWindowRadius; patchPosX++)
                        {
                            // Get the index for search window
                            int searchPosIndex = (searchPosZ + patchPosZ) * (imageHeight + 2 * patchWindowRadius) * (imageWidth + 2 * patchWindowRadius)
                                                    + (searchPosY + patchPosY) * (imageWidth + 2 * patchWindowRadius) + (searchPosX + patchPosX);
                            // Get the index for kernel
                            int kernelPosIndex = (patchPosZ + patchWindowRadius) * (2 * patchWindowRadius + 1) * (2 * patchWindowRadius + 1) +
                                                    (patchPosY + patchWindowRadius) * (2 * patchWindowRadius + 1) + (patchPosX + patchWindowRadius);
                            // Get the index for the voxel
                            int voxelPosIndex = (z + patchWindowRadius + patchPosZ) * (imageHeight + 2 * patchWindowRadius) * (imageWidth + 2 * patchWindowRadius) +
                                                    (y + patchWindowRadius + patchPosY) * (imageWidth + 2 * patchWindowRadius) + (x + patchWindowRadius + patchPosX);
                            // Compute the gaussian kernel weighted euclidean distance
                            distance += kernel[kernelPosIndex] * (inputDataWithNoise[searchPosIndex] - inputDataWithNoise[voxelPosIndex]) *
                                        (inputDataWithNoise[searchPosIndex] - inputDataWithNoise[voxelPosIndex]);
                        }
                    }
                }
                // Compute the weight for search window voxel
                weight = expf(-distance / (stdDev * stdDev));
                // Adding up weights for all voxels in search window
                sumWeights += weight;
                // Getting the maximum weight within the search window
                maxWeight = (weight > maxWeight) ? weight : maxWeight;
                // Writing to the output voxel
                outputDenoisedData[z * imageHeight * imageWidth + y * imageWidth + x] += weight * inputDataWithNoise[searchPosZ * (imageHeight + 2 * patchWindowRadius) *
                                                                                            (imageWidth + 2 * patchWindowRadius) + searchPosY *
                                                                                                (imageWidth + 2 * patchWindowRadius) + searchPosX];
            }
        }
    }
    // Writing to the output voxel
    outputDenoisedData[z * imageHeight * imageWidth + y * imageWidth + x] += maxWeight * inputDataWithNoise[(z + patchWindowRadius) * (imageHeight + 2 * patchWindowRadius) *
                                                                                (imageWidth + 2 * patchWindowRadius) + (y + patchWindowRadius) *
                                                                                    (imageWidth + 2 * patchWindowRadius) + (x + patchWindowRadius)];
    // Normalizing the denoised output
    outputDenoisedData[z * imageHeight * imageWidth + y * imageWidth + x] /= (sumWeights + maxWeight);
}