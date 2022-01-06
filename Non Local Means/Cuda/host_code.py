#!/usr/bin/env python

import numpy as np
import time
import matplotlib.pyplot as plt

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

class nlmDenoise:
    def __init__(self):
        # define block and grid dimensions. Choose block dimension post experimentation
        self.getBlockDimension = (16, 16, 4)
        self.getGridDimension = lambda imageWidth, imageHeight, slicesCount, searchWindowRadius: (
        (((imageWidth + 2 * searchWindowRadius) // 16) + 1), (((imageHeight + 2 * searchWindowRadius) // 16) + 1),
        (((slicesCount + 2 * searchWindowRadius) // 4) + 1))

    def gaussianKernel3Dcalculation(self, patchWindowRadius, sigma):
        # Calculate the patch window size from the given patch window radius
        patchWindowSize = 2 * patchWindowRadius + 1
        # Initialize the kernel sum and final kernel array to zeros
        kernelSum = 0
        kernel = np.zeros(patchWindowSize * patchWindowSize * patchWindowSize)
        # Calculate the kernel elements by looping through in all dimensions
        for z in range(-patchWindowRadius, patchWindowRadius + 1):
            for y in range(-patchWindowRadius, patchWindowRadius + 1):
                for x in range(-patchWindowRadius, patchWindowRadius + 1):
                    # Getting the kernel output index
                    kernelIndex = (z + patchWindowRadius) * patchWindowSize * patchWindowSize + (
                            y + patchWindowRadius) * patchWindowSize + (x + patchWindowRadius)
                    # Calculating the kernel value
                    kernel[kernelIndex] = np.exp(-(z * z + y * y + x * x) / (2 * sigma * sigma))
                    # Adding the calculating kernel value to the kernel sum
                    kernelSum += kernel[kernelIndex]
        for index in range(patchWindowSize * patchWindowSize * patchWindowSize):
            # Normalizing the calculated kernel values
            kernel[index] /= kernelSum
        return kernel

    def nlm_denoising_gpu(self, inputDataWithNoise, kernel, imageWidth, imageHeight, slicesCount,
                          searchWindowRadius, patchWindowRadius, stdDev):
        sourceMod = SourceModule(open('nlm_kernel.cu').read())
        nlmKernel = sourceMod.get_function("nonLocalMeans")
        # Event objects to mark the start and end points
        start = cuda.Event()
        end = cuda.Event()
        # Recording execution time including memory transfer.
        start.record()
        # Copy the input noisy data to the device
        inputDataWithNoise_gpu = gpuarray.to_gpu(inputDataWithNoise)
        # Initialize the output denoised data in the device
        outputDenoisedData_gpu = gpuarray.zeros(imageWidth * imageHeight * slicesCount, dtype = np.float32)
        kernel = kernel.astype(np.float32)
        kernel_gpu = gpuarray.to_gpu(kernel)
        # Execute the non local means parallel algorithm with constant memory implementation
        nlmKernel(inputDataWithNoise_gpu, kernel_gpu, np.int32(imageWidth), np.int32(imageHeight),
                  np.int32(slicesCount), np.int32(searchWindowRadius), np.int32(patchWindowRadius), np.float32(stdDev),
                  outputDenoisedData_gpu, block=self.getBlockDimension, grid = self.getGridDimension(imageWidth, imageHeight,
                                                                                         slicesCount, patchWindowRadius))
        outputDenoisedData = outputDenoisedData_gpu.get()
        # Wait for the event to complete and synchronize
        end.record()
        end.synchronize()
        # Record execution time
        time_ = start.time_till(end)
        return outputDenoisedData, time_

    def nlm_denoising_const_mem_gpu(self, inputDataWithNoise, kernel, imageWidth, imageHeight, slicesCount,
                          searchWindowRadius, patchWindowRadius, stdDev):
        sourceMod = SourceModule(open('nlm_const_mem_kernel.cu').read())
        nlmConstMemKernel = sourceMod.get_function("nonLocalMeansConstMem")
        # Event objects to mark the start and end points
        start = cuda.Event()
        end = cuda.Event()
        # Recording execution time including memory transfer.
        start.record()
        # Copy the input noisy data to the device
        inputDataWithNoise_gpu = gpuarray.to_gpu(inputDataWithNoise)
        # Initialize the output denoised data in the device
        outputDenoisedData_gpu = gpuarray.zeros(imageWidth * imageHeight * slicesCount, dtype = np.float32)
        kernel = kernel.astype(np.float32)
        # Get the constant memory address for the kernel
        kernel_addr = sourceMod.get_global('kernel')[0]
        # Copy the kernel to the constant memory
        cuda.memcpy_htod(kernel_addr, kernel)
        # Execute the non local means parallel algorithm with constant memory implementation
        nlmConstMemKernel(inputDataWithNoise_gpu, np.int32(imageWidth), np.int32(imageHeight),
                  np.int32(slicesCount), np.int32(searchWindowRadius), np.int32(patchWindowRadius), np.float32(stdDev),
                  outputDenoisedData_gpu, block=self.getBlockDimension, grid = self.getGridDimension(imageWidth, imageHeight,
                                                                                         slicesCount, patchWindowRadius))
        outputDenoisedData = outputDenoisedData_gpu.get()
        # Wait for the event to complete and synchronize
        end.record()
        end.synchronize()
        # Record execution time
        time_ = start.time_till(end)
        return outputDenoisedData, time_

if __name__ == "__main__":
    # Create an instance of the nlmDenoise class
    nlm = nlmDenoise()
    # Patch Window size taken as 5X5X5
    patchWindowRadius = 2
    # Search Window size taken as 11X11X11
    searchWindowRadius = 5
    # Dimensions of the test image
    imageWidth = 181
    imageHeight = 217
    slicesCount = 10
    # Default values of sigma and standard values defined
    sigma = 1
    stdDev = 10.0
    # Computing the gaussian filter kernel
    kernel = nlm.gaussianKernel3Dcalculation(patchWindowRadius, sigma)
    # Reading input binary data as a float 32 array
    inputDataWithNoise = np.fromfile("../../Data/imageNoisyPaddedInput5X5X5Slices10.bin", dtype = np.float32, sep = '')
    outputDenoisedData, time = nlm.nlm_denoising_gpu(inputDataWithNoise, kernel, imageWidth, imageHeight, slicesCount,
                          searchWindowRadius, patchWindowRadius, stdDev)
    print("Time taken to denoise by gpu is ", time)
    # Write the gpu denoised output float 32 array to a binary file
    outputDenoisedData.astype('float32').tofile('gpu_output.bin', sep = '')
    outputDenoisedData, time = nlm.nlm_denoising_const_mem_gpu(inputDataWithNoise, kernel, imageWidth, imageHeight,
                                                               slicesCount, searchWindowRadius, patchWindowRadius, stdDev)
    print("Time taken to denoise by gpu using constant memory for kernel is ", time)
    # Write the gpu denoised output float 32 array to a binary file
    outputDenoisedData.astype('float32').tofile('gpu_const_mem_output.bin', sep = '')