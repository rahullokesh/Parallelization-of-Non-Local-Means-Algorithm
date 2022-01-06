# Parallelization of Non-Local Means (NLM) Algorithm

Non-Local Means is an image processing algorithm used to denoise images. The method is based on replacing a noisy pixel with the average values of all similar pixels. We propose a CUDA kernel to perform this function. We further optimize this kernel using the constant memory approach. This repo also benchmarks the performance of the kernels. We have achieved a 104x speedup from the serial implementation.

This repository contains code for the following : 
a) Parallel CUDA kernel for faster NLM denoising.
b) Constant memory kernel to improve speed of parallel approach.
c) Notebooks for Benchmarking and analysis of serial and parallel approaches.
d) Quantify the quality of parallel denoising and compare it to the serial approach.

# Repository Structure 

Overview of what each folder contains.

## Data
This folder contains the input image binary file for testing the parallel cuda and python serial implementation. These binary files with noise and padding added to the image(imageNoisyPaddedInput5X5X5Slices2.bin and imageNoisyPaddedInput5X5X5Slices10.bin). Also, it contains image input binary files for scikit-image denoising(imageNoisyInput5X5X5Slices10.bin) which has only noise added and no padding. Lastly, there is a original image binary file which is used for PSNR and SSIM calculation for all the denoised images with respect to the original image.

## Image Generation Scripts
Matlab scripts to generate different binary input files from a .rawb image file.
* image_data_generation.m - Matlab script to generate original image data. Used in files where PSNR and SSIM is calculated for Scikit-Image Denoising, Parallel PyCuda based denoising and Python Serial based denoising. 
* image_data_padded_with_noise_generation.m - Matlab script to generate binary file of the image which for which noise and padding is added. The binary file generated from this is used as input for the parallel and serial NLM algorithms.
* image_data_with_noise_generation.m - Matlab script to generate binary file of the image with noise added without any padding. The binary file generated from this is used as input for sckit-image based NLM algorithm.
Few pre-generated binary files from these scripts are placed inside data folder. Additional ones 

## Non Local Means
  ### Cuda :
  Cuda kernels for normal parallel approach and constant memory based approach for NLM Denoising and host code to launch these kernels.
  ### Python :
  Serial python code of NLM Denoising.

## Notebooks :
.ipynb notebooks of PyCUDA parallel, python serial and sckit-image based implementation of NLM denoising for ease of use. Also, the PSNR and SSIM values are calculated for denoised image output from PyCUDA parallel, python serial and sckit-image in these notebooks.

## Running the Code 

 Step 1 : Generate binary files from the image generation generation matlab scripts and place them in the data folder. There are binary files already added to the data folder.

 Step 2 : Make sure the path for the binary files are appropriately added in the python based code and host code for parallel algorithm in the Non Local mean folder files. The path for the default files present in the data folder have already been updated.

 Step 3 : Running the python serial code
```
python NLM_Serial.py

```

 Step 4 : Running the PyCuda based parallel code which executes both normal parallel implementation and parallel constant memory based implementation
```
python host_code.py

```

Step 5 : For execution of notebook with scikit-image based NLM denoising and its appropriate PSNR and SSIM calculation, make sure to have the corresponding original input image binary file(Ex:imageInput5X5X5Slices10.bin) in the same directory of the executing notebook. Similarly, for PSNR and SSIM calculation of PyCuda parallel and Python serial based denoised images, copy corresponding output files(Ex: python_output.bin and gpu_output.bin) that are generated from the respective algorithms into the same directory of the executing notebook.
