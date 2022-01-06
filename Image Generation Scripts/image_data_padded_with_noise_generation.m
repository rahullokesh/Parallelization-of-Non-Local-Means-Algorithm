clc;
clear;
% Dimensions of the 3D images
imageWidth = 181;
imageHeight = 217;
slicesCount = 181;
% Patch window Radius
patchWindowRadius = 2;
% Slices required and start and end index
reqdSlices = 181;
sliceStartIndex = 1;
% Sigma value for noise
sigma = 10;
sliceEndIndex = sliceStartIndex + reqdSlices - 1;
% Reading the raw binary file containing the 3D image
fileRead = fopen('t2_icbm_normal_1mm_pn0_rf20.rawb','r');
% Reading the 3D image data from the file
imageData = fread(fileRead, imageWidth * imageHeight * slicesCount, ...
    'uint8=>single');
% Reshaping the 3D image data
imageData = reshape(imageData, [imageWidth imageHeight slicesCount]);
% Extracting the required number of slices from the 3D image
imageData = imageData(:, :, sliceStartIndex:sliceEndIndex);
% Adding noise to the image data
imageDataWithNoise = imageData + sigma * randn(size(imageData));
imageDataWithNoisePadded = padarray(imageDataWithNoise, [patchWindowRadius ...
    patchWindowRadius patchWindowRadius], 'symmetric');
% Writing the image noisy data which is padded to a binary file
fwrite(fopen('imageNoisyPaddedInput5X5X5Slices181.bin', 'w'), ...
    imageDataWithNoisePadded, 'float');