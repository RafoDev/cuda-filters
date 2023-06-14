#include <iostream>
#include <stdio.h>
#include <vector>
#include "../include/utils.hpp"

using namespace std;
#define CHANNELS 3
#define BLUR_SIZE 10

__global__ void blurKernel(unsigned char *Pout, unsigned char *Pin, int width, int height)
{
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;

	if (Col < width && Row < height)
	{
		int pixVal_r = 0;
		int pixVal_g = 0;
		int pixVal_b = 0;
		int pixels = 0;

		for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
		{
			for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
			{
				int currRow = Row + blurRow;
				int currCol = Col + blurCol;

				int rgbOffset = (currRow * width + currCol) * CHANNELS;

				if (currRow > -1 && currRow < height && currCol > -1 && currCol < width)
				{
					pixVal_r += Pin[rgbOffset];
					pixVal_g += Pin[rgbOffset + 1];
					pixVal_b += Pin[rgbOffset + 2];
					pixels++;
				}
			}
		}

		int blurrOffset = (Row * width + Col) * CHANNELS;

		Pout[blurrOffset] = (unsigned char)(pixVal_r / pixels);
		Pout[blurrOffset + 1] = (unsigned char)(pixVal_g / pixels);
		Pout[blurrOffset + 2] = (unsigned char)(pixVal_b / pixels);
	}
}

int main(int argc, char *argv[])
{

	if (argc != 2)
	{
		cout << "Usage: toGrayScale <filename> \n";
		return 1;
	}

	string filename(argv[1]);

	string inputPath = "../images/in/" + filename;

	int width, height, channels;

	vector<unsigned char> rgbVectorIn = imageToRGBVector(inputPath, width, height, channels);

	int size = width * height * 3 * sizeof(unsigned char);

	vector<unsigned char> rgbVectorOut(size);

	unsigned char *h_rgbVectorIn = rgbVectorIn.data();
	unsigned char *h_rgbVectorOut = rgbVectorOut.data();

	unsigned char *d_rgbVectorIn;
	unsigned char *d_rgbVectorOut;

	cudaMalloc((void **)&d_rgbVectorIn, size);
	cudaMalloc((void **)&d_rgbVectorOut, size);

	cudaMemcpy(d_rgbVectorIn, h_rgbVectorIn, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rgbVectorOut, h_rgbVectorOut, size, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
	dim3 dimBlock(16, 16, 1);

	blurKernel<<<dimGrid, dimBlock>>>(d_rgbVectorOut, d_rgbVectorIn, width, height);

	cudaMemcpy(h_rgbVectorOut, d_rgbVectorOut, size, cudaMemcpyDeviceToHost);

	cout << width << " " << height << " " << channels << '\n';

	if (!rgbVectorOut.empty())
	{
		filename = "blurred_" + filename;
		string outputPath = "../images/out/" + filename;
		RGBVectorToImage(rgbVectorOut, width, height, channels, outputPath);

		cout << "Image converted and saved successfully [images/out]." << endl;
	}
	else
	{
		cout << "Error loading image." << endl;
	}

	cudaFree(d_rgbVectorIn);
	cudaFree(d_rgbVectorOut);

	return 0;
}
