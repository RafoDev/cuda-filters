#include <iostream>
#include <stdio.h>
#include <vector>
#include "../include/utils.hpp"

using namespace std;
#define CHANNELS 3

__global__ 
void colorToGreyScaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height)
    {
        int greyOffset = Row * width + Col;
        int rgbOffset = greyOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        unsigned char tmp = 0.21f * r + 0.71f * g + 0.07f * b;

        Pout[rgbOffset] = tmp;
        Pout[rgbOffset + 1] = tmp;
        Pout[rgbOffset + 2] = tmp;
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

    std::vector<unsigned char> rgbVectorIn = imageToRGBVector(inputPath, width, height, channels);

    int size = width * height * 3 * sizeof(unsigned char);

    // unsigned char *h_rgbVector = (unsigned char *)malloc(size);
    std::vector<unsigned char> rgbVectorOut(size);
    
    unsigned char *h_rgbVectorIn = rgbVectorIn.data();
    unsigned char *h_rgbVectorOut = rgbVectorOut.data();
    

    unsigned char *d_rgbVectorIn;
    unsigned char *d_rgbVectorOut;
    
    cudaMalloc((void **)&d_rgbVectorIn, size);
    cudaMalloc((void **)&d_rgbVectorOut, size);

    cudaMemcpy(d_rgbVectorIn, h_rgbVectorIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgbVectorOut, h_rgbVectorOut, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 dimBlock(16,16,1);

    colorToGreyScaleConversion<<<dimGrid, dimBlock>>>(d_rgbVectorOut, d_rgbVectorIn, width, height);

    cudaMemcpy(h_rgbVectorOut, d_rgbVectorOut, size, cudaMemcpyDeviceToHost);

    std::cout << width << " " << height << " " << channels << '\n';

    if (!rgbVectorOut.empty())
    {
        filename = "greyScaled_" + filename;
        std::string outputPath = "../images/out/" + filename;
        RGBVectorToImage(rgbVectorOut, width, height, channels, outputPath);

        std::cout << "Image converted and saved successfully [images/out]." << std::endl;
    }
    else
    {
        std::cout << "Error loading image." << std::endl;
    }

    cudaFree(d_rgbVectorIn);
    cudaFree(d_rgbVectorOut);

    return 0;
}
