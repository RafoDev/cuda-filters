#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <vector>
#include <string>
using namespace std;

vector<unsigned char> imageToRGBVector(const string &filename, int &width, int &height, int &channels)
{
	unsigned char *image = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb);
	vector<unsigned char> rgbVector;

	if (image != nullptr)
	{
		int imageSize = width * height * channels;
		rgbVector.assign(image, image + imageSize);
		stbi_image_free(image);
	}

	return rgbVector;
}

void RGBVectorToImage(const vector<unsigned char> &rgbVector, int width, int height, int channels, const string &filename)
{
	stbi_write_png(filename.c_str(), width, height, channels, rgbVector.data(), width * channels);
}
