#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include <vector>
#include <string>

std::vector<unsigned char> imageToRGBVector(const std::string& filename, int& width, int& height, int& channels) {
    unsigned char* image = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb);
    std::vector<unsigned char> rgbVector;

    if (image != nullptr) {
        int imageSize = width * height * channels;
        rgbVector.assign(image, image + imageSize);
        stbi_image_free(image);
    }

    return rgbVector;
}

void RGBVectorToImage(const std::vector<unsigned char>& rgbVector, int width, int height, int channels, const std::string& filename) {
    stbi_write_png(filename.c_str(), width, height, channels, rgbVector.data(), width * channels);
}

#include <iostream>
#include <vector>
#include <string>


int main(int argc, char*argv[]) {
    
    if (argc != 2) {
        std::cout << "Usage: filters <filename>" << std::endl;
        return 1;
    }
    
    std::string filename(argv[1]);
    
    std::string inputPath = "../images/in/" + filename;
    int width, height, channels;
    
    std::vector<unsigned char> rgbVector = imageToRGBVector(inputPath, width, height, channels);

    std::cout<<width<<" "<<height<<" "<<channels<<'\n';

    for(auto i:rgbVector)
    {
        for(int j = 0; j < 3; j++)
        {
            std::cout<<(int)i<<' ';
        }
        std::cout<<'\n';
    }

    if (!rgbVector.empty()) {
 
        //some processing
 
        std::string outputPath = "../images/out/" + filename;
        RGBVectorToImage(rgbVector, width, height, channels, outputPath);

        std::cout << "Imagen convertida y guardada exitosamente." << std::endl;
    } else {
        std::cout << "Error al cargar la imagen." << std::endl;
    }

    return 0;
}