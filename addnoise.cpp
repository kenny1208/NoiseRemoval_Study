#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <io.h>
#include <cmath>
#include <algorithm>
#include <time.h>
using namespace std;

enum NoiseType {
    RANDOM_NOISE,
    GAUSSIAN_NOISE,
    SALT_PEPPER_NOISE
};

unsigned char rgbToGray(unsigned char r, unsigned char g, unsigned char b) {
    return static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
}

unsigned char addGaussianNoise(unsigned char original, mt19937& gen) {
    static normal_distribution<double> gauss(0.0, 25.0);
    double noise = gauss(gen);
    int result = static_cast<int>(original) + static_cast<int>(noise);
    return static_cast<unsigned char>(max(0, min(255, result)));
}

bool shouldAddSaltPepperNoise() {
    int x = rand();
    return (x % 20) == 0;
}
unsigned char getSaltPepperValue() {
    int x = rand();
    return (x % 2) == 0 ? 0 : 255;
}

unsigned char addRandomNoise(unsigned char original, mt19937& gen) {
    static uniform_int_distribution<> noise_prob(1, 20);
    static uniform_int_distribution<> dis(0, 128);

    if (noise_prob(gen) == 1) {
        double tmp = dis(gen);
        return (tmp < 0) ? 0 : (tmp > 255) ? 255 : tmp;
    }
    return original;
}

// 將24位元彩色BMP轉換為8位元灰階BMP
bool convertToGrayscale(const string& inputPath, const string& outputPath) {
    FILE* fp_r;
    FILE* fp_w;

    if (fopen_s(&fp_r, inputPath.c_str(), "rb") != 0) {
        return false;
    }

    if (fopen_s(&fp_w, outputPath.c_str(), "wb") != 0) {
        fclose(fp_r);
        return false;
    }

    unsigned char header[54];
    fread(header, 1, 54, fp_r);

    int width = 512;
    int height = 384;

    int fileSize = 54 + 1024 + width * height;
    int imageSize = width * height;

    header[2] = fileSize & 0xFF;        
    header[3] = (fileSize >> 8) & 0xFF;
    header[4] = (fileSize >> 16) & 0xFF;
    header[5] = (fileSize >> 24) & 0xFF;

    header[10] = 54 + 1024;        
    header[11] = header[12] = header[13] = 0;

    header[28] = 8;                     
    header[29] = 0;

    header[34] = imageSize & 0xFF;      
    header[35] = (imageSize >> 8) & 0xFF;
    header[36] = (imageSize >> 16) & 0xFF;
    header[37] = (imageSize >> 24) & 0xFF;

    fwrite(header, 1, 54, fp_w);

    for (int i = 0; i < 256; i++) 
    {
        unsigned char gray = (unsigned char)i;
        fputc(gray, fp_w);  
        fputc(gray, fp_w);  
        fputc(gray, fp_w);  
        fputc(0, fp_w);     
    }

    for (int y = 0; y < height; y++) 
    {
        for (int x = 0; x < width; x++) 
        {
            unsigned char pixel[3];
            fread(pixel, 1, 3, fp_r);

            unsigned char grayValue = rgbToGray(pixel[2], pixel[1], pixel[0]);
            fputc(grayValue, fp_w);
        }

        int originalPadding = (4 - (width * 3) % 4) % 4;
        for (int p = 0; p < originalPadding; p++) 
        {
            fgetc(fp_r);
        }

    }

    fclose(fp_r);
    fclose(fp_w);
    return true;
}

bool processGrayscaleImage(const string& inputPath, const string& outputPath, NoiseType noiseType) {
    FILE* fp_r;
    FILE* fp_w;

    if (fopen_s(&fp_r, inputPath.c_str(), "rb") != 0) {
        return false;
    }

    if (fopen_s(&fp_w, outputPath.c_str(), "wb") != 0) {
        fclose(fp_r);
        return false;
    }

    unsigned char buffer;
    for (int i = 0; i < 1078; i++) {
        buffer = fgetc(fp_r);
        fputc(buffer, fp_w);
    }

    random_device rd;
    mt19937 gen(rd());

    if (noiseType == SALT_PEPPER_NOISE) {
        srand(time(NULL));
    }

    int width = 512;
    int height = 384;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char pixel = fgetc(fp_r);

            switch (noiseType) {
            case RANDOM_NOISE:
                pixel = addRandomNoise(pixel, gen);
                break;
            case GAUSSIAN_NOISE:
                pixel = addGaussianNoise(pixel, gen);
                break;
            case SALT_PEPPER_NOISE:
                if (shouldAddSaltPepperNoise()) {
                    pixel = getSaltPepperValue();
                }
                break;
            }

            fputc(pixel, fp_w);
        }

    }

    fclose(fp_r);
    fclose(fp_w);
    return true;
}

vector<string> getBmpFiles() {
    vector<string> bmpFiles;
    struct _finddata_t fileData;
    intptr_t handle = _findfirst("data\\*.bmp", &fileData);

    if (handle != -1) {
        do {
            bmpFiles.push_back(fileData.name);
        } while (_findnext(handle, &fileData) == 0);
        _findclose(handle);
    }

    return bmpFiles;
}

int main() {
    vector<string> bmpFiles = getBmpFiles();

    for (const string& filename : bmpFiles) {
        string inputPath = "data\\" + filename;

        string grayPath = "gray\\" + filename;
        convertToGrayscale(inputPath, grayPath);

        string randomOutputPath = "random\\" + filename;
        processGrayscaleImage(grayPath, randomOutputPath, RANDOM_NOISE);

        string gaussianOutputPath = "gaussian\\" + filename;
        processGrayscaleImage(grayPath, gaussianOutputPath, GAUSSIAN_NOISE);

        string saltOutputPath = "salt\\" + filename;
        processGrayscaleImage(grayPath, saltOutputPath, SALT_PEPPER_NOISE);

    }

    return 0;
}