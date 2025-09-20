#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <io.h>
#include <algorithm>
#include <cmath>
using namespace std;

bool applyMedianFilter(const string& inputPath, const string& outputPath) {
    FILE* fp_r;
    FILE* fp_w;

    fopen_s(&fp_r, inputPath.c_str(), "rb");

    fopen_s(&fp_w, outputPath.c_str(), "wb");

    unsigned char buffer;
    for (int i = 0; i < 1078; i++) {
        buffer = fgetc(fp_r);
        fputc(buffer, fp_w);
    }

    int w = 512;
    int h = 384;

    vector<vector<unsigned char>> imgArr(h, vector<unsigned char>(w));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            imgArr[y][x] = fgetc(fp_r);
        }
    }

    fclose(fp_r);

    vector<vector<unsigned char>> filteredImg(h, vector<unsigned char>(w));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sum[9] = {};
            int count = 0;
            int dx[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
            int dy[] = { -1, 0, 1, -1, 1, -1, 0, 1 };

            for (int k = 0; k < 8; k++) {
                int ny = y + dy[k];
                int nx = x + dx[k];

                if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                    sum[k] = imgArr[ny][nx];
                    count++;
                }
            }

            sort(sum, sum + 9);
            filteredImg[y][x] = sum[4]; 
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            fputc(filteredImg[y][x], fp_w);
        }
    }

    fclose(fp_w);
    return true;
}

bool applyGaussianFilter(const string& inputPath, const string& outputPath) {
    FILE* fp_r;
    FILE* fp_w;

    fopen_s(&fp_r, inputPath.c_str(), "rb");

    fopen_s(&fp_w, outputPath.c_str(), "wb");

    unsigned char buffer;
    for (int i = 0; i < 1078; i++) {
        buffer = fgetc(fp_r);
        fputc(buffer, fp_w);
    }

    int w = 512;
    int h = 384;

    vector<vector<unsigned char>> imgArr(h, vector<unsigned char>(w));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            imgArr[y][x] = fgetc(fp_r);
        }
    }

    fclose(fp_r);

    double gaussianKernel[3][3] = {
        {0.0751, 0.1238, 0.0751},
        {0.1238, 0.2042, 0.1238},
        {0.0751, 0.1238, 0.0751}
    };

    vector<vector<unsigned char>> filteredImg(h, vector<unsigned char>(w));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double sum = 0.0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;

                    if (ny < 0) ny = 0;
                    if (ny >= h) ny = h - 1;
                    if (nx < 0) nx = 0;
                    if (nx >= w) nx = w - 1;

                    sum += imgArr[ny][nx] * gaussianKernel[dy + 1][dx + 1];
                }
            }

            int result = (int)(sum + 0.5);
            if (result > 255) result = 255;
            if (result < 0) result = 0;
            filteredImg[y][x] = (unsigned char)result;
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            fputc(filteredImg[y][x], fp_w);
        }
    }

    fclose(fp_w);
    return true;
}

bool applyLowPassFilter(const string& inputPath, const string& outputPath) {
    FILE* fp_r;
    FILE* fp_w;

    fopen_s(&fp_r, inputPath.c_str(), "rb");

    fopen_s(&fp_w, outputPath.c_str(), "wb");

    unsigned char buffer;
    for (int i = 0; i < 1078; i++) {
        buffer = fgetc(fp_r);
        fputc(buffer, fp_w);
    }

    int w = 512;
    int h = 384;

    vector<vector<unsigned char>> imgArr(h, vector<unsigned char>(w));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            imgArr[y][x] = fgetc(fp_r);
        }
    }

    fclose(fp_r);

    vector<vector<int>> paddedImg(h + 2, vector<int>(w + 2, 0));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            paddedImg[y + 1][x + 1] = imgArr[y][x];
        }
    }

    vector<vector<unsigned char>> filteredImg(h, vector<unsigned char>(w));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sum = 0;
            int count = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue; 

                    int py = y + 1 + dy; 
                    int px = x + 1 + dx;

                    sum += paddedImg[py][px];
                    count++;
                }
            }

            int result = sum / count;
            filteredImg[y][x] = (unsigned char)result;
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            fputc(filteredImg[y][x], fp_w);
        }
    }

    fclose(fp_w);
    return true;
}

vector<string> getBmpFiles(const string& folderPath) {
    vector<string> bmpFiles;
    struct _finddata_t fileData;
    string searchPath = folderPath + "\\*.bmp";
    intptr_t handle = _findfirst(searchPath.c_str(), &fileData);

    if (handle != -1) {
        do {
            bmpFiles.push_back(fileData.name);
        } while (_findnext(handle, &fileData) == 0);
        _findclose(handle);
    }

    return bmpFiles;
}

void processMedianFolder(const string& inputFolder, const string& outputFolder) {
    vector<string> bmpFiles = getBmpFiles(inputFolder);
    

    for (const string& filename : bmpFiles) {
        string inputPath = inputFolder + "\\" + filename;
        string outputPath = outputFolder + "\\" + filename;

        applyMedianFilter(inputPath, outputPath);
    }
}

void processGaussianFolder(const string& inputFolder, const string& outputFolder) {
    vector<string> bmpFiles = getBmpFiles(inputFolder);

    for (const string& filename : bmpFiles) {
        string inputPath = inputFolder + "\\" + filename;
        string outputPath = outputFolder + "\\" + filename;

        applyGaussianFilter(inputPath, outputPath);
    }
}

void processLowPassFolder(const string& inputFolder, const string& outputFolder) {
    vector<string> bmpFiles = getBmpFiles(inputFolder);

    for (const string& filename : bmpFiles) {
        string inputPath = inputFolder + "\\" + filename;
        string outputPath = outputFolder + "\\" + filename;

        applyLowPassFilter(inputPath, outputPath);
    }
}

int main() {

    // 中值
    processMedianFolder("random", "mid_random");
    processMedianFolder("salt", "mid_salt");
    processMedianFolder("gaussian", "mid_gaussian");

    // 高斯
    processGaussianFolder("random", "gaus_random");
    processGaussianFolder("salt", "gaus_salt");
    processGaussianFolder("gaussian", "gaus_gaussian");

    // 低通
    processLowPassFolder("random", "low_random");
    processLowPassFolder("salt", "low_salt");
    processLowPassFolder("gaussian", "low_gaussian");

    return 0;
}