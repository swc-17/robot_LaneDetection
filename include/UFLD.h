//车道线检测_UFLD方法_头文件
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "NvInfer.h"

#define DEVICE 0 // GPU id
#define BATCH_SIZE 1
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace std;
using namespace nvinfer1;

class UFLDLaneDetector{
private:
    // build engine
    static const int INPUT_C = 3;
    static const int INPUT_H = 288;
    static const int INPUT_W = 800;
    static const int OUTPUT_C = 101;
    static const int OUTPUT_H = 56;
    static const int OUTPUT_W = 4;
    static const int OUTPUT_SIZE = OUTPUT_C * OUTPUT_H * OUTPUT_W;
    const char* INPUT_BLOB_NAME = "input";
    const char* OUTPUT_BLOB_NAME = "output";
    Logger gLogger;
    string engineFile = "../models/lane_det.engine";

    // inference
    cv::Mat img;
    vector<float> processedImg = vector<float> (INPUT_C * INPUT_H * INPUT_W); 
    
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;
    cudaStream_t stream;
    int inputIndex, outputIndex;
    void *buffers[2];

    float data[BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W];
    float prob[BATCH_SIZE * OUTPUT_SIZE];
    float max_ind[BATCH_SIZE * OUTPUT_H * OUTPUT_W];
    float prob_reverse[BATCH_SIZE * OUTPUT_SIZE];
    float expect[BATCH_SIZE * OUTPUT_H * OUTPUT_W];
    int lanes[OUTPUT_W][OUTPUT_H];

    // video and visualize
    int VIS_H = INPUT_H;
    int VIS_W = INPUT_W;

    cv::VideoCapture inputVideo;
    cv::String video = "/media/745C-E7F8/videos/0802/2022-08-02-09-31-21.avi";
    
    // calbration
    cv::String calibFile = "../src/calib.yaml";
    cv::Mat map_x, map_y;     
    
    // post_process
    int col_sample_w = 8;
    std::vector<int> tusimple_row_anchor{64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                                         116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                                         168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                                         220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                                         272, 276, 280, 284};

public:
    UFLDLaneDetector();
    ~UFLDLaneDetector();
    void preProcess(cv::Mat &img, vector<float> &processedImg);
    void doInference(IExecutionContext &context, float *input, float *output, int batchSize);
    void postProcess();
    void softmax_mul(float *x, float *y, int rows, int cols, int chan);
    void argmax(float *x, float *y, int rows, int cols, int chan);
    bool display();
    bool run();
    void moduleSelfCheck();
    void moduleSelfCheckPrint();
};

