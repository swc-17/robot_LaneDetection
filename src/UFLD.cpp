//车道线检测_UFLD方法_cpp文件
#include "UFLD.h"

UFLDLaneDetector::UFLDLaneDetector()
{   
    cout << "Starting initializing model" << endl;
    
    //build tensorrt engine
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};
    
    std::ifstream file(engineFile, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    //engine = context->getEngine();
    assert(engine->getNbBindings() == 2);
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    delete[] trtModelStream;

    // init calibration 
    inputVideo.open(video);

    cv::FileStorage fs(calibFile, cv::FileStorage::READ);
    cv::Mat camMatrix, distCoeffs;
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

    int height, width;
    fs["CameraMatrix"] >> camMatrix;
    fs["DistortionCoeffs"] >> distCoeffs;
    fs["Resolution width"] >> width;
    fs["Resolution height"] >> height;
    cv::Size sz(width, height);
  
    cv::initUndistortRectifyMap(camMatrix, distCoeffs, R, camMatrix, sz, CV_32FC1, map_x, map_y);

    cout << "Initializing model done" << endl;
}

UFLDLaneDetector::~UFLDLaneDetector()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void UFLDLaneDetector::preProcess(cv::Mat &img, vector<float> &processedImg)
{
    // calibration
    cv::remap(img, img, map_x, map_y, cv::INTER_LINEAR);

    //resize
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));

    // to float
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3, 1. / 255.);

    // HWC TO CHW
    std::vector<cv::Mat> input_channels(INPUT_C);
    cv::split(img_float, input_channels);

    // normalize
    auto data = processedImg.data();
    int channelLength = INPUT_H * INPUT_W;

    static float mean[] = {0.485, 0.456, 0.406};
    static float std[] = {0.229, 0.224, 0.225};
    for (int i = 0; i < INPUT_C; ++i)
    {
        cv::Mat normed_channel = (input_channels[i] - mean[i]) / std[i];
        memcpy(data, normed_channel.data, channelLength * sizeof(float));
        data += channelLength;
    }
}

void UFLDLaneDetector::doInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    // Create GPU buffers on device
    memcpy(data, &processedImg[0], INPUT_C * INPUT_W * INPUT_H * sizeof(float));
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void UFLDLaneDetector::softmax_mul(float *x, float *y, int rows, int cols, int chan)
{
    for (int i = 0, wh = rows * cols; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float sum = 0.0;
            float expect = 0.0;
            for (int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = exp(x[k * wh + i * cols + j]);
                sum += x[k * wh + i * cols + j];
            }
            for (int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] /= sum;
            }
            for (int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = x[k * wh + i * cols + j] * (k + 1);
                expect += x[k * wh + i * cols + j];
            }
            y[i * cols + j] = expect;
        }
    }
}

void UFLDLaneDetector::argmax(float *x, float *y, int rows, int cols, int chan)
{
    for (int i = 0, wh = rows * cols; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int max = -10000000;
            int max_ind = -1;
            for (int k = 0; k < chan; k++)
            {
                if (x[k * wh + i * cols + j] > max)
                {
                    max = x[k * wh + i * cols + j];
                    max_ind = k;
                }
            }
            y[i * cols + j] = max_ind;
        }
    }
}

void UFLDLaneDetector::postProcess()
{
    for (int k = 0, wh = OUTPUT_W * OUTPUT_H; k < OUTPUT_C; k++)
    {
        for(int j = 0; j < OUTPUT_H; j ++)
        {
            for(int l = 0; l < OUTPUT_W; l++)
            {
                prob_reverse[k * wh + (OUTPUT_H - 1 - j) * OUTPUT_W + l] =
                    prob[k * wh + j * OUTPUT_W + l];
            }
        }
    }
    argmax(prob_reverse, max_ind, OUTPUT_H, OUTPUT_W, OUTPUT_C);
    softmax_mul(prob_reverse, expect, OUTPUT_H, OUTPUT_W, OUTPUT_C);
    for(int k = 0; k < OUTPUT_H; k++) {
        for(int j = 0; j < OUTPUT_W; j++) {
            max_ind[k * OUTPUT_W + j] == 100 ? expect[k * OUTPUT_W + j] = 0 :
                expect[k * OUTPUT_W + j] = expect[k * OUTPUT_W + j];
        }
    }
    std::vector<int> i_ind;
    for(int k = 0; k < OUTPUT_W; k++) {
        int ii = 0;
        for(int g = 0; g < OUTPUT_H; g++) {
            if(expect[g * OUTPUT_W + k] != 0)
                ii++;
        }
        if(ii > 2) {
            i_ind.push_back(k);
        }
    }
    memset(lanes,0,sizeof(lanes));
    for(int k = 0; k < OUTPUT_H; k++) {
        for(int ll = 0; ll < i_ind.size(); ll++) {
            if(expect[OUTPUT_W * k + i_ind[ll]] > 0) {
                lanes[i_ind[ll]][k] = int(expect[OUTPUT_W * k + i_ind[ll]] * col_sample_w * VIS_W / INPUT_W) - 1;
            }   
        }
    }
}

bool UFLDLaneDetector::display()
{   
    for(int k = 0; k < OUTPUT_H; k++) 
    { 
        for(int l = 0; l < OUTPUT_W; l++)
        {
            if(lanes[l][k] > 0)
            {
                cv::Point pp = {lanes[l][k], int( VIS_H * tusimple_row_anchor[OUTPUT_H - 1 - k] / INPUT_H) - 1 };
                cv::circle(img, pp, 8, CV_RGB(0, 255 ,0), 2);
            }
        }
    }
    cv::imshow("lane_det", img);
    if (cv::waitKey(1) == 27)
        return false;
    return true;
}

bool UFLDLaneDetector::run()
{
    while(inputVideo.grab())
    {
        inputVideo.retrieve(img);
        if(img.empty()) return false;
        
        preProcess(img, processedImg);
        doInference(*context, data, prob, BATCH_SIZE);
        postProcess();
        display();
    }
}

void UFLDLaneDetector::moduleSelfCheck()
{
    //TODO  
}

void UFLDLaneDetector::moduleSelfCheckPrint()
{
    //TODO
}
