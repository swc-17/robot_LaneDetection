#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
static const int INPUT_C = 3;
static const int INPUT_H = 288;
static const int INPUT_W = 800;
static const int OUTPUT_C = 101;
static const int OUTPUT_H = 56;
static const int OUTPUT_W = 4;
static const int OUTPUT_SIZE = OUTPUT_C * OUTPUT_H * OUTPUT_W;
const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";
static Logger gLogger;

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder,  DataType dt) {
    INetworkDefinition* network = builder->createNetwork();
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_C, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../models/lane.wts");
#if 0
    /* print layer names */
    for(std::map<std::string, Weights>::iterator iter = weightMap.begin(); iter != weightMap.end() ; iter++)
    {
        std::cout << iter->first << std::endl;
    }
#endif
    auto conv1 = network->addConvolution(*data, 64, DimsHW{ 7, 7 }, weightMap["model.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{2, 2});
    conv1->setPadding(DimsHW{3, 3});
    conv1->setNbGroups(1);

    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "model.bn1", 1e-5);
    auto relu0 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IPoolingLayer* pool0 = network->addPooling(*relu0->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
    pool0->setStride( DimsHW{ 2, 2 } );
    pool0->setPadding( DimsHW{ 1, 1 } );
    assert(pool0);

    auto basic0 = basicBlock(network, weightMap, *pool0->getOutput(0), 64, 64, 1, "model.layer1.0.");
    auto basic1 = basicBlock(network, weightMap, *basic0->getOutput(0), 64, 64, 1, "model.layer1.1.");
    auto basic2_0 = basicBlock(network, weightMap, *basic1->getOutput(0), 64, 128, 2, "model.layer2.0.");

    auto basic2_1 = basicBlock(network, weightMap, *basic2_0->getOutput(0), 128, 128, 1, "model.layer2.1.");

    auto basic3_0 = basicBlock(network, weightMap, *basic2_1->getOutput(0), 128, 256, 2, "model.layer3.0.");

    auto basic3_1 = basicBlock(network, weightMap, *basic3_0->getOutput(0), 256, 256, 1, "model.layer3.1.");

    auto basic4_0 = basicBlock(network, weightMap, *basic3_1->getOutput(0), 256, 512, 2, "model.layer4.0.");

    auto basic4_1 = basicBlock(network, weightMap, *basic4_0->getOutput(0), 512, 512, 1, "model.layer4.1.");

#if 0
    /* just for debug */
    Dims dims1 = basic4_1->getOutput(0)->getDimensions();
    for (int i = 0; i < dims1.nbDims; i++)
    {
        std::cout << dims1.d[i] << "-" << (int)dims1.type[i] << "   ";
    }
    std::cout << std::endl;
#endif

    auto conv2 = network->addConvolution(*basic4_1->getOutput(0), 8, DimsHW{ 1, 1 }, weightMap["pool.weight"], weightMap["pool.bias"]);
    assert(conv2);
    conv2->setStride(DimsHW{1, 1});
    conv2->setPadding(DimsHW{0, 0});
    conv2->setNbGroups(1);

    IShuffleLayer* permute0 = network->addShuffle(*conv2->getOutput(0));
    assert(permute0);
    permute0->setReshapeDimensions( Dims2{1, 1800});

    auto fcwts0 = network->addConstant(nvinfer1::Dims2(2048, 1800), weightMap["cls.0.weight"]);
    auto matrixMultLayer0 = network->addMatrixMultiply(*permute0->getOutput(0), false, *fcwts0->getOutput(0), true);

    assert(matrixMultLayer0 != nullptr);
    // Add elementwise layer for adding bias
    auto fcbias0 = network->addConstant(nvinfer1::Dims2(1, 2048), weightMap["cls.0.bias"]);

    auto addBiasLayer0 = network->addElementWise(*matrixMultLayer0->getOutput(0), *fcbias0->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(addBiasLayer0 != nullptr);

    auto relu = network->addActivation(*addBiasLayer0->getOutput(0), ActivationType::kRELU);

    auto fcwts1 = network->addConstant(nvinfer1::Dims2(22624, 2048), weightMap["cls.2.weight"]);
    auto matrixMultLayer1 = network->addMatrixMultiply(*relu->getOutput(0), false, *fcwts1->getOutput(0), true);

    assert(matrixMultLayer1 != nullptr);
    // Add elementwise layer for adding bias
    auto fcbias1 = network->addConstant(nvinfer1::Dims2(1, 22624), weightMap["cls.2.bias"]);

    auto addBiasLayer1 = network->addElementWise(*matrixMultLayer1->getOutput(0), *fcbias1->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(addBiasLayer1 != nullptr);

    IShuffleLayer* permute1 = network->addShuffle(*addBiasLayer1->getOutput(0));
    assert(permute1);
    permute1->setReshapeDimensions( Dims3{ 101, 56, 4 });

    permute1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*permute1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    if(builder->platformHasFastFp16()) {
        std::cout << "Platform supports fp16 mode and use it !!!" << std::endl;
        builder->setFp16Mode(true);
    } else {
        std::cout << "Platform doesn't support fp16 mode so you can't use it !!!" << std::endl;
    }
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

   return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}


int main(int argc, char** argv)
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

    IHostMemory* modelStream{ nullptr };
    APIToModel(BATCH_SIZE, &modelStream);
    assert(modelStream != nullptr);
    std::ofstream p("../models/lane_det.engine", std::ios::binary);
    if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;

}
