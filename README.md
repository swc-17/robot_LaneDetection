# robot_LaneDetection
This repo is for deploy [Ultra Fast Lane Detection](https://arxiv.org/abs/2004.11757) in C++.

## Environment 
Hardware: Nvidia AGX Xavier(ARM)  
Software: 
* JetPack 4.4
* CUDA 10.2
* CUDNN 8.0.0
* TensorRT 7.1.3  
**Tested with above but can be transfered to other environments.**


## Usage
* build: `mkdir build && cd build && cmake .. && make`
* model conversion: convert lane.wts to tensorrt engine   `./model_conversion`  
* lane_detection: `./main`

## Details
* Input: Define variable `video` in `UFLD.h` to be the input video file path
* Output: In this algorithm, row locations of lanes are predefined as `tusimple_row_anchor` in `UFLD.h`, column locations are
put into `lanes` which is a 2-dimensional array with shape (4,56), 4 is the number of lanes, 56 is the number of row-anchors, to get one lane represented by 56 points, refer to function 'display()' in `UFLD.cpp`.

