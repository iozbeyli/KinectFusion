# KinectFusion

## Code Structure
`main.cu` is the main file of the project. It initiates and executes the singleton `Visualizer.h` that involves the application logic. One can see different includes and object initializations in that file. `updateImageFrame` static function in the `Visualizer` class contains the pipeline for processing every frame. For fast development, we wrote the code in `.cuh` files. Some portions of the code are adapted from the exercises.

## Used Libraries
- Eigen 3 
- CUDA 10.2
- cuBlas
- FreeImage
- Kinect SDK 1.8
- glm 0.9.9.8
- glfw 3.3.2
- FreeGLUT

## Other Dependencies
We created a dynamic link library for Eigen solver, as the solver class could not be compiled by the NVCC version we use. The library can be downloaded from [here](https://drive.google.com/drive/folders/1NZOTIxDlY8AsV2r4TFKyiWV8bJXFaE4v?usp=sharing). It exports `solve` function and is only used in `PoseEstimator.cuh`.

## Note on normal estimation
Currently, we do not use normals estimated in the raycasting. We instead use the backprojection pipeline from the rendered depth. We attempted to implement a trilinear interpolated version that can be seen in the cg01 branch, but did not use it in the final version.

## Authors
- Ismet Melih Özbeyli
- Can Gümeli
- Johann Sutor
