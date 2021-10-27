# Survival_of_the_Synthesis-GPU_Accelerated_Frequency_Modulation_Parameter_Matcher

This repository provides the implementation of the benchmarking suite used in the paper "Survival of the Synthesis - GPU Accelerating Evolutionary Sound Matching"



## Installation

The source code provided includes cpp files and the GPU kernel/shader.

``` git clone https://github.com/Harri-Renney/Survival_of_the_Synthesis-GPU_Accelerated_Frequency_Modulation_Parameter_Matcher.git```

Once cloned, you will need to setup the build using your chosen environment and platform. Make sure to include the following dependencies:

* [libfftw3-3.lib](https://www.fftw.org/)
* [libsndfile-1.lib](http://www.mega-nerd.com/libsndfile/)
* [clFFT.lib](https://github.com/clMathLibraries/clFFT)
* [OpenCL.lib](https://www.khronos.org/opencl/)

## Datasets

The Directory "datasets" contains .csv files containing the performance profiles collected from the benchmarking suite for the following systems:

| Specification    | Mid-rangeLaptop        | High-end NVIDIA GeForce        |
| -----------      | -----------            | --------------------------     |
| CPU              | IntelCorei7-8550U      | Intel Core it-9800X(16SM)      |
| Integrated GPU   | IntelUHDGraphics620    | None                           |
| Discrete GPU     | AMDRadeon530           | GeForce RTX2080Ti (32SM)       |
| CPU RAM          | 8GB                    | 32GB                           |   

## ToDo

* Setup cross-platform build environment using cmake.
* Support different GPU processing backends CUDA, Vulkan and OpenGL.