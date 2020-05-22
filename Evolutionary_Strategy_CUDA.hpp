#ifndef EVOLUTIONARY_STRATEGY_CUDA_HPP
#define EVOLUTIONARY_STRATEGY_CUDA_HPP

#include <cstdint>
#include <random>
#include <chrono>
#include <math.h>
#include <glm/glm.hpp>
#include <array>

#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h> 
#include <cufft.h>
#include <iostream>

#include "Evolutionary_Strategy.hpp"
#include "Benchmarker.hpp"

struct Evolutionary_Strategy_CUDA_Arguments
{
	//Generic Evolutionary Strategy arguments//
	Evolutionary_Strategy_Arguments es_args;

	//CUDA details//
	dim3 globalWorkspace;
	dim3 localWorkspace;
	std::string kernelSourcePath;
};

namespace CUDA_Kernels
{
	__constant__ int POPULATION_COUNT = 1536;
	__constant__ int POPULATION_SIZE = 1536 * (4);
	__constant__ const int NUM_DIMENSIONS = 4;
	__constant__ const int NUM_WGS_FOR_PARENTS = 6;

	__constant__ const int CHUNK_SIZE_FITNESS = (32 / 2);
	__constant__ int AUDIO_WAVE_FORM_SIZE = 1024;
	__constant__ const int CHUNKS_PER_WG_SYNTH = 1;
	__constant__ const int CHUNK_SIZE_SYNTH = 32 / 1;
	__constant__ const float ONE_OVER_SAMPLE_RATE_TIMES_2_PI = 0.00014247573;

	__constant__ int FFT_OUT_SIZE = 1026;
	__constant__ int FFT_HALF_SIZE = 512;
	__constant__ float FFT_ONE_OVER_SIZE = 1 / 1026.0;
	__constant__ const float FFT_ONE_OVER_WINDOW_FACTOR = 1.0;

	__constant__ const float ALPHA = 1.4;
	__constant__ const float ONE_OVER_ALPHA = 1 / 1.4;
	__constant__ const float ROOT_TWO_OVER_PI = 0.797884524;
	__constant__ const float BETA_SCALE = 0.25;
	__constant__ const float BETA = 0.5;
	__constant__ int WAVETABLE_SIZE = 32768;

	void setConstants(uint32_t& aPopulationCount, uint32_t& aPopulationSize, uint32_t& aNumDimensions, uint32_t& aAudioWaveformSize, uint32_t& aFFTSize, uint32_t& aFFTHalfSize, float& aFFTOneOverSize, const uint32_t& aWavetableSize);

	void initPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, float* aPopulationFitness, uint2* const aRandState, uint32_t aRotationIndex);

	void recombinePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, uint32_t aRotationIndex);

	void mutatePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, uint2* rand_state, uint32_t aRotationIndex);

	void synthesisePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aOutputAudioWaves, const float* aParamMins, const float* aParamMaxs, const float* aWavetable, uint32_t aRotationIndex);

	void applyWindowPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aAudioWaves);

	void fitnessPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationFitness, float* aAudioWaveFFT, float* aTargetFFT, uint32_t aRotationIndex);

	void sortPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, float* aPopulationFitness, uint32_t aRotationIndex);
}

class Evolutionary_Strategy_CUDA : public Evolutionary_Strategy
{
private:
	std::string kernelSourcePath_;
	dim3 globalWorkspace_;
	dim3 localWorkspace_;

	//Variables for evaluating audio in chunks//
	uint32_t numChunks_;
	uint32_t chunkSize_;

	uint32_t targetAudioLength;
	float* targetAudio;
	float* targetFFT_;

	uint32_t rotationIndex_;

	//CUDA Device Buffers//
	uint32_t targetFFTSize;

	float* devicePopulationValueBuffer_;
	float* devicePopulationStepBuffer_;
	float* devicePopulationFitnessBuffer_;
	uint2* deviceRandomStatesBuffer_;
	float* deviceParamMinBuffer_;
	float* deviceParamMaxBuffer_;
	float* deviceGeneratedAudioBuffer_;
	uint32_t* deviceRoationIndex_;
	float* deviceWavetableBuffer_;
	float* deviceGeneratedFFTBuffer_;
	float* deviceTargetFFTBuffer_;

	//@ToDo - Work out use for constant memory//
	//__constant__ float* deviceParamMinBuffer_;
	//__constant__ float* deviceParamMinBuffer_;

	//CUDA Profiling//
	cudaEvent_t cudaEventStart;
	cudaEvent_t cudaEventEnd;
	float cudaTimeElapsed = 0.0f;

	static const uint8_t numKernels_ = 9;
	enum kernels_ { initPopulation = 0, recombinePopulation, mutatePopulation, synthesisePopulation, applyWindowPopulation, cudaFFT, fitnessPopulation, sortPopulation, copyPopulation };
	std::array<std::string, numKernels_> kernelNames_;
	std::chrono::nanoseconds kernelExecuteTime_[numKernels_];

	Benchmarker cudaBenchmarker_;
public:
	Evolutionary_Strategy_CUDA(Evolutionary_Strategy_CUDA_Arguments args) :
		Evolutionary_Strategy(args.es_args.numGenerations, args.es_args.pop.numParents, args.es_args.pop.numOffspring, args.es_args.pop.numDimensions, args.es_args.paramMin, args.es_args.paramMax, args.es_args.audioLengthLog2),
		cudaBenchmarker_(std::string("cudalog(pop=" + std::to_string(args.es_args.pop.populationLength) + "gens=" + std::to_string(args.es_args.numGenerations) + "audioBlockSize=" + std::to_string(1 << args.es_args.audioLengthLog2) + ").csv"), { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" }),
		kernelSourcePath_(args.kernelSourcePath),
		globalWorkspace_(dim3(population.populationLength, 1, 1)),
		localWorkspace_(args.localWorkspace),
		kernelNames_({ "initPopulation", "recombinePopulation", "mutatePopulation", "synthesisePopulation", "applyWindowPopulation", "cudaFFT", "fitnessPopulation", "sortPopulation", "copyPopulation" })
	{
		//Set sizes//
		//targetFFTSize = objective.fftHalfSize * sizeof(float);

		//Create device buffers//
		//cudaMalloc((void**)&devicetargetFFT, targetFFTSize);

		initCudaFFT();
		init();
	}

	~Evolutionary_Strategy_CUDA()
	{
		cufftDestroy(fftplan_);
		cudaFree(devicePopulationValueBuffer_);
		cudaFree(devicePopulationStepBuffer_);
		cudaFree(devicePopulationFitnessBuffer_);
		cudaFree(deviceRandomStatesBuffer_);
		cudaFree(deviceParamMinBuffer_);
		cudaFree(deviceParamMaxBuffer_);
		cudaFree(deviceGeneratedAudioBuffer_);
		cudaFree(deviceRoationIndex_);
		cudaFree(deviceWavetableBuffer_);
	}

	void init()
	{
		//@ToDo - Allocate the correct kind of memory for device//
		cudaMalloc((void**)&devicePopulationValueBuffer_, population.populationLength * population.numDimensions * sizeof(float) * 2);
		cudaMalloc((void**)&devicePopulationStepBuffer_, population.populationLength * population.numDimensions * sizeof(float) * 2);
		cudaMalloc((void**)&devicePopulationFitnessBuffer_, population.populationLength * sizeof(float) * 2);
		cudaMalloc((void**)&deviceRandomStatesBuffer_, population.populationLength * sizeof(uint2));
		cudaMalloc((void**)&deviceParamMinBuffer_, population.numDimensions * sizeof(float));
		cudaMalloc((void**)&deviceParamMaxBuffer_, population.numDimensions * sizeof(float));
		cudaMalloc((void**)&deviceGeneratedAudioBuffer_, population.populationLength * objective.audioLength * sizeof(float));
		cudaMallocHost((void**)&deviceRoationIndex_, sizeof(uint32_t));
		cudaMalloc((void**)&deviceWavetableBuffer_, objective.wavetableSize * sizeof(float));

		targetFFT_ = new float[objective.fftHalfSize];

		//Load parameter min & max//
		cudaMemcpy(deviceParamMinBuffer_, &objective.paramMins.front(), population.numDimensions * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceParamMaxBuffer_, &objective.paramMaxs.front(), population.numDimensions * sizeof(float), cudaMemcpyHostToDevice);

		//Load generated wavetable to GPU//
		cudaMemcpy(deviceWavetableBuffer_, objective.wavetable, objective.wavetableSize * sizeof(float), cudaMemcpyHostToDevice);

		initDeviceMemory();
	}
	void initDeviceMemory()
	{
		//What needed in here?
	}

	void initCudaFFT()
	{
		const int BATCH = population.populationLength;
		const int RANK = 1;
		int NX = objective.audioLength;

		//clFFT Variables//
		//clfftDim dim = CLFFT_1D;
		size_t clLengths[1] = { objective.audioLength };
		size_t in_strides[1] = { 1 };
		size_t out_strides[1] = { 1 };
		size_t in_dist = (size_t)objective.audioLength;
		size_t out_dist = (size_t)objective.audioLength / 2 + 4;

		//Update member variables with new information//
		objective.fftOutSize = out_dist * 2;
		cudaMalloc((void**)&deviceGeneratedFFTBuffer_, population.populationLength * objective.fftOutSize * sizeof(float));
		cudaMalloc((void**)&deviceTargetFFTBuffer_, objective.fftHalfSize * sizeof(float));

		//cufftPlanMany(&plan, RANK, &NX, NULL, NULL, in_dist,NULL, NULL, out_dist, CUFFT_R2C, BATCH);
		//cufftPlanMany(&fftplan_, RANK, &NX, NULL, *in_strides, in_dist, NULL, *out_strides, out_dist, CUFFT_R2C, BATCH);	//@ToDo - Need to check this works, as is not hermitian_interleaved as in ClFFT? Just real to complex?
		cufftPlan1d(&fftplan_, NX, CUFFT_R2C, BATCH);
	}
	cufftHandle fftplan_;
	void executeCudaFFT()
	{
		cufftExecR2C(fftplan_, deviceGeneratedAudioBuffer_, reinterpret_cast<cufftComplex *>(deviceGeneratedFFTBuffer_));
		cudaDeviceSynchronize();
	}

	void initPopulationCUDA()
	{
		rotationIndex_ = 0;
		cudaMemcpy(deviceRoationIndex_, &rotationIndex_, sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		//Run initialise population kernel//
		CUDA_Kernels::initPopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, devicePopulationFitnessBuffer_, deviceRandomStatesBuffer_, rotationIndex_);
		cudaDeviceSynchronize();
	}

	void initRandomStateCUDA()
	{
		//Initialize random numbers in CPU buffer//
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		//std::uniform_int_distribution<int> distribution(0, 2147483647);
		std::uniform_int_distribution<int> distribution(0, 32767);

		uint32_t numRandomStates = population.populationLength;
		glm::uvec2* rand_state = new glm::uvec2[numRandomStates];
		for (int i = 0; i < numRandomStates; ++i)
		{
			rand_state[i].x = distribution(generator);
			rand_state[i].y = distribution(generator);
		}

		//Write random states to GPU randomStatesBuffer//
		uint32_t cpySize = numRandomStates * sizeof(uint2);
		cudaMemcpy(deviceRandomStatesBuffer_, rand_state, cpySize, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		delete(rand_state);
	}

	void setTargetAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{
		//Calculate and load fft data for target audio//
		targetAudioLength = aTargetAudioLength;
		objective.calculateFFT(aTargetAudio, targetFFT_);
		cudaMemcpy(deviceTargetFFTBuffer_, targetFFT_, objective.fftHalfSize * sizeof(float), cudaMemcpyHostToDevice);
	}


	void executeGeneration()
	{
		cudaBenchmarker_.startTimer(kernelNames_[1]);
		CUDA_Kernels::recombinePopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, rotationIndex_);
		cudaDeviceSynchronize();
		cudaBenchmarker_.pauseTimer(kernelNames_[1]);

		cudaBenchmarker_.startTimer(kernelNames_[2]);
		CUDA_Kernels::mutatePopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, deviceRandomStatesBuffer_, rotationIndex_);
		cudaDeviceSynchronize();
		cudaBenchmarker_.pauseTimer(kernelNames_[2]);

		cudaBenchmarker_.startTimer(kernelNames_[3]);
		CUDA_Kernels::synthesisePopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, deviceGeneratedAudioBuffer_, deviceParamMinBuffer_, deviceParamMaxBuffer_, deviceWavetableBuffer_, rotationIndex_);
		cudaDeviceSynchronize();
		cudaBenchmarker_.pauseTimer(kernelNames_[3]);

		cudaBenchmarker_.startTimer(kernelNames_[4]);
		CUDA_Kernels::applyWindowPopulationExecute(globalWorkspace_, localWorkspace_, deviceGeneratedAudioBuffer_);
		cudaDeviceSynchronize();
		cudaBenchmarker_.pauseTimer(kernelNames_[4]);

		//CudaFFT//
		cudaBenchmarker_.startTimer(kernelNames_[5]);
		executeCudaFFT();
		cudaBenchmarker_.pauseTimer(kernelNames_[5]);

		cudaBenchmarker_.startTimer(kernelNames_[6]);
		CUDA_Kernels::fitnessPopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationFitnessBuffer_, deviceGeneratedFFTBuffer_, deviceTargetFFTBuffer_, rotationIndex_);
		cudaDeviceSynchronize();
		cudaBenchmarker_.pauseTimer(kernelNames_[6]);

		cudaBenchmarker_.startTimer(kernelNames_[7]);
		CUDA_Kernels::sortPopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, devicePopulationFitnessBuffer_, rotationIndex_);
		cudaDeviceSynchronize();
		cudaBenchmarker_.pauseTimer(kernelNames_[7]);

		cudaBenchmarker_.startTimer(kernelNames_[8]);
		rotationIndex_ = (rotationIndex_ == 0 ? 1 : 0);
		cudaBenchmarker_.pauseTimer(kernelNames_[8]);
	}
	void executeAllGenerations()
	{
		for (uint32_t i = 0; i != numGenerations; ++i)
		{
			executeGeneration();
		}
	}
	void parameterMatchAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{
		//Work out size of chunks and number of chunks to analyse//
		chunkSize_ = objective.audioLength;
		numChunks_ = aTargetAudioLength / chunkSize_;

		uint32_t modifiedFFTSize = objective.fftSize + 2;
		float modifiedFFTOneOverSize = 1 / (float)(objective.fftSize + 2);
		CUDA_Kernels::setConstants(population.populationLength, population.populationSize, population.numDimensions, objective.audioLength, modifiedFFTSize, objective.fftHalfSize, modifiedFFTOneOverSize, objective.wavetableSize);

		cudaBenchmarker_.startTimer("Total Audio Analysis Time");
		cudaBenchmarker_.pauseTimer("Total Audio Analysis Time");
		cudaBenchmarker_.startTimer("Total Audio Analysis Time");

		initRandomStateCUDA();
		for (int i = 0; i < numChunks_; i++)
		{
			//Initialise target audio and new population//
			setTargetAudio(&aTargetAudio[chunkSize_ * i], chunkSize_);

			initPopulationCUDA();

			//Execute number of ES generations on chunk//
			executeAllGenerations();

			printf("Audio chunk %d evaluated:\n", i);
			printBest();
		}
		cudaBenchmarker_.pauseTimer("Total Audio Analysis Time");

		cudaBenchmarker_.elapsedTimer(kernelNames_[1]);
		cudaBenchmarker_.elapsedTimer(kernelNames_[2]);
		cudaBenchmarker_.elapsedTimer(kernelNames_[3]);
		cudaBenchmarker_.elapsedTimer(kernelNames_[4]);
		cudaBenchmarker_.elapsedTimer(kernelNames_[5]);
		cudaBenchmarker_.elapsedTimer(kernelNames_[6]);
		cudaBenchmarker_.elapsedTimer(kernelNames_[7]);
		cudaBenchmarker_.elapsedTimer(kernelNames_[8]);
		cudaBenchmarker_.elapsedTimer("Total Audio Analysis Time");
	}

	//@ToDo - When using rotation index, need check this actually prints latest best//
	void printBest()
	{
		uint32_t tempSize = 4 * sizeof(float);
		float* tempData = new float[4];
		float* tempFitness = new float[2];
		cudaMemcpy(tempData, devicePopulationValueBuffer_, population.numDimensions * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(tempFitness, devicePopulationFitnessBuffer_, 2 * sizeof(float), cudaMemcpyDeviceToHost);
		printf("Best parameters found:\n Fc = %f\n I = %f\n Fm = %f\n A = %f\nFitness=%f\n\n", tempData[0] * objective.paramMaxs[0], tempData[1] * objective.paramMaxs[1], tempData[2] * objective.paramMaxs[2], tempData[3] * objective.paramMaxs[3], tempFitness[0]);


		delete(tempData);
	}

	void readPopulationData(void* aInputPopulationValueData, void* aOutputPopulationValueData, uint32_t aPopulationValueSize, void* aInputPopulationStepData, void* aOutputPopulationStepData, uint32_t aPopulationStepSize, void* aInputPopulationFitnessData, void* aOutputPopulationFitnessData, uint32_t aPopulationFitnessSize)
	{
		cudaMemcpy(aInputPopulationValueData, devicePopulationValueBuffer_, aPopulationValueSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(aInputPopulationStepData, devicePopulationStepBuffer_, aPopulationStepSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(aInputPopulationFitnessData, devicePopulationFitnessBuffer_, aPopulationFitnessSize, cudaMemcpyDeviceToHost);
	}
	void readSynthesizerData(void* aOutputAudioBuffer, uint32_t aOutputAudioSize, void* aInputFFTDataBuffer, void* aInputFFTTargetBuffer, uint32_t aInputFFTSize)
	{
		cudaMemcpy(aOutputAudioBuffer, deviceGeneratedAudioBuffer_, aOutputAudioSize, cudaMemcpyDeviceToHost);
	}

	static void printAvailableDevices()
	{
		printf(
			" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

		int deviceCount = 0;
		cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

		if (error_id != cudaSuccess) {
			printf("cudaGetDeviceCount returned %d\n-> %s\n",
				static_cast<int>(error_id), cudaGetErrorString(error_id));
			printf("Result = FAIL\n");
			//exit(EXIT_FAILURE);
		}

		// This function call returns 0 if there are no CUDA capable devices.
		if (deviceCount == 0) {
			printf("There are no available device(s) that support CUDA\n");
		}
		else {
			printf("Detected %d CUDA Capable device(s)\n", deviceCount);
		}

		int dev, driverVersion = 0, runtimeVersion = 0;

		for (dev = 0; dev < deviceCount; ++dev) {
			cudaSetDevice(dev);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);

			printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

			// Console log
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);
			printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
				driverVersion / 1000, (driverVersion % 100) / 10,
				runtimeVersion / 1000, (runtimeVersion % 100) / 10);
			printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
				deviceProp.major, deviceProp.minor);

			char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			sprintf_s(msg, sizeof(msg),
				"  Total amount of global memory:                 %.0f MBytes "
				"(%llu bytes)\n",
				static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
				(unsigned long long)deviceProp.totalGlobalMem);
#else
			snprintf(msg, sizeof(msg),
				"  Total amount of global memory:                 %.0f MBytes "
				"(%llu bytes)\n",
				static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
				(unsigned long long)deviceProp.totalGlobalMem);
#endif
			printf("%s", msg);

			printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
				deviceProp.multiProcessorCount,
				_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
				_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
				deviceProp.multiProcessorCount);
			printf(
				"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
				"GHz)\n",
				deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
			// This is supported in CUDA 5.0 (runtime API device properties)
			printf("  Memory Clock rate:                             %.0f Mhz\n",
				deviceProp.memoryClockRate * 1e-3f);
			printf("  Memory Bus Width:                              %d-bit\n",
				deviceProp.memoryBusWidth);

			if (deviceProp.l2CacheSize) {
				printf("  L2 Cache Size:                                 %d bytes\n",
					deviceProp.l2CacheSize);
			}

#else
			// This only available in CUDA 4.0-4.2 (but these were only exposed in the
			// CUDA Driver API)
			int memoryClock;
			getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
				dev);
			printf("  Memory Clock rate:                             %.0f Mhz\n",
				memoryClock * 1e-3f);
			int memBusWidth;
			getCudaAttribute<int>(&memBusWidth,
				CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
			printf("  Memory Bus Width:                              %d-bit\n",
				memBusWidth);
			int L2CacheSize;
			getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

			if (L2CacheSize) {
				printf("  L2 Cache Size:                                 %d bytes\n",
					L2CacheSize);
			}

#endif

			printf(
				"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
				"%d), 3D=(%d, %d, %d)\n",
				deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
				deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
				deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
			printf(
				"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
				deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
			printf(
				"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
				"layers\n",
				deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
				deviceProp.maxTexture2DLayered[2]);

			printf("  Total amount of constant memory:               %zu bytes\n",
				deviceProp.totalConstMem);
			printf("  Total amount of shared memory per block:       %zu bytes\n",
				deviceProp.sharedMemPerBlock);
			printf("  Total number of registers available per block: %d\n",
				deviceProp.regsPerBlock);
			printf("  Warp size:                                     %d\n",
				deviceProp.warpSize);
			printf("  Maximum number of threads per multiprocessor:  %d\n",
				deviceProp.maxThreadsPerMultiProcessor);
			printf("  Maximum number of threads per block:           %d\n",
				deviceProp.maxThreadsPerBlock);
			printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
				deviceProp.maxThreadsDim[2]);
			printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2]);
			printf("  Maximum memory pitch:                          %zu bytes\n",
				deviceProp.memPitch);
			printf("  Texture alignment:                             %zu bytes\n",
				deviceProp.textureAlignment);
			printf(
				"  Concurrent copy and kernel execution:          %s with %d copy "
				"engine(s)\n",
				(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
			printf("  Run time limit on kernels:                     %s\n",
				deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
			printf("  Integrated GPU sharing Host Memory:            %s\n",
				deviceProp.integrated ? "Yes" : "No");
			printf("  Support host page-locked memory mapping:       %s\n",
				deviceProp.canMapHostMemory ? "Yes" : "No");
			printf("  Alignment requirement for Surfaces:            %s\n",
				deviceProp.surfaceAlignment ? "Yes" : "No");
			printf("  Device has ECC support:                        %s\n",
				deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
				deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
				: "WDDM (Windows Display Driver Model)");
#endif
			printf("  Device supports Unified Addressing (UVA):      %s\n",
				deviceProp.unifiedAddressing ? "Yes" : "No");
			printf("  Device supports Compute Preemption:            %s\n",
				deviceProp.computePreemptionSupported ? "Yes" : "No");
			printf("  Supports Cooperative Kernel Launch:            %s\n",
				deviceProp.cooperativeLaunch ? "Yes" : "No");
			printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
				deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
			printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
				deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

			const char *sComputeMode[] = {
				"Default (multiple host threads can use ::cudaSetDevice() with device "
				"simultaneously)",
				"Exclusive (only one host thread in one process is able to use "
				"::cudaSetDevice() with this device)",
				"Prohibited (no host thread can use ::cudaSetDevice() with this "
				"device)",
				"Exclusive Process (many threads in one process is able to use "
				"::cudaSetDevice() with this device)",
				"Unknown",
				NULL };
			printf("  Compute Mode:\n");
			printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
		}

		// If there are 2 or more GPUs, query to determine whether RDMA is supported
		if (deviceCount >= 2) {
			cudaDeviceProp prop[64];
			int gpuid[64];  // we want to find the first two GPUs that can support P2P
			int gpu_p2p_count = 0;

			for (int i = 0; i < deviceCount; i++) {
				//checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));
				cudaGetDeviceProperties(&prop[i], i);

				// Only boards based on Fermi or later can support P2P
				if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
					// on Windows (64-bit), the Tesla Compute Cluster driver for windows
					// must be enabled to support this
					&& prop[i].tccDriver
#endif
					) {
					// This is an array of P2P capable GPUs
					gpuid[gpu_p2p_count++] = i;
				}
			}

			// Show all the combinations of support P2P GPUs
			int can_access_peer;

			if (gpu_p2p_count >= 2) {
				for (int i = 0; i < gpu_p2p_count; i++) {
					for (int j = 0; j < gpu_p2p_count; j++) {
						if (gpuid[i] == gpuid[j]) {
							continue;
						}
						//checkCudaErrors(
						//	cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
						cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]);
						printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
							prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
							can_access_peer ? "Yes" : "No");
					}
				}
			}
		}

		// csv masterlog info
		// *****************************
		// exe and CUDA driver name
		printf("\n");
		std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
		char cTemp[16];

		// driver version
		sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#else
		snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
			(driverVersion % 100) / 10);
#endif
		sProfileString += cTemp;

		// Runtime version
		sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#else
		snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
			(runtimeVersion % 100) / 10);
#endif
		sProfileString += cTemp;

		// Device count
		sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(cTemp, 10, "%d", deviceCount);
#else
		snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
		sProfileString += cTemp;
		sProfileString += "\n";
		printf("%s", sProfileString.c_str());

		printf("Result = PASS\n");

	}
};

#endif