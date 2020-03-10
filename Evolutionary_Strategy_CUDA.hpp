#ifndef EVOLUTIONARY_STRATEGY_CUDA_HPP
#define EVOLUTIONARY_STRATEGY_CUDA_HPP

#include <cstdint>
#include <random>
#include <chrono>
#include <math.h>
#include <glm/glm.hpp>

#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include "Evolutionary_Strategy.hpp"

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
	void initPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, float* aPopulationFitness, uint2* const aRandState, uint32_t aRotationIndex);

	void recombinePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, uint32_t aRotationIndex);

	void mutatePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, uint2* rand_state, uint32_t aRotationIndex);

	void synthesisePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aOutputAudioWaves, const float* aParamMins, const float* aParamMaxs, uint32_t aRotationIndex);

	void applyWindowPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aAudioWaves);

	void fitnessPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationFitness, float* aAudioWaveFFT, float* aTargetFFT, uint32_t aRotationIndex);

	void sortPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, float* aPopulationFitness, uint32_t aRotationIndex);

	void rotatePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, float* aPopulationFitness, uint32_t aRotationIndex);
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
	float* devicetargetFFT;

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

public:
	Evolutionary_Strategy_CUDA(Evolutionary_Strategy_CUDA_Arguments args) :
	Evolutionary_Strategy(args.es_args.numGenerations, args.es_args.pop.numParents, args.es_args.pop.numOffspring, args.es_args.pop.numDimensions, args.es_args.paramMin, args.es_args.paramMax, args.es_args.audioLengthLog2),
	kernelSourcePath_(args.kernelSourcePath),
	globalWorkspace_(args.globalWorkspace),
	localWorkspace_(args.localWorkspace)
	{
		//Set sizes//
		targetFFTSize = objective.fftHalfSize * sizeof(float);

		//Create device buffers//
		cudaMalloc((void**)&devicetargetFFT, targetFFTSize);
		
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
		cudaMalloc((void**)&deviceRoationIndex_, sizeof(uint32_t));
		cudaMalloc((void**)&deviceWavetableBuffer_, objective.wavetableSize * sizeof(float));

		targetFFT_ = new float[objective.fftHalfSize];

		initDeviceMemory();
	}
	void initDeviceMemory()
	{
		//What needed in here?
	}

	void cudaFFT()
	{
		const int BATCH = 10;	//Population Size?
		const int RANK = 1;
		int NX = 256;

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

		cufftHandle plan;
		cufftComplex *data;
		cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
		//cufftPlanMany(&plan, RANK, &NX, NULL, NULL, in_dist,NULL, NULL, out_dist, CUFFT_R2C, BATCH);
		cufftPlanMany(&plan, RANK, &NX, NULL, *in_strides, in_dist, NULL, *out_strides, out_dist, CUFFT_R2C, BATCH);	//@ToDo - Need to check this works, as is not hermitian_interleaved as in ClFFT? Just real to complex?
		cufftExecC2C(plan, data, data, CUFFT_FORWARD);
		cudaDeviceSynchronize();
		cufftDestroy(plan);
		cudaFree(data);
	}

	void initPopulationCUDA()
	{
		rotationIndex_ = 0;
		cudaMemcpy(deviceRoationIndex_, &rotationIndex_, sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		//Run initialise population kernel//
		CUDA_Kernels::initPopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, devicePopulationFitnessBuffer_, deviceRandomStatesBuffer_, *deviceRoationIndex_);
		cudaDeviceSynchronize();
	}

	void initRandomStateCL()
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
		cudaMemcpy(devicetargetFFT, targetFFT_, objective.fftHalfSize * sizeof(float), cudaMemcpyHostToDevice);
	}
	void parameterMatchAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{
		//Work out size of chunks and number of chunks to analyse//
		chunkSize_ = objective.audioLength;
		numChunks_ = aTargetAudioLength / chunkSize_;

		for (int i = 0; i < numChunks_; i++)
		{
			//Initialise target audio and new population//
			setTargetAudio(&aTargetAudio[chunkSize_ * i], chunkSize_);
			//initKernelArgumentsCL();
			initPopulationCUDA();

			//Execute number of ES generations on chunk//
			executeAllGenerations();

			printf("Audio chunk %d evaluated:\n", i);
			printBest();
		}
	}
};

#endif