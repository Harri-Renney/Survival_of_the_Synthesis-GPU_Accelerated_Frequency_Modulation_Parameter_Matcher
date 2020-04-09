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
#include <iostream>

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
	enum kernelNames_ { initPopulation = 0, recombinePopulation, mutatePopulation, synthesisePopulation, applyWindowPopulation, cudaFFT, fitnessPopulation, sortPopulation, copyPopulation };
	std::chrono::nanoseconds kernelExecuteTime_[numKernels_];
public:
	Evolutionary_Strategy_CUDA(Evolutionary_Strategy_CUDA_Arguments args) :
	Evolutionary_Strategy(args.es_args.numGenerations, args.es_args.pop.numParents, args.es_args.pop.numOffspring, args.es_args.pop.numDimensions, args.es_args.paramMin, args.es_args.paramMax, args.es_args.audioLengthLog2),
	kernelSourcePath_(args.kernelSourcePath),
	globalWorkspace_(dim3(population.populationLength, 1, 1)),
	localWorkspace_(args.localWorkspace)
	{
		//Set sizes//
		//targetFFTSize = objective.fftHalfSize * sizeof(float);

		//Create device buffers//
		//cudaMalloc((void**)&devicetargetFFT, targetFFTSize);
		
		initCudaFFT();
		init();
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

		//Load parameter min & max//
		cudaMemcpy(deviceParamMinBuffer_, &objective.paramMins.front(), population.numDimensions * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceParamMaxBuffer_, &objective.paramMaxs.front(), population.numDimensions * sizeof(float), cudaMemcpyHostToDevice);

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
		//cufftDestroy(plan);
		//cudaFree(data);
	}

	void initPopulationCUDA()
	{
		rotationIndex_ = 0;
		cudaMemcpy(deviceRoationIndex_, &rotationIndex_, sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		//Run initialise population kernel//
		CUDA_Kernels::initPopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, devicePopulationFitnessBuffer_, deviceRandomStatesBuffer_, rotationIndex_);
		cudaDeviceSynchronize();

		float* tempBuffer = new float[population.populationLength * population.numDimensions * 2];
		cudaMemcpy(tempBuffer, devicePopulationValueBuffer_, population.populationLength * population.numDimensions * sizeof(float) * 2, cudaMemcpyDeviceToHost);
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
		std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

		CUDA_Kernels::recombinePopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, rotationIndex_);
		//cudaDeviceSynchronize();
		
		auto end = std::chrono::steady_clock::now();
		auto diff = end - start;
		kernelExecuteTime_[recombinePopulation] += diff;

		//float* tempBuffer = new float[population.populationLength * population.numDimensions * 2];
		//cudaMemcpy(tempBuffer, devicePopulationValueBuffer_, population.populationLength * population.numDimensions * sizeof(float) * 2, cudaMemcpyDeviceToHost);

		//printBest();

		start = std::chrono::steady_clock::now();

		CUDA_Kernels::mutatePopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, deviceRandomStatesBuffer_, rotationIndex_);
		cudaDeviceSynchronize();

		end = std::chrono::steady_clock::now();
		diff = end - start;
		kernelExecuteTime_[mutatePopulation] += diff;

		//cudaMemcpy(tempBuffer, devicePopulationValueBuffer_, population.populationLength * population.numDimensions * sizeof(float) * 2, cudaMemcpyDeviceToHost);

		//printBest();

		start = std::chrono::steady_clock::now();

		CUDA_Kernels::synthesisePopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, deviceGeneratedAudioBuffer_, deviceParamMinBuffer_, deviceParamMaxBuffer_, rotationIndex_);
		cudaDeviceSynchronize();

		end = std::chrono::steady_clock::now();
		diff = end - start;
		kernelExecuteTime_[synthesisePopulation] += diff;

		//float* tempAudioBuffer = new float[population.populationLength * objective.audioLength];
		//cudaMemcpy(tempAudioBuffer, deviceGeneratedAudioBuffer_, population.populationLength * objective.audioLength * sizeof(float), cudaMemcpyDeviceToHost);

		//printBest();

		start = std::chrono::steady_clock::now();

		CUDA_Kernels::applyWindowPopulationExecute(globalWorkspace_, localWorkspace_, deviceGeneratedAudioBuffer_);
		cudaDeviceSynchronize();

		end = std::chrono::steady_clock::now();
		diff = end - start;
		kernelExecuteTime_[applyWindowPopulation] += diff;

		//cudaMemcpy(tempAudioBuffer, deviceGeneratedAudioBuffer_, population.populationLength * objective.audioLength * sizeof(float), cudaMemcpyDeviceToHost);

		start = std::chrono::steady_clock::now();

		//CudaFFT//
		executeCudaFFT();

		end = std::chrono::steady_clock::now();
		diff = end - start;
		kernelExecuteTime_[cudaFFT] += diff;

		//float* tempFFTBuffer = new float[population.populationLength * objective.fftOutSize];
		//cudaMemcpy(tempFFTBuffer, deviceGeneratedFFTBuffer_, population.populationLength * objective.fftOutSize * sizeof(float), cudaMemcpyDeviceToHost);

		start = std::chrono::steady_clock::now();

		//@ToDo - Make sure updated targetFFT and generatedFFT.
		CUDA_Kernels::fitnessPopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationFitnessBuffer_, deviceGeneratedFFTBuffer_, deviceTargetFFTBuffer_, rotationIndex_);
		cudaDeviceSynchronize();

		end = std::chrono::steady_clock::now();
		diff = end - start;
		kernelExecuteTime_[fitnessPopulation] += diff;

		start = std::chrono::steady_clock::now();

		//float* tempFitnessBuffer = new float[population.populationLength * 2];
		//cudaMemcpy(tempFitnessBuffer, devicePopulationFitnessBuffer_, population.populationLength * sizeof(float) * 2, cudaMemcpyDeviceToHost);

		CUDA_Kernels::sortPopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, devicePopulationFitnessBuffer_, rotationIndex_);
		cudaDeviceSynchronize();

		end = std::chrono::steady_clock::now();
		diff = end - start;
		kernelExecuteTime_[sortPopulation] += diff;

		//printBest();

		//@ToDo - Really need to run a kernel for this? Just rotatethe index host side!
		//CUDA_Kernels::rotatePopulationExecute(globalWorkspace_, localWorkspace_, devicePopulationValueBuffer_, devicePopulationStepBuffer_, devicePopulationFitnessBuffer_, rotationIndex_);
		rotationIndex_ = (rotationIndex_ == 0 ? 1 : 0);

		//delete tempBuffer;
		//delete tempAudioBuffer;
	}
	void executeAllGenerations()
	{
		for (uint32_t i = 0; i != numGenerations; ++i)
		{
			executeGeneration();
		}
		for (uint32_t i = 0; i != numKernels_; ++i)
		{
			std::chrono::duration<double> executeTime = std::chrono::duration<double>(kernelExecuteTime_[i]);
			//executeTime = executeTime / numGenerations;
			std::cout << "Time to complete kernel " << i << ": " << kernelExecuteTime_[i].count() / (float)1e6 << "ms\n";
		}
	}
	void parameterMatchAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{
		//Work out size of chunks and number of chunks to analyse//
		chunkSize_ = objective.audioLength;
		numChunks_ = aTargetAudioLength / chunkSize_;

		initRandomStateCUDA();
		for (int i = 0; i < numChunks_-1; i++)
		{
			//Initialise target audio and new population//
			setTargetAudio(&aTargetAudio[chunkSize_ * i], chunkSize_);
			
			
			initPopulationCUDA();

			//Execute number of ES generations on chunk//
			executeAllGenerations();

			printf("Audio chunk %d evaluated:\n", i);
			printBest();
		}
	}

	//@ToDo - When using rotation index, need check this actually prints latest best//
	void printBest()
	{
		uint32_t tempSize = 4 * sizeof(float);
		float* tempData = new float[4];
		float* tempFitness = new float[2];
		cudaMemcpy(tempData, devicePopulationValueBuffer_, population.numDimensions * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(tempFitness, devicePopulationFitnessBuffer_, 2*sizeof(float), cudaMemcpyDeviceToHost);
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
};

#endif