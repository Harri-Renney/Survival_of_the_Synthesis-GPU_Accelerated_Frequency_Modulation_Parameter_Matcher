#ifndef EVOLUTIONARY_STRATEGY_CUDA_HPP
#define EVOLUTIONARY_STRATEGY_CUDA_HPP

#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cufft.h>

struct Evolutionary_Strategy_CUDA_Arguments
{
	//Generic Evolutionary Strategy arguments//
	Evolutionary_Strategy_Arguments es_args;

	//CUDA details//
	dim3 globalWorkspace;
	dim3 localWorkspace;
	std::string kernelSourcePath;

	DeviceType deviceType;
};

namespace CUDA_Kernels
{
	void initPopulation(float* population_values,
		float* population_steps,
		float* population_fitnesses,
		uint2* const rand_state,
		uint32_t rotationIndex);

	void recombinePopulation(float* population_values,
		float* population_steps,
		uint32_t rotationIndex);

	void mutatePopulation(float* in_population_values,
		float* in_population_steps,
		uint2* rand_state,
		uint32_t rotationIndex);

	void synthesisePopulation(float* out_audio_waves,
		float* in_population_values,
		__constant__ float* param_mins, __constant__ float* param_maxs,
		uint32_t rotationIndex);

	void applyWindowPopulation(float* audio_waves);

	void fitnessPopulation(float* out_population_fitnesses, float* in_fft_data,
		__constant__ float* in_fft_target,
		uint32_t rotationIndex);

	void sortPopulation(float* in_population_values, float* in_population_steps,
		float* in_population_fitnesses,
		float* out_population_values, float* out_population_steps,
		float* out_population_fitnesses,
		uint32_t rotationIndex);

	void rotatePopulation(float* in_population_values, float* in_population_steps,
		float* in_population_fitnesses,
		float* out_population_values, float* out_population_steps,
		float* out_population_fitnesses,
		uint32_t rotationIndex);
}

class Evolutionary_Strategy_CUDA : Evolutionary_Strategy
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

	void runGeneralBenchmarks(uint64_t aNumRepetitions, bool isWarmup)
	{

	}

	void cudaFFT()
	{
		const int BATCH = 10;	//Population Size?
		const int RANK = 1;
		int NX = 256;

		//clFFT Variables//
		clfftDim dim = CLFFT_1D;
		size_t clLengths[1] = { objective.audioLength };
		size_t in_strides[1] = { 1 };
		size_t out_strides[1] = { 1 };
		size_t in_dist = (size_t)objective.audioLength;
		size_t out_dist = (size_t)objective.audioLength / 2 + 4;

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