#ifndef EVOLUTIONARY_STRATEGY_CUDA_HPP
#define EVOLUTIONARY_STRATEGY_CUDA_HPP

#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

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

class Evolutionary_Strategy_CUDA
{
private:
	dim3 globalWorkspace_;
	dim3 localWorkspace_;

	//CUDA Profiling//
	cudaEvent_t cudaEventStart;
	cudaEvent_t cudaEventEnd;
	float cudaTimeElapsed = 0.0f;

public:
	Evolutionary_Strategy_CUDA()
	{
	}

	void runGeneralBenchmarks(uint64_t aNumRepetitions, bool isWarmup)
	{

	}
};

#endif