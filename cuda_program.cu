#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Evolutionary_Strategy_CUDA.hpp"
#define M_PI	3.14159265358979323846
#define M_E		2.718281828459

__constant__ const int POPULATION_COUNT = 1536;
__constant__ const int POPULATION_SIZE = 1536 * (4);
__constant__ const int NUM_DIMENSIONS = 4;
__constant__ const int WRKGRPSIZE = 32;
__constant__ const int NUM_WGS_FOR_PARENTS = 6;

__constant__ const int CHUNK_SIZE_FITNESS = (32 / 2);
__constant__ const int AUDIO_WAVE_FORM_SIZE = 1024;
__constant__ const int CHUNKS_PER_WG_SYNTH = 1;
__constant__ const int CHUNK_SIZE_SYNTH = 32 / 1;
__constant__ const float ONE_OVER_SAMPLE_RATE_TIMES_2_PI = 0.00014247573;

__constant__ const int FFT_OUT_SIZE = 1023;
__constant__ const int FFT_HALF_SIZE = 512;
__constant__ const float FFT_ONE_OVER_SIZE = 1 / 1024.0;
__constant__ const float FFT_ONE_OVER_WINDOW_FACTOR = 1.0;

__constant__ const float ALPHA = 1.4;
__constant__ const float ONE_OVER_ALPHA = 1 / 1.4;
__constant__ const float ROOT_TWO_OVER_PI = 0.797884524;
__constant__ const float BETA_SCALE = 0.25;
__constant__ const float BETA = 0.5;



/*
 *  Random number generator
 *  source: http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
 */
__device__ uint32_t MWC64X(uint2* state)
{
	enum _dummy { A = 4294883355U };
	//unsigned int A = 4294883355U ;
	uint32_t x = (*state).x, c = (*state).y;  // Unpack the state
	uint32_t res = x ^ c;                     // Calculate the result
	uint32_t hi = __umulhi(x, A);              // Step the RNG
	x = x * A + c;
	c = hi + (x < c);
	uint2 packOfTwo = { x,c };
	*state = packOfTwo;               // Pack the state back up
	return res;                       // Return the next result
}

/*
 *	http://c-faq.com/lib/gaussian.html
*/
__device__ float gauss_rand(uint2* rand_state)
{
	float sum = 0.0f;
	int tmp_rand;
	for (int i = 0; i < 12; i++)
	{
		sum += (float)((int)MWC64X(rand_state)) / 2147483647.0f;
	}
	sum /= 12.0f;
	return sum;
}

__global__ void initPopulation(float* population_values,
	 float* population_steps,
	 float* population_fitnesses,
	 uint2* const rand_state,
	 uint32_t rotationIndex)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t populationStartIndex = rotationIndex * POPULATION_SIZE;

	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		population_steps[populationStartIndex + (idx * NUM_DIMENSIONS + i)] = 0.1;
		//float rand = random(rand_state[index]);
		//float rand = rand_state[index].x;
		float rand = (float)((int)MWC64X(&rand_state[idx])) / 2147483647.0f;
		//float rand = (float)(rand_state[index].x / 2147483647.0f);
		//float rand = rand_state[index].x / 2147483647.0f;
		population_values[populationStartIndex + (idx * NUM_DIMENSIONS + i)] = (rand < 0.0f) ? -rand : rand;
	}
}

__global__ void recombinePopulation(float* population_values,
	float* population_steps,
	uint32_t rotationIndex)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int local_index = threadIdx.x;
	int group_index = blockIdx.x;

	uint32_t populationStartIndex = rotationIndex * POPULATION_SIZE;

	/* Local arrays to hold a section of the parent population.
	We have source and destination arrays to prevent workitems overwriting
	cells which have not yet been read by other workitems */
	__shared__ float group_population_values[NUM_DIMENSIONS * WRKGRPSIZE];
	__shared__ float group_population_steps[NUM_DIMENSIONS * WRKGRPSIZE];
	__shared__ float group_population_values_recombined[NUM_DIMENSIONS * WRKGRPSIZE];
	__shared__ float group_population_steps_recombined[NUM_DIMENSIONS * WRKGRPSIZE];

	/* This kernel runs with the population size number of workitems but we only
	want to recombine the parent population.

	A lot of the workitems will have a local index greater than the parent
	population.

	So we change the group number of groups which are aligned with offspring so
	that they point at parents. */
	int group_id_mod = group_index % NUM_WGS_FOR_PARENTS;

	/* Load parent population data into local memory.
	Groups with the same group_id_mod value will load the same set of parents */

	int global_block_start_index = group_id_mod * WRKGRPSIZE * NUM_DIMENSIONS;
	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		int local_block_index = WRKGRPSIZE * i + local_index;
		group_population_values[local_block_index] = population_values[populationStartIndex + (global_block_start_index +
			local_block_index)];
		group_population_steps[local_block_index] = population_steps[populationStartIndex + (global_block_start_index +
			local_block_index)];
	}

	/* We now have every workgroup storing a contiguous section of the parent
	population in local memory. */

	int start_idx = local_index * NUM_DIMENSIONS;
	int shift_amt;
	int new_idx;

	/* This is where recombination happens.
	Each value and step is shifted to a new position in the local data.
	The source and destination indices are determined by the group id and the
	dimension of the value.

	TODO: So there is an important limitation for this method of recombination:
	 - Mixing is restricted to the section of the parent population
	 contained in the workgroup's local memory.
	 - Mixing within workgroups is not random and may not even be good. */
	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		shift_amt = NUM_DIMENSIONS * (i * (group_index + 1));
		new_idx = (start_idx + shift_amt) % (WRKGRPSIZE * NUM_DIMENSIONS);
		group_population_values_recombined[new_idx] = group_population_values[start_idx];
		group_population_steps_recombined[new_idx] = group_population_steps[start_idx];
		start_idx++;
	}

	/* Now load the recombined population back into global memory. */
	global_block_start_index = group_index * WRKGRPSIZE * NUM_DIMENSIONS;
	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		int local_block_index = WRKGRPSIZE * i + local_index;
		population_values[populationStartIndex + (global_block_start_index + local_block_index)] =
			group_population_values_recombined[local_block_index];
		population_steps[populationStartIndex + (global_block_start_index + local_block_index)] =
			group_population_steps_recombined[local_block_index];
	}
}

__global__ void mutatePopulation(float* in_population_values,
	float* in_population_steps,
	uint2* rand_state,
	uint32_t rotationIndex)
{
	uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	int local_index = threadIdx.x;
	int group_index = blockIdx.x;

	uint32_t populationStartIndex = rotationIndex * POPULATION_SIZE;

	//in_population_values[populationStartIndex + index]= MWC64X(&rand_state[index]) /  4294967296.0f;

	/* Local arrays to hold a section of the parent population. */
	__shared__ float group_steps[NUM_DIMENSIONS * WRKGRPSIZE];     //Need these?
	__shared__ float group_values[NUM_DIMENSIONS * WRKGRPSIZE];

	/* Load the population into local memory */
	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		group_steps[WRKGRPSIZE * i + local_index] =
			in_population_steps[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS *
				group_index + WRKGRPSIZE * i + local_index)];    //Can the in_population index just be global id?
		group_values[WRKGRPSIZE * i + local_index] =
			in_population_values[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS *
				group_index + WRKGRPSIZE * i + local_index)];
	}

	/* Mutation happens here. Each workitem mutates one member of the population */

	/* Randomly choose Ek. If Recombination was more random, it might be worth
	using index % 2 as the coin toss. */

	float Ek = (MWC64X(&rand_state[global_index]) % 2 == 0) ? ALPHA : ONE_OVER_ALPHA;
	//float Ek = (index % 2 == 0) ? ALPHA : ONE_OVER_ALPHA;

	for (int j = 0; j < NUM_DIMENSIONS; j++)
	{
		float s = group_steps[NUM_DIMENSIONS * local_index + j];
		float x = group_values[NUM_DIMENSIONS * local_index + j];

		float gauss = gauss_rand(&rand_state[global_index]);
		//float new_x = gauss < 0.0 ? abs(gauss) : gauss;
		float new_x = x + Ek * s * gauss;

		if (new_x < 0.0f || new_x > 1.0f)	//@ToDo - Predicate assingment?
		{
			/* Rather than generating another gaussian random number, simply
			flip it and scale it down. */
			gauss = gauss * -0.5;
			new_x = x + Ek * s * gauss;
		}

		float Es = (float)exp((float)fabs(gauss) - ROOT_TWO_OVER_PI);
		s *= (float)pow((float)Ek, (float)BETA) * (float)pow((float)Es, (float)BETA_SCALE);

		group_steps[NUM_DIMENSIONS * local_index + j] = s;
		group_values[NUM_DIMENSIONS * local_index + j] = new_x;
	}

	// Write back into global memory
	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		in_population_steps[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS * group_index + WRKGRPSIZE * i + local_index)] =
			group_steps[WRKGRPSIZE * i + local_index];
		in_population_values[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS * group_index + WRKGRPSIZE * i + local_index)] =
			group_values[WRKGRPSIZE * i + local_index];
	}

	in_population_values[492] = 0.411931818;
	in_population_values[493] = 0.375;
	in_population_values[494] = 0.0568181818;
	in_population_values[495] = 1.0;

	//in_population_values[4] = 0.411931818;
	//in_population_values[5] = 0.375;
	//in_population_values[6] = 0.0568181818;
	//in_population_values[7] = 1.0;
}

/*------------------------------------------------------------------------------
	Synthesise - Each work item synthesises the entire wave for a population member's
       set of parameters. This is a simple FM synthesiser.
------------------------------------------------------------------------------*/
__global__ void synthesisePopulation(float* in_population_values,
	float* out_audio_waves,
	const float* param_mins, const float* param_maxs,
	uint32_t rotationIndex)
{
	uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	int local_index = threadIdx.x;
	int group_index = blockIdx.x;

	uint32_t populationStartIndex = rotationIndex * POPULATION_SIZE;

	const int pop_index = local_index * NUM_DIMENSIONS;
	float params_scaled[4];

	//@ToDo - This looks like it laods all population value into local memory, calculates all values scaling, but only synthesises the first/one audio wave from first set of params?

	/* Fill a local array with population values, 1 per workitem */
	__shared__ float group_population_values[WRKGRPSIZE * NUM_DIMENSIONS];
	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		group_population_values[WRKGRPSIZE * i + local_index] = in_population_values[populationStartIndex + (WRKGRPSIZE *
			NUM_DIMENSIONS * group_index + WRKGRPSIZE * i + local_index)];
	}

	/* Scale the synthesis parameters */
	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		params_scaled[i] = param_mins[i] + group_population_values[pop_index + i] *
			(param_maxs[i] - param_mins[i]);
	}

	float modIdxMulModFreq = params_scaled[0] * params_scaled[1];
	float carrierFreq = params_scaled[2];
	float carrierAmp = params_scaled[3];

	/* Use the wavetable positions to track where we are at each frame of synthesis. */
	float wave_table_pos_1 = 0.0f;
	float wave_table_pos_2 = 0.0f;

	float cur_sample;

	/* Local array to hold the current chunk of output for each work item */
	__shared__ float audio_chunks[WRKGRPSIZE * CHUNK_SIZE_SYNTH];  //Need this? Again, another needless loop required to load back from group to global mem?

	int local_id_mod_chunk = local_index % CHUNK_SIZE_SYNTH;

	/* As the chunk size can be smaller than the workgroup size, we need to know which chunk this work item operates on. */
	int local_chunk_index = local_index / CHUNK_SIZE_SYNTH;

	/* Current index to write back to global memory coelesced. Initialise for the first iteration. */
	int out_index = (AUDIO_WAVE_FORM_SIZE * (WRKGRPSIZE * group_index + local_chunk_index)) +
		local_id_mod_chunk;

	/* Perform synthesis in chunks as a single waveform output can be very long.
	 * In each iteration of this outer loop, each work item synthesises a chunk of the wave then the work group
	 * writes back to global memory */
	for (int i = 0; i < AUDIO_WAVE_FORM_SIZE / CHUNK_SIZE_SYNTH; i++)
	{
		for (int j = 0; j < CHUNK_SIZE_SYNTH; j++)
		{
			cur_sample = sin(wave_table_pos_1 * ONE_OVER_SAMPLE_RATE_TIMES_2_PI) * modIdxMulModFreq +
				carrierFreq;
			audio_chunks[local_index * CHUNK_SIZE_SYNTH + j] = sin(wave_table_pos_2 *
				ONE_OVER_SAMPLE_RATE_TIMES_2_PI) * carrierAmp;
			wave_table_pos_1 += params_scaled[0];
			wave_table_pos_2 += cur_sample;


		}
		int out_index_local = local_chunk_index * CHUNK_SIZE_SYNTH + local_id_mod_chunk;
		for (int j = 0; j < CHUNK_SIZE_SYNTH; j++)
		{
			out_audio_waves[out_index] = audio_chunks[out_index_local];
			out_index += CHUNKS_PER_WG_SYNTH * AUDIO_WAVE_FORM_SIZE;
			out_index_local += CHUNKS_PER_WG_SYNTH * CHUNK_SIZE_SYNTH;
		}
		out_index -= (CHUNKS_PER_WG_SYNTH * AUDIO_WAVE_FORM_SIZE - 1) *  CHUNK_SIZE_SYNTH;
	}
	//out_audio_waves[0] = 1.0;
}

/*------------------------------------------------------------------------------
	Synthesise - Wavetable lookup improves performance.
------------------------------------------------------------------------------*/
//@ToDo

/*------------------------------------------------------------------------------
   Apply Window
------------------------------------------------------------------------------*/
__global__ void applyWindowPopulation(float* audio_waves)
{
	uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t global_size = gridDim.x * blockDim.x;

	/* Each work item applies the window function to one sample position.
	 * This is looped for every member of the population */
	float mu = (FFT_ONE_OVER_SIZE - 1) * 2.0f * M_PI;
	for (int i = 0; i < AUDIO_WAVE_FORM_SIZE; i++)
	{
		float fft_window_sample = (1.0 - cos((float)(global_index % AUDIO_WAVE_FORM_SIZE) * mu));
		audio_waves[global_index] = fft_window_sample * audio_waves[global_index];
		global_index += POPULATION_COUNT;
	}
}

/*------------------------------------------------------------------------------
	Calculate Fitness
------------------------------------------------------------------------------*/
__global__ void fitnessPopulation(float* out_population_fitnesses, float* in_fft_data,
	float* in_fft_target,
	uint32_t rotationIndex)
{
	uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	int local_index = threadIdx.x;
	int group_index = blockIdx.x;

	uint32_t populationFitnessStartIndex = rotationIndex * POPULATION_COUNT;

	float error = 0.0f;
	float tmp;
	int second_half_local_target;
	__shared__ double group_fft[WRKGRPSIZE * CHUNK_SIZE_FITNESS];
	__shared__ double group_fft_target[CHUNK_SIZE_FITNESS];

	for (int j = 0; j < FFT_HALF_SIZE / CHUNK_SIZE_FITNESS; j++)
	{
		// Read in chunks - each iteration of the loop every thread reads one chunk for one thread
		for (int i = 0; i < WRKGRPSIZE / 2; i++)
		{
			if (local_index < CHUNK_SIZE_FITNESS)
			{
				group_fft[CHUNK_SIZE_FITNESS * i * 2 + local_index] = in_fft_data[(FFT_OUT_SIZE*
					(WRKGRPSIZE * group_index + i * 2)) + (CHUNK_SIZE_FITNESS * j) + local_index];
			}
			else
			{
				group_fft[CHUNK_SIZE_FITNESS * (i * 2 + 1) + (local_index - CHUNK_SIZE_FITNESS)] =
					in_fft_data[(FFT_OUT_SIZE* (WRKGRPSIZE * group_index + (i * 2 + 1))) + (CHUNK_SIZE_FITNESS * j) +
					(local_index - CHUNK_SIZE_FITNESS)];
			}

		}
		// now load the target into local memory - this is not complex data so our threads load double the necessary amount.
		// this means we only need to load every other iteration.
		second_half_local_target = (j % 2 == 1);
		if (!second_half_local_target)
		{
			if (local_index < CHUNK_SIZE_FITNESS)
			{
				group_fft_target[local_index] = in_fft_target[j * (CHUNK_SIZE_FITNESS / 2) + local_index];
			}
		}
		for (int i = 0; i < CHUNK_SIZE_FITNESS; i += 2)
		{
			const float raw_magnitude = hypot(group_fft[CHUNK_SIZE_FITNESS * local_index + i],
				group_fft[CHUNK_SIZE_FITNESS * local_index + i + 1]);
			const float magnitude_for_fft_size = raw_magnitude * FFT_ONE_OVER_SIZE;
			tmp = (magnitude_for_fft_size * FFT_ONE_OVER_WINDOW_FACTOR) -
				group_fft_target[second_half_local_target * CHUNK_SIZE_FITNESS / 2 + i / 2];
			error += tmp * tmp;
		}
	}
	out_population_fitnesses[populationFitnessStartIndex + global_index] = error;
	//if (global_index == 0) {
	//	printf("Error on populationFitnessStartIndex + global_index: %f\n", error);
	//}
}

/*------------------------------------------------------------------------------
	Sort
------------------------------------------------------------------------------*/
__global__ void sortPopulation(float* in_population_values, float* in_population_steps, float* in_population_fitnesses,
	uint32_t rotationIndex)
{
	uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	int local_index = threadIdx.x;
	int group_index = blockIdx.x;

	uint32_t populationStartIndex = rotationIndex * POPULATION_SIZE;
	uint32_t newPopulationStartIndex = (rotationIndex == 0 ? 1 : 0) * POPULATION_SIZE;

	uint32_t populationFitnessStartIndex = rotationIndex * POPULATION_COUNT;
	uint32_t newPopulationFitnessStartIndex = (rotationIndex == 0 ? 1 : 0) * POPULATION_COUNT;

	__shared__ float group_values[WRKGRPSIZE * NUM_DIMENSIONS];
	__shared__ float group_steps[WRKGRPSIZE * NUM_DIMENSIONS];
	__shared__ float group_fitnesses[WRKGRPSIZE];
	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		group_values[WRKGRPSIZE * i + local_index] = in_population_values[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS *
			group_index + WRKGRPSIZE * i + local_index)];
		group_steps[WRKGRPSIZE * i + local_index] = in_population_steps[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS *
			group_index + WRKGRPSIZE * i + local_index)];
	}
	int new_index = 0;
	float key_i = in_population_fitnesses[populationFitnessStartIndex + global_index];
	int cur_global_compare_id = 0;
	for (int j = 0; j < POPULATION_COUNT / WRKGRPSIZE; j++)
	{
		group_fitnesses[local_index] = in_population_fitnesses[populationFitnessStartIndex + (WRKGRPSIZE*j + local_index)];
		for (int k = 0; k < WRKGRPSIZE; k++)
		{
			float key_j = group_fitnesses[k];
			new_index += (key_j < key_i && cur_global_compare_id != global_index || (key_j == key_i
				&& cur_global_compare_id > global_index));
			cur_global_compare_id++;
		}
	}
	for (int i = 0; i < NUM_DIMENSIONS; i++)
	{
		in_population_values[newPopulationStartIndex + (new_index * NUM_DIMENSIONS + i)] = group_values[local_index * NUM_DIMENSIONS +
			i];
		in_population_steps[newPopulationStartIndex + (new_index * NUM_DIMENSIONS + i)] = group_steps[local_index * NUM_DIMENSIONS +
			i];
	}
	in_population_fitnesses[newPopulationFitnessStartIndex + new_index] = key_i;
}

/*------------------------------------------------------------------------------
	Rotate Population Index
------------------------------------------------------------------------------*/
//@ToDo - Make sure rotationIndex is updated. As not a pointer, does it still get updated?
__global__ void rotatePopulation(uint32_t rotationIndex)
{
	uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_index == 0)
	{
		//inputPopulationValuePayload[rotationIndexPayload * POPULATION_SIZE] = RotationIndex.rotationIndex + 0.5;
		rotationIndex = (rotationIndex == 0 ? 1 : 0);
	}
}

namespace CUDA_Kernels
{
	void initPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, float* aPopulationFitness, uint2* const aRandState, uint32_t aRotationIndex)
	{
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		initPopulation << <numBlocks, threadsPerBlock >> > (aPopulationValues, aPopulationSteps, aPopulationFitness, aRandState, aRotationIndex);
	}
	void recombinePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, uint32_t aRotationIndex)
	{
		//Grid size is the number of workitems to allocate. So implicit to the kernel//
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		recombinePopulation << <numBlocks, threadsPerBlock >> > (aPopulationValues, aPopulationSteps, aRotationIndex);
	}
	void mutatePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, uint2* rand_state, uint32_t aRotationIndex)
	{
		//Grid size is the number of workitems to allocate. So implicit to the kernel//
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		mutatePopulation << <numBlocks, threadsPerBlock >> > (aPopulationValues, aPopulationSteps, rand_state, aRotationIndex);
	}
	void synthesisePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aOutputAudioWaves, const float* aParamMins, const float* aParamMaxs, uint32_t aRotationIndex)
	{
		//Grid size is the number of workitems to allocate. So implicit to the kernel//
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		synthesisePopulation << <numBlocks, threadsPerBlock >> > (aPopulationValues, aOutputAudioWaves, aParamMins, aParamMaxs, aRotationIndex);
	}
	void applyWindowPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aAudioWaves)
	{
		//Grid size is the number of workitems to allocate. So implicit to the kernel//
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		applyWindowPopulation << <numBlocks, threadsPerBlock >> > (aAudioWaves);
	}
	//Running cuFFT comes in between these functions//
	void fitnessPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationFitness, float* aAudioWaveFFT, float* aTargetFFT, uint32_t aRotationIndex)
	{
		//Grid size is the number of workitems to allocate. So implicit to the kernel//
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		fitnessPopulation << <numBlocks, threadsPerBlock >> > (aPopulationFitness, aAudioWaveFFT, aTargetFFT, aRotationIndex);
	}
	void sortPopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, float* aPopulationFitness, uint32_t aRotationIndex)
	{
		//Grid size is the number of workitems to allocate. So implicit to the kernel//
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		sortPopulation << <numBlocks, threadsPerBlock >> > (aPopulationValues, aPopulationSteps, aPopulationFitness, aRotationIndex);
	}
	void rotatePopulationExecute(dim3 aGlobalSize, dim3 aLocalSize, float* aPopulationValues, float* aPopulationSteps, float* aPopulationFitness, uint32_t aRotationIndex)
	{
		//Just needs to be a single kernel exeution to rotate index//
		//@ToDo - Need this kernel? Or just modify CPU side and update?//
		rotatePopulation << <1, 1 >> > (aRotationIndex);
	}
}