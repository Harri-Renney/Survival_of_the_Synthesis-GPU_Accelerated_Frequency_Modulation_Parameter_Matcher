/*
 *  Random number generator
 *  source: http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
 */
inline uint MWC64X(__global uint2* state)
{
    enum _dummy { A=4294883355U };
    //unsigned int A = 4294883355U ;
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);               // Pack the state back up
    return res;                       // Return the next result
}

/**
    http://c-faq.com/lib/gaussian.html
*/
inline float gauss_rand(__global uint2* rand_state)
{
    float sum = 0.0f;
    int tmp_rand;
    for(int i = 0; i < 12; i++)
    {
        sum += (float)((int)MWC64X(rand_state)) / 2147483647.0f;
    }
    sum /= 12.0f;
    return sum;
}

float random(const float2 state)
{
    float floor = 0.f;
    return fract(sin(dot(state.xy, (float2)(12.9898,78.233)))* 43758.5453123,&floor);
    //float2 f2Random = (float2)(12.9898,78.233);
    //float meep = dot(state.xy, f2Random.xy);
    //float floor = 0.f;
    //return fract(sin(1.0) * 43758.5453123 , &floor);
}

/*------------------------------------------------------------------------------
    Population Initialise
------------------------------------------------------------------------------*/
__kernel void initPopulation(__global float* population_values,
                                     __global float* population_steps,
                                     __global float* population_fitnesses,
                                     __global uint2* const rand_state,
                                     __global uint* rotationIndex)
{
    int index = get_global_id(0);

    uint populationStartIndex = rotationIndex[0] * POPULATION_SIZE;

    for(int i = 0; i < NUM_DIMENSIONS; i++)
    {
        population_steps[populationStartIndex + (index * NUM_DIMENSIONS + i)] = 0.1;
        //float rand = random(rand_state[index]);
        //float rand = rand_state[index].x;
        float rand = (float)((int)MWC64X(&rand_state[index])) / 2147483647.0f;
        //float rand = (float)(rand_state[index].x / 2147483647.0f);
        //float rand = rand_state[index].x / 2147483647.0f;
        population_values[populationStartIndex + (index * NUM_DIMENSIONS + i)] = (rand < 0.0f) ? -rand : rand;
    }
}

//@ToDo - Need a offspring generation to spawn the full population from parents before recombine etc.

/*------------------------------------------------------------------------------
    Recombine
------------------------------------------------------------------------------*/
__kernel void recombinePopulation(__global float* population_values,
                         __global float* population_steps,
                                     __global uint* rotationIndex)
{
    int index = get_global_id(0);
    int local_index = get_local_id(0);
    int group_id = get_group_id(0);

    uint populationStartIndex = rotationIndex[0] * POPULATION_SIZE;

    /* Local arrays to hold a section of the parent population.
    We have source and destination arrays to prevent workitems overwriting
    cells which have not yet been read by other workitems */
    __local float group_population_values[NUM_DIMENSIONS * WRKGRPSIZE];
    __local float group_population_steps[NUM_DIMENSIONS * WRKGRPSIZE];
    __local float group_population_values_recombined[NUM_DIMENSIONS * WRKGRPSIZE];
    __local float group_population_steps_recombined[NUM_DIMENSIONS * WRKGRPSIZE];

    /* This kernel runs with the population size number of workitems but we only
    want to recombine the parent population.

    A lot of the workitems will have a local index greater than the parent
    population.

    So we change the group number of groups which are aligned with offspring so
    that they point at parents. */
    int group_id_mod = group_id % NUM_WGS_FOR_PARENTS;

    /* Load parent population data into local memory.
    Groups with the same group_id_mod value will load the same set of parents */

    int global_block_start_index = group_id_mod * WRKGRPSIZE * NUM_DIMENSIONS;
    for(int i = 0; i < NUM_DIMENSIONS; i++)
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
    for(int i = 0; i < NUM_DIMENSIONS; i++)
    {
        shift_amt = NUM_DIMENSIONS * (i * (group_id + 1));
        new_idx = (start_idx + shift_amt) % ( WRKGRPSIZE * NUM_DIMENSIONS);
        group_population_values_recombined[new_idx] = group_population_values[start_idx];
        group_population_steps_recombined[new_idx] = group_population_steps[start_idx];
        start_idx++;
    }

    /* Now load the recombined population back into global memory. */
    global_block_start_index = group_id * WRKGRPSIZE * NUM_DIMENSIONS;
    for(int i = 0; i < NUM_DIMENSIONS; i++)
    {
        int local_block_index = WRKGRPSIZE * i + local_index;
        population_values[populationStartIndex + (global_block_start_index + local_block_index)]  =
            group_population_values_recombined[local_block_index];
        population_steps[populationStartIndex + (global_block_start_index + local_block_index)]  =
            group_population_steps_recombined[local_block_index];
    }
}

/*------------------------------------------------------------------------------
    Mutate
------------------------------------------------------------------------------*/

__kernel void mutatePopulation(__global float* in_population_values,
                      __global float* in_population_steps,
                      __global uint2* rand_state,
                                     __global uint* rotationIndex)
{
    int index = get_global_id(0);
    int local_index = get_local_id(0);
    int group_index = get_group_id(0);

    uint populationStartIndex = rotationIndex[0] * POPULATION_SIZE;
	
	for(int i = 0; i != NUM_DIMENSIONS; ++i)
    {
        float Ek = ( MWC64X(&rand_state[index]) % 2 == 0) ? ALPHA : ONE_OVER_ALPHA;
    
        float s = in_population_steps[populationStartIndex + index * NUM_DIMENSIONS + i];
        float x = in_population_values[populationStartIndex + index * NUM_DIMENSIONS + i];
    
        float gauss = gauss_rand(&rand_state[index]);
        float new_x = x + Ek * s * gauss;
    
        if(new_x < 0.0f || new_x > 1.0f)	//@ToDo - Predicate assingment?
        {
            /* Rather than generating another gaussian random number, simply
            flip it and scale it down. */
            gauss = gauss * -0.5;
            new_x = x + Ek * s * gauss;
        }
    
        float Es = (float)exp ( (float)fabs (gauss) - ROOT_TWO_OVER_PI );
        s *= (float)pow((float)Ek, (float)BETA) * (float)pow((float)Es, (float)BETA_SCALE);
    
        in_population_steps[populationStartIndex + index * NUM_DIMENSIONS + i] = s;
        in_population_values[populationStartIndex + index * NUM_DIMENSIONS + i] = new_x;
        
    }

    ///* Local arrays to hold a section of the parent population. */
    //__local float group_steps[NUM_DIMENSIONS * WRKGRPSIZE];     //Need these?
    //__local float group_values[NUM_DIMENSIONS * WRKGRPSIZE];
    //
    ///* Load the population into local memory */
    //for(int i = 0; i < NUM_DIMENSIONS; i++)
    //{
    //    group_steps[WRKGRPSIZE * i + local_index] =
    //        in_population_steps[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS *
    //                            group_index + WRKGRPSIZE * i + local_index)];    //Can the in_population index just be global id?
    //    group_values[WRKGRPSIZE * i + local_index] =
    //        in_population_values[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS *
    //                             group_index + WRKGRPSIZE * i + local_index)];
    //}
    //
    ///* Mutation happens here. Each workitem mutates one member of the population */
    //
    ///* Randomly choose Ek. If Recombination was more random, it might be worth
    //using index % 2 as the coin toss. */
    //
    //float Ek = ( MWC64X(&rand_state[index]) % 2 == 0) ? ALPHA : ONE_OVER_ALPHA;
    ////float Ek = (index % 2 == 0) ? ALPHA : ONE_OVER_ALPHA;
    //
    //for(int j = 0; j < NUM_DIMENSIONS; j++)
    //{
    //    float s = group_steps[NUM_DIMENSIONS * local_index + j];
    //    float x = group_values[NUM_DIMENSIONS * local_index + j];
    //
    //    float gauss = gauss_rand(&rand_state[index]);
    //    float new_x = x + Ek * s * gauss;
    //
    //    if(new_x < 0.0f || new_x > 1.0f)	//@ToDo - Predicate assingment?
    //    {
    //        /* Rather than generating another gaussian random number, simply
    //        flip it and scale it down. */
    //        gauss = gauss * -0.5;
    //        new_x = x + Ek * s * gauss;
    //    }
    //
    //    float Es = (float)exp ( (float)fabs (gauss) - ROOT_TWO_OVER_PI );
    //    s *= (float)pow((float)Ek, (float)BETA) * (float)pow((float)Es, (float)BETA_SCALE);
    //
    //    group_steps[NUM_DIMENSIONS * local_index + j] = s;
    //    group_values[NUM_DIMENSIONS * local_index + j] = new_x;
    //}
    //
    //// Write back into global memory
    //for(int i = 0; i < NUM_DIMENSIONS; i++)
    //{
    //    in_population_steps[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS * group_index + WRKGRPSIZE * i + local_index)] =
    //        group_steps[WRKGRPSIZE * i + local_index];
    //    in_population_values[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS * group_index + WRKGRPSIZE * i + local_index)] =
    //        group_values[WRKGRPSIZE * i + local_index];
    //}
	
	//in_population_values[0] = 0.411931818;
    //in_population_values[1] = 0.375;
    //in_population_values[2] = 0.0568181818;
    //in_population_values[3] = 1.0;
}



/*------------------------------------------------------------------------------
    Synthesise
     - Each work item synthesises the entire wave for a population member's
       set of parameters. This is a simple FM synthesiser.
------------------------------------------------------------------------------*/

/* We synthesise the wave in chunks. The size of the chunks is defined here in
 * terms of the work group size. The synthesis is done in chunks to avoid local
 * memory getting to big. This should really be tuned based on the hardware. */
#define CHUNKS_PER_WG_SYNTH 1
#define CHUNK_SIZE_SYNTH (WRKGRPSIZE/CHUNKS_PER_WG_SYNTH)

/* Constant used for FM synthesis */
#define ONE_OVER_SAMPLE_RATE_TIMES_2_PI 0.00014247573

__kernel void synthesisePopulation(__global float* out_audio_waves,
                          __global float* in_population_values,
                          __constant float* param_mins, __constant float* param_maxs,
                                     __global uint* rotationIndex, __global float* wavetable)
{
    int index = get_global_id(0);
    int local_index = get_local_id(0);
    int group_index = get_group_id(0);

    uint populationStartIndex = rotationIndex[0] * POPULATION_SIZE;

    const int pop_index = local_index * NUM_DIMENSIONS;
    float params_scaled[4];

    /* Scale the synthesis parameters */
    for(int i = 0; i < NUM_DIMENSIONS; i++)
    {
        params_scaled[i] = param_mins[i] + in_population_values[populationStartIndex+index*NUM_DIMENSIONS+i] * (param_maxs[i]-param_mins[i]);
    }

    float modIdxMulModFreq = params_scaled[0] * params_scaled[1];
    float carrierFreq  = params_scaled[2];
    float carrierAmp = params_scaled[3];

    /* Use the wavetable positions to track where we are at each frame of synthesis. */
    float wave_table_pos_1 = 0.0f;
    float wave_table_pos_2 = 0.0f;

    float cur_sample;

    const float wavetableIncrementOne = (WAVETABLE_SIZE / 44100.0) * params_scaled[0];

    for(int i = 0; i < AUDIO_WAVE_FORM_SIZE; i++)
	{
		 cur_sample = wavetable[(uint)wave_table_pos_1] * modIdxMulModFreq + carrierFreq;
		 out_audio_waves[index * AUDIO_WAVE_FORM_SIZE + i] = wavetable[(uint)wave_table_pos_2] 
															 * carrierAmp;
															 
		 wave_table_pos_1 += wavetableIncrementOne;
		 wave_table_pos_2 += (WAVETABLE_SIZE / 44100.0) * cur_sample;
		
		if (wave_table_pos_1 >= WAVETABLE_SIZE)
			wave_table_pos_1 -= WAVETABLE_SIZE;
		//if (wave_table_pos_1 < 0.0f)
		//	wave_table_pos_1 += WAVETABLE_SIZE;
		if (wave_table_pos_2 >= WAVETABLE_SIZE)
			wave_table_pos_2 -= WAVETABLE_SIZE;
		if (wave_table_pos_2 < 0.0f)
			wave_table_pos_2 += WAVETABLE_SIZE;
	}
}

/*------------------------------------------------------------------------------
    Synthesise - Wavetable lookup improves performance.
------------------------------------------------------------------------------*/
//__kernel void synthesisePopulation(__global float* out_audio_waves,
//                          __global float* in_population_values,
//                          __constant float* param_mins, __constant float* param_maxs,
//                                     __global uint* rotationIndex, __global float* wavetable)
//{
//    int index = get_global_id(0);
//    int local_index = get_local_id(0);
//    int group_index = get_group_id(0);
//
//    uint populationStartIndex = rotationIndex[0] * POPULATION_SIZE;
//
//    const int pop_index = local_index * NUM_DIMENSIONS;
//    float params_scaled[4];
//
//    //@ToDo - This looks like it laods all population value into local memory, calculates all values scaling, but only synthesises the first/one audio wave from first set of params?
//
//    /* Fill a local array with population values, 1 per workitem */
//    __local float group_population_values[WRKGRPSIZE * NUM_DIMENSIONS];
//    for(int i = 0; i < NUM_DIMENSIONS; i++)
//    {
//        group_population_values[WRKGRPSIZE * i + local_index] = in_population_values[populationStartIndex + (WRKGRPSIZE *
//                NUM_DIMENSIONS * group_index + WRKGRPSIZE * i + local_index)];
//    }
//
//    /* Scale the synthesis parameters */
//    for(int i = 0; i < NUM_DIMENSIONS; i++)
//    {
//        params_scaled[i] = param_mins[i] + group_population_values[pop_index + i] *
//                           (param_maxs[i] - param_mins[i]);
//    }
//
//    float modIdxMulModFreq = params_scaled[0] * params_scaled[1];
//    float carrierFreq  = params_scaled[2];
//    float carrierAmp = params_scaled[3];
//
//    /* Use the wavetable positions to track where we are at each frame of synthesis. */
//    float wave_table_pos_1 = 0.0f;
//    float wave_table_pos_2 = 0.0f;
//
//    float cur_sample;
//
//    /* Local array to hold the current chunk of output for each work item */
//    __local float audio_chunks[WRKGRPSIZE * CHUNK_SIZE_SYNTH];  //Need this? Again, another needless loop required to load back from group to global mem?
//
//    int local_id_mod_chunk = local_index % CHUNK_SIZE_SYNTH;
//
//    /* As the chunk size can be smaller than the workgroup size, we need to know which chunk this work item operates on. */
//    int local_chunk_index = local_index / CHUNK_SIZE_SYNTH;
//
//    /* Current index to write back to global memory coelesced. Initialise for the first iteration. */
//    int out_index = (AUDIO_WAVE_FORM_SIZE * (WRKGRPSIZE * group_index + local_chunk_index)) +
//                    local_id_mod_chunk;
//
//    const float wavetableIncrementOne = (WAVETABLE_SIZE / 44100.0) * params_scaled[0];
//
//    for(int i = 0; i < AUDIO_WAVE_FORM_SIZE; i++)
//	{
//		 cur_sample = wavetable[(int)wave_table_pos_1] * modIdxMulModFreq + carrierFreq;
//		 out_audio_waves[index * AUDIO_WAVE_FORM_SIZE + i] = wavetable[(int)wave_table_pos_2] 
//															 * carrierAmp;
//															 
//		 wave_table_pos_1 += wavetableIncrementOne;
//		 wave_table_pos_2 += (WAVETABLE_SIZE / 44100.0) * cur_sample;
//		
//		if (wave_table_pos_1 >= WAVETABLE_SIZE)
//			wave_table_pos_1 -= WAVETABLE_SIZE;
//		if (wave_table_pos_1 < 0.0f)
//			wave_table_pos_1 += WAVETABLE_SIZE;
//		if (wave_table_pos_2 >= WAVETABLE_SIZE)
//			wave_table_pos_2 -= WAVETABLE_SIZE;
//		if (wave_table_pos_2 < 0.0f)
//			wave_table_pos_2 += WAVETABLE_SIZE;
//	}
//                    
//    /* Perform synthesis in chunks as a single waveform output can be very long.
//     * In each iteration of this outer loop, each work item synthesises a chunk of the wave then the work group
//     * writes back to global memory */
//    //for(int i = 0; i < AUDIO_WAVE_FORM_SIZE / CHUNK_SIZE_SYNTH; i++)
//    //{
//    //    for(int j = 0; j < CHUNK_SIZE_SYNTH; j++)
//    //    {
//    //        cur_sample = wavetable[(int)wave_table_pos_1] * modIdxMulModFreq + carrierFreq;
//    //        wave_table_pos_1 += wavetableIncrementOne;
//    //        if (wave_table_pos_1 >= WAVETABLE_SIZE) {
//	//			wave_table_pos_1 -= WAVETABLE_SIZE;
//	//		}
//    //        //if (wave_table_pos_1 < 0.0f) {
//	//		//	wave_table_pos_1 += WAVETABLE_SIZE;
//	//		//}
//	//
//    //        audio_chunks[local_index * CHUNK_SIZE_SYNTH + j] = wavetable[(int)wave_table_pos_2] * carrierAmp;
//    //        wave_table_pos_2 += (WAVETABLE_SIZE / 44100.0) * cur_sample;
//	//
//    //        if (wave_table_pos_2 >= WAVETABLE_SIZE) {
//	//			wave_table_pos_2 -= WAVETABLE_SIZE;
//	//		}
//	//
//	//		if (wave_table_pos_2 < 0.0f) {
//	//			wave_table_pos_2 += WAVETABLE_SIZE;
//	//		}
//    //    }
//    //    int out_index_local = local_chunk_index * CHUNK_SIZE_SYNTH + local_id_mod_chunk;
//    //    for(int j = 0; j < CHUNK_SIZE_SYNTH; j++)
//    //    {
//    //        out_audio_waves[out_index] = audio_chunks[out_index_local];
//    //        out_index += CHUNKS_PER_WG_SYNTH * AUDIO_WAVE_FORM_SIZE;
//    //        out_index_local += CHUNKS_PER_WG_SYNTH * CHUNK_SIZE_SYNTH;
//    //    }
//    //    out_index -= (CHUNKS_PER_WG_SYNTH * AUDIO_WAVE_FORM_SIZE - 1) *  CHUNK_SIZE_SYNTH;
//    //}
//    //out_audio_waves[1] = wavetable[5];
//}



/*------------------------------------------------------------------------------
   Apply Window
------------------------------------------------------------------------------*/
__kernel void applyWindowPopulation(__global float* audio_waves)
{
    int index = get_global_id(0);
    int global_size = get_global_size(0);

    /* Each work item applies the window function to one sample position.
     * This is looped for every member of the population */
    float mu = ( FFT_ONE_OVER_SIZE - 1) * 2.0f * M_PI;
    //for(int i = 0; i < AUDIO_WAVE_FORM_SIZE; i++)
    //{
    //    //Uncoalesced.
	//	float fft_window_sample = 1.0 - cos(i  * mu);
    //    audio_waves[index*AUDIO_WAVE_FORM_SIZE+i] = fft_window_sample * audio_waves[index*AUDIO_WAVE_FORM_SIZE+i];		
    //}
    for(int i = 0; i < POPULATION_COUNT; i++)
    {
        //Coalesced.
        float fft_window_sample = 1.0 - cos(index  * mu);
		audio_waves[AUDIO_WAVE_FORM_SIZE*i+index] = fft_window_sample * audio_waves[AUDIO_WAVE_FORM_SIZE*i+index];
    }
}

/*------------------------------------------------------------------------------
    Calculate Fitness
------------------------------------------------------------------------------*/

#define CHUNK_SIZE_FITNESS (WRKGRPSIZE/2)

__kernel void fitnessPopulation(__global float* out_population_fitnesses, __global float* in_fft_data,
                           __constant float* in_fft_target,
                                     __global uint* rotationIndex)
{
    int index = get_global_id(0);
    int local_index = get_local_id(0);
    int group_index = get_group_id(0);

	uint populationFitnessStartIndex = rotationIndex[0] * POPULATION_COUNT;

    float error = 0.0f;
    float tmp;
	for (int i = 0; i < FFT_OUT_SIZE-2; i += 2)
	{
		const float raw_magnitude = hypot(in_fft_data[index * FFT_OUT_SIZE + i],
			in_fft_data[index* FFT_OUT_SIZE + i + 1]);
		const float magnitude_for_fft_size = raw_magnitude * FFT_ONE_OVER_SIZE;
		tmp = (magnitude_for_fft_size * FFT_ONE_OVER_WINDOW_FACTOR) -
			in_fft_target[i / 2];
		error += tmp * tmp;
	}
    //int second_half_local_target;
    //__local float group_fft[WRKGRPSIZE * CHUNK_SIZE_FITNESS];
    //__local float group_fft_target[CHUNK_SIZE_FITNESS];
    //
    //for(int j = 0; j < FFT_HALF_SIZE / CHUNK_SIZE_FITNESS; j++)
    //{
    //    // Read in chunks - each iteration of the loop every thread reads one chunk for one thread
    //    for(int i = 0; i < WRKGRPSIZE/2; i++)
    //    {
    //        if(local_index < CHUNK_SIZE_FITNESS)
    //        {
    //            group_fft[CHUNK_SIZE_FITNESS * i*2 + local_index] = in_fft_data[( FFT_OUT_SIZE*
    //                    (WRKGRPSIZE * group_index + i*2)) + (CHUNK_SIZE_FITNESS * j) + local_index];
    //        }
    //        else
    //        {
    //            group_fft[CHUNK_SIZE_FITNESS * (i*2 + 1) + (local_index - CHUNK_SIZE_FITNESS)] =
    //                in_fft_data[( FFT_OUT_SIZE* (WRKGRPSIZE * group_index + (i*2+1))) + (CHUNK_SIZE_FITNESS * j) +
    //                            (local_index - CHUNK_SIZE_FITNESS)];
    //        }
    //
    //    }
    //    // now load the target into local memory - this is not complex data so our threads load double the necessary amount.
    //    // this means we only need to load every other iteration.
    //    second_half_local_target = (j % 2 == 1);
    //    if(!second_half_local_target)
    //    {
    //        if(local_index < CHUNK_SIZE_FITNESS)
    //        {
    //            group_fft_target[local_index] = in_fft_target[j * (CHUNK_SIZE_FITNESS/2) + local_index];
    //        }
    //    }
    //    for(int i = 0; i < CHUNK_SIZE_FITNESS; i+=2)
    //    {
    //        const float raw_magnitude = hypot (group_fft[CHUNK_SIZE_FITNESS * local_index + i],
    //                                           group_fft[CHUNK_SIZE_FITNESS * local_index + i + 1]);
    //        const float magnitude_for_fft_size = raw_magnitude * FFT_ONE_OVER_SIZE;
    //        tmp = (magnitude_for_fft_size * FFT_ONE_OVER_WINDOW_FACTOR) -
    //              group_fft_target[second_half_local_target * CHUNK_SIZE_FITNESS / 2 + i/2];
    //        error += tmp * tmp;
    //    }
    //}
    out_population_fitnesses[populationFitnessStartIndex + index] = error;
    //out_population_fitnesses[index] = -10;
}

/*------------------------------------------------------------------------------
    Sort
------------------------------------------------------------------------------*/
__kernel void sortPopulation( __global float* in_population_values, __global float* in_population_steps,
                    __global float* in_population_fitnesses,
                                     __global uint* rotationIndex)
{
    int index = get_global_id(0);
    int local_index = get_local_id(0);
    int group_index = get_group_id(0);
    int local_size = get_local_size(0);

    uint populationStartIndex = rotationIndex[0] * POPULATION_SIZE;
    uint newPopulationStartIndex = (rotationIndex[0] == 0 ? 1 : 0) * POPULATION_SIZE;
    
	uint populationFitnessStartIndex = rotationIndex[0] * POPULATION_COUNT;
    uint newPopulationFitnessStartIndex = (rotationIndex[0] == 0 ? 1 : 0) * POPULATION_COUNT;

    __local float group_values[WRKGRPSIZE * NUM_DIMENSIONS];
    __local float group_steps[WRKGRPSIZE * NUM_DIMENSIONS];
    __local float group_fitnesses[WRKGRPSIZE];
    for(int i = 0; i < NUM_DIMENSIONS; i++)
    {
        group_values[WRKGRPSIZE * i + local_index] = in_population_values[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS *
                group_index + WRKGRPSIZE * i + local_index)];
        group_steps[WRKGRPSIZE * i + local_index] = in_population_steps[populationStartIndex + (WRKGRPSIZE * NUM_DIMENSIONS *
                group_index + WRKGRPSIZE * i + local_index)];
    }
    int new_index = 0;
    float key_i = in_population_fitnesses[populationFitnessStartIndex + index];
    int cur_global_compare_id = 0;
    for(int j = 0; j < POPULATION_COUNT/WRKGRPSIZE; j++)
    {
        group_fitnesses[local_index] = in_population_fitnesses[populationFitnessStartIndex + (WRKGRPSIZE*j + local_index)];
        for(int k = 0; k < WRKGRPSIZE; k++)
        {
            float key_j = group_fitnesses[k];
            new_index += (key_j < key_i && cur_global_compare_id != index || (key_j == key_i
                          && cur_global_compare_id > index));
            cur_global_compare_id++;
        }
    }
    for(int i = 0; i < NUM_DIMENSIONS; i++)
    {
        in_population_values[newPopulationStartIndex + (new_index * NUM_DIMENSIONS + i)] = group_values[local_index * NUM_DIMENSIONS +
                i];
        in_population_steps[newPopulationStartIndex + (new_index * NUM_DIMENSIONS + i)] = group_steps[local_index * NUM_DIMENSIONS +
                i];
    }
    in_population_fitnesses[newPopulationFitnessStartIndex + new_index] = key_i;
}
