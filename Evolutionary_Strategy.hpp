#pragma once
#ifndef EVOLUTIONARY_STRATEGY_HPP
#define EVOLUTIONARY_STRATEGY_HPP

#define M_PI 3.14159265358979323846

#include <cstdint>
#include <algorithm>
#include <numeric>

#include <fftw_cpp.hh>

struct Individual
{
	uint32_t fitness;
	
};

struct Population
{
	uint32_t numParents = 197;
	uint32_t numOffspring = 960;
	uint32_t numDimensions = 4;
	uint32_t populationLength = numParents + numOffspring;
	uint32_t populationSize = (numParents + numOffspring) * sizeof(float);

	float *data;

	float* getValue(uint32_t idxIndividual, uint32_t idxValue)
	{
		uint32_t idx = ((idxIndividual * (numDimensions*2 + 1))) + (idxValue * 2);
		return &(data[idx]);
	}
	float* getStep(uint32_t idxIndividual, uint32_t idxStep)
	{
		uint32_t idx = (idxIndividual * (numDimensions*2 + 1)) + ((idxStep * 2) + 1);
		return &(data[idx]);
	}
	float* getFitness(uint32_t idxIndividual)
	{
		uint32_t idx = ((idxIndividual * (numDimensions*2 + 1))) + (numDimensions * 2);
		return &(data[idx]);
	}
	
	void swap(int32_t first, int32_t second)
	{
		int32_t individualLength = (numDimensions * 2 + 1);
		int32_t idxF = first * individualLength;
		int32_t idxS = second * individualLength;
		for (uint32_t i = 0; i != individualLength; ++i)
		{
			float temp = data[idxS + i];
			data[idxS + i] = data[idxF + i];
			data[idxF + i] = temp;
		}
	}

	//Sorting//
	//int32_t partition(int32_t low, int32_t high)
	//{
	//	// pivot (Element to be placed at right position)
	//	float pivot = *getFitness(high);
	//
	//	int32_t i = (low - 1 < 0 ? 0 : low - 1);  // Index of smaller element
	//
	//	for (uint32_t j = low; j <= high - 1; j++)
	//	{
	//		// If current element is smaller than the pivot
	//		if (*getFitness(j) < pivot)
	//		{
	//			i++;    // increment index of smaller element
	//			swap(i, j);
	//		}
	//	}
	//	swap(i + 1 > populationSize - 1 ? populationSize - 1 : i + 1, high);
	//	return (i + 1 > populationSize - 1 ? populationSize - 1 : i + 1);
	//}
	//
	//void quickSortPopulation()
	//{
	//	int32_t low = 0;
	//	int32_t high = populationSize-1;
	//	if (low < high)
	//	{
	//		/* pi is partitioning index, arr[pi] is now
	//		   at right place */
	//		int32_t pi = partition(low, high);
	//
	//		int temp = (((int)pi - 1) < 0) ? 0 : pi - 1;
	//		quickSortPopulation(low, temp);  // Before pi
	//		quickSortPopulation(pi + 1 > populationSize-1 ? populationSize - 1 : pi + 1, high); // After pi
	//	}
	//}
	//void quickSortPopulation(int32_t low, int32_t high)
	//{
	//	if (low < high)
	//	{
	//		/* pi is partitioning index, arr[pi] is now
	//		   at right place */
	//		int32_t pi = partition(low, high);
	//
	//		int temp = (((int)pi - 1) < 0) ? 0 : pi - 1;
	//		quickSortPopulation(low, temp);  // Before pi
	//		quickSortPopulation(pi + 1 > populationSize - 1 ? populationSize - 1 : pi + 1, high); // After pi
	//	}
	//}

	void quickSortPopulation(int left, int right) {

		int i = left, j = right;

		float tmp;

		float pivot = *getFitness((left+right)/2);



		/* partition */

		while (i <= j) {

			while (*getFitness(i) < pivot)

				i++;

			while (*getFitness(j) > pivot)

				j--;

			if (i <= j) {

				swap(i, j);

				i++;

				j--;

			}

		};



		/* recursion */

		if (left < j)

			quickSortPopulation(left, j);

		if (i < right)

			quickSortPopulation(i, right);

	}
};

class Objective
{
private:
	float wavetablePosOne = 0.0f;
	float wavetablePosTwo = 0.0f;
public:
	//Parameters for the synthesiser//
	const std::vector<float> paramMins;
	const std::vector<float> paramMaxs;

	//Target audio data for comparison//
	std::vector<float> targetParams;
	float* targetWaveForm;
	float* targetFFT;
	std::complex<double> squareErrorDenominator;

	dcvector targetWaveFormComplex_;
	dcvector targetFFTComplex_;

	//Wavetable for fast synthesis//
	const uint32_t sampleRate = 44100;
	const uint32_t wavetableSize = 32768;
	float* wavetable;

	//Generated audio data buffers//
	float* audioWaveForm;
	float* audioFFT;
	const float w2srRatio = wavetableSize / (float)sampleRate;				//Wave to sample rate ratio for synthesis
	uint32_t audioLength;
	uint32_t audioLengthLog2;

	//Tmp buffers for synthesis - preallocated memory for speed//
	float* tmpFFTBuffer;
	float* tmpParamsScaled;
	dcvector tmpFFTBufferComplex;


	//Struct containing all FFT processing info//
	fftw_plan fftPlan;		//Needed?
	FFT fft;
	fftw_complex* fftOut;

	uint32_t fftOutSize;
	uint32_t fftSize;
	uint32_t fftSizeLog2;
	uint32_t fftHalfSize;
	float fftOneOverSize;
	double* fftWindow;				//Buffer for the window.
	double* fftWindowedAudio;		//Buffer for the windowed audio.
	float fftWindowFactor;				//Window factor is sum of the window samples divided by the FFT size.
	float fftOneOverWindowFactor;
public:
	Objective(uint32_t aPopulationSize, uint32_t aNumDimensions, const std::vector<float> aParamMins, const std::vector<float> aParamMaxs, uint32_t aAudioLengthLog2) :
		fft(1,1.0),
		audioLength(1 << aAudioLengthLog2),
		audioLengthLog2(aAudioLengthLog2),	//@Highlight - Want aAudioLengthLog2 = options->audio_length_log2.
		paramMins(aParamMins),
		paramMaxs(aParamMaxs)
	{	
		//Target//
		targetWaveForm = new float((1 << audioLengthLog2));
		targetFFT = new float(1 << (audioLengthLog2 - 1));
		
		targetWaveFormComplex_.resize(1 << audioLengthLog2);
		targetFFTComplex_.resize(1 << (audioLengthLog2 - 1));
		
		//Temporary//
		tmpFFTBuffer = new float((1 << (audioLengthLog2 - 1)));
		tmpParamsScaled = new float(aNumDimensions);
		
		//Generated audio//
		audioWaveForm = new float[(1 << audioLengthLog2) * aPopulationSize];
		audioFFT = new float[(1 << (audioLengthLog2 - 1)) * aPopulationSize];


		init(aPopulationSize);
	}
	void init(uint32_t aPopulationSize)	//@Highlight - Want aSizeLog2 = objective->audio_wave_form_size.
	{
		//This needed to create FFT plan?//
		//double* windowedAudio = new double[(1 << audioLengthLog2)];
		//fftw_complex* fft_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) *
		//	(1 << audioLengthLog2));

		//fft = FFT(1 << audio_length_log2, 1.0);

		fftSize = 1 << audioLengthLog2;
		fftSizeLog2 = audioLengthLog2;
		fftHalfSize = 1 << (audioLengthLog2 - 1);
		fftOneOverSize = 1.0f / fftSize;
		fftWindow = new double [fftSize];
		fftWindowedAudio = new double [fftSize];
		fftOutSize = (audioLength / 2 + 4) * 2;

		initFFTW(aPopulationSize);

		initWavetable();
	}
	void initMemory()
	{

	}
	//objective_initialise(es->objective, population_count(es->population),
	//	num_dimensions, options->audio_length_log2, param_mins, param_maxs, options);
	void initFFTW(uint32_t aPopulationSize)
	{
		//@ToDO - Method 1, C style copied from GPUMatch//
		//Initalise fft plan//
		double* tmpWindowedAudio = (double *)malloc(sizeof(double) * (audioLength));
		fftw_complex* tmpFFTOut = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (audioLength));
		fftPlan = fftw_plan_dft_r2c_1d(audioLength, tmpWindowedAudio, tmpFFTOut, FFTW_PATIENT);

		//These variables will never be used, but are needed for initialisation
		fftw_free(tmpFFTOut);
		free(tmpWindowedAudio);
		
		//Initalise fft -@ToDo Sort this out...//
		fftSizeLog2 = audioLengthLog2;
		fftSize = 1 << fftSizeLog2;
		fftHalfSize = 1 << (fftSizeLog2 - 1);
		fftOneOverSize = 1.0f / fftSize;
		fftWindow = new double[fftSize];
		fftWindowedAudio = new double[fftSize];

		/*
			@ToDo
			From clFFTpp "The problem size nx is the number of real values before being
			transformed into complex space.The output has nx / 2 + 1 complex
			values."
			So why is this different???
		*/
		
		double two_pi = 2.0 * M_PI;
		fftWindowFactor = 0.0f;
		for (uint32_t i = 0; i < fftSize; i++) {
			// Hann window equation
			fftWindow[i] = (1.0 - cos((double)i * (fftOneOverSize - 1) * two_pi));
			fftWindowFactor += fftWindow[i];
		}
		
		fftWindowFactor *= fftOneOverSize;
		fftOneOverWindowFactor = 1.f / fftWindowFactor;
		fftOut = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fftSize);

		//@ToDo - Method 2, C++ style. Using C++ header file//
		//uint32_t fftSize = 1 << audioLengthLog2;
		//uint32_t fftLength = 0;
		//FFT(fftSize, fftLength);
	}
	void initWavetable()
	{
		wavetable = new float[wavetableSize];
		float one_over_table_size_minus_1 = 1.0f / ((float)wavetableSize - 1.0f);
		for (int i = 0; i < wavetableSize; i++) {
			wavetable[i] = sinf((float)i * one_over_table_size_minus_1 * 2 * (float)M_PI);
		}
	}

	void calculateTargetFromParams(float* aTargetParams_)
	{
		for (int i = 0; i < audioLength; ++i)
		{
			targetParams.push_back(aTargetParams_[i]);
		}
		float* audioBuffer = (float*)&targetWaveFormComplex_[0];	//This probably doesn't work...
		synthesiseAudio(targetParams, audioBuffer);	//@ToDo - Does this need to take the complex<double> used in other functions? (Like FFT)
		fft.fft(targetWaveFormComplex_, targetFFTComplex_);
		//fft_calculate(plan, &(objective->fft), objective->target_wave_form, objective->target_fft);
		//vsq(objective->target_fft, 1, objective->tmp_fft_buffer, 1, objective->fft.half_size);
		//sve(objective->tmp_fft_buffer, 1, &(objective->sq_err_denominator), objective->fft.half_size);
		std::for_each(targetFFTComplex_.begin(), targetFFTComplex_.end(), [](std::complex<double>& n) {n = n * n; });
		std::accumulate(tmpFFTBufferComplex.begin(), tmpFFTBufferComplex.end(), squareErrorDenominator);
		squareErrorDenominator = 1.f / squareErrorDenominator.real();	//@ToDo - What does this actually do though? Not ever used? And no side effects...
	}
	void calculateTargetFromAudio(float* aTargetAudio)
	{
		for (int i = 0; i < audioLength; ++i)
		{
			targetWaveForm[i] = aTargetAudio[i];
			targetWaveFormComplex_[i] = aTargetAudio[i];
		}
		fft.fft(targetWaveFormComplex_, targetFFTComplex_);
		//fft_calculate(plan, &(objective->fft), objective->target_wave_form, objective->target_fft);
		//vsq(objective->target_fft, 1, objective->tmp_fft_buffer, 1, objective->fft.half_size);
		//sve(objective->tmp_fft_buffer, 1, &(objective->sq_err_denominator), objective->fft.half_size);
		std::for_each(targetFFTComplex_.begin(), targetFFTComplex_.end(), [](std::complex<double>& n) {n = n * n; });
		std::accumulate(tmpFFTBufferComplex.begin(), tmpFFTBufferComplex.end(), squareErrorDenominator);
		squareErrorDenominator = 1.f / squareErrorDenominator.real();	//@ToDo - What does this actually do though? Not ever used? And no side effects...
	}

	//@ToDo - This is done in chunks. Therefore, need to comeback and realise way to do this.
	//@ToDo - Scale Params?
	void synthesiseAudio(const std::vector<float> aParams, float* aAudioBuffer)
	{
		//float params[4] = { aParams[0], aParams[1], aParams[2], aParams[3] };
		float params[4] = { aParams[0] * paramMaxs[0], aParams[1] * paramMaxs[1], aParams[2] * paramMaxs[2], aParams[3] * paramMaxs[3] };
		float modIdxMulModFreq = params[0] * params[1];
		float carrierFreq = params[2];
		float carrierAmp = params[3];

		const float wavetableIncrementOne = w2srRatio * params[0];

		float currentSample;

		// Perform fm synthesis one sample at a time
		for (int i = 0; i < audioLength; i++)
		{
			//Oscillation 1//
			currentSample = wavetable[(unsigned int)wavetablePosOne] * modIdxMulModFreq +
				carrierFreq;
			wavetablePosOne += wavetableIncrementOne;
			if (wavetablePosOne >= wavetableSize) {
				wavetablePosOne -= wavetableSize;
			}

			// Oscillation 2 - modulated
			aAudioBuffer[i] = wavetable[(unsigned int)wavetablePosTwo] * carrierAmp;
			wavetablePosTwo += w2srRatio * currentSample;
			if (wavetablePosTwo >= wavetableSize) {
				wavetablePosTwo -= wavetableSize;
			}

			if (wavetablePosTwo < 0.0f) {
				wavetablePosTwo += wavetableSize;
			}
		}
	}
	void calculateFFTWindow(float* input, float* output)
	{
		//apply window
		for (uint32_t i = 0; i < fftSize; i++) {
			fftWindowedAudio[i] = input[i] * fftWindow[i];
		}
	}
	void calculateJustFFT(float* input, float* output)
	{
		// execute FFTW plan
		fftw_execute_dft_r2c(fftPlan, fftWindowedAudio, fftOut);

		//@ToDo - Investigate why this part would be needed. It is done in fitness evaluation? No I think it is because this is fft calculated for target once. But new audio must be done each time.
		for (uint32_t i = 0; i < fftHalfSize; i++)
		{

			const float rawMagnitude = hypotf((float)fftOut[i][0], (float)fftOut[i][1]);
			const float magnitudeForFFTSize = rawMagnitude * fftOneOverSize;
			output[i] = magnitudeForFFTSize * fftOneOverWindowFactor;
		}
	}
	void calculateFFT(float* input, float* output)
	{
		//apply window
		for (uint32_t i = 0; i < fftSize; i++) {
			fftWindowedAudio[i] = input[i] * fftWindow[i];
		}

		// execute FFTW plan
		fftw_execute_dft_r2c(fftPlan, fftWindowedAudio, fftOut);

		//@ToDo - Investigate why this part would be needed. It is done in fitness evaluation? No I think it is because this is fft calculated for target once. But new audio must be done each time.
		for(uint32_t i = 0; i < fftHalfSize; i++)
		{

			const float rawMagnitude = hypotf((float)fftOut[i][0], (float)fftOut[i][1]);
			const float magnitudeForFFTSize = rawMagnitude * fftOneOverSize;
			output[i] = magnitudeForFFTSize * fftOneOverWindowFactor;
		}
	}
	void calculateFFTSpecial(float* input, float* output)
	{
		//apply window
		for (uint32_t i = 0; i < fftSize; i++) {
			fftWindowedAudio[i] = input[i];
		}

		// execute FFTW plan
		fftw_execute(fftPlan);

		//@ToDo - Investigate why this part would be needed. It is done in fitness evaluation? No I think it is because this is fft calculated for target once. But new audio must be done each time.
		for (uint32_t i = 0; i < fftHalfSize; i++)
		{

			const float rawMagnitude = hypotf((float)fftOut[i][0], (float)fftOut[i][1]);
			const float magnitudeForFFTSize = rawMagnitude * fftOneOverSize;
			output[i] = magnitudeForFFTSize * fftOneOverWindowFactor;
		}
	}

	uint32_t getWavetableSize()
	{
		return wavetableSize;
	}
	std::vector<float> scaleParams(const std::vector<float> aParams)
	{
		std::vector<float> ret;

		for (int i = 0; i < paramMins.size(); i++)
		{
			ret.push_back(paramMins[i] + aParams[i] * (paramMaxs[i] - paramMins[i]));
		}
		return ret;
	}
};

struct Evolutionary_Strategy_Arguments
{
	//Population Details//
	Population pop;

	//Evolutionary Strategy details//
	uint32_t numGenerations;
	std::vector<float> paramMin = { 0.0,0.0,0.0,0.0 };
	std::vector<float> paramMax = { 3520.0, 8.0, 3520.0, 1.0 };
	uint32_t audioLengthLog2 = 10;
};

class Evolutionary_Strategy
{
private:
public:
	uint32_t numGenerations;
	Population population;

	Objective objective;

	const float mPI;
	const float alpha;
	const float oneOverAlpha;
	const float rootTwoOverPi;
	const float betaScale; //1.f / (float)population->num_dimensions;
	const float beta;
public:
	Evolutionary_Strategy() :
		population({ 1024, 2048, 1024 + 2048, 4, 1024 + 2048, new float[(4 + 1) * 1024 + 2048] }),
		objective(population.populationLength, population.numDimensions, { 0.0,0.0,0.0,0.0 }, { 0.0,0.0,0.0,0.0 }, 10),
		numGenerations(100),
		mPI(3.14159265358979323846),
		alpha(1.4f),
		oneOverAlpha(1.f / alpha),
		rootTwoOverPi(sqrtf(2.f / (float)mPI)),
		betaScale(1.f / population.numDimensions), //1.f / (float)population->num_dimensions;
		beta(sqrtf(betaScale))
	{}
	Evolutionary_Strategy(const uint32_t aNumGenerations, const uint32_t aNumParents, const uint32_t aNumOffspring, const uint32_t aNumDimensions, const std::vector<float> aParamMin, const std::vector<float> aParamMax, uint32_t aAudioLengthLog2) :
		population({aNumParents, aNumOffspring, aNumDimensions, aNumParents + aNumOffspring, (aNumParents + aNumOffspring) * sizeof(float), new float[(aNumDimensions+1) * aNumParents + aNumOffspring]}),
		objective(aNumParents + aNumOffspring, aNumDimensions, aParamMin, aParamMax, aAudioLengthLog2),
		numGenerations(aNumGenerations),
		mPI(3.14159265358979323846),
		alpha(1.4f),
		oneOverAlpha(1.f / alpha),
		rootTwoOverPi(sqrtf(2.f / (float)mPI)),
		betaScale(1.f / (float)population.numDimensions), //1.f / (float)population->num_dimensions;
		beta(sqrtf(betaScale))
	{

	}
	virtual void init()
	{

	}

	virtual void initTargetAudio()
	{

	}

	//Read & write evolutionary strategy data//
	virtual void writePopulationData(void* aInputPopulationValueData, void* aOutputPopulationValueData, uint32_t aPopulationValueSize, void* aInputPopulationStepData, void* aOutputPopulationStepData, uint32_t aPopulationStepSize, void* aInputPopulationFitnessData, void* aOutputPopulationFitnessData, uint32_t aPopulationFitnessSize)
	{

	}
	virtual void readPopulationData(void* aInputPopulationValueData, void* aOutputPopulationValueData, uint32_t aPopulationValueSize, void* aInputPopulationStepData, void* aOutputPopulationStepData, uint32_t aPopulationStepSize, void* aInputPopulationFitnessData, void* aOutputPopulationFitnessData, uint32_t aPopulationFitnessSize)
	{

	}

	//Read & write synthesizer data//
	virtual void writeSynthesizerData(void* aOutputAudioBuffer, uint32_t aOutputAudioSize, void* aInputFFTDataBuffer, void* aInputFFTTargetBuffer, uint32_t aInputFFTSize)
	{

	}
	virtual void readSynthesizerData(void* aOutputAudioBuffer, uint32_t aOutputAudioSize, void* aInputFFTDataBuffer, void* aInputFFTTargetBuffer, uint32_t aInputFFTSize)
	{

	}

	//Execute evolutionary algorithm//
	virtual void executeGeneration()
	{

	}
	virtual void executeAllGenerations()
	{

	}
	virtual void parameterMatchAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{

	}

	//Audio files//
	virtual void readAudioFile()
	{

	}
	virtual void generateAudioFile()
	{

	}
	virtual void analyseAudio()
	{

	}
	virtual void setTargetFFT(float* aTargetAudio)
	{

	}
	virtual void printBest()
	{

	}
};

#endif