#ifndef EVOLUTIONARY_STRATEGY_CPU_HPP
#define EVOLUTIONARY_STRATEGY_CPU_HPP

#include "Evolutionary_Strategy.hpp"
#include "Benchmarker.hpp"

#include <vector>
#include <chrono>
#include <glm/glm.hpp>
#include <random>

static void outputAudioFileTwo(const char* aPath, float* aAudioBuffer, uint32_t aAudioLength)
{
	//SF_INFO sfinfo;
	//sfinfo.channels = 1;
	//sfinfo.samplerate = 44100;
	//sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	//
	////printf("writing: %s\n", file_path);
	//SNDFILE *outfile = sf_open(file_path, SFM_WRITE, &sfinfo);
	//if (sf_error(outfile) != SF_ERR_NO_ERROR) {
	//	printf("error: %s\n", sf_strerror(outfile));
	//}
	//sf_write_float(outfile, &audio_buffer[0], audio_length);
	//sf_write_sync(outfile);
	//sf_close(outfile);

	AudioFile<float> audioFile;
	AudioFile<float>::AudioBuffer buffer;

	buffer.resize(1);
	buffer[0].resize(aAudioLength);
	audioFile.setBitDepth(24);
	audioFile.setSampleRate(44100);

	for (int k = 0; k != aAudioLength; ++k)
		buffer[0][k] = (float)aAudioBuffer[k];

	audioFile.setAudioBuffer(buffer);
	audioFile.save(aPath);
}

struct Evolutionary_Strategy_CPU_Arguments
{
	//Generic Evolutionary Strategy arguments//
	Evolutionary_Strategy_Arguments es_args;
};

class Evolutionary_Strategy_CPU : public Evolutionary_Strategy
{
private:
	float* fftAudioData;
	float* audioData;
	std::vector<float> fftAudioData_;
	std::vector<float> audioData_;
	std::vector<float> a;
	std::vector<glm::uvec2> randomStates_;

	uint32_t numChunks_;
	uint32_t chunkSize_;

	uint32_t targetAudioLength;
	float* targetAudio;
	std::vector<float> targetFFT_;

	static const uint8_t numKernels_ = 9;
	enum kernelNames_ { initPopulation = 0, recombinePopulation, mutatePopulation, synthesisePopulation, applyWindowPopulation, CPUFFT, fitnessPopulation, sortPopulation, rotatePopulation };
	std::array<std::string, numKernels_> kernelNames_;
	std::chrono::nanoseconds kernelExecuteTime_[numKernels_];

	Benchmarker cpuBenchmarker_;

	void initRandomStates()
	{
		//Initialize random numbers in CPU buffer//
		unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		//std::uniform_int_distribution<int> distribution(0, 2147483647);
		std::uniform_int_distribution<int> distribution(0, 32767);

		uint32_t numRandomStates = population.populationSize * population.numDimensions;
		randomStates_.resize(numRandomStates);
		for (int i = 0; i < numRandomStates; ++i)
		{
			randomStates_[i].x = distribution(generator);
			randomStates_[i].y = distribution(generator);
		}
	}
	void initPopulationCPU()
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::uniform_real_distribution<float> distribution(0, 1.0);

		size_t populationSize = population.populationSize * ((population.numDimensions * 2+1));
		population.data = new float[populationSize];
		for (uint32_t i = 0; i != population.populationSize; ++i)
		{
			for (uint32_t j = 0; j != population.numDimensions; ++j)
			{
				*population.getValue(i, j) = distribution(generator);
				*population.getStep(i, j) = 0.1;
			}
			*population.getFitness(i) = 0.0;
		}
	}
	void calculateAudioFFT()
	{
		//for (uint32_t i = 0; i != population.populationSize * audioLength; i += audioLength)
		//{
		//	objective.calculateFFT(audioData[i], fftAudioData[i]);
		//}
	}

	void recombine()
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::uniform_int_distribution<int> distribution(0, population.populationSize / 2);

		uint32_t randomParentIdx;
		for (uint32_t i = 0; i != population.populationSize; ++i)
		{
			for (uint32_t j = 0; j != population.numDimensions; ++j)
			{
				randomParentIdx = distribution(generator) * 2;
				//population.data[i+j] = population.data[randomParentIdx];
				//population.data[i+j+1] = population.data[randomParentIdx+1];

				*population.getValue(i, j) = *population.getValue(randomParentIdx, j);
				*population.getStep(i, j) = *population.getStep(randomParentIdx, j);
			}
		}
	}
	void mutate()
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generatorValue(seed);
		std::default_random_engine generatorStep(seed);
		std::uniform_real_distribution<float> distributionStep(-0.01, 0.01);

		uint32_t randomParentIdx;
		for (uint32_t i = 0; i != population.populationSize; ++i)
		{
			for (uint32_t j = 0; j != population.numDimensions; ++j)
			{
				float step = *population.getStep(i, j);
				std::uniform_real_distribution<float> distributionValue(-abs(step), abs(step));
				
				float newVal = *population.getValue(i, j) + distributionValue(generatorValue);
				while (newVal > 1.0 || newVal < 0.0)
					newVal = -newVal / 2.0;
				
				*population.getValue(i, j) = newVal;
				*population.getStep(i, j) = step + distributionStep(generatorStep);

				//std::uniform_real_distribution<float> distributionValue(-abs(1.0), abs(1.0));
				//float newVal = distributionValue(generatorValue);
				//while (newVal > 1.0 || newVal < 0.0)
				//	newVal = -newVal / 2.0;
				//*population.getValue(i, j) = newVal;
			}
			//if (i == 200)
			//{
			//	*population.getValue(i, 0) = 0.41193181818;
			//	*population.getValue(i, 1) = 0.375;
			//	*population.getValue(i, 2) = 0.05681818181;
			//	*population.getValue(i, 3) = 1.0;
			//}
			//*population.getValue(i, 0) = 0.41193181818;
			//*population.getValue(i, 1) = 0.375;
			//*population.getValue(i, 2) = 0.05681818181;
			//*population.getValue(i, 3) = 1.0;

		}
	}
	void synthesis()
	{
		for (uint32_t i = 0; i != population.populationSize; ++i)
		{
			const std::vector<float> currentParams = { *population.getValue(i, 0), *population.getValue(i, 1), *population.getValue(i, 2), *population.getValue(i, 3) };
			//const std::vector<float> currentParams = { 0.000689655172, 0.33333, 0.005, 1.0 };
			objective.synthesiseAudio(currentParams, &audioData_[i * objective.audioLength]);
		}
	}
	void applyFFTWindow()
	{
		for (uint32_t i = 0; i != population.populationLength; ++i)
		{
			objective.calculateFFTWindow(&audioData_[i * objective.audioLength], &fftAudioData_[i * objective.fftHalfSize]);
		}
	}
	void fft()
	{
		for (uint32_t i = 0; i != population.populationLength; ++i)
		{
			objective.calculateJustFFT(&audioData_[i * objective.audioLength], &fftAudioData_[i * objective.fftHalfSize]);
		}
	}
	void fitness()
	{
		float error = 0.0f;
		float errorTemp = 0.0f;
		for (uint32_t i = 0; i != population.populationLength; ++i)
		{
			float fft_magnitude_sum = 0.0;
			for (uint32_t j = 0; j < objective.fftHalfSize; j+=2)
			{
					//GPU paper//
					//const float raw_magnitude = hypot(fftAudioData_[(i * objective.fftHalfSize) + (j / 2)],
					//	fftAudioData_[(i * objective.fftHalfSize) + (j / 2) + 1]);
					//const float magnitude_for_fft_size = raw_magnitude * objective.fftOneOverSize;
					//errorTemp = (magnitude_for_fft_size * objective.fftOneOverWindowFactor) -
					//	targetFFT_[j/2];
					//error += errorTemp * errorTemp;

				const float raw_magnitude = std::hypot(fftAudioData_[i * objective.fftOutSize + j],
					fftAudioData_[i * objective.fftOutSize + j + 1]);
				const float magnitude_for_fft_size = raw_magnitude * objective.fftOneOverSize;
				errorTemp = (magnitude_for_fft_size * objective.fftOneOverWindowFactor) -
					targetFFT_[j / 2];
				error += errorTemp * errorTemp;

				//float temp = fftAudioData_[(i * objective.fftHalfSize) + (j)] - targetFFT_[j];
				//error += temp * temp;


					//error += ((fftAudioData_[(i * objective.fftHalfSize) + (j)] * targetFFT_[j])) / objective.fftHalfSize;

				//From paper//
				//const float fft_magnitude = fftAudioData_[(i * objective.fftHalfSize) + (j)] / (objective.audioLength * objective.fftWindowFactor);
				//error += powf((targetFFT_[j] - fft_magnitude), 2);
				//fft_magnitude_sum += targetFFT_[j] / (objective.audioLength * objective.fftWindowFactor);
			}

			//*population.getFitness(i) = sqrt(error/pow(fft_magnitude_sum, 2));
			*population.getFitness(i) = error;

			error = 0;
			errorTemp = 0;


			//const float raw_magnitude = hypot(group_fft[CHUNK_SIZE_FITNESS * local_index + i],
			//	group_fft[CHUNK_SIZE_FITNESS * local_index + i + 1]);
			//const float magnitude_for_fft_size = raw_magnitude * FFT_ONE_OVER_SIZE;
			//tmp = (magnitude_for_fft_size * FFT_ONE_OVER_WINDOW_FACTOR) -
			//	group_fft_target[second_half_local_target * CHUNK_SIZE_FITNESS / 2 + i / 2];
			//error = errorTemp * errorTemp;
			//error = fftAudioData_[i];
			//*population.getFitness(i) = error;
		}
	}
	void sort()
	{
		population.quickSortPopulation(0, population.populationSize-1);
	}
	void rotate()
	{
		//rotationIndex_ = (rotationIndex_ == 0 ? 1 : 0);
	}

public:
	//Evolutionary_Strategy_CPU(uint32_t aNumGenerations, uint32_t aNumParents, uint32_t aNumOffspring, uint32_t aNumDimensions, const std::vector<float> aParamMin, const std::vector<float> aParamMax, uint32_t aAudioLengthLog2) :
	//	Evolutionary_Strategy(aNumGenerations, aNumParents, aNumOffspring, aNumDimensions, aParamMin, aParamMax, aAudioLengthLog2)
	//{
	//
	//}
	Evolutionary_Strategy_CPU(Evolutionary_Strategy_CPU_Arguments args) :
		Evolutionary_Strategy(args.es_args.numGenerations, args.es_args.pop.numParents, args.es_args.pop.numOffspring, args.es_args.pop.numDimensions, args.es_args.paramMin, args.es_args.paramMax, args.es_args.audioLengthLog2),
		cpuBenchmarker_(std::string("cpulog(pop=" + std::to_string(args.es_args.pop.populationLength) + "gens=" + std::to_string(args.es_args.numGenerations) + "audioBlockSize=" + std::to_string(1 << args.es_args.audioLengthLog2) + ").csv"), { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" }),
		kernelNames_({ "initPopulation", "recombinePopulation", "mutatePopulation", "synthesisePopulation", "applyWindowPopulation", "FFTW", "fitnessPopulation", "sortPopulation", "rotatePopulation" })
	{
		//randomStates_.resize(population.populationSize);
		//initRandomStates();
		//initPopulationCPU();

		audioData_.resize(population.populationSize * objective.audioLength);
		fftAudioData_.resize(population.populationSize * objective.fftHalfSize);
		targetFFT_.resize(objective.fftHalfSize);
	}
	void init()
	{
	}

	void initTargetAudio()
	{

	}

	//Read & write evolutionary strategy data//
	void writePopulationData(void* aInputPopulationValueData, void* aOutputPopulationValueData, uint32_t aPopulationValueSize, void* aInputPopulationStepData, void* aOutputPopulationStepData, uint32_t aPopulationStepSize, void* aInputPopulationFitnessData, void* aOutputPopulationFitnessData, uint32_t aPopulationFitnessSize)
	{

	}
	void readPopulationData(void* aInputPopulationValueData, void* aOutputPopulationValueData, uint32_t aPopulationValueSize, void* aInputPopulationStepData, void* aOutputPopulationStepData, uint32_t aPopulationStepSize, void* aInputPopulationFitnessData, void* aOutputPopulationFitnessData, uint32_t aPopulationFitnessSize)
	{
		for (uint32_t i = 0; i != population.populationSize; ++i)
		{
			for (uint32_t j = 0; j != population.numDimensions; ++j)
			{
				((float*)aInputPopulationValueData)[i+j] = *population.getValue(i, j);
				((float*)aInputPopulationStepData)[i + j] = *population.getStep(i, j);
			}
			((float*)aInputPopulationFitnessData)[i] = *population.getFitness(i);
		}
	}

	//Read & write synthesizer data//
	void writeSynthesizerData(void* aOutputAudioBuffer, uint32_t aOutputAudioSize, void* aInputFFTDataBuffer, void* aInputFFTTargetBuffer, uint32_t aInputFFTSize)
	{

	}
	void readSynthesizerData(void* aOutputAudioBuffer, uint32_t aOutputAudioSize, void* aInputFFTDataBuffer, void* aInputFFTTargetBuffer, uint32_t aInputFFTSize)
	{

	}

	//Execute evolutionary algorithm//
	void executeGeneration()
	{
		std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

		cpuBenchmarker_.startTimer(kernelNames_[1]);
		recombine();
		cpuBenchmarker_.pauseTimer(kernelNames_[1]);

		//auto end = std::chrono::steady_clock::now();
		//auto diff = end - start;
		//kernelExecuteTime_[recombinePopulation] += diff;
		//
		//start = std::chrono::steady_clock::now();

		cpuBenchmarker_.startTimer(kernelNames_[2]);
		mutate();
		cpuBenchmarker_.pauseTimer(kernelNames_[2]);

		//end = std::chrono::steady_clock::now();
		//diff = end - start;
		//kernelExecuteTime_[mutatePopulation] += diff;
		//
		//start = std::chrono::steady_clock::now();
		
		cpuBenchmarker_.startTimer(kernelNames_[3]);
		synthesis();
		cpuBenchmarker_.pauseTimer(kernelNames_[3]);

		//end = std::chrono::steady_clock::now();
		//diff = end - start;
		//kernelExecuteTime_[synthesisePopulation] += diff;

		//outputAudioFileTwo("testio.wav", &audioData_[0], objective.audioLength);

		//start = std::chrono::steady_clock::now();

		cpuBenchmarker_.startTimer(kernelNames_[4]);
		applyFFTWindow();
		cpuBenchmarker_.pauseTimer(kernelNames_[4]);

		cpuBenchmarker_.startTimer(kernelNames_[5]);
		fft();
		cpuBenchmarker_.pauseTimer(kernelNames_[5]);

		cpuBenchmarker_.startTimer(kernelNames_[6]);
		fitness();
		cpuBenchmarker_.pauseTimer(kernelNames_[6]);

		//end = std::chrono::steady_clock::now();
		//diff = end - start;
		//kernelExecuteTime_[fitnessPopulation] += diff;
		//
		//start = std::chrono::steady_clock::now();

		cpuBenchmarker_.startTimer(kernelNames_[7]);
		sort();
		cpuBenchmarker_.pauseTimer(kernelNames_[7]);

		cpuBenchmarker_.startTimer(kernelNames_[8]);
		rotate();
		cpuBenchmarker_.pauseTimer(kernelNames_[8]);

		//end = std::chrono::steady_clock::now();
		//diff = end - start;
		//kernelExecuteTime_[sortPopulation] += diff;
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

		cpuBenchmarker_.startTimer("Total Audio Analysis Time");
		cpuBenchmarker_.pauseTimer("Total Audio Analysis Time");
		cpuBenchmarker_.startTimer("Total Audio Analysis Time");

		initRandomStates();
		for (int i = 0; i < numChunks_; i++)
		{
			//Initialise target audio and new population//
			setTargetAudio(&aTargetAudio[chunkSize_ * i], chunkSize_);
			initPopulationCPU();

			//Execute number of ES generations on chunk//
			executeAllGenerations();

			printf("Audio chunk %d evaluated:\n", i);
			printBest();
		}

		cpuBenchmarker_.pauseTimer("Total Audio Analysis Time");

		cpuBenchmarker_.elapsedTimer(kernelNames_[1]);
		cpuBenchmarker_.elapsedTimer(kernelNames_[2]);
		cpuBenchmarker_.elapsedTimer(kernelNames_[3]);
		cpuBenchmarker_.elapsedTimer(kernelNames_[4]);
		cpuBenchmarker_.elapsedTimer(kernelNames_[5]);
		cpuBenchmarker_.elapsedTimer(kernelNames_[6]);
		cpuBenchmarker_.elapsedTimer(kernelNames_[7]);
		cpuBenchmarker_.elapsedTimer(kernelNames_[8]);
		cpuBenchmarker_.elapsedTimer("Total Audio Analysis Time");
	}

	//Audio files//
	void readAudioFile()
	{

	}
	void generateAudioFile()
	{

	}
	void analyseAudio()
	{

	}
	void setTargetAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{
		//Calculate and load fft data for target audio//
		targetAudioLength = aTargetAudioLength;
		objective.calculateFFT(aTargetAudio, &targetFFT_[0]);
	}
	void printBest()
	{
		float tempData[4];
		tempData[0] = *population.getValue(0, 0);
		tempData[1] = *population.getValue(0, 1);
		tempData[2] = *population.getValue(0, 2);
		tempData[3] = *population.getValue(0, 3);
		printf("Best parameters found:\n Fc = %f\n I = %f\n Fm = %f\n A = %f\n\n\n", tempData[0] * objective.paramMaxs[0], tempData[1] * objective.paramMaxs[1], tempData[2] * objective.paramMaxs[2], tempData[3] * objective.paramMaxs[3]);
		printf("Fitness: %f\n", *population.getFitness(0));
	}
};

#endif