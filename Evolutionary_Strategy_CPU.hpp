#ifndef EVOLUTIONARY_STRATEGY_CPU_HPP
#define EVOLUTIONARY_STRATEGY_CPU_HPP

#include "Evolutionary_Strategy.hpp"

#include <vector>
#include <chrono>
#include <glm/glm.hpp>
#include <random>

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
	void initRandomStates()
	{
		//Initialize random numbers in CPU buffer//
		unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		//std::uniform_int_distribution<int> distribution(0, 2147483647);
		std::uniform_int_distribution<int> distribution(0, 32767);

		uint32_t numRandomStates = population.populationSize;
		randomStates_.resize(numRandomStates);
		for (int i = 0; i < numRandomStates; ++i)
		{
			randomStates_[i].x = distribution(generator);
			randomStates_[i].y = distribution(generator);
		}
	}
	void initPopulation()
	{
		size_t populationSize = population.populationSize * ((population.numDimensions * 2) + 1);
		population.data = new float[populationSize];
		for (uint32_t i = 0; i != populationSize; i)
		{
			for (uint32_t j = 0; j != population.numDimensions; ++j)
			{
				population.data[i++] = randomStates_[0].x;
				population.data[i++] = 0.1;
			}
			population.data[i++] = 0.0;
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
		uint32_t randomParentIdx = 0;	//@ToDo - Pick randomly.
		for (uint32_t i = 0; i != population.populationSize; ++i)
		{
			for (uint32_t j = 0; j != population.numDimensions; ++j)
			{
				population.data[i+j] = population.data[randomParentIdx];
				population.data[i+j+1] = population.data[randomParentIdx+1];
			}
		}
		
	}
	void mutate()
	{

	}
	void synthesis()
	{

	}
	void applyWindow()
	{

	}
	void fitness()
	{

	}
	void sort()
	{

	}
	void swapPopulations()
	{

	}

public:
	Evolutionary_Strategy_CPU(uint32_t aNumGenerations, uint32_t aNumParents, uint32_t aNumOffspring, uint32_t aNumDimensions, const std::vector<float> aParamMin, const std::vector<float> aParamMax, uint32_t aAudioLengthLog2) :
		Evolutionary_Strategy(aNumGenerations, aNumParents, aNumOffspring, aNumDimensions, aParamMin, aParamMax, aAudioLengthLog2)
	{

	}
	Evolutionary_Strategy_CPU(Evolutionary_Strategy_CPU_Arguments args) :
		Evolutionary_Strategy(args.es_args.numGenerations, args.es_args.pop.numParents, args.es_args.pop.numOffspring, args.es_args.pop.numDimensions, args.es_args.paramMin, args.es_args.paramMax, args.es_args.audioLengthLog2)
	{
		//randomStates_.resize(population.populationSize);
		initRandomStates();
		initPopulation();

		audioData_.resize(population.populationSize * objective.audioLength);
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

	}
	void executeAllGenerations()
	{

	}
	void parameterMatchAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{

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
	void setTargetFFT(float* aTargetAudio)
	{

	}
};

#endif