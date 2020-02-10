#ifndef EVOLUTIONARY_STRATEGY_OPENCL_HPP
#define EVOLUTIONARY_STRATEGY_OPENCL_HPP

//#define CL_HPP_TARGET_OPENCL_VERSION 210
//#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>
//#include <CL/cl.hpp>
#include <CL/cl_gl.h>

#include <math.h>
#include <array>
#include <fstream>
#include <random>

#include <clFFT.h>

#include "Evolutionary_Strategy.hpp"

enum DeviceType { INTEGRATED, DISCRETE };

struct Evolutionary_Strategy_OpenCL_Arguments
{
	//Generic Evolutionary Strategy arguments//
	Evolutionary_Strategy_Arguments es_args;

	//OpenCL details//
	uint32_t workgroupX = 32;
	uint32_t workgroupY = 1;
	uint32_t workgroupZ = 1;
	uint32_t workgroupSize = workgroupX * workgroupY * workgroupZ;
	std::string kernelSourcePath;

	DeviceType deviceType;
};

class Evolutionary_Strategy_OpenCL : public Evolutionary_Strategy
{
private:
	cl_int errorStatus_ = 0;
	cl_uint num_platforms, num_devices;
	cl::Platform platform_;
	cl::Context context_;
	cl::Device device_;
	cl::CommandQueue commandQueue_;
	cl::Program kernelProgram_;
	std::string kernelSourcePath_;

	//Kernels//
	static const uint8_t numKernels_ = 9;
	enum kernelNames_ { initPopulation = 0, recombinePopulation, mutatePopulation, synthesisePopulation, applyWindowPopulation, openCLFFT, fitnessPopulation, sortPopulation, copyPopulation };
	std::array<std::string, numKernels_> kernelNames_;
	std::array<cl::Kernel, numKernels_> kernels_;
	std::chrono::nanoseconds kernelExecuteTime_[numKernels_];

	//Buffers//
	static const uint8_t numBuffers_ = 14;
	enum storageBufferNames_ { inputPopulationValueBuffer = 0, inputPopulationStepBuffer, inputPopulationFitnessBuffer, outputPopulationValueBuffer, outputPopulationStepBuffer, outputPopulationFitnessBuffer, randomStatesBuffer, paramMinBuffer, paramMaxBuffer, outputAudioBuffer, inputFFTDataBuffer, inputFFTTargetBuffer, rotationIndexBuffer, wavetableBuffer};
	std::array<cl::Buffer, numBuffers_> storageBuffers_;
	std::array<uint32_t, numBuffers_> storageBufferSizes_;

	//Shader workgroup details//
	uint32_t globalSize_;
	uint32_t workgroupX;
	uint32_t workgroupY;
	uint32_t workgroupZ;
	uint32_t workgroupSize;
	uint32_t numWorkgroupsPerParent;

	uint32_t numChunks_;
	uint32_t chunkSize_;

	uint32_t targetAudioLength;
	float* targetAudio;
	float* targetFFT_;

	clfftPlanHandle planHandle;

	uint32_t rotationIndex_;

	DeviceType deviceType_;

	const std::string compilerArguments_ =
		"-cl-fast-relaxed-math "
		"-D WRKGRPSIZE=%d "
		"-D NUM_DIMENSIONS=%d "
		"-D AUDIO_WAVE_FORM_SIZE=%d "
		"-D POPULATION_COUNT=%d "
		"-D POPULATION_SIZE=%d "
		"-D NUM_WGS_FOR_PARENTS=%d "
		"-D ALPHA=%f "
		"-D ONE_OVER_ALPHA=%f "
		"-D ROOT_TWO_OVER_PI=%f "
		"-D BETA_SCALE=%f "
		"-D BETA=%f "
		"-D FFT_ONE_OVER_SIZE=%f "
		"-D FFT_ONE_OVER_WINDOW_FACTOR=%f "
		"-D FFT_OUT_SIZE=%d "
		"-D FFT_HALF_SIZE=%d "
		"-D WAVETABLE_SIZE=%d";
public:
	Evolutionary_Strategy_OpenCL(uint32_t aNumGenerations, uint32_t aNumParents, uint32_t aNumOffspring, uint32_t aNumDimensions, const std::vector<float> aParamMin, const std::vector<float> aParamMax, std::string aKernelSourcePath, uint32_t aAudioLengthLog2) :
		Evolutionary_Strategy(aNumGenerations, aNumParents, aNumOffspring, aNumDimensions, aParamMin, aParamMax, aAudioLengthLog2),
		kernelSourcePath_(aKernelSourcePath),
		workgroupX(32),
		workgroupY(1),
		workgroupZ(1),
		workgroupSize(workgroupX*workgroupY*workgroupZ),
		numWorkgroupsPerParent(population.numParents / workgroupSize),
		kernelNames_({ "initPopulation", "recombinePopulation", "mutatePopulation", "synthesisePopulation", "applyWindowPopulation", "openCLFFT", "fitnessPopulation", "sortPopulation", "copyPopulation" }),
		globalSize_(population.populationSize)
	{
		init();
	}
	Evolutionary_Strategy_OpenCL(Evolutionary_Strategy_OpenCL_Arguments args) :
		Evolutionary_Strategy(args.es_args.numGenerations, args.es_args.pop.numParents, args.es_args.pop.numOffspring, args.es_args.pop.numDimensions, args.es_args.paramMin, args.es_args.paramMax, args.es_args.audioLengthLog2),
		kernelSourcePath_(args.kernelSourcePath),
		workgroupX(args.workgroupX),
		workgroupY(args.workgroupY),
		workgroupZ(args.workgroupZ),
		workgroupSize(args.workgroupX*args.workgroupY*args.workgroupZ),
		numWorkgroupsPerParent(population.numParents / workgroupSize),
		kernelNames_({ "initPopulation", "recombinePopulation", "mutatePopulation", "synthesisePopulation", "applyWindowPopulation", "openCLFFT", "fitnessPopulation", "sortPopulation", "copyPopulation" }),
		globalSize_(population.populationSize),
		deviceType_(args.deviceType)
	{
		init();
	}
	void init()
	{
		initBufferSizesCL();
		initMemory();

		//Initialise OpenCL context//
		initContextCL(1, 0);

		initCLFFT();
		initBuffersCL();
		initKernelsCL(kernelSourcePath_);
		initKernelArgumentsCL();
		initRandomStateCL();
	}
	void initMemory()
	{
		targetFFT_ = new float[objective.fftHalfSize];
	}
	void initCLFFT()
	{
		//clFFT Variables//
		clfftDim dim = CLFFT_1D;
		size_t clLengths[1] = { objective.audioLength };
		size_t in_strides[1] = { 1 };
		size_t out_strides[1] = { 1 };
		size_t in_dist = (size_t)objective.audioLength;
		size_t out_dist = (size_t)objective.audioLength / 2 + 4;

		//Update member variables with new information//
		objective.fftOutSize = out_dist * 2;
		storageBufferSizes_[inputFFTDataBuffer] = population.populationSize * objective.fftOutSize * sizeof(float);
		storageBufferSizes_[inputFFTTargetBuffer] = objective.fftHalfSize * sizeof(float);

		//Setup clFFT//
		clfftSetupData fftSetup;
		errorStatus_ = clfftInitSetupData(&fftSetup);
		errorStatus_ = clfftSetup(&fftSetup);

		//Create a default plan for a complex FFT//
		errorStatus_ = clfftCreateDefaultPlan(&planHandle, context_(), dim, clLengths);

		//Set plan parameters//
		errorStatus_ = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
		errorStatus_ = clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
		errorStatus_ = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
		errorStatus_ = clfftSetPlanBatchSize(planHandle, (size_t)population.populationSize);
		clfftSetPlanInStride(planHandle, dim, in_strides);
		clfftSetPlanOutStride(planHandle, dim, out_strides);
		clfftSetPlanDistance(planHandle, in_dist, out_dist);

		//Bake clFFT plan//
		errorStatus_ = clfftBakePlan(planHandle, 1, &commandQueue_(), NULL, NULL);
		if (errorStatus_)
			std::cout << "ERROR creating clFFT plan. Status code: " << errorStatus_ << std::endl;
	}
	//@ToDo - Right now pick platform. Can extend to pick best available.
	void initContextCL(uint8_t aPlatform, uint8_t aDevice)
	{
		//Discover platforms//
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Create contex properties for first platform//
		cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[aPlatform])(), 0 };	//Need to specify platform 3 for dedicated graphics - Harri Laptop.

		//Create context context using platform for GPU device//
		if (deviceType_ == DeviceType::INTEGRATED)
			context_ = cl::Context(CL_DEVICE_TYPE_CPU, contextProperties);
		if(deviceType_ == DeviceType::DISCRETE)
			context_ = cl::Context(CL_DEVICE_TYPE_GPU, contextProperties);
		else
			context_ = cl::Context(CL_DEVICE_TYPE_ALL, contextProperties);

		//Get device list from context//
		std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();

		//Create command queue for first device - Profiling enabled//
		commandQueue_ = cl::CommandQueue(context_, devices[aDevice], CL_QUEUE_PROFILING_ENABLE, &errorStatus_);	//Need to specify device 1[0] of platform 3[2] for dedicated graphics - Harri Laptop.
		if (errorStatus_)
			std::cout << "ERROR creating command queue for device. Status code: " << errorStatus_ << std::endl;
	}
	void initKernelsCL(std::string aPath)
	{
		//Read in program source//
		std::ifstream sourceFileName(aPath.c_str());
		std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName), (std::istreambuf_iterator<char>()));

		//Create program source object from std::string source code//
		std::vector<std::string> programSources;
		programSources.push_back(sourceFile);
		cl::Program::Sources source(programSources);	//Apparently this takes a vector of strings as the program source.

		//Create program from source code//
		cl::Program evolutionaryStrategyProgram(context_, source, &errorStatus_);
		if (errorStatus_)
			std::cout << "ERROR creating program from source. Status code: " << errorStatus_ << std::endl;
		
		//Set constants for access in kernels//
		//Using --cl-fast-relaxed-math for performance - From tests, accuracy not noticably affected//
		char option_buffer[1024];
		snprintf(option_buffer, sizeof option_buffer,
			compilerArguments_.c_str(),
			workgroupSize,
			population.numDimensions,
			objective.audioLength,
			population.populationSize,	//@ToDo - SHOULD THIS BE POPULATION SIZE OR NUMBER OF PARENTS!
			population.populationSize * population.numDimensions,
			numWorkgroupsPerParent,
			alpha, 
			oneOverAlpha, 
			rootTwoOverPi, 
			betaScale, 
			beta,
			objective.fftOneOverSize,
			objective.fftOneOverWindowFactor,
			objective.fftOutSize, objective.fftHalfSize,
			objective.getWavetableSize());

		//Build program//
		evolutionaryStrategyProgram.build(option_buffer);

		//Create kernels from source code//
		for(auto iter = kernels_.begin(); iter != kernels_.end(); ++iter)
		{
			uint32_t idx = std::distance(kernels_.begin(), iter);
			cl::Kernel& currentKernel = *iter;

			currentKernel = cl::Kernel(evolutionaryStrategyProgram, kernelNames_[idx].c_str(), &errorStatus_);
			if (errorStatus_)
				std::cout << "ERROR creating kernel " << idx << ". Status code: " << errorStatus_ << std::endl;
		}
	}
	void initBufferSizesCL()
	{

		//Initialise buffer sizes//
		for (int i = 0; i != randomStatesBuffer; ++i)
			storageBufferSizes_[i] = population.populationSize * population.numDimensions * sizeof(float) * 2;

		storageBufferSizes_[randomStatesBuffer] = population.populationSize * sizeof(glm::uvec2);
		storageBufferSizes_[inputPopulationFitnessBuffer] = population.populationSize * sizeof(float) * 2;
		storageBufferSizes_[outputPopulationFitnessBuffer] = population.populationSize * sizeof(float) * 2;
		storageBufferSizes_[paramMinBuffer] = population.numDimensions * sizeof(float);
		storageBufferSizes_[paramMaxBuffer] = population.numDimensions * sizeof(float);
		storageBufferSizes_[outputAudioBuffer] = population.populationSize * objective.audioLength * sizeof(float);
		storageBufferSizes_[rotationIndexBuffer] = sizeof(uint32_t);
		storageBufferSizes_[wavetableBuffer] = objective.wavetableSize * sizeof(float);
	}
	void initBuffersCL()
	{
		//Create GPU OpenCL buffers//
		uint32_t memoryFlags = CL_MEM_READ_WRITE;
		for (auto iter = storageBuffers_.begin(); iter != storageBuffers_.end(); ++iter)
		{
			uint32_t idx = std::distance(storageBuffers_.begin(), iter);
			cl::Buffer& currentBuffer = *iter;

			//Set access flag appropriatly for each buffer//
			if (idx == paramMinBuffer || idx == paramMaxBuffer || idx == inputFFTTargetBuffer)
				memoryFlags = CL_MEM_READ_ONLY;
			else
				memoryFlags = CL_MEM_READ_WRITE;

			currentBuffer = cl::Buffer(context_, memoryFlags, storageBufferSizes_[idx]);
			//@ToDo - Error when trying to use error status                           ^//
			//if (errorStatus_)
			//	std::cout << "ERROR creating buffer " << idx << ". Status code: " << errorStatus_ << std::endl;
		}

		//Write intial data to buffers @ToDO - Do elsewhere//
		commandQueue_.enqueueWriteBuffer(storageBuffers_[paramMinBuffer], CL_TRUE, 0, population.numDimensions * sizeof(float), objective.paramMins.data());
		commandQueue_.enqueueWriteBuffer(storageBuffers_[paramMaxBuffer], CL_TRUE, 0, population.numDimensions * sizeof(float), objective.paramMaxs.data());
		commandQueue_.enqueueWriteBuffer(storageBuffers_[wavetableBuffer], CL_TRUE, 0, objective.wavetableSize * sizeof(float), objective.wavetable);
	}
	void initKernelArgumentsCL()
	{
		//Set initPopulation kernel arguments//
		kernels_[initPopulation].setArg(0, sizeof(cl_mem), &(storageBuffers_[inputPopulationValueBuffer]));
		kernels_[initPopulation].setArg(1, sizeof(cl_mem), &(storageBuffers_[inputPopulationStepBuffer]));
		kernels_[initPopulation].setArg(2, sizeof(cl_mem), &(storageBuffers_[inputPopulationFitnessBuffer]));
		kernels_[initPopulation].setArg(3, sizeof(cl_mem), &(storageBuffers_[randomStatesBuffer]));
		kernels_[initPopulation].setArg(4, sizeof(cl_mem), &(storageBuffers_[rotationIndexBuffer]));

		//Set recombinePopulation kernel arguments//
		kernels_[recombinePopulation].setArg(0, sizeof(cl_mem), &(storageBuffers_[inputPopulationValueBuffer]));
		kernels_[recombinePopulation].setArg(1, sizeof(cl_mem), &(storageBuffers_[inputPopulationStepBuffer]));
		kernels_[recombinePopulation].setArg(2, sizeof(cl_mem), &(storageBuffers_[rotationIndexBuffer]));

		//Set mutatePopulation kernel arguments//
		kernels_[mutatePopulation].setArg(0, sizeof(cl_mem), &(storageBuffers_[inputPopulationValueBuffer]));
		kernels_[mutatePopulation].setArg(1, sizeof(cl_mem), &(storageBuffers_[inputPopulationStepBuffer]));
		kernels_[mutatePopulation].setArg(2, sizeof(cl_mem), &(storageBuffers_[randomStatesBuffer]));
		kernels_[mutatePopulation].setArg(3, sizeof(cl_mem), &(storageBuffers_[rotationIndexBuffer]));

		//Set synthesisePopulation kernel arguments//
		kernels_[synthesisePopulation].setArg(0, sizeof(cl_mem), &(storageBuffers_[outputAudioBuffer]));
		kernels_[synthesisePopulation].setArg(1, sizeof(cl_mem), &(storageBuffers_[inputPopulationValueBuffer]));
		kernels_[synthesisePopulation].setArg(2, sizeof(cl_mem), &(storageBuffers_[paramMinBuffer]));
		kernels_[synthesisePopulation].setArg(3, sizeof(cl_mem), &(storageBuffers_[paramMaxBuffer]));
		kernels_[synthesisePopulation].setArg(4, sizeof(cl_mem), &(storageBuffers_[rotationIndexBuffer]));
		kernels_[synthesisePopulation].setArg(5, sizeof(cl_mem), &(storageBuffers_[wavetableBuffer]));

		//Set applyWindowPopulation kernel arguments//
		kernels_[applyWindowPopulation].setArg(0, sizeof(cl_mem), &(storageBuffers_[outputAudioBuffer]));

		//Set fitnessPopulation kernel arguments//
		kernels_[fitnessPopulation].setArg(0, sizeof(cl_mem), &(storageBuffers_[inputPopulationFitnessBuffer]));
		kernels_[fitnessPopulation].setArg(1, sizeof(cl_mem), &(storageBuffers_[inputFFTDataBuffer]));
		kernels_[fitnessPopulation].setArg(2, sizeof(cl_mem), &(storageBuffers_[inputFFTTargetBuffer]));
		kernels_[fitnessPopulation].setArg(3, sizeof(cl_mem), &(storageBuffers_[rotationIndexBuffer]));

		//Set sortPopulation kernel arguments//
		kernels_[sortPopulation].setArg(0, sizeof(cl_mem), &(storageBuffers_[inputPopulationValueBuffer]));
		kernels_[sortPopulation].setArg(1, sizeof(cl_mem), &(storageBuffers_[inputPopulationStepBuffer]));
		kernels_[sortPopulation].setArg(2, sizeof(cl_mem), &(storageBuffers_[inputPopulationFitnessBuffer]));
		kernels_[sortPopulation].setArg(3, sizeof(cl_mem), &(storageBuffers_[outputPopulationValueBuffer]));
		kernels_[sortPopulation].setArg(4, sizeof(cl_mem), &(storageBuffers_[outputPopulationStepBuffer]));
		kernels_[sortPopulation].setArg(5, sizeof(cl_mem), &(storageBuffers_[outputPopulationFitnessBuffer]));
		kernels_[sortPopulation].setArg(6, sizeof(cl_mem), &(storageBuffers_[rotationIndexBuffer]));

		//Set copyPopulation kernel arguments//
		kernels_[copyPopulation].setArg(0, sizeof(cl_mem), &(storageBuffers_[outputPopulationValueBuffer]));
		kernels_[copyPopulation].setArg(1, sizeof(cl_mem), &(storageBuffers_[outputPopulationStepBuffer]));
		kernels_[copyPopulation].setArg(2, sizeof(cl_mem), &(storageBuffers_[outputPopulationFitnessBuffer]));
		kernels_[copyPopulation].setArg(3, sizeof(cl_mem), &(storageBuffers_[inputPopulationValueBuffer]));
		kernels_[copyPopulation].setArg(4, sizeof(cl_mem), &(storageBuffers_[inputPopulationStepBuffer]));
		kernels_[copyPopulation].setArg(5, sizeof(cl_mem), &(storageBuffers_[inputPopulationFitnessBuffer]));
		kernels_[copyPopulation].setArg(6, sizeof(cl_mem), &(storageBuffers_[rotationIndexBuffer]));
	}

	void initPopulationCL()
	{
		rotationIndex_ = 0;
		commandQueue_.enqueueWriteBuffer(storageBuffers_[rotationIndexBuffer], CL_TRUE, 0, sizeof(uint32_t), &rotationIndex_);
		commandQueue_.finish();

		//Run initialise population kernel//
		commandQueue_.enqueueNDRangeKernel(kernels_[initPopulation], cl::NullRange, globalSize_, workgroupSize, NULL);
		commandQueue_.finish();
	}

	void initRandomStateCL()
	{
		//Initialize random numbers in CPU buffer//
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		//std::uniform_int_distribution<int> distribution(0, 2147483647);
		std::uniform_int_distribution<int> distribution(0, 32767);

		uint32_t numRandomStates = population.populationSize;
		glm::uvec2* rand_state = new glm::uvec2[numRandomStates];
		for (int i = 0; i < numRandomStates; ++i)
		{
			rand_state[i].x = distribution(generator);
			rand_state[i].y = distribution(generator);
		}

		//Write random states to GPU randomStatesBuffer//
		uint32_t cpySize = numRandomStates * sizeof(glm::uvec2);
		commandQueue_.enqueueWriteBuffer(storageBuffers_[randomStatesBuffer], CL_TRUE, 0, cpySize, rand_state);
		commandQueue_.finish();

		delete(rand_state);
	}
	void writePopulationData(void* aInputPopulationValueData, void* aOutputPopulationValueData, uint32_t aPopulationValueSize, void* aInputPopulationStepData, void* aOutputPopulationStepData, uint32_t aPopulationStepSize, void* aInputPopulationFitnessData, void* aOutputPopulationFitnessData, uint32_t aPopulationFitnessSize)
	{
		//Write population input and output values//
		commandQueue_.enqueueWriteBuffer(storageBuffers_[inputPopulationValueBuffer], CL_TRUE, 0, aPopulationValueSize, aInputPopulationValueData);
		commandQueue_.enqueueWriteBuffer(storageBuffers_[outputPopulationValueBuffer], CL_TRUE, 0, aPopulationValueSize, aOutputPopulationValueData);

		//Write population input and output step size//
		commandQueue_.enqueueWriteBuffer(storageBuffers_[inputPopulationStepBuffer], CL_TRUE, 0, aPopulationStepSize, aInputPopulationStepData);
		commandQueue_.enqueueWriteBuffer(storageBuffers_[outputPopulationStepBuffer], CL_TRUE, 0, aPopulationStepSize, aOutputPopulationStepData);

		//Write population input and output fitness//
		commandQueue_.enqueueWriteBuffer(storageBuffers_[inputPopulationFitnessBuffer], CL_TRUE, 0, aPopulationFitnessSize, aInputPopulationFitnessData);
		commandQueue_.enqueueWriteBuffer(storageBuffers_[outputPopulationFitnessBuffer], CL_TRUE, 0, aPopulationFitnessSize, aOutputPopulationFitnessData);
	}
	void readPopulationData(void* aInputPopulationValueData, void* aOutputPopulationValueData, uint32_t aPopulationValueSize, void* aInputPopulationStepData, void* aOutputPopulationStepData, uint32_t aPopulationStepSize, void* aInputPopulationFitnessData, void* aOutputPopulationFitnessData, uint32_t aPopulationFitnessSize)
	{
		//Read population input and output values//
		commandQueue_.enqueueReadBuffer(storageBuffers_[inputPopulationValueBuffer], CL_TRUE, 0, aPopulationValueSize, aInputPopulationValueData);
		commandQueue_.enqueueReadBuffer(storageBuffers_[outputPopulationValueBuffer], CL_TRUE, 0, aPopulationValueSize, aOutputPopulationValueData);

		//Read population input and output step size//
		commandQueue_.enqueueReadBuffer(storageBuffers_[inputPopulationStepBuffer], CL_TRUE, 0, aPopulationStepSize, aInputPopulationStepData);
		commandQueue_.enqueueReadBuffer(storageBuffers_[outputPopulationStepBuffer], CL_TRUE, 0, aPopulationStepSize, aOutputPopulationStepData);

		//Read population input and output fitness//
		commandQueue_.enqueueReadBuffer(storageBuffers_[inputPopulationFitnessBuffer], CL_TRUE, 0, aPopulationFitnessSize, aInputPopulationFitnessData);
		commandQueue_.enqueueReadBuffer(storageBuffers_[outputPopulationFitnessBuffer], CL_TRUE, 0, aPopulationFitnessSize, aOutputPopulationFitnessData);
	}

	void writeSynthesizerData(void* aOutputAudioBuffer, uint32_t aOutputAudioSize, void* aInputFFTDataBuffer, void* aInputFFTTargetBuffer, uint32_t aInputFFTSize)
	{
		//Write audio data buffer//
		commandQueue_.enqueueWriteBuffer(storageBuffers_[outputAudioBuffer], CL_TRUE, 0, aOutputAudioSize, aOutputAudioBuffer);

		//Write FFT buffers//
		commandQueue_.enqueueWriteBuffer(storageBuffers_[inputFFTDataBuffer], CL_TRUE, 0, aInputFFTSize, aInputFFTDataBuffer);
		commandQueue_.enqueueWriteBuffer(storageBuffers_[inputFFTTargetBuffer], CL_TRUE, 0, aInputFFTSize/2, aInputFFTTargetBuffer);
	}
	void readSynthesizerData(void* aOutputAudioBuffer, uint32_t aOutputAudioSize, void* aInputFFTDataBuffer, void* aInputFFTTargetBuffer, uint32_t aInputFFTSize)
	{
		//Read audio data buffer//
		commandQueue_.enqueueReadBuffer(storageBuffers_[outputAudioBuffer], CL_TRUE, 0, aOutputAudioSize, aOutputAudioBuffer);

		//Read FFT buffers//
		commandQueue_.enqueueReadBuffer(storageBuffers_[inputFFTDataBuffer], CL_TRUE, 0, aInputFFTSize, aInputFFTDataBuffer);
		commandQueue_.enqueueReadBuffer(storageBuffers_[inputFFTTargetBuffer], CL_TRUE, 0, aInputFFTSize/2, aInputFFTTargetBuffer);
	}

	void executeMutate()
	{
		commandQueue_.enqueueNDRangeKernel(kernels_[mutatePopulation], cl::NullRange, globalSize_, workgroupSize, NULL);
		commandQueue_.finish();
	}
	void executeFitness()
	{
		commandQueue_.enqueueNDRangeKernel(kernels_[fitnessPopulation], cl::NullRange, globalSize_, workgroupSize, NULL);
		commandQueue_.finish();
	}
	void executeSynthesise()
	{
		commandQueue_.enqueueNDRangeKernel(kernels_[synthesisePopulation], cl::NullRange, globalSize_, workgroupSize, NULL);
		commandQueue_.finish();
	}

	void executeGeneration()
	{
		//Iterate and execute through kernels//
		for (auto iter = kernels_.begin() + recombinePopulation; iter != kernels_.end(); ++iter)
		{
			uint32_t idx = std::distance(kernels_.begin(), iter);
			if (idx == openCLFFT)
			{
				std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
				executeOpenCLFFT();
				auto end = std::chrono::steady_clock::now();
				auto diff = end - start;
				kernelExecuteTime_[idx] += diff;
			}
			//else if (idx == copyPopulation)
			//{
			//	std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
			//
			//	commandQueue_.enqueueReadBuffer(storageBuffers_[rotationIndexBuffer], CL_TRUE, 0, sizeof(uint32_t), &rotationIndex_);
			//	rotationIndex_ = rotationIndex_ == 0 ? 1 : 0;
			//	commandQueue_.enqueueWriteBuffer(storageBuffers_[rotationIndexBuffer], CL_TRUE, 0, sizeof(uint32_t), &rotationIndex_);
			//
			//	auto end = std::chrono::steady_clock::now();
			//	auto diff = end - start;
			//	kernelExecuteTime_[idx] += diff;
			//}
			else
			{
				cl::Kernel currentKernel = *iter;
				std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

				commandQueue_.enqueueNDRangeKernel(currentKernel, cl::NullRange, globalSize_, workgroupSize, NULL);
				//clFinish(commandQueue_());
				commandQueue_.finish();

				auto end = std::chrono::steady_clock::now();
				auto diff = end - start;
				kernelExecuteTime_[idx] += diff;
			}
		}
		//std::for_each(kernels_.begin(), kernels_.end(), &commandQueue_.enqueueNDRangeKernel);
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
			std::cout << "Time to complete kernel " << i << ": " << executeTime.count() << "\n";
		}
	}
	void executeOpenCLFFT()
	{
		//Execute the baked clFFT plan//
		errorStatus_ = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &commandQueue_(), 0, NULL, NULL, &storageBuffers_[outputAudioBuffer](), &storageBuffers_[inputFFTDataBuffer](), NULL);
		commandQueue_.finish();
	}

	void setTargetAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{
		//Calculate and load fft data for target audio//
		targetAudioLength = aTargetAudioLength;
		objective.calculateFFT(aTargetAudio, targetFFT_);
		commandQueue_.enqueueWriteBuffer(storageBuffers_[inputFFTTargetBuffer], CL_TRUE, 0, objective.fftHalfSize*sizeof(float), targetFFT_);
		commandQueue_.finish();
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
			initPopulationCL();

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
		commandQueue_.enqueueReadBuffer(storageBuffers_[inputPopulationValueBuffer], CL_TRUE, 0, tempSize, tempData);
		printf("Best parameters found:\n Fc = %f\n I = %f\n Fm = %f\n A = %f\n\n\n", tempData[0] * objective.paramMaxs[0], tempData[1] * objective.paramMaxs[1], tempData[2] * objective.paramMaxs[2], tempData[3] * objective.paramMaxs[3]);

		delete(tempData);
	}

	//Static Functions//
	static void printAvailableDevices()
	{
		cl::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Print all available devices//
		int platform_id = 0;
		std::cout << "Number of Platforms: " << platforms.size() << std::endl << std::endl;
		for (cl::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it)
		{
			cl::Platform platform(*it);

			std::cout << "Platform ID: " << platform_id++ << std::endl;
			std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
			std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

			cl::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

			int device_id = 0;
			for (cl::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2)
			{
				cl::Device device(*it2);

				std::cout << "\tDevice " << device_id++ << ": " << std::endl;
				std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
				std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
				std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
				std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
				std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
				std::cout << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
				std::cout << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
				std::cout << "\t\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
				std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
				std::cout << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;

				//If an AMD platform//
				if (strstr(platform.getInfo<CL_PLATFORM_NAME>().c_str(), "AMD"))
				{
					std::cout << "\tAMD Specific:" << std::endl;
					//std::cout << "\t\tAMD Wavefront size: " << device.getInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD>() << std::endl;
				}
			}
			std::cout << std::endl;
		}
	}
};

#endif