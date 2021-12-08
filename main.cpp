#include <iostream>

//Parsing parameters as json file//
#include <json.hpp>

//Save samples in wav file//
#include "AudioFile.h"

#include "Evolutionary_Strategy_CPU.hpp"
#include "Evolutionary_Strategy_Vulkan.hpp"
#include "Evolutionary_Strategy_OpenCL.hpp"
//#include "Evolutionary_Strategy_CUDA.hpp"

#include "sndfile.h"    //Audio file I/O

using nlohmann::json;

enum class Implementation { None, CPU = 1, OpenCL = 2, Vulkan = 3, CUDA = 4};
enum class InputType { None, Params = 1, Audio = 2 };

static void show_usage(std::string name);
static float* readAudioFile(const char* aPath, uint32_t* aAudioLength);
static void outputAudioFile(const char* aPath, float* aAudioBuffer, uint32_t aAudioLength);

int main(int argc, char* argv[])
{
	try
	{
		// Eg. ./main -audio-in ./input/0.1-0.15-0.4-1.0.wav -ll2 10//
		// const char *opt_audio_in = "-audio-in";
		// const char *opt_params_in = "-params-in";
		// const char *opt_params_out = "-params-out";
		// const char *opt_term_thresh = "-tt";
		// const char *opt_audio_length_log2 = "-ll2";
		// const char *opt_num_parents = "-parents";
		// const char *opt_work_group_size = "-wgsize";
		// const char *opt_print_on = "-no-print";
		// const char *opt_help = "-help";

		//Evolutionary_Strategy_CUDA cudaES();

		//Parse command line arguments//
		if (argc < 2) {
			show_usage(argv[0]);
			return 1;
		}
		std::string jsonPath;
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if ((arg == "-h") || (arg == "--help")) {
				show_usage(argv[0]);
				return 0;
			}
			else if ((arg == "-j") || (arg == "--json"))
				jsonPath = argv[i + 1];
		}

		//Read json file into program object//
		std::ifstream ifs(jsonPath);
		json j = json::parse(ifs);
		//std::cout << j << std::endl;

		//Implementation type//
		Implementation implementation = Implementation::None;
		if (j["type"]["implementation"] == "CPU")
			implementation = Implementation::CPU;
		else if (j["type"]["implementation"] == "OpenCL")
			implementation = Implementation::OpenCL;
		else if (j["type"]["implementation"] == "Vulkan")
			implementation = Implementation::Vulkan;
		else if (j["type"]["implementation"] == "CUDA")
			implementation = Implementation::CUDA;

		//Synth matching input type//
		InputType inputType = InputType::None;
		if (j["type"]["input"] == "params")
			inputType = InputType::Params;
		else if (j["type"]["input"] == "audio")
			inputType = InputType::Audio;

		//General Parameters//
		const bool isDebug = j["general"]["isDebug"];
		const bool isAudio = j["general"]["isAudio"];
		const std::string outputAudioPath = j["general"]["outputAudioPath"];
		const bool isBenchmarking = j["general"]["isBenchmarking"];
		const bool isLog = j["general"]["isLog"];

		//Audio Parameters//
		const uint32_t sampleRate = j["audio"]["sampleRate"];
		const uint32_t audioLengthLog2 = j["audio"]["audioLengthLog2"];
		const uint32_t wavetableSize = j["audio"]["wavetableSize"];

		//Evolutionary Strategy Parameters//
		const uint32_t numGenerations = j["evolutionary"]["numGenerations"];
		const uint32_t numParents = j["evolutionary"]["numParents"];
		const uint32_t numOffspring = j["evolutionary"]["numOffspring"];
		const uint32_t numDimensions = j["evolutionary"]["numDimensions"];
		//const std::vector<float> paramMins;
		//const std::vector<float> paramInputs = std::vector<float>{ {j["type"]["params"][0], j["type"]["params"][1], j["type"]["params"][2], j["type"]["params"][3]} };
		const std::vector<float> paramInputs = j["type"]["params"];
		const std::vector<float> paramMins = j["evolutionary"]["paramMins"];
		const std::vector<float> paramMaxs = j["evolutionary"]["paramMaxs"];
		const uint32_t fitnessThreshold = j["evolutionary"]["fitnessThreshold"];

		Evolutionary_Strategy* es;
		es = new Evolutionary_Strategy();

		if (implementation == Implementation::CPU)
		{
			Evolutionary_Strategy_CPU_Arguments args;
			args.es_args.pop.numParents = numParents;
			args.es_args.pop.numOffspring = numOffspring;
			args.es_args.pop.numDimensions = numDimensions;
			args.es_args.pop.populationLength = numParents + numOffspring;
			args.es_args.pop.populationSize = numParents + numOffspring * sizeof(float);
			args.es_args.numGenerations = numGenerations;
			args.es_args.paramMin = paramMins;
			args.es_args.paramMax = paramMaxs;
			args.es_args.audioLengthLog2 = audioLengthLog2;
		
			es = new Evolutionary_Strategy_CPU(args);
		}
		if (implementation == Implementation::OpenCL)
		{
			Evolutionary_Strategy_OpenCL_Arguments args;
			args.es_args.pop.numParents = numParents;
			args.es_args.pop.numOffspring = numOffspring;
			args.es_args.pop.numDimensions = numDimensions;
			args.es_args.numGenerations = numGenerations;
			args.es_args.pop.populationLength = numParents + numOffspring;
			args.es_args.pop.populationSize = numParents + numOffspring * sizeof(float);
			args.es_args.paramMin = paramMins;
			args.es_args.paramMax = paramMaxs;
			args.es_args.audioLengthLog2 = audioLengthLog2;
			args.workgroupX = j["type"]["OpenCL"]["workgroupSize"];
			args.workgroupY = 1;
			args.workgroupZ = 1;
			args.kernelSourcePath = "kernels/ocl_program.cl";
			args.deviceType = NVIDIA;

			es = new Evolutionary_Strategy_OpenCL(args);

			Evolutionary_Strategy_OpenCL::printAvailableDevices();
		}
		if (implementation == Implementation::Vulkan)
		{
			Evolutionary_Strategy_Vulkan_Arguments args;
			args.es_args.pop.numParents = numParents;
			args.es_args.pop.numOffspring = numOffspring;
			args.es_args.pop.populationLength = numParents + numOffspring;
			args.es_args.pop.populationSize = numParents + numOffspring * sizeof(float);
			args.es_args.pop.numDimensions = numDimensions;
			args.es_args.numGenerations = numGenerations;
			args.es_args.paramMin = paramMins;
			args.es_args.paramMax = paramMaxs;
			args.es_args.audioLengthLog2 = audioLengthLog2;
			args.workgroupX = j["type"]["Vulkan"]["workgroupSize"];
			args.workgroupY = 1;
			args.workgroupZ = 1;
			args.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;

			es = new Evolutionary_Strategy_Vulkan(args);
		}
		//if (implementation == Implementation::CUDA)
		//{
		//	Evolutionary_Strategy_CUDA_Arguments args;
		//	args.es_args.pop.numParents = numParents;
		//	args.es_args.pop.numOffspring = numOffspring;
		//	args.es_args.pop.numDimensions = numDimensions;
		//	args.es_args.pop.populationLength = numParents + numOffspring;
		//	args.es_args.pop.populationSize = numParents + numOffspring * sizeof(float);
		//	args.es_args.numGenerations = numGenerations;
		//	args.es_args.paramMin = paramMins;
		//	args.es_args.paramMax = paramMaxs;
		//	args.es_args.audioLengthLog2 = audioLengthLog2;
		//	args.localWorkspace = dim3(j["type"]["CUDA"]["workgroupSize"], 1, 1);

		//	es = new Evolutionary_Strategy_CUDA(args);

		//	Evolutionary_Strategy_CUDA::printAvailableDevices();
		//}

		uint32_t populationValueSize = (numParents + numOffspring) * numDimensions;
		uint32_t populationStepSize = (numParents + numOffspring ) * numDimensions;
		uint32_t populationFitnessSize = (numParents + numOffspring);

		float* inputPopulationValues = new float[populationValueSize];
		float* inputPopulationSteps = new float[populationStepSize];
		float* inputPopulationFitness = new float[populationFitnessSize];

		float* outputPopulationValues = new float[populationValueSize];
		float* outputPopulationSteps = new float[populationStepSize];
		float* outputPopulationFitness = new float[populationFitnessSize];

		uint32_t audioLength = 1 << audioLengthLog2;
		float* outputAudioData = new float[audioLength*20];
		uint32_t fftLength = audioLength;
		float* fftAudioData = new float[fftLength];
		float* fftTargetData = new float[fftLength];

		uint32_t logGeneratedAudioLength = 11;
		uint32_t targetAudioLength = 1 << logGeneratedAudioLength;
		float* targetAudio = new float[targetAudioLength];
		if (inputType == InputType::Audio)
		{
			std::string audioPath = j["type"]["audio"];
			targetAudio = readAudioFile(audioPath.c_str(), &targetAudioLength);
		}
		//Synthesise and output audio from generated parameters//
		Objective obj = Objective(numParents, 4, paramMins, paramMaxs, logGeneratedAudioLength);
		obj.initWavetable();
		if (inputType == InputType::Params)
		{
			//Simple
			const std::vector<float> params = { paramInputs[0] / paramMaxs[0], paramInputs[1] / paramMaxs[1], paramInputs[2] / paramMaxs[2], paramInputs[3] / paramMaxs[3] };
			obj.synthesiseAudio(params, targetAudio);
			//Triple parallel
			/*const std::vector<float> params = { paramInputs[0] / paramMaxs[0], paramInputs[1] / paramMaxs[1], paramInputs[2] / paramMaxs[2], paramInputs[3] / paramMaxs[3],
			paramInputs[4] / paramMaxs[0], paramInputs[5] / paramMaxs[1], paramInputs[6] / paramMaxs[2], paramInputs[7] / paramMaxs[3] ,
			paramInputs[8] / paramMaxs[0], paramInputs[9] / paramMaxs[1], paramInputs[10] / paramMaxs[2], paramInputs[11] / paramMaxs[3] };
			obj.synthesiseAudioTriple(params, targetAudio);*/
			//Double Series
			/*const std::vector<float> params = { paramInputs[0] / paramMaxs[0], paramInputs[1] / paramMaxs[1], paramInputs[2] / paramMaxs[2], paramInputs[3] / paramMaxs[3],
			paramInputs[4] / paramMaxs[4], paramInputs[5] / paramMaxs[5] };*/
			//obj.synthesiseAudioDoubleSeries(params, targetAudio);
			outputAudioFile("inputGenerated.wav", targetAudio, (targetAudioLength));
		}

		//Start total compute time//
		std::chrono::time_point<std::chrono::steady_clock> start;
		start = std::chrono::steady_clock::now();

		es->parameterMatchAudio(targetAudio, targetAudioLength);

		//Print total compute time//
		auto end = std::chrono::steady_clock::now();
		auto diff = end - start;
		std::cout << "Total time to complete: " << std::chrono::duration<double>(diff).count() << "s" << std::endl;
		std::cout << "Total time to complete: " << std::chrono::duration <double, std::milli>(diff).count() << "ms" << std::endl << std::endl;

		es->readPopulationData(inputPopulationValues, outputPopulationValues, populationValueSize * sizeof(float), inputPopulationSteps, outputPopulationSteps, populationStepSize * sizeof(float), inputPopulationFitness, outputPopulationFitness, populationFitnessSize * sizeof(float));
		//es->readSynthesizerData(outputAudioData, audioLength * sizeof(float)*20, fftAudioData, fftTargetData, fftLength * sizeof(float));
		//((Evolutionary_Strategy_Vulkan*)es)->readTestingData(testingValues, populationValueSize);
		for (int i = 0; i != es->population.populationLength; ++i)
		{
			//std::cout << i << ": " << (testingValues)[i] << std::endl;
			//std::cout << i << ": " << (fftAudioData)[i] << std::endl;
			//std::cout << i << ": " << (fftTargetData)[i] << std::endl;
			//printf("%d: %f\n", i, (inputPopulationFitness)[i]);
			//printf("%d: %f\n", i, (inputPopulationValues)[i]);
			//printf("%d: %f\n", i, (outputAudioData)[i]);
			//std::cout << i << ": " << inputPopulationValues[i + es->population.populationSize * *((Evolutionary_Strategy_Vulkan*)(es))->rotationIndex_] << std::endl;
		}
		//std::cout << "Input value rotation check." << std::endl;
		//std::cout << inputPopulationValues[es->population.populationLength * 4 * 0] << std::endl;
		//std::cout << inputPopulationValues[es->population.populationLength * 4 * 1] << std::endl;
		//outputAudioFile("gpuOutput.wav", outputAudioData, audioLength*20);	//@ToDo - Why not working?
		
		//Simple
		//float params[] = { 500.0f / paramMax[0], 8.0f / paramMax[1], 2500.0f / paramMax[2], 1.0f / paramMax[3] };
		std::vector<float> bestParamsUnscaled = { inputPopulationValues[0], inputPopulationValues[1], inputPopulationValues[2], inputPopulationValues[3] };
		float* audioBuffer = new float[1 << 14];	//Why need this much memory?
		obj.synthesiseAudio(bestParamsUnscaled, audioBuffer);
		//Triple Parallel
		//std::vector<float> bestParamsUnscaled = { inputPopulationValues[0], inputPopulationValues[1], inputPopulationValues[2], inputPopulationValues[3],
		//inputPopulationValues[4], inputPopulationValues[5], inputPopulationValues[6], inputPopulationValues[7] ,
		//inputPopulationValues[8], inputPopulationValues[9], inputPopulationValues[10], inputPopulationValues[11] };
		//std::vector<float> bestParamsScaled = es->objective.scaleParams(bestParamsUnscaled);
		//float* audioBuffer = new float[1 << 14];	//Why need this much memory?
		//obj.synthesiseAudioTriple(bestParamsUnscaled, audioBuffer);
		//Double Series
		//std::vector<float> bestParamsUnscaled = { inputPopulationValues[0], inputPopulationValues[1], inputPopulationValues[2], inputPopulationValues[3],
		//inputPopulationValues[4], inputPopulationValues[5] };
		//std::vector<float> bestParamsScaled = es->objective.scaleParams(bestParamsUnscaled);
		//float* audioBuffer = new float[1 << 14];	//Why need this much memory?
		//obj.synthesiseAudioDoubleSeries(bestParamsUnscaled, audioBuffer);
		outputAudioFile(outputAudioPath.c_str(), audioBuffer, (1<<14));

		printf("Overall best parameters found\n Fitness = %f\n", inputPopulationFitness[0]);
		//for(uint32_t i = 0; i != 1000; ++i)
		//	printf("Fitness %d = %f\n", i, inputPopulationFitness[i]);
		es->printBest();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		char a;
		std::cin >> a;
		system("pause");
		return EXIT_FAILURE;
	}


	//Benchmark 1: OverallExecution

	//Benchmark 2: ProgramStageExecution

	//Benchmark 3: AudioAnalysisChunkSize

	//Benchmark 4: Populationscaling

	//Benchmark 5: Optimizations On/Off

	//Benchmark 6: Discrete vs Integrate

	system("pause");
	return EXIT_SUCCESS;
}

static float* readAudioFile(const char* aPath, uint32_t* aAudioLength)
{
	SNDFILE* in_file;
	SF_INFO info;
	float *buf;

	/* Open the WAV file. */
	info.format = 0;
	in_file = sf_open(aPath, SFM_READ, &info);
	if (in_file == NULL) {
		printf("Failed to open the file.\n");
		sf_close(in_file);
		return NULL;
	}
	/* Print some of the info, and figure out how much data to read. */
	int f = info.frames;
	//int sr = info.samplerate;
	int c = info.channels;
	//printf("frames=%d\n",f);
	//printf("samplerate=%d\n",sr);
	//printf("channels=%d\n",c);
	*aAudioLength = f * c;
	//printf("num_items=%d\n",*audio_length);
	/* Allocate space for the data to be read, then read it. */
	buf = (float *)malloc(*aAudioLength * sizeof(float));
	sf_read_float(in_file, buf, *aAudioLength);
	sf_close(in_file);
	return buf;
}

static void outputAudioFile(const char* aPath, float* aAudioBuffer, uint32_t aAudioLength)
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

static void show_usage(std::string name)
{
	std::cerr << "Usage: " << "GPU_SOUND_MATCH_UWE"
		<< "Options:\n"
		<< "\t-h,--help\t\t\tShow this help message\n"
		<< "\t-j,--json Use json file(Required)\tSpecify the json file path"
		<< "\tExample json file: "
		<< "\t"
		<< "\n{"
		<< "\n"
		<< "\n	\"general\": {"
		<< "\n		\"sampleRate\": 44100,"
		<< "\n		\"bufferSize\" : 512,"
		<< "\n		\"numberBuffers\" : 100,"
		<< "\n		\"isDebug\" : false,"
		<< "\n		\"isAudio\" : true,"
		<< "\n		\"isBenchmarking\" : true,"
		<< "\n		\"isLog\" : false"
		<< "\n		},"
		<< "\n"
		<< "\n		\"implementation\": {"
		<< "\n			\"type\": \"OpenL\","
		<< "\n			\"modelWidth\" : 32,"
		<< "\n			\"modelHeight\" : 32,"
		<< "\n			\"propagationFactor\" : 0.35,"
		<< "\n			\"dampingCoefficient\" : 0.005,"
		<< "\n			\"boundaryGain\" : 0.02,"
		<< "\n			\"listenerPosition\" : [8, 8],"
		<< "\n			\"excitationPosition\" : [16, 16],"
		<< "\n"
		<< "\n			\"OpenCL\" : {"
		<< "\n			\"workGroupDimensions\": [16, 16],"
		<< "\n				\"kernelSource\" : \"Kernels/fdtdGlobal.cl\""
		<< "\n			},"
		<< "\n"
		<< "\n			\"OpenGL\" : {"
		<< "\n				\"fboVsSource\": \"Shaders/fbo_vs.glsl\","
		<< "\n				\"fboFsSource\" : \"Shaders/fbo_fs.glsl\","
		<< "\n				\"renderVsSource\" : \"Shaders/render_vs.glsl\","
		<< "\n				\"renderFsSource\" : \"Shaders/render_fs.glsl\","
		<< "\n				\"isVisualize\" : true"
		<< "\n			},"
		<< "\n"
		<< "\n		}"
		<< "\n"
		<< "\n}"
		<< std::endl;
}