#ifndef EVOLUTIONARY_STRATEGY_VULKAN_HPP
#define EVOLUTIONARY_STRATEGY_VULKAN_HPP

#include <math.h>
#include <array>
#include <random>
#include <chrono>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

//Graphic math types and functions//
#include <glm\glm.hpp>

#include "Evolutionary_Strategy.hpp"

#include "Vulkan_Helper.hpp"

//#define CL_HPP_TARGET_OPENCL_VERSION 200
//#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <clFFT.h>

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct Evolutionary_Strategy_Vulkan_Arguments
{
	//Generic Evolutionary Strategy arguments//
	Evolutionary_Strategy_Arguments es_args;

	//Shader workgroup details//
	uint32_t workgroupX = 32;
	uint32_t workgroupY = 1;
	uint32_t workgroupZ = 1;
	uint32_t workgroupSize = workgroupX * workgroupY * workgroupZ;
	uint32_t numWorkgroupsPerParent;
	VkPhysicalDeviceType deviceType;
};

struct Specialization_Constants
{
	uint32_t workgroupX = 32;
	uint32_t workgroupY = 1;
	uint32_t workgroupZ = 1;
	uint32_t workgroupSize = 32;
	uint32_t numDimensions = 4;

	uint32_t populationCount = 1536;
	uint32_t numWorkgroupsPerParent = 1;	//population->num_parents / cl->work_group_size
	uint32_t chunkSizeFitness = 1;

	uint32_t audioWaveFormSize = 1024;

	uint32_t fftOutSize = 1;
	uint32_t fftHalfSize = 1;
	float fftOneOverSize = 1.0;
	float fftOneOverWindowFactor = 1.0;

	float mPI = 3.14159265358979323846;
	float alpha = 1.4f;
	float oneOverAlpha = 1.f / alpha;
	float rootTwoOverPi = sqrtf(2.f / (float)mPI);
	float betaScale = 1.f / numDimensions; //1.f / (float)population->num_dimensions;
	float beta = sqrtf(betaScale);

	uint32_t chunksPerWorkgroupSynth = 2;
	uint32_t chunkSizeSynth = workgroupSize / chunksPerWorkgroupSynth;
	float OneOverSampleRateTimeTwoPi = 0.00014247573;

	uint32_t populationSize = 1536 * 4;
} specializationData;

class Evolutionary_Strategy_Vulkan : public Evolutionary_Strategy
{
private:
	//Shader workgroup details//
	uint32_t globalSize;
	uint32_t localSize;
	uint32_t workgroupX;
	uint32_t workgroupY;
	uint32_t workgroupZ;
	uint32_t workgroupSize;
	uint32_t numWorkgroupsX;
	uint32_t numWorkgroupsY;
	uint32_t numWorkgroupsZ;

	uint32_t chunks;
	uint32_t chunkSize;
	uint32_t chunkSizeFitness = 1;	//@ToDo - Need this? Or just define in shader/kernel. Only important to GPU implementations so in subclass.

	float* populationAudioDate;
	float* populationFFTData;

	//////////
	//Vulkan//

	//Instance//
	VkInstance instance_;

	//Physical Device//
	VkPhysicalDevice physicalDevice_;
	VkPhysicalDeviceType deviceType_;

	//Logical Device//
	VkDevice logicalDevice_;

	//Compute Queue//
	VkQueue computeQueue_;		//A queue supporting compute operations.
	uint32_t queueFamilyIndex_;

	//Pipeline//
	//FFT comes inbetween applyWindowPopulation & fitnessPopulation//
	static const int numPipelines_ = 9;
	enum computePipelineNames_ { initPopulation = 0, recombinePopulation, mutatePopulation, synthesisePopulation, applyWindowPopulation, VulkanFFT, fitnessPopulation, sortPopulation, rotatePopulation};
	std::vector<std::string> shaderNames_;
	VkPipeline computePipelines_[numPipelines_];
	VkPipelineLayout computePipelineLayouts_[numPipelines_];

	//Command Buffer//
	VkCommandPool commandPoolInit_;
	VkCommandBuffer commandBufferInit_;
	VkQueryPool queryPoolInit_;

	VkCommandPool commandPoolESOne_;
	VkCommandBuffer commandBufferESOne_;
	VkQueryPool queryPoolESOne_[5];

	VkCommandPool commandPoolESTwo_;
	VkCommandBuffer commandBufferESTwo_;
	VkQueryPool queryPoolESTwo_[3];

	float shaderExecuteTime_[numPipelines_];

	VkCommandPool commandPoolFFT_;
	VkCommandBuffer commandBufferFFT_;


	//Descriptor//
	VkDescriptorPool descriptorPool_;
	VkDescriptorSet descriptorSet_;
	VkDescriptorSetLayout descriptorSetLayout_;

	std::vector<VkDescriptorPool> descriptorPools_;
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts_;
	std::vector<VkDescriptorSet> descriptorSets_;

	VkDescriptorPool descriptorPoolFFT_;
	VkDescriptorSet descriptorSetFFT_;
	VkDescriptorSetLayout descriptorSetLayoutFFT_;

	//Constants//
	static const int numSpecializationConstants_ = 23;
	void* specializationConstantData_;
	std::array<VkSpecializationMapEntry, numSpecializationConstants_> specializationConstantEntries_;
	VkSpecializationInfo specializationConstantInfo_;

	//Fences//
	std::vector<VkFence> fences_;

	//Population Buffers//
	static const int numBuffers_ = 13;
	enum storageBufferNames_ { inputPopulationValueBuffer = 0, inputPopulationStepBuffer, inputPopulationFitnessBuffer, outputPopulationValueBuffer, outputPopulationStepBuffer, outputPopulationFitnessBuffer, randomStatesBuffer, paramMinBuffer, paramMaxBuffer, outputAudioBuffer, inputFFTDataBuffer, inputFFTTargetBuffer, rotationIndexBuffer };
	std::array<VkBuffer, numBuffers_> storageBuffers_;
	std::array<VkDeviceMemory, numBuffers_> storageBuffersMemory_;
	std::array<uint32_t, numBuffers_> storageBufferSizes_;

	//Staging Buffer//
	VkQueue bufferQueue;		//A queue for supporting memory buffer operations.
	VkCommandPool commandPoolStaging_;

	uint32_t stagingBufferSrcSize_;
	VkBuffer stagingBufferSrc;
	VkDeviceMemory stagingBufferMemorySrc;
	void* stagingBufferSrcPointer;

	uint32_t stagingBufferDstSize_;
	VkBuffer stagingBufferDst;
	VkDeviceMemory stagingBufferMemoryDst;
	void* stagingBufferDstPointer;

	//Validation & Debug Variables//
	VkDebugReportCallbackEXT debugReportCallback_;
	std::vector<const char *> enabledValidationLayers;
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
		VkDebugReportFlagsEXT                       flags,
		VkDebugReportObjectTypeEXT                  objectType,
		uint64_t                                    object,
		size_t                                      location,
		int32_t                                     messageCode,
		const char*                                 pLayerPrefix,
		const char*                                 pMessage,
		void*                                       pUserData)
	{

		printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

		return VK_FALSE;
	}
	void initConstantsVK()
	{
		//Setup specialization constant entries//
		specializationConstantEntries_[0].constantID = 1;
		specializationConstantEntries_[0].size = sizeof(specializationData.workgroupX);
		specializationConstantEntries_[0].offset = 0;

		specializationConstantEntries_[1].constantID = 2;
		specializationConstantEntries_[1].size = sizeof(specializationData.workgroupY);
		specializationConstantEntries_[1].offset = offsetof(Specialization_Constants, workgroupY);

		specializationConstantEntries_[2].constantID = 3;
		specializationConstantEntries_[2].size = sizeof(specializationData.workgroupZ);
		specializationConstantEntries_[2].offset = offsetof(Specialization_Constants, workgroupZ);

		specializationConstantEntries_[3].constantID = 4;
		specializationConstantEntries_[3].size = sizeof(specializationData.workgroupSize);
		specializationConstantEntries_[3].offset = offsetof(Specialization_Constants, workgroupSize);

		specializationConstantEntries_[4].constantID = 5;
		specializationConstantEntries_[4].size = sizeof(specializationData.numDimensions);
		specializationConstantEntries_[4].offset = offsetof(Specialization_Constants, numDimensions);

		specializationConstantEntries_[5].constantID = 6;
		specializationConstantEntries_[5].size = sizeof(specializationData.populationCount);
		specializationConstantEntries_[5].offset = offsetof(Specialization_Constants, populationCount);

		specializationConstantEntries_[6].constantID = 7;
		specializationConstantEntries_[6].size = sizeof(specializationData.numWorkgroupsPerParent);
		specializationConstantEntries_[6].offset = offsetof(Specialization_Constants, numWorkgroupsPerParent);

		specializationConstantEntries_[7].constantID = 8;
		specializationConstantEntries_[7].size = sizeof(specializationData.chunkSizeFitness);
		specializationConstantEntries_[7].offset = offsetof(Specialization_Constants, chunkSizeFitness);

		specializationConstantEntries_[8].constantID = 9;
		specializationConstantEntries_[8].size = sizeof(specializationData.audioWaveFormSize);
		specializationConstantEntries_[8].offset = offsetof(Specialization_Constants, audioWaveFormSize);

		specializationConstantEntries_[9].constantID = 10;
		specializationConstantEntries_[9].size = sizeof(specializationData.fftOutSize);
		specializationConstantEntries_[9].offset = offsetof(Specialization_Constants, fftOutSize);

		specializationConstantEntries_[10].constantID = 11;
		specializationConstantEntries_[10].size = sizeof(specializationData.fftHalfSize);
		specializationConstantEntries_[10].offset = offsetof(Specialization_Constants, fftHalfSize);

		specializationConstantEntries_[11].constantID = 12;
		specializationConstantEntries_[11].size = sizeof(specializationData.fftOneOverSize);
		specializationConstantEntries_[11].offset = offsetof(Specialization_Constants, fftOneOverSize);

		specializationConstantEntries_[12].constantID = 13;
		specializationConstantEntries_[12].size = sizeof(specializationData.fftOneOverWindowFactor);
		specializationConstantEntries_[12].offset = offsetof(Specialization_Constants, fftOneOverWindowFactor);

		specializationConstantEntries_[13].constantID = 14;
		specializationConstantEntries_[13].size = sizeof(specializationData.mPI);
		specializationConstantEntries_[13].offset = offsetof(Specialization_Constants, mPI);

		specializationConstantEntries_[14].constantID = 15;
		specializationConstantEntries_[14].size = sizeof(specializationData.alpha);
		specializationConstantEntries_[14].offset = offsetof(Specialization_Constants, alpha);

		specializationConstantEntries_[15].constantID = 16;
		specializationConstantEntries_[15].size = sizeof(specializationData.oneOverAlpha);
		specializationConstantEntries_[15].offset = offsetof(Specialization_Constants, oneOverAlpha);

		specializationConstantEntries_[16].constantID = 17;
		specializationConstantEntries_[16].size = sizeof(specializationData.rootTwoOverPi);
		specializationConstantEntries_[16].offset = offsetof(Specialization_Constants, rootTwoOverPi);

		specializationConstantEntries_[17].constantID = 18;
		specializationConstantEntries_[17].size = sizeof(specializationData.betaScale);
		specializationConstantEntries_[17].offset = offsetof(Specialization_Constants, betaScale);

		specializationConstantEntries_[18].constantID = 19;
		specializationConstantEntries_[18].size = sizeof(specializationData.beta);
		specializationConstantEntries_[18].offset = offsetof(Specialization_Constants, beta);

		specializationConstantEntries_[19].constantID = 20;
		specializationConstantEntries_[19].size = sizeof(specializationData.chunksPerWorkgroupSynth);
		specializationConstantEntries_[19].offset = offsetof(Specialization_Constants, chunksPerWorkgroupSynth);

		specializationConstantEntries_[20].constantID = 21;
		specializationConstantEntries_[20].size = sizeof(specializationData.chunkSizeSynth);
		specializationConstantEntries_[20].offset = offsetof(Specialization_Constants, chunkSizeSynth);

		specializationConstantEntries_[21].constantID = 22;
		specializationConstantEntries_[21].size = sizeof(specializationData.OneOverSampleRateTimeTwoPi);
		specializationConstantEntries_[21].offset = offsetof(Specialization_Constants, OneOverSampleRateTimeTwoPi);

		specializationConstantEntries_[22].constantID = 23;
		specializationConstantEntries_[22].size = sizeof(specializationData.populationSize);
		specializationConstantEntries_[22].offset = offsetof(Specialization_Constants, populationSize);

		//Setup specialization constant data//
		specializationData.workgroupX = workgroupX;
		specializationData.workgroupY = workgroupY;
		specializationData.workgroupZ = workgroupZ;
		specializationData.workgroupSize = workgroupSize;
		specializationData.numDimensions = population.numDimensions;
		specializationData.populationCount = population.populationLength;
		specializationData.numWorkgroupsPerParent = population.numParents / workgroupSize;
		specializationData.chunkSizeFitness = workgroupSize / 2;
		specializationData.audioWaveFormSize = objective.audioLength;
		specializationData.fftOutSize = objective.fftOutSize;
		specializationData.fftHalfSize = objective.fftHalfSize;
		specializationData.fftOneOverSize = objective.fftOneOverSize;
		specializationData.fftOneOverWindowFactor = objective.fftOneOverWindowFactor;

		specializationData.mPI = mPI;
		specializationData.alpha = alpha;
		specializationData.oneOverAlpha = oneOverAlpha;
		specializationData.rootTwoOverPi = rootTwoOverPi;
		specializationData.betaScale = betaScale;
		specializationData.beta = beta;

		specializationData.chunksPerWorkgroupSynth = 2;
		specializationData.chunkSizeSynth = workgroupSize / specializationData.chunksPerWorkgroupSynth;
		specializationData.OneOverSampleRateTimeTwoPi = 0.00014247573;

		specializationData.populationSize = population.populationLength * population.numDimensions;
	}
	//@ToDo - Use local memory buffers correctly//
	void initBuffersVK()
	{
		//Creating each buffer at time//
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[inputPopulationValueBuffer], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[inputPopulationValueBuffer], storageBuffersMemory_[inputPopulationValueBuffer]);
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[inputPopulationStepBuffer], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[inputPopulationStepBuffer], storageBuffersMemory_[inputPopulationStepBuffer]);
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[inputPopulationFitnessBuffer], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[inputPopulationFitnessBuffer], storageBuffersMemory_[inputPopulationFitnessBuffer]);
		
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[outputPopulationValueBuffer], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[outputPopulationValueBuffer], storageBuffersMemory_[outputPopulationValueBuffer]);
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[outputPopulationStepBuffer], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[outputPopulationStepBuffer], storageBuffersMemory_[outputPopulationStepBuffer]);
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[outputPopulationFitnessBuffer], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[outputPopulationFitnessBuffer], storageBuffersMemory_[outputPopulationFitnessBuffer]);
		
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[outputAudioBuffer], VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffers_[outputAudioBuffer], storageBuffersMemory_[outputAudioBuffer]);

		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[randomStatesBuffer], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[randomStatesBuffer], storageBuffersMemory_[randomStatesBuffer]);
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[paramMinBuffer], VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[paramMinBuffer], storageBuffersMemory_[paramMinBuffer]);
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[paramMaxBuffer], VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[paramMaxBuffer], storageBuffersMemory_[paramMaxBuffer]);
		
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[inputFFTDataBuffer], VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[inputFFTDataBuffer], storageBuffersMemory_[inputFFTDataBuffer]);
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[inputFFTTargetBuffer], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[inputFFTTargetBuffer], storageBuffersMemory_[inputFFTTargetBuffer]);
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, storageBufferSizes_[rotationIndexBuffer], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, storageBuffers_[rotationIndexBuffer], storageBuffersMemory_[rotationIndexBuffer]);

		//Create Staging Buffer//
		stagingBufferSrcSize_ = storageBufferSizes_[inputFFTDataBuffer];
		stagingBufferDstSize_ = storageBufferSizes_[inputFFTDataBuffer];

		VKHelper::createBuffer(physicalDevice_, logicalDevice_, stagingBufferSrcSize_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT
			, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSrc, stagingBufferMemorySrc);
		VKHelper::createBuffer(physicalDevice_, logicalDevice_, stagingBufferDstSize_, VK_BUFFER_USAGE_TRANSFER_DST_BIT
			, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferDst, stagingBufferMemoryDst);
	}
	void initRandomStateBuffer()
	{
		//Initialize random numbers in CPU buffer//
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		//std::uniform_int_distribution<int> distribution(0, 2147483647);
		std::uniform_int_distribution<int> distribution(0, 32767);
		//std::uniform_real_distribution<float> distribution(0.0, 1.0);

		glm::uvec2* rand_state = new glm::uvec2[population.populationLength];
		for (int i = 0; i < population.populationLength; ++i)
		{
			rand_state[i].x = distribution(generator);
			rand_state[i].y = distribution(generator);
		}

		void* data;
		uint32_t cpySize = population.populationLength * sizeof(glm::uvec2);
		vkMapMemory(logicalDevice_, storageBuffersMemory_[randomStatesBuffer], 0, cpySize, 0, &data);
			memcpy(data, rand_state, static_cast<size_t>(cpySize));
		vkUnmapMemory(logicalDevice_, storageBuffersMemory_[randomStatesBuffer]);
	}

	void createInstance()
	{
		std::vector<const char *> enabledExtensions;

		/*
		By enabling validation layers, Vulkan will emit warnings if the API
		is used incorrectly. We shall enable the layer VK_LAYER_LUNARG_standard_validation,
		which is basically a collection of several useful validation layers.
		*/
		if (enableValidationLayers)
		{
			//Get all supported layers with vkEnumerateInstanceLayerProperties//
			uint32_t layerCount;
			vkEnumerateInstanceLayerProperties(&layerCount, NULL);
			std::vector<VkLayerProperties> layerProperties(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

			//Check if VK_LAYER_LUNARG_standard_validation is among supported layers//
			bool foundLayer = false;
			if (std::find_if(layerProperties.begin(), layerProperties.end(), [](const VkLayerProperties& m) -> bool { return strcmp(m.layerName, "VK_LAYER_LUNARG_standard_validation"); }) != layerProperties.end())
				foundLayer = true;

			if (!foundLayer) {
				throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
			}
			enabledValidationLayers.push_back("VK_LAYER_LUNARG_standard_validation");

			/*
				We need to enable an extension named VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
				in order to be able to print the warnings emitted by the validation layer.
				Check if the extension is among the supported extensions.
			*/
			uint32_t extensionCount;

			vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, NULL);
			std::vector<VkExtensionProperties> extensionProperties(extensionCount);
			vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, extensionProperties.data());

			bool foundExtension = false;
			if (std::find_if(extensionProperties.begin(), extensionProperties.end(), [](const VkExtensionProperties& m) -> bool { return strcmp(m.extensionName, VK_EXT_DEBUG_REPORT_EXTENSION_NAME); }) != extensionProperties.end())
				foundExtension = true;

			if (!foundExtension) {
				throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
			}
			enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}

		//Create Vulkan instance//
		VkApplicationInfo applicationInfo = {};
		applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		applicationInfo.pApplicationName = "Hello world app";
		applicationInfo.applicationVersion = 0;
		applicationInfo.pEngineName = "VkSoundMatch";
		applicationInfo.engineVersion = 0;
		applicationInfo.apiVersion = VK_MAKE_VERSION(1,2,0);

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.flags = 0;
		createInfo.pApplicationInfo = &applicationInfo;

		// Give our desired layers and extensions to vulkan.
		createInfo.enabledLayerCount = enabledValidationLayers.size();
		createInfo.ppEnabledLayerNames = enabledValidationLayers.data();
		createInfo.enabledExtensionCount = enabledExtensions.size();
		createInfo.ppEnabledExtensionNames = enabledExtensions.data();

		/*
		Actually create the instance.
		Having created the instance, we can actually start using vulkan.
		*/
		VK_CHECK_RESULT(vkCreateInstance(&createInfo, NULL, &instance_));

		/*
		Register a callback function for the extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME, so that warnings emitted from the validation
		layer are actually printed.
		*/
		if (enableValidationLayers) {
			VkDebugReportCallbackCreateInfoEXT createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
			createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
			createInfo.pfnCallback = &debugReportCallbackFn;

			// We have to explicitly load this function.
			auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance_, "vkCreateDebugReportCallbackEXT");
			if (vkCreateDebugReportCallbackEXT == nullptr) {
				throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
			}

			// Create and register callback.
			VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance_, &createInfo, NULL, &debugReportCallback_));
		}

	}

	//@ToDo - Correctly find the AMD GPU, not use intel one.//
	void findPhysicalDevice()
	{
		//Collect physical devices available to VUlkan//
		uint32_t deviceCount;
		vkEnumeratePhysicalDevices(instance_, &deviceCount, NULL);
		if (deviceCount == 0) {
			throw std::runtime_error("could not find a device with vulkan support");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

		//Iterate through and choose appropriate device//
		for (VkPhysicalDevice device : devices) {
			if (true) { // As above stated, we do no feature checks, so just accept.
				VkPhysicalDeviceProperties deviceProperties;
				VkPhysicalDeviceFeatures deviceFeatures;
				vkGetPhysicalDeviceProperties(device, &deviceProperties);
				vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
				printf("Device found: %s\n", deviceProperties.deviceName);
				printf("Device maxDescriptorSetStorageBuffers: %d\n", deviceProperties.limits.maxDescriptorSetStorageBuffers);
		
				if(deviceProperties.deviceType == deviceType_)
				{
					physicalDevice_ = device;
					break;
				}
			}
		}
	}

	// Returns the index of a queue family that supports compute operations. 
	uint32_t getComputeQueueFamilyIndex()
	{
		uint32_t queueFamilyCount;

		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, NULL);

		// Retrieve all queue families.
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, queueFamilies.data());

		// Now find a family that supports compute.
		uint32_t i = 0;
		for (; i < queueFamilies.size(); ++i) {
			VkQueueFamilyProperties props = queueFamilies[i];

			if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
				// found a queue with compute. We're done!
				break;
			}
		}

		if (i == queueFamilies.size()) {
			throw std::runtime_error("could not find a queue family that supports operations");
		}

		return i;
	}

	void createDevice()
	{
		/*
		We create the logical device in this function.
		*/

		/*
		When creating the device, we also specify what queues it has.
		*/
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueFamilyIndex_ = getComputeQueueFamilyIndex(); // find queue family with compute capability.
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex_;
		queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
		float queuePriorities = 1.0;  // we only have one queue, so this is not that imporant. 
		queueCreateInfo.pQueuePriorities = &queuePriorities;

		/*
		Now we create the logical device. The logical device allows us to interact with the physical
		device.
		*/
		VkDeviceCreateInfo deviceCreateInfo = {};

		// Specify any desired device features here. We do not need any for this application, though.
		VkPhysicalDeviceFeatures deviceFeatures = {};

		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.enabledLayerCount = enabledValidationLayers.size();  // need to specify validation layers here as well.
		deviceCreateInfo.ppEnabledLayerNames = enabledValidationLayers.data();
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo; // when creating the logical device, we also specify what queues it has.
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

		VK_CHECK_RESULT(vkCreateDevice(physicalDevice_, &deviceCreateInfo, NULL, &logicalDevice_)); // create logical device.

		// Get a handle to the only member of the queue family.
		vkGetDeviceQueue(logicalDevice_, queueFamilyIndex_, 0, &computeQueue_);
	}

	// find memory type with desired properties.
	uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memoryProperties;

		vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties);

		/*
		How does this search work?
		See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
		*/
		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
			if ((memoryTypeBits & (1 << i)) &&
				((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
				return i;
		}
		return -1;
	}

	void createDescriptorSetLayout()
	{
		/*
		Here we specify a descriptor set layout. This allows us to bind our descriptors to
		resources in the shader.

		*/

		/*
		Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
		0. This binds to

		  layout(std140, binding = 0) buffer buf

		in the compute shader.
		*/
		VkDescriptorSetLayoutBinding populationValueLayoutBinding = {};
		populationValueLayoutBinding.binding = 0; // binding = 0
		populationValueLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		populationValueLayoutBinding.descriptorCount = 1;
		populationValueLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding populationStepLayoutBinding = {};
		populationStepLayoutBinding.binding = 1; // binding = 0
		populationStepLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		populationStepLayoutBinding.descriptorCount = 1;
		populationStepLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding populationFitnessLayoutBinding = {};
		populationFitnessLayoutBinding.binding = 2; // binding = 0
		populationFitnessLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		populationFitnessLayoutBinding.descriptorCount = 1;
		populationFitnessLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding populationValueTempLayoutBinding = {};
		populationValueTempLayoutBinding.binding = 3; // binding = 0
		populationValueTempLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		populationValueTempLayoutBinding.descriptorCount = 1;
		populationValueTempLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding populationStepTempLayoutBinding = {};
		populationStepTempLayoutBinding.binding = 4; // binding = 0
		populationStepTempLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		populationStepTempLayoutBinding.descriptorCount = 1;
		populationStepTempLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding populationFitnessTempLayoutBinding = {};
		populationFitnessTempLayoutBinding.binding = 5; // binding = 0
		populationFitnessTempLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		populationFitnessTempLayoutBinding.descriptorCount = 1;
		populationFitnessTempLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding randStateLayoutBinding = {};
		randStateLayoutBinding.binding = 6; // binding = 0
		randStateLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		randStateLayoutBinding.descriptorCount = 1;
		randStateLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding paramMinLayoutBinding = {};
		paramMinLayoutBinding.binding = 7; // binding = 0
		paramMinLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		paramMinLayoutBinding.descriptorCount = 1;
		paramMinLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding paramMaxLayoutBinding = {};
		paramMaxLayoutBinding.binding = 8; // binding = 0
		paramMaxLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		paramMaxLayoutBinding.descriptorCount = 1;
		paramMaxLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding audioWaveLayoutBinding = {};
		audioWaveLayoutBinding.binding = 9; // binding = 0
		audioWaveLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		audioWaveLayoutBinding.descriptorCount = 1;
		audioWaveLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding FFTOutputLayoutBinding = {};
		FFTOutputLayoutBinding.binding = 10; // binding = 0
		FFTOutputLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		FFTOutputLayoutBinding.descriptorCount = 1;
		FFTOutputLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding FFTTargetLayoutBinding = {};
		FFTTargetLayoutBinding.binding = 11; // binding = 0
		FFTTargetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		FFTTargetLayoutBinding.descriptorCount = 1;
		FFTTargetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding rotationIndexLayoutBinding = {};
		rotationIndexLayoutBinding.binding = 12; // binding = 0
		rotationIndexLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		rotationIndexLayoutBinding.descriptorCount = 1;
		rotationIndexLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		//Layout create information from binding layouts//
		std::array<VkDescriptorSetLayoutBinding, numBuffers_> bindings = { populationValueLayoutBinding, populationStepLayoutBinding, populationFitnessLayoutBinding, populationValueTempLayoutBinding, populationStepTempLayoutBinding, populationFitnessTempLayoutBinding, randStateLayoutBinding, paramMinLayoutBinding, paramMaxLayoutBinding, audioWaveLayoutBinding, FFTOutputLayoutBinding, FFTTargetLayoutBinding, rotationIndexLayoutBinding };

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
		descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(bindings.size()); // only a single binding in this descriptor set layout. 
		descriptorSetLayoutCreateInfo.pBindings = bindings.data();

		//Create the descriptor set layout//
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logicalDevice_, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout_));
	}

	void createDescriptorSet()
	{
		/*
		So we will allocate a descriptor set here.
		But we need to first create a descriptor pool to do that.
		*/

		/*
		Our descriptor pool can only allocate a single storage buffer.
		*/
		std::array<VkDescriptorPoolSize, numBuffers_> poolSizes = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[0].descriptorCount = 1;
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[1].descriptorCount = 1;
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[2].descriptorCount = 1;
		poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[3].descriptorCount = 1;
		poolSizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[4].descriptorCount = 1;
		poolSizes[5].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[5].descriptorCount = 1;
		poolSizes[6].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[6].descriptorCount = 1;
		poolSizes[7].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[7].descriptorCount = 1;
		poolSizes[8].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[8].descriptorCount = 1;
		poolSizes[9].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[9].descriptorCount = 1;
		poolSizes[10].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[10].descriptorCount = 1;
		poolSizes[11].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[11].descriptorCount = 1;
		poolSizes[12].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[12].descriptorCount = 1;

		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
		descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolCreateInfo.maxSets = 1; // we only need to allocate one descriptor set from the pool.
		descriptorPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		descriptorPoolCreateInfo.pPoolSizes = poolSizes.data();

		// create descriptor pool.
		VK_CHECK_RESULT(vkCreateDescriptorPool(logicalDevice_, &descriptorPoolCreateInfo, NULL, &descriptorPool_));

		/*
		With the pool allocated, we can now allocate the descriptor set.
		*/
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
		descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		descriptorSetAllocateInfo.descriptorPool = descriptorPool_; // pool to allocate from.
		descriptorSetAllocateInfo.descriptorSetCount = 1; // allocate a single descriptor set.
		descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout_;

		// allocate descriptor set.
		VK_CHECK_RESULT(vkAllocateDescriptorSets(logicalDevice_, &descriptorSetAllocateInfo, &descriptorSet_));

		/*
		Next, we need to connect our actual storage buffer with the descrptor.
		We use vkUpdateDescriptorSets() to update the descriptor set.
		*/

		std::array<VkDescriptorBufferInfo, numBuffers_> descriptorBuffersInfo;
		std::array<VkWriteDescriptorSet, numBuffers_> descriptorWrites = {};
		for (uint32_t i = 0; i != numBuffers_; ++i)
		{
			descriptorBuffersInfo[i].buffer = storageBuffers_[i];
			descriptorBuffersInfo[i].offset = 0;
			descriptorBuffersInfo[i].range = storageBufferSizes_[i];

			descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[i].dstSet = descriptorSet_;
			descriptorWrites[i].dstBinding = i;
			descriptorWrites[i].dstArrayElement = 0;
			descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[i].descriptorCount = 1;	//@ToDo - May need higher count.
			descriptorWrites[i].pBufferInfo = &(descriptorBuffersInfo[i]);
		}

		// perform the update of the descriptor set.
		vkUpdateDescriptorSets(logicalDevice_, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, NULL);
	}
	void createComputePipelines()	//create pipelines
	{
		VkSpecializationInfo specializationInfo{};
		specializationInfo.dataSize = sizeof(specializationData);
		specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationConstantEntries_.size());
		specializationInfo.pMapEntries = specializationConstantEntries_.data();
		specializationInfo.pData = &specializationData;

		VkPushConstantRange pushConstantRange[1] = {};
		pushConstantRange[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange[0].offset = 0;
		pushConstantRange[0].size = sizeof(uint32_t);

		for (uint8_t i = 0; i != numPipelines_; ++i)
		{
			/*
			Now let us actually create the compute pipeline.
			A compute pipeline is very simple compared to a graphics pipeline.
			It only consists of a single stage with a compute shader.

			So first we specify the compute shader stage, and it's entry point(main).
			*/
			uint32_t filelength;
			std::string fileName = "shaders/" + shaderNames_[i];
			std::vector<char> code = VKHelper::readFile(fileName);
			VkShaderModule shaderModule = VKHelper::createShaderModule(logicalDevice_, code);
			VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
			shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderStageCreateInfo.module = shaderModule;
			shaderStageCreateInfo.pName = "main";
			shaderStageCreateInfo.pSpecializationInfo = &specializationInfo;

			/*
			The pipeline layout allows the pipeline to access descriptor sets.
			So we just specify the descriptor set layout we created earlier.

			All pipelines are going to access the same descriptors to start with.
			*/
			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
			pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout_;
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRange;
			VK_CHECK_RESULT(vkCreatePipelineLayout(logicalDevice_, &pipelineLayoutCreateInfo, NULL, &(computePipelineLayouts_[i])));

			VkComputePipelineCreateInfo pipelineCreateInfo = {};
			pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pipelineCreateInfo.stage = shaderStageCreateInfo;
			pipelineCreateInfo.layout = (computePipelineLayouts_[i]);
			VK_CHECK_RESULT(vkCreateComputePipelines(logicalDevice_, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &(computePipelines_[i])));

			vkDestroyShaderModule(logicalDevice_, shaderModule, NULL);

			std::cout << "Created shader: " << fileName.c_str() << std::endl;
		}
	}
	void createCopyCommandPool()
	{
		VkCommandPoolCreateInfo commandPoolCreateInfo = {};
		commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		commandPoolCreateInfo.flags = 0;
		// the queue family of this command pool. All command buffers allocated from this command pool,
		// must be submitted to queues of this family ONLY. 
		commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex_;
		VK_CHECK_RESULT(vkCreateCommandPool(logicalDevice_, &commandPoolCreateInfo, NULL, &commandPoolStaging_));
	}
	void calculateAudioFFT()
	{
		//int counter = 0;
		//for (uint32_t i = 0; i != population.populationSize * objective.audioLength; i += objective.audioLength)
		//{
		//	objective.calculateFFTSpecial(&populationAudioDate[i], &populationFFTData[counter]);
		//	counter += objective.fftHalfSize;
		//}

		executeOpenCLFFT();
	}
	//@ToDo - Right now pick platform. Can extend to pick best available.
	int errorStatus_ = 0;
	cl_uint num_platforms, num_devices;
	cl::Platform platform_;
	cl::Context context_;
	cl::Device device_;
	cl::CommandQueue commandQueue_;
	cl::Program kernelProgram_;
	std::string kernelSourcePath_;
	cl::NDRange globalws_;
	cl::NDRange localws_;
	clfftPlanHandle planHandle;
	void initContextCL(uint8_t aPlatform, uint8_t aDevice)
	{
		//Discover platforms//
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Create contex properties for first platform//
		cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[aPlatform])(), 0 };	//Need to specify platform 3 for dedicated graphics - Harri Laptop.

		//Create context context using platform for GPU device//
		context_ = cl::Context(CL_DEVICE_TYPE_ALL, contextProperties);

		//Get device list from context//
		std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();

		//Create command queue for first device - Profiling enabled//
		commandQueue_ = cl::CommandQueue(context_, devices[aDevice], CL_QUEUE_PROFILING_ENABLE, &errorStatus_);	//Need to specify device 1[0] of platform 3[2] for dedicated graphics - Harri Laptop.
		if (errorStatus_)
			std::cout << "ERROR creating command queue for device. Status code: " << errorStatus_ << std::endl;

		globalws_ = cl::NDRange(globalSize);
		localws_ = cl::NDRange(workgroupX, workgroupY, workgroupZ);

		/* FFT library realted declarations */
		clfftDim dim = CLFFT_1D;
		size_t clLengths[1] = { objective.audioLength };

		/* Setup clFFT. */
		clfftSetupData fftSetup;
		errorStatus_ = clfftInitSetupData(&fftSetup);
		errorStatus_ = clfftSetup(&fftSetup);

		/* Create a default plan for a complex FFT. */
		errorStatus_ = clfftCreateDefaultPlan(&planHandle, context_(), dim, clLengths);

		/* Set plan parameters. */
		errorStatus_ = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
		errorStatus_ = clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
		errorStatus_ = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
		errorStatus_ = clfftSetPlanBatchSize(planHandle, (size_t)population.populationLength);

		size_t in_strides[1] = { 1 };
		size_t out_strides[1] = { 1 };
		size_t in_dist = (size_t)objective.audioLength;
		size_t out_dist = (size_t)objective.audioLength / 2 + 4;

		objective.fftOutSize = out_dist * 2;
		objective.fftHalfSize = 1 << (objective.audioLengthLog2 - 1);
		storageBufferSizes_[inputFFTDataBuffer] = population.populationLength * objective.fftOutSize * sizeof(float);
		storageBufferSizes_[inputFFTTargetBuffer] = objective.fftHalfSize * sizeof(float);	//objective.fftSizeHalf

		clfftSetPlanInStride(planHandle, dim, in_strides);
		clfftSetPlanOutStride(planHandle, dim, out_strides);
		clfftSetPlanDistance(planHandle, in_dist, out_dist);

		/* Bake the plan. */
		errorStatus_ = clfftBakePlan(planHandle, 1, &commandQueue_(), NULL, NULL);
	}
	cl::Buffer inputBuffer;
	cl::Buffer outputBuffer;
	void initBuffersCL()
	{
		inputBuffer = cl::Buffer(context_, CL_MEM_READ_WRITE, storageBufferSizes_[outputAudioBuffer]);
		outputBuffer = cl::Buffer(context_, CL_MEM_READ_WRITE, storageBufferSizes_[inputFFTDataBuffer]);
	}
public:
	uint32_t* rotationIndex_;
	Evolutionary_Strategy_Vulkan(uint32_t aNumGenerations, uint32_t aNumParents, uint32_t aNumOffspring, uint32_t aNumDimensions, const std::vector<float> aParamMin, const std::vector<float> aParamMax, uint32_t aAudioLengthLog2) :
		Evolutionary_Strategy(aNumGenerations, aNumParents, aNumOffspring, aNumDimensions, aParamMin, aParamMax, aAudioLengthLog2),
		shaderNames_({ "initPopulation.spv", "recombinePopulation.spv", "mutatePopulation.spv", "SynthesisePopulation.spv", "applyWindowPopulation.spv", "VulkanFFT.spv", "fitnessPopulation.spv", "sortPopulation.spv", "rotatePopulation.spv" })
	{

	}
	Evolutionary_Strategy_Vulkan(Evolutionary_Strategy_Vulkan_Arguments args) :
		Evolutionary_Strategy(args.es_args.numGenerations, args.es_args.pop.numParents, args.es_args.pop.numOffspring, args.es_args.pop.numDimensions, args.es_args.paramMin, args.es_args.paramMax, args.es_args.audioLengthLog2),
		shaderNames_({ "initPopulation.spv", "recombinePopulation.spv", "mutatePopulation.spv", "SynthesisePopulation.spv", "applyWindowPopulation.spv", "VulkanFFT.spv", "fitnessPopulation.spv", "sortPopulation.spv", "rotatePopulation.spv" }),
		workgroupX(args.workgroupX),
		workgroupY(args.workgroupY),
		workgroupZ(args.workgroupZ),
		workgroupSize(args.workgroupX*args.workgroupY*args.workgroupZ),
		globalSize(population.populationLength),
		numWorkgroupsX(globalSize / workgroupX),
		numWorkgroupsY(1),
		numWorkgroupsZ(1),
		chunkSizeFitness(workgroupSize / 2),
		deviceType_(args.deviceType)
	{
		for (int i = 0; i != randomStatesBuffer; ++i)
			storageBufferSizes_[i] = (population.numParents + population.numOffspring) * population.numDimensions * sizeof(float) * 2;

		storageBufferSizes_[randomStatesBuffer] = (population.numParents + population.numOffspring) * sizeof(glm::vec2);
		storageBufferSizes_[inputPopulationFitnessBuffer] = (population.numParents + population.numOffspring) * sizeof(float) * 2;
		storageBufferSizes_[outputPopulationFitnessBuffer] = (population.numParents + population.numOffspring) * sizeof(float) * 2;
		storageBufferSizes_[paramMinBuffer] = population.numDimensions * sizeof(float);
		storageBufferSizes_[paramMaxBuffer] = population.numDimensions * sizeof(float);
		storageBufferSizes_[outputAudioBuffer] = population.populationLength * objective.audioLength * sizeof(float);
		storageBufferSizes_[rotationIndexBuffer] = sizeof(uint32_t);

		rotationIndex_ = new uint32_t;
		*rotationIndex_ = 0;

		initContextCL(1, 0);
		initBuffersCL();
		initCLFFT();

		populationAudioDate = new float[population.populationLength * objective.audioLength];
		populationFFTData = new float[population.populationLength * objective.fftOutSize];

		createInstance();
		findPhysicalDevice();
		createDevice();
		initBuffersVK();
		createDescriptorSetLayout();
		createDescriptorSet();
		initConstantsVK();
		createComputePipelines();
		createCopyCommandPool();

		std::vector<VkPipelineLayout> pipelineLayoutsTemp = { computePipelineLayouts_[initPopulation] };
		std::vector<VkPipeline> pipelinesTemp = { computePipelines_[initPopulation] };
		createCommandBuffer(commandPoolInit_, commandBufferInit_, pipelineLayoutsTemp, pipelinesTemp, &queryPoolInit_);

		pipelineLayoutsTemp = { computePipelineLayouts_[recombinePopulation], computePipelineLayouts_[mutatePopulation], computePipelineLayouts_[synthesisePopulation], computePipelineLayouts_[applyWindowPopulation] };
		pipelinesTemp = { computePipelines_[recombinePopulation], computePipelines_[mutatePopulation], computePipelines_[synthesisePopulation], computePipelines_[applyWindowPopulation] };
		createCommandBuffer(commandPoolESOne_, commandBufferESOne_, pipelineLayoutsTemp, pipelinesTemp, queryPoolESOne_);

		pipelineLayoutsTemp = { computePipelineLayouts_[fitnessPopulation], computePipelineLayouts_[sortPopulation], computePipelineLayouts_[rotatePopulation] };
		pipelinesTemp = { computePipelines_[fitnessPopulation], computePipelines_[sortPopulation], computePipelines_[rotatePopulation] };
		createCommandBuffer(commandPoolESTwo_, commandBufferESTwo_, pipelineLayoutsTemp, pipelinesTemp, queryPoolESTwo_);

		//createPopulationInitialiseCommandBuffer();
		//createESCommandBufferOne();
		//createESCommandBufferTwo();
		initRandomStateBuffer();
		writeLocalBuffer(storageBuffers_[paramMinBuffer], 4 * sizeof(float), (void*)objective.paramMins.data());
		writeLocalBuffer(storageBuffers_[paramMaxBuffer], 4 * sizeof(float), (void*)objective.paramMaxs.data());
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
		storageBufferSizes_[inputFFTDataBuffer] = population.populationLength * objective.fftOutSize * sizeof(float);
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
		errorStatus_ = clfftSetPlanBatchSize(planHandle, (size_t)population.populationLength);
		clfftSetPlanInStride(planHandle, dim, in_strides);
		clfftSetPlanOutStride(planHandle, dim, out_strides);
		clfftSetPlanDistance(planHandle, in_dist, out_dist);

		//Bake clFFT plan//
		errorStatus_ = clfftBakePlan(planHandle, 1, &commandQueue_(), NULL, NULL);
		if (errorStatus_)
			std::cout << "ERROR creating clFFT plan. Status code: " << errorStatus_ << std::endl;
	}
	void writePopulationData()
	{

	}
	void readTestingData(void* aTestingData, size_t aTestingDataSize)
	{
		copyBuffer(storageBuffers_[inputPopulationValueBuffer], stagingBufferDst, aTestingDataSize);
		VKHelper::readBuffer(logicalDevice_, aTestingDataSize, stagingBufferMemoryDst, aTestingData);
	}
	void readPopulationData(void* aInputPopulationValueData, void* aOutputPopulationValueData, uint32_t aPopulationValueSize, void* aInputPopulationStepData, void* aOutputPopulationStepData, uint32_t aPopulationStepSize, void* aInputPopulationFitnessData, void* aOutputPopulationFitnessData, uint32_t aPopulationFitnessSize)
	{
		VKHelper::readBuffer(logicalDevice_, aPopulationValueSize, storageBuffersMemory_[inputPopulationValueBuffer], aInputPopulationValueData);
		VKHelper::readBuffer(logicalDevice_, aPopulationValueSize, storageBuffersMemory_[outputPopulationValueBuffer], aOutputPopulationValueData);
		VKHelper::readBuffer(logicalDevice_, aPopulationStepSize, storageBuffersMemory_[inputPopulationStepBuffer], aInputPopulationStepData);
		VKHelper::readBuffer(logicalDevice_, aPopulationStepSize, storageBuffersMemory_[outputPopulationStepBuffer], aOutputPopulationStepData);
		VKHelper::readBuffer(logicalDevice_, aPopulationFitnessSize, storageBuffersMemory_[inputPopulationFitnessBuffer], aInputPopulationFitnessData);
		VKHelper::readBuffer(logicalDevice_, aPopulationFitnessSize, storageBuffersMemory_[outputPopulationFitnessBuffer], aOutputPopulationFitnessData);
	}
	void readPopulationDataStaging(void* aInputPopulationValueData, void* aOutputPopulationValueData, uint32_t aPopulationValueSize, void* aInputPopulationStepData, void* aOutputPopulationStepData, uint32_t aPopulationStepSize, void* aInputPopulationFitnessData, void* aOutputPopulationFitnessData, uint32_t aPopulationFitnessSize)
	{
		copyBuffer(storageBuffers_[inputPopulationValueBuffer], stagingBufferDst, aPopulationValueSize);
		VKHelper::readBuffer(logicalDevice_, aPopulationValueSize, stagingBufferMemoryDst, aInputPopulationValueData);
		copyBuffer(storageBuffers_[outputPopulationValueBuffer], stagingBufferDst, aPopulationValueSize);
		VKHelper::readBuffer(logicalDevice_, aPopulationValueSize, stagingBufferMemoryDst, aOutputPopulationValueData);
		copyBuffer(storageBuffers_[inputPopulationStepBuffer], stagingBufferDst, aPopulationValueSize);
		VKHelper::readBuffer(logicalDevice_, aPopulationStepSize, stagingBufferMemoryDst, aInputPopulationStepData);
		copyBuffer(storageBuffers_[outputPopulationStepBuffer], stagingBufferDst, aPopulationValueSize);
		VKHelper::readBuffer(logicalDevice_, aPopulationStepSize, stagingBufferMemoryDst, aOutputPopulationStepData);
		copyBuffer(storageBuffers_[inputPopulationFitnessBuffer], stagingBufferDst, aPopulationValueSize);
		VKHelper::readBuffer(logicalDevice_, aPopulationFitnessSize, stagingBufferMemoryDst, aInputPopulationFitnessData);
		copyBuffer(storageBuffers_[outputPopulationFitnessBuffer], stagingBufferDst, aPopulationValueSize);
		VKHelper::readBuffer(logicalDevice_, aPopulationFitnessSize, stagingBufferMemoryDst, aOutputPopulationFitnessData);
	}

	void writeSynthesizerData(void* aOutputAudioBuffer, uint32_t aOutputAudioSize, void* aInputFFTDataBuffer, void* aInputFFTTargetBuffer, uint32_t aInputFFTSize)
	{

	}
	void readSynthesizerData(void* aOutputAudioBuffer, uint32_t aOutputAudioSize, void* aInputFFTDataBuffer, void* aInputFFTTargetBuffer, uint32_t aInputFFTSize)
	{
		VKHelper::readBuffer(logicalDevice_, aOutputAudioSize, storageBuffersMemory_[outputAudioBuffer], aOutputAudioBuffer);
		VKHelper::readBuffer(logicalDevice_, aInputFFTSize, storageBuffersMemory_[inputFFTDataBuffer], aInputFFTDataBuffer);
		VKHelper::readBuffer(logicalDevice_, aInputFFTSize/2, storageBuffersMemory_[inputFFTTargetBuffer], aInputFFTTargetBuffer);
	}

	void readParamData()
	{

	}
	void writeParamData(void* aParamMinBuffer, void* aParamMaxBuffer)
	{
		VKHelper::writeBuffer(logicalDevice_, population.numDimensions*sizeof(float), storageBuffersMemory_[paramMinBuffer], aParamMinBuffer);
		VKHelper::writeBuffer(logicalDevice_, population.numDimensions*sizeof(float), storageBuffersMemory_[paramMaxBuffer], aParamMaxBuffer);
	}

	void executeMutate()
	{
		//commandQueue_.enqueueNDRangeKernel(kernels_[mutatePopulation], cl::NullRange/*globaloffset*/, globalws_, localws_, NULL);
		//commandQueue_.finish();
	}
	void executeFitness()
	{
		//commandQueue_.enqueueNDRangeKernel(kernels_[fitnessPopulation], cl::NullRange/*globaloffset*/, globalws_, localws_, NULL);
		//commandQueue_.finish();
	}
	void executeSynthesise()
	{
		//commandQueue_.enqueueNDRangeKernel(kernels_[synthesisePopulation], cl::NullRange/*globaloffset*/, globalws_, localws_, NULL);
		//commandQueue_.finish();
	}

	void executeGeneration()
	{
		VKHelper::runCommandBuffer(logicalDevice_, computeQueue_, commandBufferESOne_); 

		//VKHelper::readBuffer(logicalDevice_, storageBufferSizes_[outputAudioBuffer], storageBuffersMemory_[outputAudioBuffer], populationAudioDate);
		//copyBuffer(storageBuffers_[outputAudioBuffer], stagingBufferDst, storageBufferSizes_[outputAudioBuffer]);
		//VKHelper::readBuffer(logicalDevice_, storageBufferSizes_[outputAudioBuffer], stagingBufferMemoryDst, populationAudioDate);
		std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
		readLocalBuffer(storageBuffers_[outputAudioBuffer], storageBufferSizes_[outputAudioBuffer], populationAudioDate);
		

		calculateAudioFFT();	//@ToDo - Work out how to calculate FFT for GPU acceptable format.

		writeLocalBuffer(storageBuffers_[inputFFTDataBuffer], storageBufferSizes_[inputFFTDataBuffer], populationFFTData);
		auto end = std::chrono::steady_clock::now();
		auto diff = end - start;
		shaderExecuteTime_[5] += diff.count() / (float)1e6;
		//VKHelper::writeBuffer(logicalDevice_, storageBufferSizes_[outputAudioBuffer], stagingBufferMemorySrc, populationAudioDate);
		//copyBuffer(stagingBufferSrc, storageBuffers_[outputAudioBuffer], storageBufferSizes_[outputAudioBuffer]);
		//VKHelper::writeBuffer(logicalDevice_, storageBufferSizes_[inputFFTDataBuffer], storageBuffersMemory_[inputFFTDataBuffer], populationFFTData);

		VKHelper::runCommandBuffer(logicalDevice_, computeQueue_, commandBufferESTwo_);
	}
	void executeAllGenerations()
	{
		*rotationIndex_ = 0;
		for (uint32_t i = 0; i != numGenerations; ++i)
		{
			//initRandomStateBuffer();
			executeGeneration();
			//*rotationIndex_ = (*rotationIndex_ == 0 ? 1 : 0);
			//
			//VkCommandBuffer commandBuffer = beginSingleTimeCommands();
			//for(uint32_t i = 0; i != numPipelines_; ++i)
			//	vkCmdPushConstants(commandBuffer, computePipelineLayouts_[i], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), rotationIndex_);
			////VkCmdPushConstants(commandBufferESOne_, computePipelineLayouts_[0], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), rotationIndex_);
			//endSingleTimeCommands(commandBuffer);

			uint64_t timestamp[2];

			vkGetQueryPoolResults(logicalDevice_, queryPoolInit_, 0, 1, sizeof(uint64_t), &(timestamp[0]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			vkGetQueryPoolResults(logicalDevice_, queryPoolInit_, 1, 1, sizeof(uint64_t), &(timestamp[1]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			uint64_t diff = timestamp[1] - timestamp[0];
			shaderExecuteTime_[0] += diff / (float)1e6;
			//printf("Timestamp information [%d]: %f\n", 0, diff / (float)1e6);

			vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[0], 0, 1, sizeof(uint64_t), &(timestamp[0]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[0], 1, 1, sizeof(uint64_t), &(timestamp[1]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			diff = timestamp[1] - timestamp[0];
			shaderExecuteTime_[1] += diff / (float)1e6;
			//printf("Timestamp information [%d]: %f\n", 1, diff / (float)1e6);

			vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[1], 0, 1, sizeof(uint64_t), &(timestamp[0]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[1], 1, 1, sizeof(uint64_t), &(timestamp[1]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			diff = timestamp[1] - timestamp[0];
			shaderExecuteTime_[2] += diff / (float)1e6;
			//printf("Timestamp information [%d]: %f\n", 2, diff / (float)1e6);

			vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[2], 0, 1, sizeof(uint64_t), &(timestamp[0]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[2], 1, 1, sizeof(uint64_t), &(timestamp[1]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			diff = timestamp[1] - timestamp[0];
			shaderExecuteTime_[3] += diff / (float)1e6;
			//printf("Timestamp information [%d]: %f\n", 3, diff / (float)1e6);

			vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[3], 0, 1, sizeof(uint64_t), &(timestamp[0]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[3], 1, 1, sizeof(uint64_t), &(timestamp[1]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			diff = timestamp[1] - timestamp[0];
			shaderExecuteTime_[4] += diff / (float)1e6;
			//printf("Timestamp information [%d]: %f\n", 4, diff / (float)1e6);

			//vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[4], 0, 1, sizeof(uint64_t), &(timestamp[0]), 0, VK_QUERY_RESULT_64_BIT);
			//vkGetQueryPoolResults(logicalDevice_, queryPoolESOne_[4], 1, 1, sizeof(uint64_t), &(timestamp[1]), 0, VK_QUERY_RESULT_64_BIT);
			//diff = timestamp[1] - timestamp[0];
			//shaderExecuteTime_[5] += diff / (float)1e6;

			vkGetQueryPoolResults(logicalDevice_, queryPoolESTwo_[0], 0, 1, sizeof(uint64_t), &(timestamp[0]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			vkGetQueryPoolResults(logicalDevice_, queryPoolESTwo_[0], 1, 1, sizeof(uint64_t), &(timestamp[1]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			diff = timestamp[1] - timestamp[0];
			shaderExecuteTime_[6] += diff / (float)1e6;

			vkGetQueryPoolResults(logicalDevice_, queryPoolESTwo_[1], 0, 1, sizeof(uint64_t), &(timestamp[0]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			vkGetQueryPoolResults(logicalDevice_, queryPoolESTwo_[1], 1, 1, sizeof(uint64_t), &(timestamp[1]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			diff = timestamp[1] - timestamp[0];
			shaderExecuteTime_[7] += diff / (float)1e6;

			vkGetQueryPoolResults(logicalDevice_, queryPoolESTwo_[2], 0, 1, sizeof(uint64_t), &(timestamp[0]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			vkGetQueryPoolResults(logicalDevice_, queryPoolESTwo_[2], 1, 1, sizeof(uint64_t), &(timestamp[1]), 0, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			diff = timestamp[1] - timestamp[0];
			shaderExecuteTime_[8] += diff / (float)1e6;
		}

		float totalKernelRunTime = 0.0;
		for (uint32_t i = 0; i != numPipelines_; ++i)
		{
			//executeTime = executeTime / numGenerations;
			std::cout << "Time to complete kernel " << i << ": " << shaderExecuteTime_[i] << "ms\n";
			totalKernelRunTime += shaderExecuteTime_[i];
		}
		std::cout << "Time to complete all kernels for this run: " << totalKernelRunTime << "ms\n";
	}
	void parameterMatchAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{
		chunkSize = objective.audioLength;
		chunks = aTargetAudioLength / chunkSize;

		for (int i = 0; i < chunks; i++)
		{
			setTargetAudio(&aTargetAudio[i*chunkSize], chunkSize);
			initPopulationVK();
			uint32_t rI = 0;
			VKHelper::writeBuffer(logicalDevice_, sizeof(uint32_t), storageBuffersMemory_[rotationIndexBuffer], &rI);	//Need this?
			executeAllGenerations();

			uint32_t tempSize = 4 * sizeof(float);
			float* tempData = new float[4];
			float tempFitness;
			VKHelper::readBuffer(logicalDevice_, tempSize, storageBuffersMemory_[inputPopulationValueBuffer], tempData);
			VKHelper::readBuffer(logicalDevice_, sizeof(float), storageBuffersMemory_[inputPopulationFitnessBuffer], &tempFitness);
			printf("Generation %d parameters:\n Param0 = %f\n Param1 = %f\n Param2 = %f\n Param3 = %f\nFitness=%f\n\n\n", i, tempData[0] * 3520.0, tempData[1] * 8.0, tempData[2] * 3520.0, tempData[3] * 1.0, tempFitness);
		}
	}
	void executeOpenCLFFT()
	{
		commandQueue_.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, storageBufferSizes_[outputAudioBuffer], populationAudioDate);

		/* Execute the plan. */
		errorStatus_ = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &commandQueue_(), 0, NULL, NULL, &inputBuffer(), &outputBuffer(), NULL);
		commandQueue_.finish();

		commandQueue_.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, storageBufferSizes_[inputFFTDataBuffer], populationFFTData);
	}

	void setTargetAudio(float* aTargetAudio, uint32_t aTargetAudioLength)
	{
		float* targetFFT = new float[objective.fftHalfSize];
		objective.calculateFFT(aTargetAudio, targetFFT);

		//@ToDo - Check this works. Not any problems with passing as values//
		VKHelper::writeBuffer(logicalDevice_, objective.fftHalfSize * sizeof(float), storageBuffersMemory_[inputFFTTargetBuffer], targetFFT);

		delete(targetFFT);
	}
	void initPopulationVK()
	{
		VKHelper::runCommandBuffer(logicalDevice_, computeQueue_, commandBufferInit_);
	}


	//Command execution functions//
	VkCommandBuffer beginSingleTimeCommands()
	{
		//Memory transfer executed like drawing commands//
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPoolStaging_;			//@ToDo - Is this okay to be renderCommandPool only? Or need one for each??
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(logicalDevice_, &allocInfo, &commandBuffer);

		//Start recording command buffer//
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}
	void endSingleTimeCommands(VkCommandBuffer aCommandBuffer)
	{
		vkEndCommandBuffer(aCommandBuffer);

		//Execute command buffer to complete transfer - Can use fences for synchronized, simultaneous execution//
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCommandBuffer;

		vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(computeQueue_);

		//Cleanup used comand pool//
		vkFreeCommandBuffers(logicalDevice_, commandPoolStaging_, 1, &aCommandBuffer);	//@ToDo - Is this okay to be renderCommandPool only? Or need one for each??
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPoolStaging_;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(logicalDevice_, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);
			VkBufferCopy copyRegion = {};
			copyRegion.srcOffset = 0; // Optional
			copyRegion.dstOffset = 0; // Optional
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(computeQueue_);

		vkFreeCommandBuffers(logicalDevice_, commandPoolStaging_, 1, &commandBuffer);
	}

	void readLocalBuffer(VkBuffer aBuffer, uint32_t aSize, void* aOutput)
	{
		copyBuffer(aBuffer, stagingBufferDst, aSize);
		VKHelper::readBuffer(logicalDevice_, aSize, stagingBufferMemoryDst, aOutput);
	}
	void writeLocalBuffer(VkBuffer aBuffer, uint32_t aSize, void* aInput)
	{
		VKHelper::writeBuffer(logicalDevice_, aSize, stagingBufferMemorySrc, aInput);
		copyBuffer(stagingBufferSrc, aBuffer, aSize);
	}

	void createCommandBuffer(VkCommandPool& aCommandPool, VkCommandBuffer& aCommandBuffer, std::vector<VkPipelineLayout>& aPipelineLayouts, std::vector<VkPipeline>& aPipelines, /*std::vector<VkQueryPool>&*/ VkQueryPool aQueryPools[])
	{
		//Timestamp initilize//
		VkQueryPoolCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		createInfo.pNext = nullptr;
		createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		createInfo.queryCount = 2;

		for (uint32_t i = 0; i != aPipelineLayouts.size(); ++i)
		{
			VkResult res = vkCreateQueryPool(logicalDevice_, &createInfo, nullptr, &(aQueryPools[i]));
			assert(res == VK_SUCCESS);
		}

		/*
		We are getting closer to the end. In order to send commands to the device(GPU),
		we must first record commands into a command buffer.
		To allocate a command buffer, we must first create a command pool. So let us do that.
		*/
		VkCommandPoolCreateInfo commandPoolCreateInfo = {};
		commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		commandPoolCreateInfo.flags = 0;
		// the queue family of this command pool. All command buffers allocated from this command pool,
		// must be submitted to queues of this family ONLY. 
		commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex_;
		VK_CHECK_RESULT(vkCreateCommandPool(logicalDevice_, &commandPoolCreateInfo, NULL, &aCommandPool));

		/*
		Now allocate a command buffer from the command pool.
		*/
		VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
		commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocateInfo.commandPool = aCommandPool; // specify the command pool to allocate from. 
		// if the command buffer is primary, it can be directly submitted to queues. 
		// A secondary buffer has to be called from some primary command buffer, and cannot be directly 
		// submitted to a queue. To keep things simple, we use a primary command buffer. 
		commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferAllocateInfo.commandBufferCount = 1; // allocate a single command buffer. 
		VK_CHECK_RESULT(vkAllocateCommandBuffers(logicalDevice_, &commandBufferAllocateInfo, &aCommandBuffer)); // allocate command buffer.

		/*
		Now we shall start recording commands into the newly allocated command buffer.
		*/
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; // the buffer is only submitted and used once in this application.
		VK_CHECK_RESULT(vkBeginCommandBuffer(aCommandBuffer, &beginInfo)); // start recording commands.
		for (uint32_t i = 0; i != aPipelineLayouts.size(); ++i)
		{
			vkCmdResetQueryPool(aCommandBuffer, aQueryPools[i], 0, 2);
			/*
			We need to bind a pipeline, AND a descriptor set before we dispatch.

			The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
			*/
			vkCmdBindDescriptorSets(aCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, aPipelineLayouts[i], 0, 1, &descriptorSet_, 0, NULL);
			vkCmdBindPipeline(aCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, aPipelines[i]);

			//Include loop and update push constants every iteration//
			//vkCmdPushConstants(commandBufferInit_, computePipelineLayouts_[initPopulation], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), rotationIndex_);
			vkCmdWriteTimestamp(aCommandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, aQueryPools[i], 0);

			/*
			Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
			The number of workgroups is specified in the arguments.
			If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
			*/
			vkCmdDispatch(aCommandBuffer, numWorkgroupsX, numWorkgroupsY, numWorkgroupsZ);
			vkCmdPipelineBarrier(aCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, NULL, 0, NULL, 0, NULL, 0, NULL);

			vkCmdWriteTimestamp(aCommandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, aQueryPools[i], 1);
		}

		VK_CHECK_RESULT(vkEndCommandBuffer(aCommandBuffer)); // end recording commands.
	}
};

#endif