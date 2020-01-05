#ifndef VULKAN_HELPER_HPP
#define VULKAN_HELPER_HPP

#include <vulkan/vulkan.h>

#include <vector>
#include <cstdio>
#include <cassert>
#include <fstream>

// Used for validating return values of Vulkan API calls.
#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

namespace VKHelper
{
	static std::vector<char> readFile(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
			throw std::runtime_error("failed to open file!");

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);
		file.seekg(0);
		file.read(buffer.data(), fileSize);
		file.close();

		return buffer;
	}
	//Create shader module from shader source code//
	VkShaderModule createShaderModule(VkDevice aLogicalDevice, const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(aLogicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			throw std::runtime_error("Failed to create shader module!");

		return shaderModule;
	}

	//Find memory type with desired properties/
	uint32_t findMemoryType(VkPhysicalDevice aPhysicalDevice, uint32_t aMemoryTypeBits, VkMemoryPropertyFlags aProperties)
	{
		VkPhysicalDeviceMemoryProperties memoryProperties;

		vkGetPhysicalDeviceMemoryProperties(aPhysicalDevice, &memoryProperties);

		/*
		How does this search work?
		See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
		*/
		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
			if ((aMemoryTypeBits & (1 << i)) &&
				((memoryProperties.memoryTypes[i].propertyFlags & aProperties) == aProperties))
				return i;
		}
		return -1;
	}
	//Generic buffer creation//
	void createBuffer(VkPhysicalDevice aPhysicalDevice, VkDevice aLogicalDevice, VkDeviceSize aSize, VkBufferUsageFlags aUsage, VkMemoryPropertyFlags aProperties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
	{
		//Define a buffer object//
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = aSize;
		bufferInfo.usage = aUsage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		//Create the buffer//
		VK_CHECK_RESULT(vkCreateBuffer(aLogicalDevice, &bufferInfo, nullptr, &buffer))

			//Query memory requirements//
			VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(aLogicalDevice, buffer, &memRequirements);

		//Specify memory requirements//
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(aPhysicalDevice, memRequirements.memoryTypeBits, aProperties);

		//Apparently not meant to use one allocator for each object/buffer created - For large numbers of objects, create custom allocator which that splits up single allocations by using offset parameters//
		VK_CHECK_RESULT(vkAllocateMemory(aLogicalDevice, &allocInfo, nullptr, &bufferMemory))

		//Bind the created buffer to GPU memory//
		VK_CHECK_RESULT(vkBindBufferMemory(aLogicalDevice, buffer, bufferMemory, 0));
	}
	//void copyBuffer(VkPhysicalDevice aPhysicalDevice, VkDevice aLogicalDevice, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	//{
	//	VkCommandBufferAllocateInfo allocInfo = {};
	//	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	//	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	//	allocInfo.commandPool = commandPool;
	//	allocInfo.commandBufferCount = 1;

	//	VkCommandBuffer commandBuffer;
	//	vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
	//}

	void runCommandBuffer(VkDevice aLogicalDevice, VkQueue aComputeQueue, VkCommandBuffer aCommandBuffer)
	{
		/*
		Now we shall finally submit the recorded command buffer to a queue.
		*/

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1; // submit a single command buffer
		submitInfo.pCommandBuffers = &aCommandBuffer; // the command buffer to submit.

		/*
		  We create a fence.
		*/
		VkFence fence;
		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = 0;
		VK_CHECK_RESULT(vkCreateFence(aLogicalDevice, &fenceCreateInfo, NULL, &fence));

		/*
		We submit the command buffer on the queue, at the same time giving a fence.
		*/
		VK_CHECK_RESULT(vkQueueSubmit(aComputeQueue, 1, &submitInfo, fence));
		VK_CHECK_RESULT(vkWaitForFences(aLogicalDevice, 1, &fence, VK_TRUE, 100000000000));

		vkDestroyFence(aLogicalDevice, fence, NULL);
	}

	void readBuffer(VkDevice aLogicalDevice, VkDeviceSize aSize, VkDeviceMemory aBufferMemory, void* aData)
	{
		void* mappedMemory = NULL;
		//Map the buffer memory, so that we can read from it on the CPU//
		vkMapMemory(aLogicalDevice, aBufferMemory, 0, aSize, 0, &mappedMemory);
		std::memcpy(aData, mappedMemory, aSize);
		//Unmap memory as finished with//
		vkUnmapMemory(aLogicalDevice, aBufferMemory);
	}
	void writeBuffer(VkDevice aLogicalDevice, VkDeviceSize aSize, VkDeviceMemory aBufferMemory, void* aData)
	{
		void* mappedMemory = NULL;
		//Map the buffer memory, so that we can read from it on the CPU//
		vkMapMemory(aLogicalDevice, aBufferMemory, 0, aSize, 0, &mappedMemory);
		std::memcpy(mappedMemory, aData, aSize);
		//Unmap memory as finished with//
		vkUnmapMemory(aLogicalDevice, aBufferMemory);
	}

	void writeLocalBuffer(VkDevice aLogicalDevice, VkDeviceSize aSize, VkDeviceMemory aBufferMemory, VkDeviceMemory aStagingBufferMemory, void* aData)
	{

	}
}

#endif