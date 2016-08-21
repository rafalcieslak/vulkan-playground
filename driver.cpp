#include "driver.hpp"
#include <iostream>
#include <set>
#include <unordered_set>
#include <limits>
#include <cstring>

#include "utils.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>


#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    static vk::VertexInputBindingDescription getBindingDescription(){
        return vk::VertexInputBindingDescription()
            .setBinding(0)
            .setStride(sizeof(Vertex))
            .setInputRate(vk::VertexInputRate::eVertex);
    }
    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, color);
        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};

const vk::SampleCountFlagBits multisample_level = vk::SampleCountFlagBits::e8;

float rot_speed = 1.5f;
float float_speed = 0.6f;

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct UBO2{
    float x[600];
};

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif



void Driver::init(GLFWwindow* win){
    window = win;
    createInstance();
    setupDebugCallback();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain(); // recreatable
    createImageViews(); // recreatable
    createRenderPass(); // recreateable
    createDescriptorSetLayout();
    createGraphicsPipeline(); // recreatable
    createFramebuffers(); // recreatable
    createCommandPool();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffer();
    createDescriptorPool();
    createDescriptorSet();
    createCommandBuffers(); // recreatable
    createSemaphores();
}

void Driver::recreateSwapChain(){
    device->waitIdle();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandBuffers();
}


void Driver::createInstance(){
    // Test whether requested validation layers are available
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    // Create ApplicationInfo
    vk::ApplicationInfo appInfo = vk::ApplicationInfo()
        .setPApplicationName("Hello Triangle")
        .setApplicationVersion(VK_MAKE_VERSION(1,0,0))
        .setPEngineName("None")
        .setEngineVersion(VK_MAKE_VERSION(1,0,0))
        .setApiVersion(VK_API_VERSION_1_0);
    vk::InstanceCreateInfo createInfo = vk::InstanceCreateInfo()
        .setPApplicationInfo(&appInfo);

    std::vector<const char*> requiredExtensions = getRequiredExtensions();
    createInfo
        .setEnabledExtensionCount(requiredExtensions.size())
        .setPpEnabledExtensionNames(requiredExtensions.data())
        .setEnabledLayerCount(0);
    // Enable validation layers, if requested
    if(enableValidationLayers){
        createInfo.enabledLayerCount = validationLayers.size();
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }else{
        createInfo.enabledLayerCount = 0;
    }

    // Crete VK instance!
    instance = vk::createInstance(createInfo);

    std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties(nullptr);
    std::cout << "Available extensions:" << std::endl;
    for(const auto& ext : extensions) std::cout << ext.extensionName << std::endl;
    std::cout << "Checking whether required extensions are among the available ones:" << std::endl;
    for(const std::string& ext_name : requiredExtensions){
        std::cout << ext_name << ": ";
        bool found = false;
        for(const auto& ext : extensions){
            if(ext.extensionName == ext_name) found = true;
        }
        std::cout << ((found)?"OK":"NOT FOUND") << std::endl;
        if(!found){
            std::cout << "Instance creation failed: Extension " << ext_name << " is required by GLFW, but is not available in the instance" << std::endl;
        }
    }
    std::cout << "All required extensions are present." << std::endl;
}


// Returns a list of required extensions.
std::vector<const char*> Driver::getRequiredExtensions(){
    std::vector<const char*> extensions;
    unsigned int glfwExtensionCount = 0;
    const char** glfwExtensions;
    // Required by GLFW
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    for (unsigned int i = 0; i < glfwExtensionCount; i++) {
        extensions.push_back(glfwExtensions[i]);
    }
    // Required for output from validation layers
    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }
    return extensions;
}


bool Driver::checkValidationLayerSupport(){
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
    for (const char* layerName : validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            std::cout << "Failure: Validation layer " << layerName << " is not available" << std::endl;
            return false;
        }
    }
    return true;
}

void Driver::setupDebugCallback(){
    if(!enableValidationLayers) return;
    vk::DebugReportCallbackCreateInfoEXT createInfo = vk::DebugReportCallbackCreateInfoEXT()
        // Enable only errors and warnings
        .setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning)
        .setPfnCallback(debugCallback);
    E_vkCreateDebugReportCallbackEXT(instance(), &createInfo, nullptr, &callback);
}

void Driver::E_vkCreateDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackCreateInfoEXT* createInfo, vk::AllocationCallbacks* allocator, vk::DebugReportCallbackEXT* callback){
    auto func = (PFN_vkCreateDebugReportCallbackEXT)instance.getProcAddr("vkCreateDebugReportCallbackEXT");
    if(!func) throw std::runtime_error("Failed to get procedure address for vkCreateDebugReportCallbackEXT");
    func(*reinterpret_cast<VkInstance*>(&instance), reinterpret_cast<VkDebugReportCallbackCreateInfoEXT*>(createInfo), reinterpret_cast<VkAllocationCallbacks*>(allocator), reinterpret_cast<VkDebugReportCallbackEXT*>(callback));
}

void Driver::E_DestroyDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackEXT* callback, vk::AllocationCallbacks* allocator){
    auto func = (PFN_vkDestroyDebugReportCallbackEXT)instance.getProcAddr("vkDestroyDebugReportCallbackEXT");
    if(!func) throw std::runtime_error("Failed to get procedure address for DestroyDebugReportCallbackEXT");
    func(*reinterpret_cast<VkInstance*>(&instance), *reinterpret_cast<VkDebugReportCallbackEXT*>(callback), reinterpret_cast<VkAllocationCallbacks*>(allocator));
}

VkBool32 Driver::debugCallback(VkDebugReportFlagsEXT /*flags*/,
                              VkDebugReportObjectTypeEXT /*objType*/,
                              uint64_t /*obj*/,
                              size_t /*location*/,
                              int32_t /*code*/,
                              const char* layerPrefix,
                              const char* msg,
                              void* /*userData*/) {

    std::cerr << "=== Debug Callback ===" << std::endl;
    std::cerr << "Layer: " << layerPrefix << ", message: " << msg << std::endl;

    return VK_FALSE;
}

void Driver::createSurface(){
    if (glfwCreateWindowSurface(*reinterpret_cast<VkInstance*>(&(instance())), window, nullptr, reinterpret_cast<VkSurfaceKHR*>(&surface)) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

void Driver::pickPhysicalDevice(){
    std::vector<vk::PhysicalDevice> devices = instance->enumeratePhysicalDevices();
    if(devices.empty()) throw std::runtime_error("No Vulkan-capable devices found.");
    for (const auto& device : devices) {
        std::cout << "Physical device found: " << device.getProperties().deviceName << std::endl;
        if (isDeviceSuitable(device)) {
            std::cout << "This device seems suitable" << std::endl;
            physicalDevice = device;
            break;
        }
    }
    if(!physicalDevice) throw std::runtime_error("Vulkan-capable devices found, but none of them is capable.");
}


Driver::QueueFamilyIndices Driver::findQueueFamilies(const vk::PhysicalDevice& device) {
    QueueFamilyIndices indices;
    auto queueFamilies = device.getQueueFamilyProperties();
    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }
        if(device.getSurfaceSupportKHR(i, surface)){
            indices.presentFamily = i;
        }
        if (indices.isComplete()) {
            break;
        }
        i++;
    }
    return indices;
}

bool Driver::isDeviceSuitable(const vk::PhysicalDevice& device){
    vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
    vk::PhysicalDeviceFeatures deviceFeatures = device.getFeatures();
    QueueFamilyIndices indices = findQueueFamilies(device);
    bool supportsExtensions = checkDeviceExtensionSupport(device);
    bool swapChainAdequate = false;
    if (supportsExtensions) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }
    return deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
        deviceFeatures.geometryShader && indices.isComplete() &&
        supportsExtensions && swapChainAdequate;
}

bool Driver::checkDeviceExtensionSupport(const vk::PhysicalDevice& device){

    std::vector<vk::ExtensionProperties> availableExtensions =
        device.enumerateDeviceExtensionProperties(nullptr);
    std::unordered_set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }
    return requiredExtensions.empty();
}

void Driver::createLogicalDevice(){
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    std::set<int> uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

    float queuePriority = 1.0f;

    for (int queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo = vk::DeviceQueueCreateInfo()
            .setQueueFamilyIndex(queueFamily)
            .setQueueCount(1)
            .setPQueuePriorities(&queuePriority);
        queueCreateInfos.push_back(queueCreateInfo);
    }
    vk::PhysicalDeviceFeatures requestedFeatures = {}; // None
    vk::DeviceCreateInfo createInfo = vk::DeviceCreateInfo()
        .setPQueueCreateInfos(queueCreateInfos.data())
        .setQueueCreateInfoCount(queueCreateInfos.size()) // count of the above queues
        .setPEnabledFeatures(&requestedFeatures);
    // Enable validation layers for the device, if requested
    if(enableValidationLayers){
        createInfo.enabledLayerCount = validationLayers.size();
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }else{
        createInfo.enabledLayerCount = 0;
    }
    createInfo.enabledExtensionCount = deviceExtensions.size();
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    device = physicalDevice.createDevice(createInfo, nullptr);
    graphicsQueue = device->getQueue(indices.graphicsFamily, 0);
    presentQueue  = device->getQueue(indices.presentFamily , 0);
}

Driver::SwapChainSupportDetails Driver::querySwapChainSupport(vk::PhysicalDevice device) {
    SwapChainSupportDetails details;
    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);
    return details;
}

vk::SurfaceFormatKHR Driver::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    // Case 1: The surface has no preferred format
    if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined) {
        return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
    }
    // Case 2: Surface prefers some formats, see if our desires is available on the list
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Unorm &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }
    // Case 3: Desired format unavailable, use whatever is available
    return availableFormats[0];
}

vk::PresentModeKHR Driver::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> availablePresentModes) {
    return vk::PresentModeKHR::eImmediate; // Other present modes cause tearing and X lag.

    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }
    return vk::PresentModeKHR::eFifo; // Guaranteed to be available
}

vk::Extent2D Driver::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        vk::Extent2D actualExtent = {(uint32_t)width, (uint32_t)height};

        actualExtent.width = std::max(capabilities.minImageExtent.width,
                                      std::min(capabilities.maxImageExtent.width,
                                               actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height,
                                       std::min(capabilities.maxImageExtent.height,
                                                actualExtent.height));

        return actualExtent;
    }
}


void Driver::createSwapChain() {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
    // Decide on number of images in the swap chain
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }
    std::cout << "Desired image count: " << imageCount << std::endl;
    vk::SwapchainCreateInfoKHR createInfo = vk::SwapchainCreateInfoKHR()
        .setSurface(surface)
        .setMinImageCount(imageCount)
        .setImageFormat(surfaceFormat.format)
        .setImageColorSpace(surfaceFormat.colorSpace)
        .setImageExtent(extent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {(uint32_t) indices.graphicsFamily, (uint32_t) indices.presentFamily};

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
    }
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    // Ignore the alpha bit, do not blend with other windows...
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    vk::SwapchainKHR oldSwapChain = swapChain;
    createInfo.oldSwapchain = oldSwapChain;

    // Create the swap chain!
    //vk::SwapchainKHR newSwapChain = device->createSwapchainKHR(createInfo);
    //swapChain = newSwapChain;
    swapChain = device->createSwapchainKHR(createInfo);

    // Rertieve images from the chain
    swapChainImages = device->getSwapchainImagesKHR(swapChain);
    // Store swapchain image format
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;

    auto deviceProperties = physicalDevice.getProperties();
    if(!(deviceProperties.limits.framebufferColorSampleCounts & vk::SampleCountFlags(multisample_level))){
        std::cout << "!!! Multisample level not supported by device" << std::endl;
            }

    resolveImages.resize(swapChainImages.size());
    resolveImagesMemory.resize(swapChainImages.size(), VDeleterPure<vk::DeviceMemory>{device, &vk::Device::freeMemory});
    for(uint32_t i = 0; i < resolveImages.size(); i++){
        vk::ImageCreateInfo imageInfo = vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setFormat(swapChainImageFormat)
            .setExtent(vk::Extent3D(swapChainExtent.width, swapChainExtent.height, 1))
            .setMipLevels(1)
            .setArrayLayers(1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal)
            .setSamples(multisample_level)
            .setUsage(vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment)
            .setInitialLayout(vk::ImageLayout::eUndefined);
        resolveImages[i] = device->createImage(imageInfo);

        vk::MemoryRequirements memReq = device->getImageMemoryRequirements(resolveImages[i]);
        vk::MemoryAllocateInfo allocInfo = vk::MemoryAllocateInfo()
            .setAllocationSize(memReq.size);
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        resolveImagesMemory[i] = device->allocateMemory(allocInfo);
        device->bindImageMemory(resolveImages[i], resolveImagesMemory[i], 0);
    }

}

void Driver::createImageViews(){
    swapChainImageViews.resize(swapChainImages.size(), VDeleterPure<vk::ImageView>{device, &vk::Device::destroyImageView});
    resolveImageViews.resize(swapChainImages.size(), VDeleterPure<vk::ImageView>{device, &vk::Device::destroyImageView});
    for (uint32_t i = 0; i < swapChainImages.size(); i++) {
        {
        vk::ImageViewCreateInfo createInfo = vk::ImageViewCreateInfo()
            .setImage(swapChainImages[i])
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(swapChainImageFormat);

        createInfo.components.r = vk::ComponentSwizzle::eIdentity;
        createInfo.components.g = vk::ComponentSwizzle::eIdentity;
        createInfo.components.b = vk::ComponentSwizzle::eIdentity;
        createInfo.components.a = vk::ComponentSwizzle::eIdentity;
        createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        swapChainImageViews[i] = device->createImageView(createInfo);
        }
        {
        vk::ImageViewCreateInfo createInfo = vk::ImageViewCreateInfo()
            .setImage(resolveImages[i])
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(swapChainImageFormat);

        createInfo.components.r = vk::ComponentSwizzle::eIdentity;
        createInfo.components.g = vk::ComponentSwizzle::eIdentity;
        createInfo.components.b = vk::ComponentSwizzle::eIdentity;
        createInfo.components.a = vk::ComponentSwizzle::eIdentity;
        createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        resolveImageViews[i] = device->createImageView(createInfo);
        }
    }
}

void Driver::createRenderPass(){
    vk::AttachmentDescription colorAttachment = vk::AttachmentDescription()
        .setFormat(swapChainImageFormat)
        .setSamples(multisample_level)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::AttachmentDescription resolveAttachment = vk::AttachmentDescription()
        .setFormat(swapChainImageFormat)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    vk::AttachmentReference colorAttachmentRef = vk::AttachmentReference()
        .setAttachment(0)
        .setLayout(vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference resolveAttachmentRef = vk::AttachmentReference()
        .setAttachment(1)
        .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subPass = vk::SubpassDescription()
        .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
        .setColorAttachmentCount(1)
        .setPColorAttachments(&colorAttachmentRef)
        .setPResolveAttachments(&resolveAttachmentRef);

    vk::SubpassDependency dependency = vk::SubpassDependency()
        .setSrcSubpass(VK_SUBPASS_EXTERNAL)
        .setDstSubpass(0)
        .setSrcStageMask(vk::PipelineStageFlagBits::eBottomOfPipe)
        .setSrcAccessMask(vk::AccessFlagBits::eMemoryRead)
        .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                          vk::AccessFlagBits::eColorAttachmentWrite);

    std::array<vk::AttachmentDescription, 2> attachments = {colorAttachment, resolveAttachment};
    vk::RenderPassCreateInfo renderPassInfo = vk::RenderPassCreateInfo()
        .setAttachmentCount(2)
        .setPAttachments(attachments.data())
        .setSubpassCount(1)
        .setPSubpasses(&subPass)
        .setDependencyCount(1)
        .setPDependencies(&dependency);
    renderPass = device->createRenderPass(renderPassInfo);
}

void Driver::createDescriptorSetLayout(){
    std::array<vk::DescriptorSetLayoutBinding, 2> uboLayoutBinding;
    uboLayoutBinding[0]
        .setBinding(0)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);
    uboLayoutBinding[1]
        .setBinding(1)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);
    vk::DescriptorSetLayoutCreateInfo layoutInfo = vk::DescriptorSetLayoutCreateInfo()
        .setBindingCount(2)
        .setPBindings(uboLayoutBinding.data());
    descriptorSetLayout = device->createDescriptorSetLayout(layoutInfo);


    std::array<vk::DescriptorSetLayoutBinding, 1> uboLayoutBinding2;
    uboLayoutBinding2[0]
        .setBinding(1)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);
    vk::DescriptorSetLayoutCreateInfo layoutInfo2 = vk::DescriptorSetLayoutCreateInfo()
        .setBindingCount(1)
        .setPBindings(uboLayoutBinding2.data());
    descriptorSetLayout2 = device->createDescriptorSetLayout(layoutInfo2);
}

void Driver::createGraphicsPipeline(){
    // Creating wrappers for shader bytecode
    auto vertShaderCode = readBinaryFile("shaders/vert.spv");
    auto fragShaderCode = readBinaryFile("shaders/frag.spv");
    VDeleterPure<vk::ShaderModule> vertShaderModule{device, &vk::Device::destroyShaderModule};
    VDeleterPure<vk::ShaderModule> fragShaderModule{device, &vk::Device::destroyShaderModule};
    createShaderModule(vertShaderCode, vertShaderModule);
    createShaderModule(fragShaderCode, fragShaderModule);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo = vk::PipelineShaderStageCreateInfo()
        .setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(vertShaderModule)
        .setPName("main");
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo = vk::PipelineShaderStageCreateInfo()
        .setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(fragShaderModule)
        .setPName("main");

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo = vk::PipelineVertexInputStateCreateInfo()
        .setVertexBindingDescriptionCount(1)
        .setPVertexBindingDescriptions(&bindingDescription)
        .setVertexAttributeDescriptionCount(2)
        .setPVertexAttributeDescriptions(attributeDescriptions.data());
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly = vk::PipelineInputAssemblyStateCreateInfo()
        .setTopology(vk::PrimitiveTopology::eTriangleList)
        .setPrimitiveRestartEnable(false);

    vk::Viewport viewport = vk::Viewport()
        .setX(0.0f).setY(0.0f)
        .setWidth(swapChainExtent.width)
        .setHeight(swapChainExtent.height)
        .setMinDepth(0.0f).setMaxDepth(1.0f);
    vk::Rect2D scissor = vk::Rect2D()
        .setOffset({0,0})
        .setExtent(swapChainExtent);
    vk::PipelineViewportStateCreateInfo viewportState = vk::PipelineViewportStateCreateInfo()
        .setViewportCount(1).setPViewports(&viewport)
        .setScissorCount(1).setPScissors(&scissor);

    vk::PipelineRasterizationStateCreateInfo rasterizer = vk::PipelineRasterizationStateCreateInfo()
        .setDepthClampEnable(false)
        .setRasterizerDiscardEnable(false)
        .setPolygonMode(vk::PolygonMode::eFill)
        .setLineWidth(1.0f)
        .setCullMode(vk::CullModeFlagBits::eBack)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setDepthBiasEnable(VK_FALSE);
    vk::PipelineMultisampleStateCreateInfo multisample = vk::PipelineMultisampleStateCreateInfo()
        .setSampleShadingEnable(false)
        .setRasterizationSamples(multisample_level)
        .setMinSampleShading(1.0f)
        .setPSampleMask(nullptr)
        .setAlphaToCoverageEnable(false)
        .setAlphaToOneEnable(false);
    vk::PipelineColorBlendAttachmentState colorBlendAttachment = vk::PipelineColorBlendAttachmentState()
        .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                           vk::ColorComponentFlagBits::eG |
                           vk::ColorComponentFlagBits::eB |
                           vk::ColorComponentFlagBits::eA)
        .setBlendEnable(false); // Note: This could also set the blending function parameters
    vk::PipelineColorBlendStateCreateInfo colorBlending = vk::PipelineColorBlendStateCreateInfo()
        .setLogicOpEnable(false)
        .setLogicOp(vk::LogicOp::eCopy)
        .setAttachmentCount(1)
        .setPAttachments(&colorBlendAttachment);
    vk::DescriptorSetLayout setLayouts[] = {descriptorSetLayout,descriptorSetLayout2};
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo = vk::PipelineLayoutCreateInfo()
        .setSetLayoutCount(1)
        .setPSetLayouts(setLayouts)
        .setPushConstantRangeCount(0)
        .setPPushConstantRanges(nullptr);
    pipelineLayout = device->createPipelineLayout(pipelineLayoutInfo);

    // --- Finally, assemble the pipeline ---
    vk::GraphicsPipelineCreateInfo pipelineInfo = vk::GraphicsPipelineCreateInfo()
        .setStageCount(2)
        .setPStages(shaderStages)
        .setPVertexInputState(&vertexInputInfo)
        .setPInputAssemblyState(&inputAssembly)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisample)
        .setPDepthStencilState(nullptr)
        .setPColorBlendState(&colorBlending)
        .setPDynamicState(nullptr)
        .setLayout(pipelineLayout)
        .setRenderPass(renderPass)
        .setSubpass(0)
        .setBasePipelineHandle(VK_NULL_HANDLE)
        .setBasePipelineIndex(-1);
    graphicsPipeline = device->createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo);
}

void Driver::createShaderModule(const std::vector<char>& bytecode, VDeleterPure<vk::ShaderModule>& shaderModule) {
    vk::ShaderModuleCreateInfo createInfo = vk::ShaderModuleCreateInfo()
        .setCodeSize(bytecode.size())
        .setPCode((uint32_t*)bytecode.data());
    shaderModule = device->createShaderModule(createInfo);
}

void Driver::createFramebuffers(){
    swapChainFramebuffers.resize(swapChainImageViews.size(), VDeleterPure<vk::Framebuffer>{device,&vk::Device::destroyFramebuffer});
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        vk::ImageView attachments[] = { resolveImageViews[i], swapChainImageViews[i] };
        vk::FramebufferCreateInfo framebufferInfo = vk::FramebufferCreateInfo()
            .setRenderPass(renderPass)
            .setAttachmentCount(2)
            .setPAttachments(attachments)
            .setWidth(swapChainExtent.width)
            .setHeight(swapChainExtent.height)
            .setLayers(1);
        swapChainFramebuffers[i]  = device->createFramebuffer(framebufferInfo);
    }
}

void Driver::createCommandPool(){
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
    vk::CommandPoolCreateInfo poolInfo = vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(queueFamilyIndices.graphicsFamily)
        .setFlags(vk::CommandPoolCreateFlags());
    commandPool = device->createCommandPool(poolInfo);
}


void Driver::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, VDeleterPure<vk::Buffer>& buffer, VDeleterPure<vk::DeviceMemory>& bufferMemory) {

    vk::BufferCreateInfo bufferInfo = vk::BufferCreateInfo()
        .setSize(size)
        .setUsage(usage)
        .setSharingMode(vk::SharingMode::eExclusive);

    buffer = device->createBuffer(bufferInfo);

    vk::MemoryRequirements memRequirements = device->getBufferMemoryRequirements(buffer);


    uint32_t memTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    vk::MemoryAllocateInfo allocInfo = vk::MemoryAllocateInfo()
        .setAllocationSize(memRequirements.size)
        .setMemoryTypeIndex(memTypeIndex);

    bufferMemory = device->allocateMemory(allocInfo);

    device->bindBufferMemory(buffer, bufferMemory, 0);
}

void Driver::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
    vk::CommandBufferAllocateInfo allocInfo = vk::CommandBufferAllocateInfo()
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandPool(commandPool)
        .setCommandBufferCount(1);
    vk::CommandBuffer commandBuffer = device->allocateCommandBuffers(allocInfo)[0];

    commandBuffer.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy copyRegion = vk::BufferCopy()
        .setSrcOffset(0)
        .setDstOffset(0)
        .setSize(size);
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, {copyRegion});
    commandBuffer.end();

    graphicsQueue.submit({vk::SubmitInfo()
                .setCommandBufferCount(1)
                .setPCommandBuffers(&commandBuffer)
                }, nullptr);
    graphicsQueue.waitIdle();

    device->freeCommandBuffers(commandPool, {commandBuffer});
}

void Driver::createVertexBuffer(){
    vk::DeviceSize size = sizeof(vertices[0]) * vertices.size();
    VDeleterPure<vk::Buffer> stagingBuffer{device, &vk::Device::destroyBuffer};
    VDeleterPure<vk::DeviceMemory> stagingBufferMemory{device, &vk::Device::freeMemory};

    createBuffer(size,
                 vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                 vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void* memoryMap;
    memoryMap = device->mapMemory(stagingBufferMemory, 0, size, vk::MemoryMapFlags());
    memcpy(memoryMap, vertices.data(), (uint32_t)size);
    device->unmapMemory(stagingBufferMemory);

    createBuffer(size,
                 vk::BufferUsageFlagBits::eTransferDst |
                 vk::BufferUsageFlagBits::eVertexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                 vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, size);
}


void Driver::createIndexBuffer(){
    vk::DeviceSize size = sizeof(indices[0]) * indices.size();
    VDeleterPure<vk::Buffer> stagingBuffer{device, &vk::Device::destroyBuffer};
    VDeleterPure<vk::DeviceMemory> stagingBufferMemory{device, &vk::Device::freeMemory};

    createBuffer(size,
                 vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                 vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void* memoryMap;
    memoryMap = device->mapMemory(stagingBufferMemory, 0, size, vk::MemoryMapFlags());
    memcpy(memoryMap, indices.data(), (uint32_t)size);
    device->unmapMemory(stagingBufferMemory);

    createBuffer(size,
                 vk::BufferUsageFlagBits::eTransferDst |
                 vk::BufferUsageFlagBits::eIndexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                 indexBuffer, indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, size);
}

void Driver::createUniformBuffer(){

    const size_t allign = 0x100;
    std::cout << "sizeof ubo1 = " << sizeof(UniformBufferObject) << std::endl;
    std::cout << "allign = " << allign << std::endl;
    // TODO: Missing 1 allignment if already perfectly alligned
    uint32_t l = sizeof(UniformBufferObject)/allign;
    ubo2_offset = (l+1)*allign;
    std::cout << "offset = " << ubo2_offset << std::endl;

    uniformBufferTotalSize = ubo2_offset + sizeof(UBO2);
    vk::DeviceSize bsize = uniformBufferTotalSize;

    createBuffer(bsize,
                 vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                 vk::MemoryPropertyFlagBits::eHostCoherent,
                 uniformStagingBuffer, uniformStagingBufferMemory);
    createBuffer(bsize,
                 vk::BufferUsageFlagBits::eTransferDst |
                 vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                 uniformBuffer, uniformBufferMemory);
}

void Driver::createDescriptorPool(){
    vk::DescriptorPoolSize poolSize = vk::DescriptorPoolSize()
        .setDescriptorCount(2)
        .setType(vk::DescriptorType::eUniformBuffer);
    vk::DescriptorPoolCreateInfo createInfo = vk::DescriptorPoolCreateInfo()
        .setPoolSizeCount(1)
        .setPPoolSizes(&poolSize)
        .setMaxSets(1);
    descriptorPool = device->createDescriptorPool(createInfo);
}

void Driver::createDescriptorSet(){
    vk::DescriptorSetLayout layouts[] = {descriptorSetLayout};
    vk::DescriptorSetAllocateInfo allocInfo = vk::DescriptorSetAllocateInfo()
        .setDescriptorPool(descriptorPool)
        .setDescriptorSetCount(1)
        .setPSetLayouts(layouts);
    descriptorSet = device->allocateDescriptorSets(allocInfo)[0];

    std::array<vk::DescriptorBufferInfo, 2> bufferInfo;
    bufferInfo[0].setBuffer(uniformBuffer);
    bufferInfo[0].setOffset(0);
    bufferInfo[0].setRange(sizeof(UniformBufferObject));
    bufferInfo[1].setBuffer(uniformBuffer);
    std::cout << "offset = " << ubo2_offset << std::endl;
    bufferInfo[1].setOffset(ubo2_offset);
    bufferInfo[1].setRange(sizeof(UBO2));
    vk::WriteDescriptorSet descWrite = vk::WriteDescriptorSet()
        .setDstSet(descriptorSet)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(2)
        .setPBufferInfo(bufferInfo.data());
    device->updateDescriptorSets({descWrite},{});
}

void Driver::createCommandBuffers(){
    if (commandBuffers.size() > 0) {
        device->freeCommandBuffers(commandPool, commandBuffers.size(), commandBuffers.data());
    }

    vk::CommandBufferAllocateInfo allocInfo = vk::CommandBufferAllocateInfo()
        .setCommandPool(commandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount((uint32_t)swapChainFramebuffers.size());
    commandBuffers = device->allocateCommandBuffers(allocInfo);

    for (size_t i = 0; i < commandBuffers.size(); i++) {
        vk::CommandBufferBeginInfo beginInfo = vk::CommandBufferBeginInfo()
            .setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse)
            .setPInheritanceInfo(nullptr);

        commandBuffers[i].begin(beginInfo);

        vk::ClearValue clearColor = vk::ClearValue()
            .setColor(vk::ClearColorValue(std::array<float,4>{{0.0f,0.0f,0.0f,1.0f}}));
        vk::RenderPassBeginInfo renderPassInfo = vk::RenderPassBeginInfo()
            .setRenderPass(renderPass)
            .setFramebuffer(swapChainFramebuffers[i])
            .setClearValueCount(1)
            .setPClearValues(&clearColor);
        renderPassInfo.renderArea.setOffset({0,0});
        renderPassInfo.renderArea.setExtent(swapChainExtent);

        // Recording draw commands
        commandBuffers[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
        commandBuffers[i].bindVertexBuffers(0, {vertexBuffer}, {0});
        commandBuffers[i].bindIndexBuffer(indexBuffer,0,vk::IndexType::eUint16);
        commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, {descriptorSet}, {});
        commandBuffers[i].drawIndexed(indices.size(), 1, 0, 0, 0);
        commandBuffers[i].endRenderPass();

        commandBuffers[i].end();
    }
}

void Driver::createSemaphores(){
    vk::SemaphoreCreateInfo semaphoreInfo;
    imageAvailableSemaphore = device->createSemaphore(semaphoreInfo);
    renderFinishedSemaphore = device->createSemaphore(semaphoreInfo);
    vk::FenceCreateInfo fenceInfo;
    fences.resize(commandBuffers.size(), VDeleterPure<vk::Fence>(device, &vk::Device::destroyFence));
    for(unsigned int i = 0; i < fences.size(); i++){
        fences[i] = device->createFence(fenceInfo);
    }
}


uint32_t Driver::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}


void Driver::updateUniformBuffer(){
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0f;

    UniformBufferObject ubo = {};
    float angle = rot_speed * time * glm::radians(90.0f);
    ubo.model = glm::rotate(glm::mat4(), angle, glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;
    UBO2 ubo2 = {};
    ubo2.x[0] = 1.0f * glm::sin(time * float_speed);

    char* mappedMemory;
    mappedMemory = (char*)device->mapMemory(uniformStagingBufferMemory, 0, uniformBufferTotalSize, vk::MemoryMapFlags());
    memcpy(mappedMemory, &ubo, sizeof(ubo));
    memcpy(mappedMemory + ubo2_offset, &ubo2, sizeof(ubo2));
    device->unmapMemory(uniformStagingBufferMemory);

    copyBuffer(uniformStagingBuffer, uniformBuffer, uniformBufferTotalSize);
}

void Driver::drawFrame(){
    vk::ResultValue<unsigned int> res = device->acquireNextImageKHR(swapChain,
                                                                    std::numeric_limits<uint64_t>::max(),
                                                                    imageAvailableSemaphore,
                                                                    VK_NULL_HANDLE); // No fence
    uint32_t imageIndex = res.value;
    if (res.result == vk::Result::eErrorOutOfDateKHR || res.result == vk::Result::eSuboptimalKHR) {
        recreateSwapChain();
    }

    //std::cout << "Waiting for fence " << imageIndex << " " << fences[imageIndex]().m_fence << std::endl;
    //device->waitForFences({fences[imageIndex]}, true, UINT64_MAX);
    device->resetFences({fences[imageIndex]});

    vk::Semaphore waitSemaphores[] = {imageAvailableSemaphore};
    vk::Semaphore signalSemaphores[] = {renderFinishedSemaphore};
    vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

    vk::SubmitInfo submitInfo = vk::SubmitInfo()
        .setWaitSemaphoreCount(1)
        .setPWaitSemaphores(waitSemaphores)
        .setPWaitDstStageMask(waitStages)
        .setCommandBufferCount(1)
        .setPCommandBuffers(&commandBuffers[imageIndex])
        .setSignalSemaphoreCount(1)
        .setPSignalSemaphores(signalSemaphores);
    graphicsQueue.submit(1, &submitInfo, fences[imageIndex]);

    vk::SwapchainKHR swapChains[] = {swapChain};
    vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR()
        .setWaitSemaphoreCount(1)
        .setPWaitSemaphores(signalSemaphores)
        .setSwapchainCount(1)
        .setPSwapchains(swapChains)
        .setPImageIndices(&imageIndex);

    presentQueue.presentKHR(presentInfo);
}
