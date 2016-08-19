#define private public
#include "vulkan/vulkan.hpp"

#include <iostream>
#include <stdexcept>
#include <set>
#include <unordered_set>
#include <limits>
#include <cstring>

#include "deleter.hpp"
#include "utils.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

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
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{1.0f, -0.5f}, {0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
};


const int WIDTH = 800;
const int HEIGHT = 600;

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


class HelloTriangleApplication{
public:
    void run(){
        initWindow();
        initVulkan();
        mainLoop();
    }

private:
    unsigned int framecount = 0;

    GLFWwindow* window;
    VDeleter<vk::Instance> instance;
    VDeleterPure<vk::DebugReportCallbackEXT> callback{instance, E_DestroyDebugReportCallbackEXT};
    VDeleterPure<vk::SurfaceKHR> surface{instance, &vk::Instance::destroySurfaceKHR};
    vk::PhysicalDevice physicalDevice;
    VDeleter<vk::Device> device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    VDeleterPure<vk::SwapchainKHR> swapChain{device, &vk::Device::destroySwapchainKHR};
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<VDeleterPure<vk::ImageView>> swapChainImageViews;
    VDeleterPure<vk::RenderPass> renderPass{device, &vk::Device::destroyRenderPass};
    VDeleterPure<vk::PipelineLayout> pipelineLayout{device, &vk::Device::destroyPipelineLayout};
    VDeleterPure<vk::Pipeline> graphicsPipeline{device, &vk::Device::destroyPipeline};
    std::vector<VDeleterPure<vk::Framebuffer>> swapChainFramebuffers;
    VDeleterPure<vk::CommandPool> commandPool{device, &vk::Device::destroyCommandPool};
    std::vector<vk::CommandBuffer> commandBuffers;
    VDeleterPure<vk::Semaphore> imageAvailableSemaphore{device, &vk::Device::destroySemaphore};
    VDeleterPure<vk::Semaphore> renderFinishedSemaphore{device, &vk::Device::destroySemaphore};
    std::vector<VDeleterPure<vk::Fence>> fences;
    VDeleterPure<vk::Buffer> vertexBuffer{device, &vk::Device::destroyBuffer};
    VDeleterPure<vk::DeviceMemory> vertexBufferMemory{device, &vk::Device::freeMemory};

    void initWindow(){
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetWindowSizeCallback(window, HelloTriangleApplication::onWindowResized);

    }
    void initVulkan(){
        createInstance();
        setupDebugCallback();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createCommandBuffers();
        createSemaphores();
    }
    void recreateSwapChain() {
        device->waitIdle();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
    }

    void createInstance(){
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
    std::vector<const char*> getRequiredExtensions() {
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


    bool checkValidationLayerSupport(){
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

    void setupDebugCallback(){
        if(!enableValidationLayers) return;
        vk::DebugReportCallbackCreateInfoEXT createInfo = vk::DebugReportCallbackCreateInfoEXT()
            // Enable only errors and warnings
            .setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning)
            .setPfnCallback(debugCallback);
        E_vkCreateDebugReportCallbackEXT(instance(), &createInfo, nullptr, &callback);
    }

    static void E_vkCreateDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackCreateInfoEXT* createInfo, vk::AllocationCallbacks* allocator, vk::DebugReportCallbackEXT* callback){
        auto func = (PFN_vkCreateDebugReportCallbackEXT)instance.getProcAddr("vkCreateDebugReportCallbackEXT");
        if(!func) throw std::runtime_error("Failed to get procedure address for vkCreateDebugReportCallbackEXT");
        func(*reinterpret_cast<VkInstance*>(&instance), reinterpret_cast<VkDebugReportCallbackCreateInfoEXT*>(createInfo), reinterpret_cast<VkAllocationCallbacks*>(allocator), reinterpret_cast<VkDebugReportCallbackEXT*>(callback));
    }

    static void E_DestroyDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackEXT* callback, vk::AllocationCallbacks* allocator){
        auto func = (PFN_vkDestroyDebugReportCallbackEXT)instance.getProcAddr("vkDestroyDebugReportCallbackEXT");
        if(!func) throw std::runtime_error("Failed to get procedure address for DestroyDebugReportCallbackEXT");
        func(*reinterpret_cast<VkInstance*>(&instance), *reinterpret_cast<VkDebugReportCallbackEXT*>(callback), reinterpret_cast<VkAllocationCallbacks*>(allocator));
    }

    static VkBool32 debugCallback(VkDebugReportFlagsEXT /*flags*/,
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

    void createSurface(){
        if (glfwCreateWindowSurface(*reinterpret_cast<VkInstance*>(&(instance())), window, nullptr, reinterpret_cast<VkSurfaceKHR*>(&surface)) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice(){
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

    struct QueueFamilyIndices {
        int graphicsFamily = -1;
        int presentFamily = -1;
        bool isComplete() const { return graphicsFamily >= 0 && presentFamily >= 0; }
    };

    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) {
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

    bool isDeviceSuitable(const vk::PhysicalDevice& device){
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

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device){

        std::vector<vk::ExtensionProperties> availableExtensions =
            device.enumerateDeviceExtensionProperties(nullptr);
        std::unordered_set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    void createLogicalDevice(){
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


    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device) {
        SwapChainSupportDetails details;
        details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
        details.formats = device.getSurfaceFormatsKHR(surface);
        details.presentModes = device.getSurfacePresentModesKHR(surface);
        return details;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
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

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
            }
        return vk::PresentModeKHR::eFifo; // Guaranteed to be available
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            vk::Extent2D actualExtent = {WIDTH, HEIGHT};

            actualExtent.width = std::max(capabilities.minImageExtent.width,
                                          std::min(capabilities.maxImageExtent.width,
                                                   actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height,
                                           std::min(capabilities.maxImageExtent.height,
                                                    actualExtent.height));

            return actualExtent;
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        // Decide on number of images in the swap chain
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
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

    }

    void createImageViews(){
        swapChainImageViews.resize(swapChainImages.size(), VDeleterPure<vk::ImageView>{device, &vk::Device::destroyImageView});
        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
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
    }

    void createRenderPass(){
        vk::AttachmentDescription colorAttachment = vk::AttachmentDescription()
            .setFormat(swapChainImageFormat)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);
        vk::AttachmentReference colorAttachmentRef = vk::AttachmentReference()
            .setAttachment(0)
            .setLayout(vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subPass = vk::SubpassDescription()
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachmentCount(1)
            .setPColorAttachments(&colorAttachmentRef);

        vk::SubpassDependency dependency = vk::SubpassDependency()
            .setSrcSubpass(VK_SUBPASS_EXTERNAL)
            .setDstSubpass(0)
            .setSrcStageMask(vk::PipelineStageFlagBits::eBottomOfPipe)
            .setSrcAccessMask(vk::AccessFlagBits::eMemoryRead)
            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                              vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo renderPassInfo = vk::RenderPassCreateInfo()
            .setAttachmentCount(1)
            .setPAttachments(&colorAttachment)
            .setSubpassCount(1)
            .setPSubpasses(&subPass)
            .setDependencyCount(1)
            .setPDependencies(&dependency);
        renderPass = device->createRenderPass(renderPassInfo);
    }

    void createGraphicsPipeline(){
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
            .setFrontFace(vk::FrontFace::eClockwise)
            .setDepthBiasEnable(VK_FALSE);
        // TODO: Tinker with multisampling
        vk::PipelineMultisampleStateCreateInfo multisample = vk::PipelineMultisampleStateCreateInfo()
            .setSampleShadingEnable(false)
            .setRasterizationSamples(vk::SampleCountFlagBits::e1)
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
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo = vk::PipelineLayoutCreateInfo()
            .setSetLayoutCount(0)
            .setPSetLayouts(nullptr)
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

    void createShaderModule(const std::vector<char>& bytecode, VDeleterPure<vk::ShaderModule>& shaderModule) {
        vk::ShaderModuleCreateInfo createInfo = vk::ShaderModuleCreateInfo()
            .setCodeSize(bytecode.size())
            .setPCode((uint32_t*)bytecode.data());
        shaderModule = device->createShaderModule(createInfo);
    }

    void createFramebuffers(){
        swapChainFramebuffers.resize(swapChainImageViews.size(), VDeleterPure<vk::Framebuffer>{device,&vk::Device::destroyFramebuffer});
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            vk::ImageView attachments[] = { swapChainImageViews[i] };
            vk::FramebufferCreateInfo framebufferInfo = vk::FramebufferCreateInfo()
                .setRenderPass(renderPass)
                .setAttachmentCount(1)
                .setPAttachments(attachments)
                .setWidth(swapChainExtent.width)
                .setHeight(swapChainExtent.height)
                .setLayers(1);
            swapChainFramebuffers[i]  = device->createFramebuffer(framebufferInfo);
        }
    }

    void createCommandPool(){
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        vk::CommandPoolCreateInfo poolInfo = vk::CommandPoolCreateInfo()
            .setQueueFamilyIndex(queueFamilyIndices.graphicsFamily)
            .setFlags(vk::CommandPoolCreateFlags());
        commandPool = device->createCommandPool(poolInfo);
    }

    void createVertexBuffer(){
        vk::BufferCreateInfo bufferInfo = vk::BufferCreateInfo()
            .setSize(sizeof(vertices[0]) * vertices.size())
            .setUsage(vk::BufferUsageFlagBits::eVertexBuffer)
            .setSharingMode(vk::SharingMode::eExclusive);
        vertexBuffer = device->createBuffer(bufferInfo);
        vk::MemoryRequirements memRequirements = device->getBufferMemoryRequirements(vertexBuffer);

        std::cout << "VB req | size: " << memRequirements.size << ", allignment " << memRequirements.alignment << std::endl;

        uint32_t memTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
                                               vk::MemoryPropertyFlagBits::eHostVisible |
                                               vk::MemoryPropertyFlagBits::eHostCoherent);
        vk::MemoryAllocateInfo allocInfo = vk::MemoryAllocateInfo()
            .setAllocationSize(memRequirements.size)
            .setMemoryTypeIndex(memTypeIndex);

        vertexBufferMemory = device->allocateMemory(allocInfo);

        device->bindBufferMemory(vertexBuffer, vertexBufferMemory, 0);

        void* memoryMap;
        memoryMap = device->mapMemory(vertexBufferMemory, 0, allocInfo.allocationSize, vk::MemoryMapFlags());
        memcpy(memoryMap, vertices.data(), (uint32_t)bufferInfo.size);
        device->unmapMemory(vertexBufferMemory);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers(){
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
            commandBuffers[i].draw(vertices.size(), 1, 0, 0);
            commandBuffers[i].endRenderPass();

            commandBuffers[i].end();
        }
    }

    void createSemaphores(){
        vk::SemaphoreCreateInfo semaphoreInfo;
        imageAvailableSemaphore = device->createSemaphore(semaphoreInfo);
        renderFinishedSemaphore = device->createSemaphore(semaphoreInfo);
        vk::FenceCreateInfo fenceInfo;
        fences.resize(commandBuffers.size(), VDeleterPure<vk::Fence>(device, &vk::Device::destroyFence));
        for(unsigned int i = 0; i < fences.size(); i++){
            fences[i] = device->createFence(fenceInfo);
        }
    }

    static void onWindowResized(GLFWwindow* window, int width, int height) {
        if (width == 0 || height == 0) return;

        HelloTriangleApplication* app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->recreateSwapChain();
    }

    void mainLoop(){
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            framecount++;
        }
        device->waitIdle();
        std::cout << "Total frames: " << framecount << std::endl;
    }

    void drawFrame(){
        vk::ResultValue<unsigned int> res = device->acquireNextImageKHR(swapChain,
                                                 std::numeric_limits<uint64_t>::max(),
                                                 imageAvailableSemaphore,
                                                 VK_NULL_HANDLE); // No fence
        uint32_t imageIndex = res.value;
        if (res.result == vk::Result::eErrorOutOfDateKHR || res.result == vk::Result::eSuboptimalKHR) {
            recreateSwapChain();
        }

        //std::cout << "Waiting for fence " << imageIndex << " " << fences[imageIndex]().m_fence << std::endl;
        device->waitForFences({fences[imageIndex]}, true, UINT64_MAX);
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
};

int main(){
    HelloTriangleApplication app;

    try{
        app.run();
    }catch (const std::runtime_error& e){
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
