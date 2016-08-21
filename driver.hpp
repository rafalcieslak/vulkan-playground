#ifndef __DRIVER_HPP__
#define __DRIVER_HPP__

#include "vulkan/vulkan.hpp"
#include "deleter.hpp"

struct GLFWwindow;

class Driver{
public:
    void init(GLFWwindow* win);
    void drawFrame();
    void recreateSwapChain();
private:
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
    std::vector<vk::Image> resolveImages;
    std::vector<VDeleterPure<vk::DeviceMemory>> resolveImagesMemory;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<VDeleterPure<vk::ImageView>> resolveImageViews;
    std::vector<VDeleterPure<vk::ImageView>> swapChainImageViews;
    VDeleterPure<vk::RenderPass> renderPass{device, &vk::Device::destroyRenderPass};
    VDeleterPure<vk::DescriptorSetLayout> descriptorSetLayout{device, &vk::Device::destroyDescriptorSetLayout};
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
    VDeleterPure<vk::Buffer> indexBuffer{device, &vk::Device::destroyBuffer};
    VDeleterPure<vk::DeviceMemory> indexBufferMemory{device, &vk::Device::freeMemory};
    VDeleterPure<vk::Buffer> uniformStagingBuffer{device, &vk::Device::destroyBuffer};
    VDeleterPure<vk::DeviceMemory> uniformStagingBufferMemory{device, &vk::Device::freeMemory};
    VDeleterPure<vk::Buffer> uniformBuffer{device, &vk::Device::destroyBuffer};
    VDeleterPure<vk::DeviceMemory> uniformBufferMemory{device, &vk::Device::freeMemory};
    VDeleterPure<vk::DescriptorPool> descriptorPool{device, &vk::Device::destroyDescriptorPool};
    vk::DescriptorSet descriptorSet;

    void createInstance();
    void setupDebugCallback();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffer();
    void createDescriptorPool();
    void createDescriptorSet();
    void createCommandBuffers();
    void createSemaphores();

    void updateUniformBuffer();

    struct QueueFamilyIndices {
        int graphicsFamily = -1;
        int presentFamily = -1;
        bool isComplete() const { return graphicsFamily >= 0 && presentFamily >= 0; }
    };
    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    static std::vector<const char*> getRequiredExtensions();
    bool checkValidationLayerSupport();
    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device);
    bool isDeviceSuitable(const vk::PhysicalDevice& device);
    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device);
    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device);
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> availablePresentModes);
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
    void createShaderModule(const std::vector<char>& bytecode, VDeleterPure<vk::ShaderModule>& shaderModule);
    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, VDeleterPure<vk::Buffer>& buffer, VDeleterPure<vk::DeviceMemory>& bufferMemory);
    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);

    static VkBool32 debugCallback(VkDebugReportFlagsEXT /*flags*/,
                                  VkDebugReportObjectTypeEXT /*objType*/,
                                  uint64_t /*obj*/,
                                  size_t /*location*/,
                                  int32_t /*code*/,
                                  const char* layerPrefix,
                                  const char* msg,
                                  void* /*userData*/);
    static void E_vkCreateDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackCreateInfoEXT* createInfo, vk::AllocationCallbacks* allocator, vk::DebugReportCallbackEXT* callback);
    static void E_DestroyDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackEXT* callback, vk::AllocationCallbacks* allocator);
};

#endif // __DRIVER_HPP__
