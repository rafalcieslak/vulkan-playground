#define private public
#include "vulkan/vulkan.hpp"

#include <iostream>
#include <stdexcept>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "driver.hpp"

const int WIDTH = 800;
const int HEIGHT = 600;

class Application{
public:
    void run(){
        initWindow();
        driver.init(window);
        mainLoop();
    }

private:
    unsigned int framecount = 0;

    GLFWwindow* window;
    Driver driver;

    void initWindow(){
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetWindowSizeCallback(window, Application::onWindowResized);
    }

    static void onWindowResized(GLFWwindow* window, int width, int height) {
        if (width == 0 || height == 0) return;

        Application* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        app->driver.recreateSwapChain();
    }

    void mainLoop(){
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            driver.updateUniformBuffer();
            driver.drawFrame();
            framecount++;
        }
        driver.device->waitIdle();
        std::cout << "Total frames: " << framecount << std::endl;
    }
};

int main(){
    Application app;

    try{
        app.run();
    }catch (const std::runtime_error& e){
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
