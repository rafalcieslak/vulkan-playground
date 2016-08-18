VULKAN_INSTALL_PATH = /opt/vulkan
GLFW_INSTALL_PATH = /opt/glfw-3.2


CXX = clang++
COPTS = -Wall -Wextra -std=c++14 -g

CFLAGS = $(COPTS) -I$(VULKAN_INSTALL_PATH)/include -I$(GLFW_INSTALL_PATH)/include
LDFLAGS = -L$(VULKAN_INSTALL_PATH)/lib -L$(GLFW_INSTALL_PATH) `PKG_CONFIG_PATH=$(GLFW_INSTALL_PATH)/lib/pkgconfig pkg-config --static --libs glfw3` -lvulkan

main: main.cpp
	$(CXX) $(CFLAGS) -o main main.cpp utils.cpp $(LDFLAGS)

reference: reference.cpp
	$(CXX) $(CFLAGS) -o reference reference.cpp $(LDFLAGS)

.PHONY: test clean

test: main
	VK_LAYER_PATH=$(VULKAN_INSTALL_PATH)/etc/explicit_layer.d ./main

gdb: main
	VK_LAYER_PATH=$(VULKAN_INSTALL_PATH)/etc/explicit_layer.d gdb ./main

test-ref: reference
	VK_LAYER_PATH=$(VULKAN_INSTALL_PATH)/etc/explicit_layer.d ./reference

clean:
	rm -f main
