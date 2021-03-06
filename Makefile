VULKAN_INSTALL_PATH = /opt/vulkan
GLFW_INSTALL_PATH = /opt/glfw-3.2


CXX = clang++
COPTS = -Wall -Wextra -std=c++14
COPT = -O3
CDBG = -g

CFLAGS = $(COPTS) -I$(VULKAN_INSTALL_PATH)/include -I$(GLFW_INSTALL_PATH)/include
LDFLAGS = -L$(VULKAN_INSTALL_PATH)/lib -L$(GLFW_INSTALL_PATH) `PKG_CONFIG_PATH=$(GLFW_INSTALL_PATH)/lib/pkgconfig pkg-config --static --libs glfw3` -lvulkan

main: main.cpp utils.cpp driver.cpp shaders/vert.spv shaders/frag.spv
	$(CXX) $(CFLAGS) $(COPT) -o main main.cpp utils.cpp driver.cpp $(LDFLAGS)

main-dbg: main.cpp utils.cpp driver.cpp
	$(CXX) $(CFLAGS) $(CDBG) -o main-dbg main.cpp utils.cpp driver.cpp $(LDFLAGS)

shaders/vert.spv: shaders/basic.vert
	$(VULKAN_INSTALL_PATH)/bin/glslangValidator -V shaders/basic.vert -o shaders/vert.spv

shaders/frag.spv: shaders/basic.frag
	$(VULKAN_INSTALL_PATH)/bin/glslangValidator -V shaders/basic.frag -o shaders/frag.spv

reference: reference.cpp
	$(CXX) $(CFLAGS) -o reference reference.cpp $(LDFLAGS)

.PHONY: test clean

test: main
	VK_LAYER_PATH=$(VULKAN_INSTALL_PATH)/etc/explicit_layer.d ./main

gdb: main-dbg
	VK_LAYER_PATH=$(VULKAN_INSTALL_PATH)/etc/explicit_layer.d gdb ./main-dbg

test-ref: reference
	VK_LAYER_PATH=$(VULKAN_INSTALL_PATH)/etc/explicit_layer.d ./reference

clean:
	rm -f main reference main-gdb
