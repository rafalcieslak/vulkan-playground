#ifndef __DELETER_HPP__
#define __DELETER_HPP__

#include <functional>
#include "vulkan/vulkan.hpp"

template <typename T>
class VDeleter {
public:
    VDeleter(){}

    ~VDeleter() {
        cleanup();
    }

    T* operator &() {
        cleanup();
        return &object;
    }

    void operator=(T&& other){
        //std::swap(object,other);
        cleanup();
        object = other;
    }

    inline operator T() const {
        return object;
    }

    inline T* operator->(){
        return &object;
    }

    T& operator()(){
        return object;
    }

    const T& operator()() const{
        return object;
    }

private:
    T object{VK_NULL_HANDLE};

    void cleanup() {
        if (object) {
            object.destroy();
        }
        object = T();
    }
};


template <typename T>
class VDeleterPure {
public:
    typedef void (vk::Device::*vkDeviceMemDeleterFun)(T, const vk::AllocationCallbacks*) const;
    typedef void (vk::Instance::*vkInstanceMemDeleterFun)(T, const vk::AllocationCallbacks*) const;

    VDeleterPure(){}

    VDeleterPure(std::function<void(T, VkAllocationCallbacks*)> deletef) {
        this->deleter = [=](T obj) {
            deletef(obj, nullptr);
        };
    }

    VDeleterPure(const VDeleter<VkInstance>& instance, std::function<void(VkInstance, T, VkAllocationCallbacks*)> deletef) {
        this->deleter = [&instance, deletef](T obj) {
            deletef(instance, obj, nullptr);
        };
    }

    VDeleterPure(VDeleter<vk::Instance>& instance, std::function<void(vk::Instance&, T*, vk::AllocationCallbacks*)> deletef) {
        this->deleter = [&instance, deletef](T obj) {
            deletef(instance(), &obj, nullptr);
        };
    }



    VDeleterPure(VDeleter<vk::Instance>& instance, vkInstanceMemDeleterFun mfptr){
        this->deleter = [&instance, mfptr](T obj){
            ((&instance())->*mfptr)(obj, nullptr);
        };
    }
    VDeleterPure(VDeleter<vk::Device>& device, vkDeviceMemDeleterFun mfptr){
        this->deleter = [&device, mfptr](T obj){
            ((&device())->*mfptr)(obj, nullptr);
        };
    }


    ~VDeleterPure() {
        cleanup();
    }

    T* operator &() {
        cleanup();
        return &object;
    }

    void operator=(T&& other){
        //object = std::move(other);
        cleanup();
        object = other;
    }

    inline operator T() const {
        return object;
    }

    const T& operator()() const{
        return object;
    }

    T& operator()(){
        return object;
    }

private:
    T object{VK_NULL_HANDLE};
    std::function<void(T)> deleter;

    void cleanup() {
        if (object) {
            deleter(object);
        }
        object = T();
    }
};





#endif
