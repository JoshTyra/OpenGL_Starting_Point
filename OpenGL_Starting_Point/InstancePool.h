#pragma once
#include <array>
#include <vector>
#include <optional>
#include <concepts>
#include <algorithm>

template<typename T, size_t MaxInstances>
class InstancePool {
public:
    struct InstanceData {
        std::optional<T> data;
        size_t index;
        bool active;

        InstanceData() : index(0), active(false) {}
        InstanceData(const InstanceData&) = delete;
        InstanceData& operator=(const InstanceData&) = delete;
        InstanceData(InstanceData&&) noexcept = default;
        InstanceData& operator=(InstanceData&&) noexcept = default;
    };

private:
    std::array<InstanceData, MaxInstances> instances;
    std::vector<size_t> freeIndices;
    size_t activeCount;

public:
    InstancePool() : activeCount(0) {
        for (size_t i = 0; i < MaxInstances; ++i) {
            instances[i].index = i;
            instances[i].active = false;
            freeIndices.push_back(i);
        }
    }

    template<typename... Args>
        requires std::constructible_from<T, Args...>
    std::optional<size_t> addInstance(Args&&... args) {
        if (freeIndices.empty()) {
            return std::nullopt;
        }

        size_t index = freeIndices.back();
        freeIndices.pop_back();

        instances[index].data.emplace(std::forward<Args>(args)...);
        instances[index].active = true;
        ++activeCount;

        return index;
    }

    void removeInstance(size_t index) {
        if (index < MaxInstances && instances[index].active) {
            instances[index].data.reset();  // This will call the destructor of T
            instances[index].active = false;
            freeIndices.push_back(index);
            --activeCount;
        }
    }

    T* getInstance(size_t index) {
        if (index < MaxInstances && instances[index].active) {
            return &(*instances[index].data);
        }
        return nullptr;
    }

    const T* getInstance(size_t index) const {
        if (index < MaxInstances && instances[index].active) {
            return &(*instances[index].data);
        }
        return nullptr;
    }

    size_t getActiveCount() const {
        return activeCount;
    }

    bool isActive(size_t index) const {
        return index < MaxInstances && instances[index].active;
    }

    void moveInstance(size_t fromIndex, size_t toIndex) {
        if (isActive(fromIndex) && !isActive(toIndex)) {
            if (instances[fromIndex].data.has_value()) {
                instances[toIndex].data.emplace(std::move(*instances[fromIndex].data));
                instances[fromIndex].data.reset();
            }
            instances[toIndex].index = toIndex;
            instances[toIndex].active = true;
            instances[fromIndex].active = false;
        }
    }

    template<std::invocable<T&> F>
    void foreach(F&& func) {
        for (auto& instance : instances) {
            if (instance.active) {
                func(*instance.data);
            }
        }
    }

    template<std::invocable<T&, size_t> F>
    void foreach(F&& func) {
        for (size_t i = 0; i < MaxInstances; ++i) {
            if (instances[i].active) {
                func(*instances[i].data, i);
            }
        }
    }

    template<std::invocable<const T&> F>
    void foreachConst(F&& func) const {
        for (const auto& instance : instances) {
            if (instance.active) {
                func(*instance.data);
            }
        }
    }
};