#pragma once
#include <chrono>
#include <deque>
#include <numeric>

class FrameTimeTracker {
    std::deque<double> frameTimes;
    std::chrono::high_resolution_clock::time_point lastFrameTime;
    const size_t maxSamples = 60;

public:
    FrameTimeTracker() : lastFrameTime(std::chrono::high_resolution_clock::now()) {}

    void update() {
        auto currentTime = std::chrono::high_resolution_clock::now();
        double deltaTime = std::chrono::duration<double, std::milli>(currentTime - lastFrameTime).count();
        lastFrameTime = currentTime;

        frameTimes.push_back(deltaTime);
        if (frameTimes.size() > maxSamples) {
            frameTimes.pop_front();
        }
    }

    double getAverageFrameTime() const {
        if (frameTimes.empty()) return 0.0;
        return std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0) / frameTimes.size();
    }

    double getMinFrameTime() const {
        if (frameTimes.empty()) return 0.0;
        return *std::min_element(frameTimes.begin(), frameTimes.end());
    }

    double getMaxFrameTime() const {
        if (frameTimes.empty()) return 0.0;
        return *std::max_element(frameTimes.begin(), frameTimes.end());
    }
};