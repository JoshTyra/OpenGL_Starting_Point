#pragma once
#include <functional>
#include <unordered_map>
#include <string>
#include <tuple>

class StateMachine {
public:
    using StateFunction = std::function<void(float deltaTime)>;
    using TransitionFunction = std::function<bool()>;
    using KeyframeRange = std::tuple<float, float>; // start frame, end frame

    void addState(const std::string& stateName, StateFunction stateFunction, float startFrame, float endFrame) {
        states[stateName] = stateFunction;
        keyframeRanges[stateName] = std::make_tuple(startFrame, endFrame);
    }

    void addTransition(const std::string& fromState, const std::string& toState, TransitionFunction condition) {
        transitions[fromState][toState] = condition;
    }

    void setInitialState(const std::string& stateName) {
        currentState = stateName;
        updateCurrentKeyframeRange();
    }

    void update(float deltaTime) {
        // Execute current state function
        if (states.find(currentState) != states.end()) {
            states[currentState](deltaTime);
        }

        // Check for transitions
        if (transitions.find(currentState) != transitions.end()) {
            for (const auto& transition : transitions[currentState]) {
                if (transition.second()) {
                    currentState = transition.first;
                    updateCurrentKeyframeRange();
                    break;
                }
            }
        }
    }

    std::string getCurrentState() const {
        return currentState;
    }

    KeyframeRange getCurrentKeyframeRange() const {
        return currentKeyframeRange;
    }

private:
    std::unordered_map<std::string, StateFunction> states;
    std::unordered_map<std::string, std::unordered_map<std::string, TransitionFunction>> transitions;
    std::unordered_map<std::string, KeyframeRange> keyframeRanges;
    std::string currentState;
    KeyframeRange currentKeyframeRange;

    void updateCurrentKeyframeRange() {
        if (keyframeRanges.find(currentState) != keyframeRanges.end()) {
            currentKeyframeRange = keyframeRanges[currentState];
        }
    }
};