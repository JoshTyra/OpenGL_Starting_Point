#pragma once

#include <boost/statechart/state_machine.hpp>
#include <boost/statechart/simple_state.hpp>
#include <boost/statechart/transition.hpp>
#include <boost/mpl/list.hpp>
#include <chrono>
#include <iostream>

namespace sc = boost::statechart;
namespace mpl = boost::mpl;

// Forward declarations of states
struct Idle;
struct Running;
struct Wandering;

// Events
struct StartRunning : sc::event<StartRunning> {};
struct StopRunning : sc::event<StopRunning> {};
struct StartWandering : sc::event<StartWandering> {};
struct StopWandering : sc::event<StopWandering> {};
struct PathComplete : sc::event<PathComplete> {};

struct AnimationStateMachine : sc::state_machine<AnimationStateMachine, Idle>
{
    float startFrame = 0.0f;
    float endFrame = 58.0f;

    AnimationStateMachine() {
        std::cout << "[DEBUG] Constructing AnimationStateMachine, this=" << this << std::endl;
    }

    ~AnimationStateMachine() {
        std::cout << "[DEBUG] Destroying AnimationStateMachine, this=" << this << std::endl;
    }

    void setAnimationFrames(float start, float end) {
        std::cout << "[DEBUG] Setting animation frames: " << start << " to " << end << ", this=" << this << std::endl;
        startFrame = start;
        endFrame = end;
    }
};

struct Idle : sc::simple_state<Idle, AnimationStateMachine>
{
    typedef sc::transition<StartWandering, Wandering> reactions;

    Idle() {
        std::cout << "[DEBUG] Constructing Idle State, this=" << this << std::endl;
    }

    ~Idle() {
        std::cout << "[DEBUG] Destroying Idle State, this=" << this << std::endl;
    }

    void on_entry() {
        std::cout << "[DEBUG] Entering Idle State, this=" << this << std::endl;
        try {
            AnimationStateMachine& context = this->template context<AnimationStateMachine>();
            std::cout << "[DEBUG] Retrieved context, context=" << &context << std::endl;
            context.setAnimationFrames(0.0f, 58.0f);
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception in Idle on_entry: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[ERROR] Unknown exception in Idle on_entry" << std::endl;
        }
    }
};

struct Wandering : sc::simple_state<Wandering, AnimationStateMachine>
{
    typedef sc::transition<PathComplete, Idle> reactions;

    Wandering() {
        std::cout << "[DEBUG] Constructing Wandering State, this=" << this << std::endl;
    }

    ~Wandering() {
        std::cout << "[DEBUG] Destroying Wandering State, this=" << this << std::endl;
    }

    void on_entry() {
        std::cout << "[DEBUG] Entering Wandering State, this=" << this << std::endl;
        try {
            AnimationStateMachine& context = this->template context<AnimationStateMachine>();
            std::cout << "[DEBUG] Retrieved context, context=" << &context << std::endl;
            context.setAnimationFrames(59.0f, 78.0f);
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception in Wandering on_entry: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[ERROR] Unknown exception in Wandering on_entry" << std::endl;
        }
    }
};

struct Running : sc::simple_state<Running, AnimationStateMachine>
{
    typedef sc::transition<StopRunning, Idle> reactions;

    Running() {
        std::cout << "[DEBUG] Constructing Running State, this=" << this << std::endl;
    }

    ~Running() {
        std::cout << "[DEBUG] Destroying Running State, this=" << this << std::endl;
    }

    void on_entry() {
        std::cout << "[DEBUG] Entering Running State, this=" << this << std::endl;
        try {
            AnimationStateMachine& context = this->template context<AnimationStateMachine>();
            std::cout << "[DEBUG] Retrieved context, context=" << &context << std::endl;
            context.setAnimationFrames(59.0f, 78.0f);
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception in Running on_entry: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[ERROR] Unknown exception in Running on_entry" << std::endl;
        }
    }
};

