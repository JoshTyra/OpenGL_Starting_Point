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
        //std::cout << "[DEBUG] Constructing AnimationStateMachine, this=" << this << std::endl;
    }

    ~AnimationStateMachine() {
        //std::cout << "[DEBUG] Destroying AnimationStateMachine, this=" << this << std::endl;
    }

    void setAnimationFrames(float start, float end) {
        //std::cout << "[DEBUG] Setting animation frames: " << start << " to " << end << ", this=" << this << std::endl;
        startFrame = start;
        endFrame = end;
    }
};

// Idle state definition
struct Idle : sc::simple_state<Idle, AnimationStateMachine>
{
    Idle() {
        //std::cout << "[DEBUG] Constructing Idle State, this=" << this << std::endl;
    }

    ~Idle() {
        //std::cout << "[DEBUG] Destroying Idle State, this=" << this << std::endl;
    }
};

// Running state definition
struct Running : sc::simple_state<Running, AnimationStateMachine>
{
    typedef sc::transition<StopRunning, Idle> StopRunningTransition;
    typedef mpl::list<StopRunningTransition> reactions;

    Running() {
        std::cout << "Entering Running State" << std::endl;
        context<AnimationStateMachine>().setAnimationFrames(59.0f, 78.0f);
    }

    ~Running() { std::cout << "Exiting Running State" << std::endl; }
};

// Wandering state definition
struct Wandering : sc::simple_state<Wandering, AnimationStateMachine>
{
    typedef sc::transition<PathComplete, Idle> reactions;

    Wandering() {
        std::cout << "Entering Wandering State" << std::endl;
        context<AnimationStateMachine>().setAnimationFrames(59.0f, 78.0f);
    }

    ~Wandering() { std::cout << "Exiting Wandering State" << std::endl; }
};
