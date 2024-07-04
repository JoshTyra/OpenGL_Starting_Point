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
    }

    ~AnimationStateMachine() {
    }

    void setAnimationFrames(float start, float end) {
        startFrame = start;
        endFrame = end;
    }
};

struct Idle : sc::simple_state<Idle, AnimationStateMachine>
{
    typedef sc::transition<StartWandering, Wandering> reactions;

    Idle() {
    }

    ~Idle() {
    }

    void on_entry() {
        AnimationStateMachine& context = this->template context<AnimationStateMachine>();
        context.setAnimationFrames(0.0f, 58.0f);
    }
};

struct Wandering : sc::simple_state<Wandering, AnimationStateMachine>
{
    typedef sc::transition<PathComplete, Idle> reactions;

    Wandering() {
    }

    ~Wandering() {
    }

    void on_entry() {
        AnimationStateMachine& context = this->template context<AnimationStateMachine>();
        context.setAnimationFrames(59.0f, 78.0f);
    }
};

struct Running : sc::simple_state<Running, AnimationStateMachine>
{
    typedef sc::transition<StopRunning, Idle> reactions;

    Running() {
    }

    ~Running() {
    }

    void on_entry() {
        AnimationStateMachine& context = this->template context<AnimationStateMachine>();
        std::cout << "[DEBUG] Retrieved context, context=" << &context << std::endl;
        context.setAnimationFrames(59.0f, 78.0f);
    }
};

