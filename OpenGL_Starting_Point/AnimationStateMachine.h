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

// State machine definition
struct AnimationStateMachine : sc::state_machine<AnimationStateMachine, Idle> {};

// Idle state definition
struct Idle : sc::simple_state<Idle, AnimationStateMachine>
{
    typedef sc::transition<StartWandering, Wandering> reactions;
};

// Running state definition
struct Running : sc::simple_state<Running, AnimationStateMachine>
{
    typedef sc::transition<StopRunning, Idle> StopRunningTransition;
    typedef mpl::list<StopRunningTransition> reactions;

    Running() { std::cout << "Entering Running State" << std::endl; }
    ~Running() { std::cout << "Exiting Running State" << std::endl; }
};

// Wandering state definition
struct Wandering : sc::simple_state<Wandering, AnimationStateMachine>
{
    typedef sc::transition<PathComplete, Idle> reactions;
};