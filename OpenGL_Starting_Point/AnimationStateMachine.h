#pragma once

#include <boost/statechart/state_machine.hpp>
#include <boost/statechart/simple_state.hpp>
#include <boost/statechart/transition.hpp>
#include <boost/mpl/list.hpp>
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

// State machine definition
struct AnimationStateMachine : sc::state_machine<AnimationStateMachine, Idle> {};

// Idle state definition
struct Idle : sc::simple_state<Idle, AnimationStateMachine>
{
    typedef mpl::list<
        sc::transition<StartRunning, Running>,
        sc::transition<StartWandering, Wandering>
    > reactions;

    Idle() { std::cout << "Entering Idle State" << std::endl; }
    ~Idle() { std::cout << "Exiting Idle State" << std::endl; }
};

// Running state definition
struct Running : sc::simple_state<Running, AnimationStateMachine>
{
    typedef mpl::list<
        sc::transition<StopRunning, Idle>,
        sc::transition<StopRunning, Wandering>
    > reactions;

    Running() { std::cout << "Entering Running State" << std::endl; }
    ~Running() { std::cout << "Exiting Running State" << std::endl; }
};

// Wandering state definition
struct Wandering : sc::simple_state<Wandering, AnimationStateMachine>
{
    typedef mpl::list<
        sc::transition<StopWandering, Idle>,
        sc::transition<StartRunning, Running>
    > reactions;

    Wandering() { std::cout << "Entering Wandering State" << std::endl; }
    ~Wandering() { std::cout << "Exiting Wandering State" << std::endl; }
};