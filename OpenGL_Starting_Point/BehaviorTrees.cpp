// BehaviorTrees.cpp
#include "BehaviorTrees.h"
#include <GLFW/glfw3.h> // For glfwGetTime() and key checks

namespace BT {

    IdleNode::IdleNode(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config) {}

    BT::PortsList IdleNode::providedPorts() {
        return { BT::InputPort<NPC*>("npc") };
    }

    BT::NodeStatus IdleNode::tick() {
        NPC* npc = nullptr;
        if (getInput<NPC*>("npc", npc)) {
            if (npc != nullptr) {
                npc->setAnimation(AnimationType::Idle);
                return BT::NodeStatus::SUCCESS;
            }
        }
        return BT::NodeStatus::FAILURE;
    }

    RunningNode::RunningNode(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config) {}

    BT::PortsList RunningNode::providedPorts() {
        return { BT::InputPort<NPC*>("npc") };
    }

    BT::NodeStatus RunningNode::tick() {
        NPC* npc = nullptr;
        if (getInput<NPC*>("npc", npc)) {
            if (npc != nullptr) {
                npc->setAnimation(AnimationType::Run);
                return BT::NodeStatus::SUCCESS;
            }
        }
        return BT::NodeStatus::FAILURE;
    }

    ShouldRun::ShouldRun(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ConditionNode(name, config) {}

    BT::PortsList ShouldRun::providedPorts() {
        return { BT::InputPort<NPC*>("npc") };
    }

    BT::NodeStatus ShouldRun::tick() {
        auto blackboard = config().blackboard;
        if (!blackboard) {
            std::cerr << "Blackboard does not exist in ShouldRun::tick" << std::endl;
            return BT::NodeStatus::FAILURE;
        }

        NPC* npc = nullptr;
        try {
            npc = blackboard->get<NPC*>("npc");
            if (!npc) {
                std::cerr << "Failed to get NPC from blackboard in ShouldRun::tick" << std::endl;
                return BT::NodeStatus::FAILURE;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception while accessing blackboard in ShouldRun::tick: " << e.what() << std::endl;
            return BT::NodeStatus::FAILURE;
        }

        float currentTime = static_cast<float>(glfwGetTime());
        if (glfwGetKey(glfwGetCurrentContext(), GLFW_KEY_R) == GLFW_PRESS) {
            if (currentTime - lastToggleTime > toggleDelay) {
                npc->setRunning(!npc->isRunning());
                lastToggleTime = currentTime;
            }
        }

        bool isRunning = npc->isRunning();
        return isRunning ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    }

    void registerNodes(BT::BehaviorTreeFactory& factory) {
        factory.registerNodeType<IdleNode>("Idle");
        factory.registerNodeType<RunningNode>("Running");
        factory.registerNodeType<ShouldRun>("ShouldRun");
    }

    const char* getMainTreeXML() {
        return R"(
        <root BTCPP_format="4">
            <BehaviorTree ID="MainTree">
                <Fallback name="root_fallback">
                    <Sequence name="running_sequence">
                        <ShouldRun npc="{npc}"/>
                        <Running npc="{npc}"/>
                    </Sequence>
                    <Sequence name="idle_sequence">
                        <Idle npc="{npc}"/>
                    </Sequence>
                </Fallback>
            </BehaviorTree>
        </root>
        )";
    }
} // namespace BT