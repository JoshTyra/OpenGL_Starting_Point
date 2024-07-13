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
                npc->animation.startFrame = 0.0f;
                npc->animation.endFrame = 58.0f;
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
                npc->animation.startFrame = 59.0f;
                npc->animation.endFrame = 78.0f;
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
        NPC* npc = nullptr;
        if (getInput<NPC*>("npc", npc)) {
            if (npc != nullptr) {
                float currentTime = static_cast<float>(glfwGetTime());
                if (glfwGetKey(glfwGetCurrentContext(), GLFW_KEY_R) == GLFW_PRESS) {
                    if (currentTime - lastToggleTime > toggleDelay) {
                        npc->movement.isRunning = !npc->movement.isRunning;
                        lastToggleTime = currentTime;
                    }
                }
                return npc->movement.isRunning ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
            }
        }
        return BT::NodeStatus::FAILURE;
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