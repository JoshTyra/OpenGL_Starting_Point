// BehaviorTrees.h
#pragma once
#include <behaviortree_cpp/bt_factory.h>
#include <string>
#include "NPC.h" // Assuming you have an NPC class defined

namespace BT {

    class IdleNode : public BT::SyncActionNode {
    public:
        IdleNode(const std::string& name, const BT::NodeConfiguration& config);
        static BT::PortsList providedPorts();
        BT::NodeStatus tick() override;
    };

    class RunningNode : public BT::SyncActionNode {
    public:
        RunningNode(const std::string& name, const BT::NodeConfiguration& config);
        static BT::PortsList providedPorts();
        BT::NodeStatus tick() override;
    };

    class ShouldRun : public BT::ConditionNode {
    public:
        ShouldRun(const std::string& name, const BT::NodeConfiguration& config);
        static BT::PortsList providedPorts();
        BT::NodeStatus tick() override;

    private:
        float lastToggleTime = 0.0f;
        const float toggleDelay = 0.3f;
    };

    void registerNodes(BT::BehaviorTreeFactory& factory);
    const char* getMainTreeXML();

} // namespace BT