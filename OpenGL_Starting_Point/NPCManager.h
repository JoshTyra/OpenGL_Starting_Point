// NPCManager.h
#pragma once
#include "NPC.h"
#include "PhysicsWorld.h"
#include <vector>
#include <memory>
#include "BehaviorTrees.h"
#include <random>
#include <chrono>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <behaviortree_cpp/bt_factory.h>

class NPCManager {
public:
    NPCManager(size_t maxNPCs, PhysicsWorld& physicsWorld);
    ~NPCManager();

    void update(float deltaTime);
    int addNPC(const glm::vec3& position);
    void removeNPC(int id);
    NPC* getNPC(int id);
    const std::vector<std::unique_ptr<NPC>>& getNPCs() const { return m_npcs; }
    void initializeNPCs(float worldSize, const glm::mat4& originalModelMatrix);
    void setupBehaviorTrees(BT::BehaviorTreeFactory& factory);
    void checkAndRemoveFallenNPCs(float threshold);
    void updateNPCs(float deltaTime);
    void debugPrintNPCs() const;
    static const size_t MAX_NPCS = 8;

private:
    std::vector<std::unique_ptr<NPC>> m_npcs;
    PhysicsWorld& m_physicsWorld;
    size_t m_maxNPCs;
    int m_nextID;
};
