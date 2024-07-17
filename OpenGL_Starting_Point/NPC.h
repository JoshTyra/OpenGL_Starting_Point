// NPC.h
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <span>
#include <behaviortree_cpp/bt_factory.h>

enum class NPCState {
    Idle,
    Moving,
    Interacting,
    Attacking,
    Dead
};

enum class AnimationType {
    Idle,
    Walk,
    Run,
    Attack,
    Die,
    Interact
};

struct NPCStats {
    float health{ 100.0f };
    float speed{ 5.0f };
    float attackPower{ 10.0f };
    float defense{ 5.0f };
};

struct NPCAnimation {
    float animationTime{ 0.0f };
    int currentAnimationIndex{ 0 };
    float startFrame{ 0.0f };
    float endFrame{ 0.0f };
    float blendFactor{ 0.0f };

    void update(float deltaTime);
    void setAnimation(int index, float start, float end);
};

struct NPCMovement {
    glm::vec3 position{ 0.0f };
    glm::vec3 velocity{ 0.0f };
    glm::vec3 acceleration{ 0.0f };
    float rotationAngle{ 0.0f };
    glm::mat4 rotationMatrix{ 1.0f };

    void updatePosition(float deltaTime, float speed);
    void updateRotation(float deltaTime, float rotationSpeed);
};

class NPC {
public:
    NPC(int id, const glm::vec3& startPosition, const glm::mat4& initialTransform);
    ~NPC() = default;

    NPC(const NPC&) = delete;
    NPC& operator=(const NPC&) = delete;
    NPC(NPC&&) noexcept = default;
    NPC& operator=(NPC&&) noexcept = default;

    void update(float deltaTime);

    void setState(NPCState newState);
    [[nodiscard]] NPCState getState() const noexcept { return currentState; }

    void setDestination(const glm::vec3& destination);
    void stopMoving() noexcept;
    [[nodiscard]] bool isMoving() const noexcept { return currentState == NPCState::Moving; }

    void setAnimation(AnimationType type);
    void updateAnimation(float deltaTime);
    void setAnimationFrames(float start, float end);
    bool isRunning() const { return currentState == NPCState::Moving; }
    void setRunning(bool running);

    void takeDamage(float amount);
    void attack(NPC* target);
    [[nodiscard]] bool isAlive() const noexcept { return stats.health > 0; }

    void interact(NPC* other);

    // Updated Behavior Tree methods
    void setupBehaviorTree(BT::Tree tree);
    void updateBehavior(float deltaTime);

    [[nodiscard]] int getID() const noexcept { return instanceID; }
    [[nodiscard]] const glm::mat4& getModelMatrix() const noexcept { return modelMatrix; }
    [[nodiscard]] const glm::vec3& getPosition() const noexcept { return movement.position; }
    [[nodiscard]] const glm::vec3& getColor() const noexcept { return color; }
    [[nodiscard]] const NPCAnimation& getAnimation() const noexcept { return animation; }
    [[nodiscard]] const NPCStats& getStats() const noexcept { return stats; }

    void setPosition(const glm::vec3& newPosition) noexcept;
    void setColor(const glm::vec3& newColor) noexcept { color = newColor; }
    void setStats(const NPCStats& newStats) noexcept { stats = newStats; }

    // Add this method to access the blackboard
    [[nodiscard]] BT::Blackboard::Ptr getBlackboard() const noexcept { return blackboard; }

private:
    int instanceID;
    NPCState currentState{ NPCState::Idle };
    NPCAnimation animation;
    NPCMovement movement;
    NPCStats stats;
    glm::mat4 modelMatrix{ 1.0f };
    glm::mat4 initialTransform;
    glm::vec3 color;

    // Updated Behavior Tree members
    BT::Tree behaviorTree;
    BT::Blackboard::Ptr blackboard;

    std::vector<glm::vec3> currentPath;
    size_t currentPathIndex{ 0 };

    void updateModelMatrix() noexcept;
    void updatePathFinding(float deltaTime);
    static glm::vec3 getRandomColor();
};

class NPCManager {
public:
    static constexpr size_t MAX_NPCS = 8;

    explicit NPCManager(size_t maxNPCs = MAX_NPCS);
    ~NPCManager() = default;

    NPCManager(const NPCManager&) = delete;
    NPCManager& operator=(const NPCManager&) = delete;
    NPCManager(NPCManager&&) noexcept = default;
    NPCManager& operator=(NPCManager&&) noexcept = default;

    void initializeNPCs(float worldSize, const glm::mat4& originalModelMatrix);
    void updateNPCs(float deltaTime);
    void setupBehaviorTrees(BT::BehaviorTreeFactory& factory);

    void addNPC(const glm::vec3& position, const glm::mat4& initialTransform);
    void removeNPC(int id);
    [[nodiscard]] NPC* getNPC(int id);
    [[nodiscard]] std::span<const std::unique_ptr<NPC>> getNPCs() const noexcept { return npcs; }

    void handleNPCInteractions();
    void updatePathfinding();

private:
    std::vector<std::unique_ptr<NPC>> npcs;
    size_t maxNPCs;
    float worldSize{ 0.0f };

    void cleanupDeadNPCs();
};