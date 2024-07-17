#include "NPC.h"
#include <algorithm>
#include <random>
#include <glm/gtc/random.hpp>
#include "BehaviorTrees.h"

NPC::NPC(int id, const glm::vec3& startPosition, const glm::mat4& initialTransform)
    : instanceID(id),
    initialTransform(initialTransform),
    color(getRandomColor()) {
    movement.position = startPosition;
    updateModelMatrix();

    // Initialize animation data
    animation.animationTime = 0.0f;
    animation.currentAnimationIndex = 0;
    animation.startFrame = 0.0f;
    animation.endFrame = 58.0f;  // Set this to a valid value based on your animation
    animation.blendFactor = 0.0f;
}

void NPC::update(float deltaTime) {
    updateBehavior(deltaTime);
    movement.updatePosition(deltaTime, stats.speed);
    movement.updateRotation(deltaTime, 2.0f);

    // Update animation time
    animation.animationTime += deltaTime;
    if (animation.animationTime > animation.endFrame) {
        animation.animationTime = animation.startFrame + std::fmod(animation.animationTime - animation.startFrame, animation.endFrame - animation.startFrame);
    }

    updatePathFinding(deltaTime);
    updateModelMatrix();
}

void NPC::setState(NPCState newState) {
    if (currentState != newState) {
        currentState = newState;
        switch (currentState) {
        case NPCState::Idle:
            setAnimation(AnimationType::Idle);
            break;
        case NPCState::Moving:
            setAnimation(AnimationType::Walk);
            break;
            // Add cases for other states as needed
        }
    }
}

void NPC::setAnimationFrames(float start, float end) {
    animation.startFrame = start;
    animation.endFrame = end;
}

void NPC::setRunning(bool running) {
    if (running) {
        setState(NPCState::Moving);
    }
    else {
        setState(NPCState::Idle);
    }
}

void NPC::setDestination(const glm::vec3& destination) {
    currentPath.clear();
    currentPath.push_back(destination);
    currentPathIndex = 0;
    setState(NPCState::Moving);
}

void NPC::stopMoving() noexcept {
    currentPath.clear();
    movement.velocity = glm::vec3(0.0f);
    setState(NPCState::Idle);
}

void NPC::setAnimation(AnimationType type) {
    switch (type) {
    case AnimationType::Idle:
        animation.setAnimation(0, 0.0f, 58.0f);
        break;
    case AnimationType::Walk:
        animation.setAnimation(1, 59.0f, 78.0f);
        break;
        // Add cases for other animation types
    }
}

void NPC::takeDamage(float amount) {
    stats.health = std::max(0.0f, stats.health - amount);
    if (stats.health <= 0) {
        setState(NPCState::Dead);
    }
}

void NPC::attack(NPC* target) {
    if (target && target->isAlive()) {
        target->takeDamage(stats.attackPower);
        setState(NPCState::Attacking);
        setAnimation(AnimationType::Attack);
    }
}

void NPC::interact(NPC* other) {
    setState(NPCState::Interacting);
    // Implement interaction logic
}

void NPC::setupBehaviorTree(BT::Tree tree) {
    behaviorTree = std::move(tree);
    blackboard = BT::Blackboard::create();
    blackboard->set("npc", this);
}

void NPC::updateBehavior(float deltaTime) {
    if (behaviorTree.rootNode()) {
        blackboard->set("deltaTime", deltaTime);
        behaviorTree.tickOnce();
    }
}

void NPC::setPosition(const glm::vec3& newPosition) noexcept {
    movement.position = newPosition;
    updateModelMatrix();
}

void NPC::updateModelMatrix() noexcept {
    modelMatrix = glm::translate(glm::mat4(1.0f), movement.position) *
        movement.rotationMatrix *
        initialTransform;
}

void NPC::updatePathFinding(float deltaTime) {
    if (currentPathIndex < currentPath.size()) {
        glm::vec3 direction = glm::normalize(currentPath[currentPathIndex] - movement.position);
        movement.velocity = direction * stats.speed;

        if (glm::distance(movement.position, currentPath[currentPathIndex]) < 0.1f) {
            ++currentPathIndex;
            if (currentPathIndex >= currentPath.size()) {
                stopMoving();
            }
        }
    }
}

glm::vec3 NPC::getRandomColor() {
    static const std::vector<std::string> colorCodes = {
        "#C13E3E", "#3639C9", "#C9BA36", "#208A20", "#B53C8A",
        "#DF9735", "#744821", "#EB7EC5", "#D2D2D2", "#758550",
        "#707E71", "#01FFFF", "#6493ED", "#C69C6C"
    };
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_int_distribution<> dis(0, colorCodes.size() - 1);

    const std::string& hex = colorCodes[dis(gen)];
    int r = std::stoi(hex.substr(1, 2), nullptr, 16);
    int g = std::stoi(hex.substr(3, 2), nullptr, 16);
    int b = std::stoi(hex.substr(5, 2), nullptr, 16);
    return glm::vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}

// Implement NPCAnimation methods
void NPCAnimation::update(float deltaTime) {
    animationTime += deltaTime;
    if (animationTime > endFrame) {
        animationTime = startFrame + std::fmod(animationTime - startFrame, endFrame - startFrame);
    }
}

void NPCAnimation::setAnimation(int index, float start, float end) {
    currentAnimationIndex = index;
    startFrame = start;
    endFrame = end;
    animationTime = start;
}

// Implement NPCMovement methods
void NPCMovement::updatePosition(float deltaTime, float speed) {
    velocity += acceleration * deltaTime;
    position += velocity * speed * deltaTime;
    acceleration = glm::vec3(0.0f);
}

void NPCMovement::updateRotation(float deltaTime, float rotationSpeed) {
    if (glm::length(velocity) > 0.001f) {
        glm::vec3 direction = glm::normalize(velocity);
        float targetAngle = std::atan2(direction.x, direction.z);
        float angleDiff = targetAngle - rotationAngle;

        // Normalize the angle difference
        angleDiff = std::fmod(angleDiff + glm::pi<float>(), glm::two_pi<float>()) - glm::pi<float>();

        float rotationAmount = rotationSpeed * deltaTime;
        if (std::abs(angleDiff) < rotationAmount) {
            rotationAngle = targetAngle;
        }
        else {
            rotationAngle += (angleDiff > 0 ? rotationAmount : -rotationAmount);
        }

        rotationMatrix = glm::rotate(glm::mat4(1.0f), rotationAngle, glm::vec3(0, 1, 0));
    }
}

// NPCManager implementation
NPCManager::NPCManager(size_t maxNPCs) : maxNPCs(std::min(maxNPCs, MAX_NPCS)) {}

void NPCManager::initializeNPCs(float worldSize, const glm::mat4& originalModelMatrix) {
    this->worldSize = worldSize;
    size_t gridSide = static_cast<size_t>(std::ceil(std::sqrt(maxNPCs)));
    float spacing = worldSize / gridSide;

    for (size_t i = 0; i < maxNPCs; ++i) {
        size_t row = i / gridSide;
        size_t col = i % gridSide;

        float x = -worldSize / 2 + spacing * col + spacing / 2;
        float z = -worldSize / 2 + spacing * row + spacing / 2;

        glm::vec3 position(x, 0.0f, z);
        addNPC(position, originalModelMatrix);
    }
}

void NPCManager::updateNPCs(float deltaTime) {
    for (auto& npc : npcs) {
        npc->update(deltaTime);
    }
    cleanupDeadNPCs();
}

void NPCManager::setupBehaviorTrees(BT::BehaviorTreeFactory& factory) {
    for (auto& npc : npcs) {
        auto tree = factory.createTreeFromText(BT::getMainTreeXML());
        npc->setupBehaviorTree(std::move(tree));
    }
}

void NPCManager::addNPC(const glm::vec3& position, const glm::mat4& initialTransform) {
    if (npcs.size() < maxNPCs) {
        npcs.push_back(std::make_unique<NPC>(npcs.size(), position, initialTransform));
    }
}

void NPCManager::removeNPC(int id) {
    npcs.erase(std::remove_if(npcs.begin(), npcs.end(),
        [id](const auto& npc) { return npc->getID() == id; }),
        npcs.end());
}

NPC* NPCManager::getNPC(int id) {
    auto it = std::find_if(npcs.begin(), npcs.end(),
        [id](const auto& npc) { return npc->getID() == id; });
    return it != npcs.end() ? it->get() : nullptr;
}

void NPCManager::handleNPCInteractions() {
    // Implement NPC interaction logic here
}

void NPCManager::updatePathfinding() {
    // Implement global pathfinding updates here
}

void NPCManager::cleanupDeadNPCs() {
    npcs.erase(std::remove_if(npcs.begin(), npcs.end(),
        [](const auto& npc) { return !npc->isAlive(); }),
        npcs.end());
}