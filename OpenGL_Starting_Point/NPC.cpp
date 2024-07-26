#include "NPC.h"
#include <algorithm>
#include <random>
#include <glm/gtc/random.hpp>
#include "BehaviorTrees.h"

NPC::NPC(int id, const glm::vec3& startPosition, const glm::mat4& initialTransform, PhysicsWorld& physicsWorld)
    :uniqueID(id),
    initialTransform(initialTransform),
    color(getRandomColor()),
    physicsWorld(physicsWorld),
    rotationAngle(0.0f),
    currentState(NPCState::Idle),
    modelMatrix(1.0f),
    physicsBodyIndex(-1)
{
    movement.position = startPosition;
    updateModelMatrix();

    // Initialize animation data
    animation.animationTime = 0.0f;
    animation.currentAnimationIndex = 0;
    animation.startFrame = 0.0f;
    animation.endFrame = 58.0f;
    animation.blendFactor = 0.0f;

    // Initialize stats with default values
    stats = NPCStats{};
}

void NPC::update(float deltaTime) {
    // Update behavior tree
    updateBehavior(deltaTime);

    // Get the physics transform
    glm::mat4 physicsTransform = physicsWorld.getTransform(physicsBodyIndex);

    // Update position and rotation based on physics
    btRigidBody* body = physicsWorld.getRigidBody(physicsBodyIndex);
    if (body) {
        btTransform transform = body->getWorldTransform();
        btVector3 up = transform.getBasis() * btVector3(0, 1, 0);

        // If the character is too tilted, apply a torque to bring it upright
        if (up.dot(btVector3(0, 1, 0)) < 0.8f) {
            btVector3 uprightTorque = up.cross(btVector3(0, 1, 0)) * 10.0f;
            body->applyTorque(uprightTorque);
        }

        // Update rotation angle based on physics body's rotation around Y-axis
        btScalar yaw, pitch, roll;
        transform.getRotation().getEulerZYX(yaw, pitch, roll);
        rotationAngle = yaw;
    }

    // Update animation time
    animation.animationTime += deltaTime;
    if (animation.animationTime > animation.endFrame) {
        animation.animationTime = animation.startFrame + std::fmod(animation.animationTime - animation.startFrame, animation.endFrame - animation.startFrame);
    }

    // Update blend factor
    const float blendDuration = 0.2f;
    if (animation.blendFactor < 1.0f) {
        animation.blendFactor = std::min(animation.blendFactor + deltaTime / blendDuration, 1.0f);
    }

    // Update pathfinding
    updatePathFinding(deltaTime);

    // Update visual representation based on physics
    updateFromPhysics(physicsTransform);
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
    int newAnimationIndex = -1;
    float newStartFrame = 0.0f, newEndFrame = 0.0f;

    switch (type) {
    case AnimationType::Idle:
        newAnimationIndex = 0;
        newStartFrame = 0.0f;
        newEndFrame = 58.0f;
        break;
    case AnimationType::Walk:
    case AnimationType::Run:
        newAnimationIndex = 1;
        newStartFrame = 59.0f;
        newEndFrame = 78.0f;
        break;
    case AnimationType::Attack:
        // Add appropriate values for Attack animation
        break;
    case AnimationType::Die:
        // Add appropriate values for Die animation
        break;
    case AnimationType::Interact:
        // Add appropriate values for Interact animation
        break;
    default:
        std::cerr << "Unknown animation type encountered: " << static_cast<int>(type) << std::endl;
        return;
    }

    // Check if a valid animation index was set
    if (newAnimationIndex != -1 && newAnimationIndex != animation.currentAnimationIndex) {
        animation.blendFactor = 0.0f;
        animation.currentAnimationIndex = newAnimationIndex;
        animation.startFrame = newStartFrame;
        animation.endFrame = newEndFrame;
        animation.animationTime = newStartFrame; // Reset animation time to start of new animation
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

    // Use the tree's root blackboard instead of creating a new one
    blackboard = behaviorTree.rootBlackboard();

    // Set the NPC pointer in the blackboard
    blackboard->set("npc", this);
}

void NPC::updateBehavior(float deltaTime) {
    if (behaviorTree.rootNode()) {
        blackboard->set("deltaTime", deltaTime);
        auto status = behaviorTree.tickOnce();
    }
    else {
        std::cerr << "No root node for NPC " << getUniqueID() << std::endl;
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

void NPC::applyForce(const glm::vec3& force) {
    physicsWorld.applyForce(physicsBodyIndex, force);
}

void NPC::applyImpulse(const glm::vec3& impulse) {
    btRigidBody* body = physicsWorld.getRigidBody(physicsBodyIndex);
    if (body) {
        body->activate(true);
        btVector3 btImpulse(impulse.x, impulse.y, impulse.z);
        body->applyCentralImpulse(btImpulse);

        std::cout << "Applied impulse to NPC with ID: " << uniqueID << std::endl;
        std::cout << "Applied impulse: " << btImpulse.x() << ", " << btImpulse.y() << ", " << btImpulse.z() << std::endl;
        std::cout << "Body mass: " << 1.0f / body->getInvMass() << std::endl;
        std::cout << "Body velocity after impulse: "
            << body->getLinearVelocity().x() << ", "
            << body->getLinearVelocity().y() << ", "
            << body->getLinearVelocity().z() << std::endl;
        std::cout << "Body is active: " << (body->isActive() ? "Yes" : "No") << std::endl;
    }
    else {
        std::cerr << "Failed to apply impulse to NPC with ID: " << uniqueID
            << ". Rigid body not found. Physics body index: " << physicsBodyIndex << std::endl;
    }
}

void NPC::setPhysicsBodyIndex(int index) {
    if (index >= 0) {
        physicsBodyIndex = index;
        btRigidBody* body = physicsWorld.getRigidBody(physicsBodyIndex);
        if (body) {
            body->setUserIndex(uniqueID);
        }
        else {
            std::cerr << "Warning: No rigid body found for physics body index " << index << std::endl;
        }
    }
    else {
        std::cerr << "Invalid physics body index: " << index << std::endl;
    }
}

int NPC::getPhysicsBodyIndex() const {
    return physicsBodyIndex;
}

void NPC::updateFromPhysics(const glm::mat4& physicsTransform) {
    const float interpolationFactor = 0.3f; // Adjust for smoother or more responsive movement

    glm::vec3 newPosition = glm::vec3(physicsTransform[3]);
    glm::vec3 currentPosition = glm::vec3(modelMatrix[3]);

    glm::vec3 interpolatedPosition = glm::mix(currentPosition, newPosition, interpolationFactor);

    glm::mat4 uprightRotation = glm::rotate(glm::mat4(1.0f), rotationAngle, glm::vec3(0, 1, 0));
    modelMatrix = glm::translate(glm::mat4(1.0f), interpolatedPosition) * uprightRotation * initialTransform;

    // Update movement struct if needed
    movement.position = interpolatedPosition;
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
NPCManager::NPCManager(size_t maxNPCs, PhysicsWorld& physicsWorld)
    : maxNPCs(std::min(maxNPCs, MAX_NPCS)), physicsWorld(physicsWorld) {}

void NPCManager::initializeNPCs(float worldSize, const glm::mat4& originalModelMatrix) {
    this->worldSize = worldSize;

    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-worldSize / 2, worldSize / 2);

    for (size_t i = 0; i < maxNPCs; ++i) {
        // Generate random x and z coordinates within the world bounds
        float x = dis(gen);
        float z = dis(gen);

        glm::vec3 position(x, 0.0f, z);
        addNPC(position, originalModelMatrix);
    }
}

void NPCManager::updateNPCs(float deltaTime) {
    for (auto& [id, npc] : npcs) {
        glm::mat4 physicsTransform = physicsWorld.getTransform(npc->getPhysicsBodyIndex());
        npc->updateFromPhysics(physicsTransform);
        npc->update(deltaTime);
    }
    cleanupDeadNPCs();
}

void NPCManager::setupBehaviorTrees(BT::BehaviorTreeFactory& factory) {
    for (auto& [id, npc] : npcs) {
        auto tree = factory.createTreeFromText(BT::getMainTreeXML());
        npc->setupBehaviorTree(std::move(tree));
    }
}

void NPCManager::addNPC(const glm::vec3& position, const glm::mat4& initialTransform) {
    if (npcs.size() >= maxNPCs) {
        std::cerr << "Warning: Maximum number of NPCs reached. Cannot add more." << std::endl;
        return;
    }

    int uniqueID = nextUniqueID++;

    // Capsule parameters
    float radius = 0.45f;
    float height = 0.75f;
    float mass = 10.0f;
    float yOffset = 0.85f;

    int physicsBodyIndex = physicsWorld.addCapsuleRigidBody(position, radius, height, mass, yOffset, uniqueID);

    try {
        auto npc = std::make_unique<NPC>(uniqueID, position, initialTransform, physicsWorld);
        npc->setPhysicsBodyIndex(physicsBodyIndex);

        BT::BehaviorTreeFactory factory;
        registerNodes(factory);

        auto tree = factory.createTreeFromText(BT::getMainTreeXML());
        if (tree.rootNode()) {
            npc->setupBehaviorTree(std::move(tree));
            npcs[uniqueID] = std::move(npc);

            // Set the user index of the rigid body to the NPC's unique ID
            btRigidBody* body = physicsWorld.getRigidBody(physicsBodyIndex);
            if (body) {
                body->setUserIndex(uniqueID);
            }
        }
        else {
            std::cerr << "Error: Failed to create a valid behavior tree for NPC ID: " << npc->getUniqueID() << std::endl;
            physicsWorld.removeRigidBody(physicsBodyIndex);
            return;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception when creating NPC or behavior tree: " << e.what() << std::endl;
        physicsWorld.removeRigidBody(physicsBodyIndex);
        return;
    }
}

void NPCManager::removeNPC(int id) {
    auto it = npcs.find(id);
    if (it != npcs.end()) {
        int physicsBodyIndex = it->second->getPhysicsBodyIndex();
        physicsWorld.removeRigidBody(physicsBodyIndex);
        npcs.erase(it);

        // Update physics body indices for remaining NPCs
        for (auto& [npcId, npc] : npcs) {
            int currentIndex = npc->getPhysicsBodyIndex();
            if (currentIndex > physicsBodyIndex) {
                npc->setPhysicsBodyIndex(currentIndex - 1);
            }
        }
    }
}

NPC* NPCManager::getNPC(int id) {
    auto it = npcs.find(id);
    return it != npcs.end() ? it->second.get() : nullptr;
}

void NPCManager::handleNPCInteractions() {
    // Implement NPC interaction logic here
}

void NPCManager::updatePathfinding() {
    // Implement global pathfinding updates here
}

void NPCManager::cleanupDeadNPCs() {
    for (auto it = npcs.begin(); it != npcs.end(); ) {
        if (!it->second->isAlive()) {
            physicsWorld.removeRigidBody(it->second->getPhysicsBodyIndex());
            it = npcs.erase(it);
        }
        else {
            ++it;
        }
    }
}

void NPCManager::checkAndRemoveFallenNPCs(float threshold) {
    std::vector<int> npcToRemove;
    for (const auto& [id, npc] : npcs) {
        glm::vec3 position = npc->getPosition();
        if (position.y < threshold) {
            std::cout << "Marking NPC with ID " << id << " for removal (position: " << position.y << ")" << std::endl;
            npcToRemove.push_back(id);
        }
    }
    for (int id : npcToRemove) {
        removeNPC(id);
    }
}

void NPCManager::debugPrintNPCs() const {
    std::cout << "Current NPCs:" << std::endl;
    for (const auto& [id, npc] : npcs) {
        std::cout << "NPC ID: " << id
            << ", Physics Body Index: " << npc->getPhysicsBodyIndex()
            << ", Position: " << glm::to_string(npc->getPosition())
            << std::endl;
    }
}