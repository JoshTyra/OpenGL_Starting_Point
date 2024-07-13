// NPC.cpp
#include "NPC.h"
#include <glm/gtc/matrix_transform.hpp>
#include <random>

NPC::NPC(int id, const glm::vec3& startPosition, const glm::mat4& initialTransform)
    : instanceID(id), color(getRandomColor()), initialTransform(initialTransform) {
    movement.position = startPosition;
    modelMatrix = glm::translate(glm::mat4(1.0f), startPosition) * initialTransform;
}

void NPCAnimation::update(float deltaTime) {
    animationTime += deltaTime;
}

void NPCAnimation::setAnimation(int index, float start, float end) {
    currentAnimationIndex = index;
    startFrame = start;
    endFrame = end;
}

void NPCMovement::updatePosition(float deltaTime, float speed) {
    if (currentPathIndex < currentPath.size()) {
        glm::vec3 direction = glm::normalize(currentPath[currentPathIndex] - position);
        position += direction * speed * deltaTime;

        if (glm::distance(position, currentPath[currentPathIndex]) < 0.1f) {
            currentPathIndex++;
        }
    }
}

void NPCMovement::setDestination(const glm::vec3& destination) {
    currentDestination = destination;
    currentPath.clear();
    currentPath.push_back(destination);
    currentPathIndex = 0;
}

void NPCMovement::updateRotation(float deltaTime, float rotationSpeed) {
    if (currentPathIndex < currentPath.size()) {
        glm::vec3 direction = glm::normalize(currentPath[currentPathIndex] - position);
        float targetAngle = atan2(direction.x, direction.z);
        float angleDiff = targetAngle - currentRotationAngle;

        // Normalize the angle difference
        if (angleDiff > glm::pi<float>()) angleDiff -= 2 * glm::pi<float>();
        if (angleDiff < -glm::pi<float>()) angleDiff += 2 * glm::pi<float>();

        float rotationAmount = rotationSpeed * deltaTime;
        if (abs(angleDiff) < rotationAmount) {
            currentRotationAngle = targetAngle;
        }
        else {
            currentRotationAngle += (angleDiff > 0 ? rotationAmount : -rotationAmount);
        }

        currentRotationMatrix = glm::rotate(glm::mat4(1.0f), currentRotationAngle, glm::vec3(0, 1, 0));
    }
}

void NPC::update(float deltaTime) {
    animation.update(deltaTime);
    movement.updatePosition(deltaTime, movement.isRunning ? 5.0f : 2.0f);
    movement.updateRotation(deltaTime, 2.0f);

    // Create the new model matrix
    glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), movement.position);
    glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), movement.currentRotationAngle, glm::vec3(0.0f, 1.0f, 0.0f));

    // Combine the transformations, including the initial transform
    modelMatrix = translationMatrix * rotationMatrix * initialTransform;
}

glm::mat4 NPC::getModelMatrix() const {
    return modelMatrix;
}

NPCManager::NPCManager() {
    npcs.reserve(MAX_NPCS);
}

void NPCManager::initializeNPCs(float worldSize, int numInstances, const glm::mat4& originalModelMatrix) {
    int gridSide = static_cast<int>(std::ceil(std::sqrt(numInstances)));
    float spacing = worldSize / gridSide;

    for (int i = 0; i < numInstances; ++i) {
        int row = i / gridSide;
        int col = i % gridSide;

        float x = -worldSize / 2 + spacing * col + spacing / 2;
        float z = -worldSize / 2 + spacing * row + spacing / 2;

        glm::vec3 position(x, 0.0f, z);

        npcs.emplace_back(i, position, originalModelMatrix);

        // Set initial animation
        npcs.back().animation.currentAnimationIndex = i % 2; // Alternating between 0 and 1
        if (npcs.back().animation.currentAnimationIndex == 0) {
            npcs.back().animation.startFrame = 0.0f;
            npcs.back().animation.endFrame = 58.0f;
        }
        else {
            npcs.back().animation.startFrame = 59.0f;
            npcs.back().animation.endFrame = 78.0f;
        }
    }
}

void NPCManager::updateNPCs(float deltaTime) {
    for (auto& npc : npcs) {
        npc.update(deltaTime);
    }
}

glm::vec3 NPC::getRandomColor() {
    static const std::vector<std::string> colorCodes = {
        "#C13E3E", "#3639C9", "#C9BA36", "#208A20", "#B53C8A",
        "#DF9735", "#744821", "#EB7EC5", "#D2D2D2", "#758550",
        "#707E71", "#01FFFF", "#6493ED", "#C69C6C"
    };
    static std::random_device rd;
    static std::mt19937 engine(rd());
    static std::uniform_int_distribution<int> distribution(0, colorCodes.size() - 1);
    return hexToRGB(colorCodes[distribution(engine)]);
}

glm::vec3 NPC::hexToRGB(const std::string& hex) {
    int r = std::stoi(hex.substr(1, 2), nullptr, 16);
    int g = std::stoi(hex.substr(3, 2), nullptr, 16);
    int b = std::stoi(hex.substr(5, 2), nullptr, 16);
    return glm::vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}