// NPC.h
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <vector>
#include <string>

class NPCAnimation {
public:
    float animationTime = 0.0f;
    int currentAnimationIndex = 0;
    float startFrame = 0.0f;
    float endFrame = 58.0f;
    float blendFactor = 0.0f;

    void update(float deltaTime);
    void setAnimation(int index, float start, float end);
};

class NPCMovement {
public:
    glm::vec3 position;
    std::vector<glm::vec3> currentPath;
    int currentPathIndex = 0;
    glm::vec3 currentDestination;
    float currentRotationAngle = 0.0f;
    glm::mat4 currentRotationMatrix = glm::mat4(1.0f);
    bool isRunning = false;

    void updatePosition(float deltaTime, float speed);
    void setDestination(const glm::vec3& destination);
    void updateRotation(float deltaTime, float rotationSpeed);
};

class NPC {
public:
    NPCAnimation animation;
    NPCMovement movement;
    glm::mat4 modelMatrix;
    glm::mat4 initialTransform;
    glm::vec3 color;
    float idleTimer = 0.0f;
    int instanceID;

    NPC(int id, const glm::vec3& startPosition, const glm::mat4& initialTransform);
    void update(float deltaTime);
    glm::mat4 getModelMatrix() const;
    glm::vec3 getColor() const { return color; }
    float getAnimationTime() const { return animation.animationTime; }
    float getStartFrame() const { return animation.startFrame; }
    float getEndFrame() const { return animation.endFrame; }
    int getCurrentAnimationIndex() const { return animation.currentAnimationIndex; }
    float getBlendFactor() const { return animation.blendFactor; }

    static glm::vec3 getRandomColor();

private:
    static glm::vec3 hexToRGB(const std::string& hex);
};

class NPCManager {
public:
    #ifdef _DEBUG
        static const int MAX_NPCS = 8;
    #else
        static const int MAX_NPCS = 100;
    #endif

    NPCManager();
    void initializeNPCs(float worldSize, int numInstances, const glm::mat4& originalModelMatrix);
    void updateNPCs(float deltaTime);
    std::vector<NPC>& getNPCs() { return npcs; }  // Non-const version
    const std::vector<NPC>& getNPCs() const { return npcs; }  // Const version for const contexts

private:
    std::vector<NPC> npcs;
};