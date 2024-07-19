#pragma once
#ifndef MODELLOADER_H
#define MODELLOADER_H

#include <GL/glew.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <string_view>
#include <optional>
#include <array>
#include <span>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>

struct Vertex {
    glm::vec3 Position;
    glm::vec2 TexCoord;
    glm::vec3 Normal;
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
    glm::ivec2 BoneIDs;  // Changed from ivec4 to ivec2
    glm::vec2 Weights;   // Changed from vec4 to vec2
};

struct BoneInfo {
    glm::mat4 BoneOffset;
    glm::mat4 FinalTransformation;

    BoneInfo() : BoneOffset(1.0f), FinalTransformation(1.0f) {}
};

struct Animation {
    std::string name;
    double duration;
    double ticksPerSecond;
    double startTime; // Start time in ticks
    double endTime;   // End time in ticks
    std::map<std::string, const aiNodeAnim*> channels;
    float blendWeight;

    Animation() : duration(0.0), ticksPerSecond(0.0), startTime(0.0), endTime(0.0), blendWeight(0.0f) {}

    Animation(const std::string& animName, double dur, double ticksPerSec, double start, double end, float weight)
        : name(animName), duration(dur), ticksPerSecond(ticksPerSec), startTime(start), endTime(end), blendWeight(weight) {}
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

struct Mesh {
    unsigned int VAO, VBO, EBO;
    std::vector<unsigned int> indices;
    int meshBufferIndex;
};

class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    void loadModel(const std::string& path);
    void updateBoneTransforms(float timeInSeconds, std::string_view animationName, float blendFactor, float startFrame, float endFrame, std::vector<glm::mat4>& outBoneTransforms);
    void setCurrentAnimation(std::string_view name);
    [[nodiscard]] const std::vector<Mesh>& getLoadedMeshes() const noexcept;
    [[nodiscard]] const AABB& getLoadedModelAABB() const noexcept;
    const std::vector<glm::mat4>& getBoneTransforms() const;
    void processAnimations();
    void updateHeadRotation(float deltaTime, const std::string& animationName, int currentAnimationIndex);
    void setBoneTransformsTBO(GLuint tbo, GLuint tboTexture);
    [[nodiscard]] GLuint getBoneTransformsTBO() const noexcept;
    [[nodiscard]] size_t getNumBones() const noexcept;

    GLuint boneTransformsTBO;
    GLuint boneTransformsTBOTexture;

private:
    void processNode(aiNode* node, const aiScene* scene);
    void processMesh(aiMesh* mesh, const aiScene* scene, const aiMatrix4x4& nodeTransformation);
    void storeMesh(std::span<const Vertex> vertices, std::span<const unsigned int> indices, int meshBufferIndex);
    void readNodeHierarchy(float animationTime, const aiNode* node, const glm::mat4& parentTransform, std::string_view animationName, float startFrame, float endFrame, std::vector<glm::mat4>& outBoneTransforms);
    const aiNodeAnim* findNodeAnim(const Animation& animation, const std::string& nodeName);
    void calcInterpolatedScaling(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim);
    void calcInterpolatedRotation(aiQuaternion& out, float animationTime, const aiNodeAnim* nodeAnim);
    void calcInterpolatedPosition(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim);
    unsigned int findScaling(float animationTime, const aiNodeAnim* nodeAnim);
    unsigned int findRotation(float animationTime, const aiNodeAnim* nodeAnim);
    unsigned int findPosition(float animationTime, const aiNodeAnim* nodeAnim);
    AABB computeAABB(const std::vector<Vertex>& vertices);
    AABB transformAABB(const AABB& aabb, const glm::mat4& transform);
    glm::quat fastSlerp(const glm::quat& start, const glm::quat& end, float t);

    std::vector<Vertex> aggregatedVertices;
    std::vector<Mesh> loadedMeshes;
    AABB loadedModelAABB;
    std::unordered_map<std::string, int> boneMapping;
    std::vector<BoneInfo> boneInfo;
    int numBones;
    std::vector<glm::mat4> boneTransforms;
    const aiScene* scene;
    Assimp::Importer importer;
    std::unordered_map<std::string, Animation> animations;
    std::optional<Animation*> currentAnimation;

    glm::quat currentHeadRotation;
    glm::quat targetHeadRotation;
    float headRotationTimer;
    float headRotationElapsedTime;
    float headRotationDuration;
    bool headRotationInProgress;
    std::array<glm::vec2, 5> headPoses;
    static constexpr float HEAD_ROTATION_DURATION = 3.0f;
};

#endif // MODELLOADER_H