#include "AnimatedModel.h"

ModelLoader::ModelLoader()
    : numBones(0), scene(nullptr), currentAnimation(nullptr),
    currentHeadRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)),
    targetHeadRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)),
    headRotationTimer(0.0f),
    headRotationElapsedTime(0.0f),
    headRotationDuration(3.0f),  // Change head rotation duration as needed
    headRotationInProgress(false),
    headPoses({
        glm::vec2(0.0f, 0.0f),  // Neutral pose
        glm::vec2(glm::radians(45.0f), glm::radians(5.0f)),  // Slight right tilt
        glm::vec2(glm::radians(-45.0f), glm::radians(-5.0f)),  // Slight left tilt
        glm::vec2(glm::radians(30.0f), glm::radians(10.0f)),  // Slight upward tilt
        glm::vec2(glm::radians(-30.0f), glm::radians(-10.0f))  // Slight downward tilt
        }) {}

ModelLoader::~ModelLoader() {}

void normalizeWeights(Vertex& vertex) {
    float totalWeight = vertex.Weights[0] + vertex.Weights[1] + vertex.Weights[2] + vertex.Weights[3];
    if (totalWeight > 0.0f) {
        vertex.Weights[0] /= totalWeight;
        vertex.Weights[1] /= totalWeight;
        vertex.Weights[2] /= totalWeight;
        vertex.Weights[3] /= totalWeight;
    }
}

// Function to print bone information
void printBoneInfo(const aiScene* scene) {
    for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
        aiMesh* mesh = scene->mMeshes[i];
        std::cout << "Mesh " << i << " has " << mesh->mNumBones << " bones.\n";
        for (unsigned int j = 0; j < mesh->mNumBones; ++j) {
            aiBone* bone = mesh->mBones[j];
            std::cout << "Bone " << j << ": " << bone->mName.C_Str() << "\n";
        }
    }
}

void ModelLoader::loadModel(const std::string& path) {
    scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
        return;
    }

    std::cout << "Model loaded successfully." << std::endl;

    printBoneInfo(scene);

    if (scene->mNumAnimations > 0) {
        std::cout << "Number of animations: " << scene->mNumAnimations << std::endl;
        for (unsigned int i = 0; i < scene->mNumAnimations; ++i) {
            std::cout << "Animation " << i << " duration: " << scene->mAnimations[i]->mDuration << std::endl;
        }
    }
    else {
        std::cout << "No animations found in the model." << std::endl;
    }

    aggregatedVertices.clear();
    processNode(scene->mRootNode, scene);
    loadedModelAABB = computeAABB(aggregatedVertices);
}

void ModelLoader::updateBoneTransforms(float timeInSeconds, const std::string& animationName, float blendFactor, float startFrame, float endFrame, std::vector<glm::mat4>& outBoneTransforms) {
    if (!scene || animationName.empty()) {
        outBoneTransforms.resize(boneInfo.size(), glm::mat4(1.0f));
        std::cout << "No scene or empty animation name" << std::endl;
        return;
    }

    glm::mat4 identity = glm::mat4(1.0f);
    std::vector<glm::mat4> blendedBoneTransforms(boneInfo.size(), glm::mat4(0.0f));

    auto it = animations.find(animationName);
    if (it == animations.end()) {
        //std::cout << "Animation not found: " << animationName << std::endl;
        outBoneTransforms.resize(boneInfo.size(), glm::mat4(1.0f));
        return;
    }

    const auto& animation = it->second;
    float animationDuration = endFrame - startFrame;
    float localAnimationTime = startFrame + fmod(timeInSeconds * animation.ticksPerSecond, animationDuration);

    std::vector<glm::mat4> currentBoneTransforms(boneInfo.size(), glm::mat4(1.0f));
    readNodeHierarchy(localAnimationTime, scene->mRootNode, identity, animationName, startFrame, endFrame, currentBoneTransforms);

    outBoneTransforms = currentBoneTransforms;  // Directly assign the calculated transforms
}

const std::vector<Mesh>& ModelLoader::getLoadedMeshes() const {
    return loadedMeshes;
}

const AABB& ModelLoader::getLoadedModelAABB() const {
    return loadedModelAABB;
}

void ModelLoader::processNode(aiNode* node, const aiScene* scene) {
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, scene, node->mTransformation);
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}

void ModelLoader::processMesh(aiMesh* mesh, const aiScene* scene, const aiMatrix4x4& nodeTransformation) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // Process vertices
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        aiVector3D transformedPosition = nodeTransformation * mesh->mVertices[i];
        vertex.Position = glm::vec3(transformedPosition.x, transformedPosition.y, transformedPosition.z);
        vertex.Normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        vertex.TexCoord = mesh->mTextureCoords[0] ? glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : glm::vec2(0.0f);
        vertex.Tangent = glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z);
        vertex.Bitangent = glm::vec3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z);
        vertex.BoneIDs = glm::ivec4(0);
        vertex.Weights = glm::vec4(0.0f);
        vertices.push_back(vertex);
    }

    // Process indices
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    // Process bones
    for (unsigned int i = 0; i < mesh->mNumBones; i++) {
        aiBone* bone = mesh->mBones[i];
        int boneIndex = 0;

        if (boneMapping.find(bone->mName.C_Str()) == boneMapping.end()) {
            boneIndex = numBones;
            numBones++;
            BoneInfo bi;
            boneInfo.push_back(bi);
            boneInfo[boneIndex].BoneOffset = glm::transpose(glm::make_mat4(&bone->mOffsetMatrix.a1));
            boneMapping[bone->mName.C_Str()] = boneIndex;
        }
        else {
            boneIndex = boneMapping[bone->mName.C_Str()];
        }

        for (unsigned int j = 0; j < bone->mNumWeights; j++) {
            int vertexID = bone->mWeights[j].mVertexId;
            float weight = bone->mWeights[j].mWeight;

            for (int k = 0; k < 4; ++k) {
                if (vertices[vertexID].Weights[k] == 0.0f) {
                    vertices[vertexID].BoneIDs[k] = boneIndex;
                    vertices[vertexID].Weights[k] = weight;
                    break;
                }
            }
        }
    }

    // Normalize the weights for all vertices
    for (auto& vertex : vertices) {
        normalizeWeights(vertex);
    }

    // Aggregate vertices for AABB computation
    aggregatedVertices.insert(aggregatedVertices.end(), vertices.begin(), vertices.end());

    // Store the processed mesh
    int meshBufferIndex = 0;
    if (mesh->mName.C_Str() == std::string("helmet:visor:LOD0")) {
        meshBufferIndex = 1;
    }
    storeMesh(vertices, indices, meshBufferIndex);
}

void ModelLoader::storeMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices, int meshBufferIndex) {
    Mesh mesh;

    glGenVertexArrays(1, &mesh.VAO);
    glGenBuffers(1, &mesh.VBO);
    glGenBuffers(1, &mesh.EBO);

    glBindVertexArray(mesh.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    // Vertex Positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Position));
    glEnableVertexAttribArray(0);

    // Vertex Texture Coords
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoord));
    glEnableVertexAttribArray(1);

    // Vertex Normals
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    glEnableVertexAttribArray(2);

    // Vertex Tangents
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
    glEnableVertexAttribArray(3);

    // Vertex Bitangents
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));
    glEnableVertexAttribArray(4);

    // Bone IDs
    glVertexAttribIPointer(5, 4, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, BoneIDs));
    glEnableVertexAttribArray(5);

    // Bone Weights
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Weights));
    glEnableVertexAttribArray(6);

    glBindVertexArray(0);

    mesh.meshBufferIndex = meshBufferIndex;
    mesh.indices = indices;
    loadedMeshes.push_back(mesh);
}

void ModelLoader::readNodeHierarchy(float animationTime, const aiNode* node, const glm::mat4& parentTransform, const std::string& animationName, float startFrame, float endFrame, std::vector<glm::mat4>& outBoneTransforms) {
    std::string nodeName(node->mName.data);

    glm::mat4 nodeTransformation = glm::transpose(glm::make_mat4(&node->mTransformation.a1));
    const aiNodeAnim* nodeAnim = findNodeAnim(animations[animationName], nodeName);

    if (nodeAnim) {
        // Ensure animationTime is within the correct range
        animationTime = glm::clamp(animationTime, startFrame, endFrame);

        aiVector3D scaling;
        calcInterpolatedScaling(scaling, animationTime, nodeAnim);
        glm::mat4 scalingM = glm::scale(glm::mat4(1.0f), glm::vec3(scaling.x, scaling.y, scaling.z));

        aiQuaternion rotationQ;
        calcInterpolatedRotation(rotationQ, animationTime, nodeAnim);
        glm::mat4 rotationM = glm::mat4_cast(glm::quat(rotationQ.w, rotationQ.x, rotationQ.y, rotationQ.z));

        aiVector3D translation;
        calcInterpolatedPosition(translation, animationTime, nodeAnim);
        glm::mat4 translationM = glm::translate(glm::mat4(1.0f), glm::vec3(translation.x, translation.y, translation.z));

        nodeTransformation = translationM * rotationM * scalingM;
    }

    glm::mat4 globalTransformation = parentTransform * nodeTransformation;

    if (boneMapping.find(nodeName) != boneMapping.end()) {
        int boneIndex = boneMapping[nodeName];
        if (nodeName == "head") {
            glm::mat4 headRotationM = glm::mat4_cast(currentHeadRotation);
            globalTransformation = globalTransformation * headRotationM;
        }
        outBoneTransforms[boneIndex] = globalTransformation * boneInfo[boneIndex].BoneOffset;
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        readNodeHierarchy(animationTime, node->mChildren[i], globalTransformation, animationName, startFrame, endFrame, outBoneTransforms);
    }
}

const aiNodeAnim* ModelLoader::findNodeAnim(const Animation& animation, const std::string& nodeName) {
    auto it = animation.channels.find(nodeName);
    return it != animation.channels.end() ? it->second : nullptr;
}

void ModelLoader::calcInterpolatedScaling(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim) {
    if (nodeAnim->mNumScalingKeys == 1) {
        out = nodeAnim->mScalingKeys[0].mValue;
        return;
    }

    unsigned int scalingIndex = findScaling(animationTime, nodeAnim);
    unsigned int nextScalingIndex = (scalingIndex + 1);
    assert(nextScalingIndex < nodeAnim->mNumScalingKeys);
    float deltaTime = (float)(nodeAnim->mScalingKeys[nextScalingIndex].mTime - nodeAnim->mScalingKeys[scalingIndex].mTime);
    float factor = (animationTime - (float)nodeAnim->mScalingKeys[scalingIndex].mTime) / deltaTime;
    assert(factor >= 0.0f && factor <= 1.0f);
    const aiVector3D& start = nodeAnim->mScalingKeys[scalingIndex].mValue;
    const aiVector3D& end = nodeAnim->mScalingKeys[nextScalingIndex].mValue;
    aiVector3D delta = end - start;
    out = start + factor * delta;
}

void ModelLoader::calcInterpolatedRotation(aiQuaternion& out, float animationTime, const aiNodeAnim* nodeAnim) {
    if (nodeAnim->mNumRotationKeys == 1) {
        out = nodeAnim->mRotationKeys[0].mValue;
        return;
    }

    unsigned int rotationIndex = findRotation(animationTime, nodeAnim);
    unsigned int nextRotationIndex = (rotationIndex + 1);
    assert(nextRotationIndex < nodeAnim->mNumRotationKeys);
    float deltaTime = (float)(nodeAnim->mRotationKeys[nextRotationIndex].mTime - nodeAnim->mRotationKeys[rotationIndex].mTime);
    float factor = (animationTime - (float)nodeAnim->mRotationKeys[rotationIndex].mTime) / deltaTime;
    assert(factor >= 0.0f && factor <= 1.0f);
    const aiQuaternion& startRotationQ = nodeAnim->mRotationKeys[rotationIndex].mValue;
    const aiQuaternion& endRotationQ = nodeAnim->mRotationKeys[nextRotationIndex].mValue;
    aiQuaternion::Interpolate(out, startRotationQ, endRotationQ, factor);
    out = out.Normalize();
}

void ModelLoader::calcInterpolatedPosition(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim) {
    if (nodeAnim->mNumPositionKeys == 1) {
        out = nodeAnim->mPositionKeys[0].mValue;
        return;
    }

    unsigned int positionIndex = findPosition(animationTime, nodeAnim);
    unsigned int nextPositionIndex = (positionIndex + 1);
    assert(nextPositionIndex < nodeAnim->mNumPositionKeys);
    float deltaTime = (float)(nodeAnim->mPositionKeys[nextPositionIndex].mTime - nodeAnim->mPositionKeys[positionIndex].mTime);
    float factor = (animationTime - (float)nodeAnim->mPositionKeys[positionIndex].mTime) / deltaTime;
    assert(factor >= 0.0f && factor <= 1.0f);
    const aiVector3D& start = nodeAnim->mPositionKeys[positionIndex].mValue;
    const aiVector3D& end = nodeAnim->mPositionKeys[nextPositionIndex].mValue;
    aiVector3D delta = end - start;
    out = start + factor * delta;
}

unsigned int ModelLoader::findScaling(float animationTime, const aiNodeAnim* nodeAnim) {
    for (unsigned int i = 0; i < nodeAnim->mNumScalingKeys - 1; i++) {
        if (animationTime < (float)nodeAnim->mScalingKeys[i + 1].mTime) {
            return i;
        }
    }
    return 0;
}

unsigned int ModelLoader::findRotation(float animationTime, const aiNodeAnim* nodeAnim) {
    for (unsigned int i = 0; i < nodeAnim->mNumRotationKeys - 1; i++) {
        if (animationTime < (float)nodeAnim->mRotationKeys[i + 1].mTime) {
            return i;
        }
    }
    return 0;
}

unsigned int ModelLoader::findPosition(float animationTime, const aiNodeAnim* nodeAnim) {
    for (unsigned int i = 0; i < nodeAnim->mNumPositionKeys - 1; i++) {
        if (animationTime < (float)nodeAnim->mPositionKeys[i + 1].mTime) {
            return i;
        }
    }
    return 0;
}

AABB ModelLoader::computeAABB(const std::vector<Vertex>& vertices) {
    glm::vec3 min = vertices[0].Position;
    glm::vec3 max = vertices[0].Position;

    for (const auto& vertex : vertices) {
        min = glm::min(min, vertex.Position);
        max = glm::max(max, vertex.Position);
    }

    return { min, max };
}

AABB ModelLoader::transformAABB(const AABB& aabb, const glm::mat4& transform) {
    glm::vec3 corners[8] = {
         aabb.min,
         glm::vec3(aabb.min.x, aabb.min.y, aabb.max.z),
         glm::vec3(aabb.min.x, aabb.max.y, aabb.min.z),
         glm::vec3(aabb.min.x, aabb.max.y, aabb.max.z),
         glm::vec3(aabb.max.x, aabb.min.y, aabb.min.z),
         glm::vec3(aabb.max.x, aabb.min.y, aabb.max.z),
         glm::vec3(aabb.max.x, aabb.max.y, aabb.min.z),
         aabb.max
    };

    glm::vec3 newMin = transform * glm::vec4(corners[0], 1.0f);
    glm::vec3 newMax = newMin;

    for (int i = 1; i < 8; ++i) {
        glm::vec3 transformedCorner = transform * glm::vec4(corners[i], 1.0f);
        newMin = glm::min(newMin, transformedCorner);
        newMax = glm::max(newMax, transformedCorner);
    }

    return { newMin, newMax };
}

const std::vector<glm::mat4>& ModelLoader::getBoneTransforms() const {
    return boneTransforms;
}

void ModelLoader::processAnimations() {
    if (!scene || scene->mNumAnimations == 0) {
        std::cout << "No animations found to process." << std::endl;
        return;
    }

    for (unsigned int i = 0; i < scene->mNumAnimations; ++i) {
        const aiAnimation* aiAnim = scene->mAnimations[i];

        Animation anim1(aiAnim->mName.C_Str(), aiAnim->mDuration,
            aiAnim->mTicksPerSecond != 0 ? aiAnim->mTicksPerSecond : 25.0,
            0, 58, 1.0f);
        Animation anim2(aiAnim->mName.C_Str(), aiAnim->mDuration,
            aiAnim->mTicksPerSecond != 0 ? aiAnim->mTicksPerSecond : 25.0,
            59, 78, 1.0f);
        Animation anim3(aiAnim->mName.C_Str(), aiAnim->mDuration,
            aiAnim->mTicksPerSecond != 0 ? aiAnim->mTicksPerSecond : 25.0,
            79, 137, 1.0f);

        for (unsigned int j = 0; j < aiAnim->mNumChannels; ++j) {
            const aiNodeAnim* nodeAnim = aiAnim->mChannels[j];
            anim1.channels[nodeAnim->mNodeName.data] = nodeAnim;
            anim2.channels[nodeAnim->mNodeName.data] = nodeAnim;
            anim3.channels[nodeAnim->mNodeName.data] = nodeAnim;
        }

        animations["combat_sword_idle"] = anim1;
        animations["combat_sword_move_front"] = anim2;
        animations["ui_pr_idle"] = anim3;
    }
}

void ModelLoader::setCurrentAnimation(const std::string& name) {
    auto it = animations.find(name);
    if (it != animations.end()) {
        currentAnimation = &it->second;
    }
    else {
        std::cerr << "Animation " << name << " not found." << std::endl;
        currentAnimation = nullptr; // Set to nullptr if the animation is not found
    }
}

void ModelLoader::updateHeadRotation(float deltaTime, const std::string& animationName, int currentAnimationIndex) {
    if (animationName == "combat_sword_idle" || animationName == "ui_pr_idle") {
        headRotationTimer += deltaTime;

        // Change head pose every 3 seconds
        if (headRotationTimer >= 3.0f && !headRotationInProgress) {
            glm::vec2 targetPose = headPoses[rand() % headPoses.size()];

            glm::quat targetHeadRotationY = glm::angleAxis(targetPose.x, glm::vec3(-1.0f, 0.0f, 0.0f));
            glm::quat targetHeadRotationX = glm::angleAxis(targetPose.y, glm::vec3(1.0f, 0.0f, 0.0f));
            targetHeadRotation = targetHeadRotationY * targetHeadRotationX;

            headRotationElapsedTime = 0.0f;
            headRotationInProgress = true;
            headRotationTimer = 0.0f;  // Reset the timer
        }

        // Update head rotation if in progress
        if (headRotationInProgress) {
            headRotationElapsedTime += deltaTime;
            float t = headRotationElapsedTime / headRotationDuration;
            if (t >= 1.0f) {
                t = 1.0f;
                headRotationInProgress = false;
                currentHeadRotation = targetHeadRotation;
            }
            else {
                currentHeadRotation = glm::slerp(currentHeadRotation, targetHeadRotation, t);
            }
        }
    }
    else {
        // Only set the flag to false, don't reset the rotation
        headRotationInProgress = false;
    }
}

void ModelLoader::setBoneTransformsTBO(GLuint tbo, GLuint tboTexture) {
    boneTransformsTBO = tbo;
    boneTransformsTBOTexture = tboTexture;
}

GLuint ModelLoader::getBoneTransformsTBO() const {
    return boneTransformsTBO;
}

size_t ModelLoader::getNumBones() const {
    return boneInfo.size();
}
