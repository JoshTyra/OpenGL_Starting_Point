#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"



// Include necessary headers

#include <iostream>

#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <glm/gtx/string_cast.hpp>

#include <random>

#include "Rendering/Frustum.h"

#include "Camera.h"

#include "FileSystemUtils.h"

#include <assimp/Importer.hpp>

#include <assimp/scene.h>

#include <assimp/postprocess.h>

#include <random>

#include <chrono>

#include <map>



// Constants and global variables

const int WIDTH = 2560;

const int HEIGHT = 1080;

float lastX = WIDTH / 2.0f;

float lastY = HEIGHT / 2.0f;

bool firstMouse = true;

float deltaTime = 0.0f; // Time between current frame and last frame

float lastFrame = 0.0f; // Time of last frame



Camera camera(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f, 6.0f, 0.1f, 45.0f);

Frustum frustum;

glm::mat4 projectionMatrix;

glm::mat4 viewMatrix;



struct Vertex {

    glm::vec3 Position;

    glm::vec2 TexCoord;

    glm::vec3 Normal;

    glm::vec3 Tangent;

    glm::vec3 Bitangent;

    glm::ivec4 BoneIDs;

    glm::vec4 Weights;

};



struct BoneInfo {

    glm::mat4 BoneOffset;

    glm::mat4 FinalTransformation;



    BoneInfo() : BoneOffset(1.0f), FinalTransformation(1.0f) {}

};



std::map<std::string, int> boneMapping; // maps a bone name to its index

std::vector<BoneInfo> boneInfo;

int numBones = 0;

std::vector<glm::mat4> boneTransforms;

const aiScene* scene = nullptr;

Assimp::Importer importer;



struct AABB {

    glm::vec3 min;

    glm::vec3 max;

};



struct Mesh {

    unsigned int VAO, VBO, EBO;

    std::vector<unsigned int> indices;

};



unsigned int characterShaderProgram;

unsigned int characterTexture;

unsigned int characterNormalMap;

unsigned int cubemapTexture;

unsigned int characterMaskTexture;

unsigned int waterCubeMapTexture;

unsigned int characterInstanceVBO;

unsigned int characterInstanceColorVBO;



const glm::vec3 staticNodeRotationAxis(1.0f, 0.0f, 0.0f);

const float staticNodeRotationAngle = glm::radians(-90.0f);

float characterRotationSpeed = 0.5f; // Adjust the rotation speed as desired

std::vector<glm::vec3> characterPositions;

std::vector<glm::vec3> characterInstanceColors;



std::vector<Vertex> aggregatedVertices; // Global vector to hold all vertices of the model

std::vector<Mesh> loadedMeshes;

AABB loadedModelAABB;



// Skybox shit

unsigned int skyboxVAO, skyboxVBO;

unsigned int skyboxShaderProgram;



void initializeOpenGL(GLFWwindow* window);

void render(GLFWwindow* window);

void processInput(GLFWwindow* window);

void mouseCallback(GLFWwindow* window, double xpos, double ypos);

void framebufferSizeCallback(GLFWwindow* window, int width, int height);

void initializeCharacterGridLayout();

void initializeShaders();

void loadModel(const std::string& path);

void processNode(aiNode* node, const aiScene* scene);

void processMesh(aiMesh* mesh, const aiScene* scene, const aiMatrix4x4& nodeTransformation);

void storeMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);

AABB computeAABB(const std::vector<Vertex>& vertices);

AABB transformAABB(const AABB& aabb, const glm::mat4& transform);

unsigned int loadTexture(const char* path);

float randomFloat();

unsigned int compileShader(unsigned int type, const char* source);

unsigned int createShaderProgram(unsigned int vertexShader, unsigned int fragmentShader);

void initializeSkybox();

void calcInterpolatedPosition(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim);

void calcInterpolatedRotation(aiQuaternion& out, float animationTime, const aiNodeAnim* nodeAnim);

void calcInterpolatedScaling(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim);

const aiNodeAnim* findNodeAnim(const aiAnimation* animation, const std::string nodeName);

void readNodeHierarchy(float animationTime, const aiNode* node, const glm::mat4& parentTransform);

void updateBoneTransforms(float timeInSeconds, const aiScene* scene);

unsigned int findPosition(float animationTime, const aiNodeAnim* nodeAnim);

unsigned int findRotation(float animationTime, const aiNodeAnim* nodeAnim);

unsigned int findScaling(float animationTime, const aiNodeAnim* nodeAnim);



unsigned int loadCubemap(std::vector<std::string> faces) {

    unsigned int textureID;

    glGenTextures(1, &textureID);

    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);



    int width, height, nrChannels;

    for (unsigned int i = 0; i < faces.size(); i++) {

        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);

        if (data) {

            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,

                0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

            stbi_image_free(data);

        }

        else {

            std::cerr << "Cubemap texture failed to load at path: " << faces[i] << std::endl;

            stbi_image_free(data);

        }

    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);



    return textureID;

}



glm::vec3 hexToRGB(const std::string& hex) {

    int r = std::stoi(hex.substr(1, 2), nullptr, 16);

    int g = std::stoi(hex.substr(3, 2), nullptr, 16);

    int b = std::stoi(hex.substr(5, 2), nullptr, 16);

    return glm::vec3(r / 255.0f, g / 255.0f, b / 255.0f);

}



std::vector<std::string> colorCodes = {

    "#C13E3E", // Multiplayer Red

    "#3639C9", // Multiplayer Blue

    "#C9BA36", // Multiplayer Gold/Yellow

    "#208A20", // Multiplayer Green

    "#B53C8A", // Multiplayer Purple

    "#DF9735", // Multiplayer Orange

    "#744821", // Multiplayer Brown

    "#EB7EC5", // Multiplayer Pink

    "#D2D2D2", // Multiplayer White

    "#758550", // Campaign Color Lighter

    "#55613A"  // Campaign Color Darker

    "#000000", // Halo ce multiplayer black

    "#707E71", // Halo ce multiplayer gray

    "#01FFFF", // Halo ce multiplayer cyan

    "#6493ED", // Halo ce multiplayer cobalt

    "#C69C6C" // Halo ce multiplayer tan

};



glm::vec3 getRandomColor() {

    static std::random_device rd;

    static std::mt19937 engine(rd());

    static std::uniform_int_distribution<int> distribution(0, colorCodes.size() - 1);

    return hexToRGB(colorCodes[distribution(engine)]);

}



int main() {

    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_DEPTH_BITS, 32); // Request a 32-bit depth buffer

    glfwWindowHint(GLFW_SAMPLES, 4); // Enable 4x multisampling

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Frustum Culling", nullptr, nullptr);

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1); // Enable VSync to cap frame rate to monitor's refresh rate

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    glfwSetCursorPosCallback(window, mouseCallback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);



    glewInit();



    std::vector<std::string> faces{

    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_right.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_left.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_top.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_down.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_front.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_back.tga")

    };



    cubemapTexture = loadCubemap(faces);





    std::vector<std::string> waterFaces{

    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_right.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_left.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_up.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_down.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_front.tga"),

    FileSystemUtils::getAssetFilePath("textures/cubemaps/snow_back.tga")

    };



    waterCubeMapTexture = loadCubemap(waterFaces);



    std::string staticModelPath = FileSystemUtils::getAssetFilePath("models/combat_br_idle.fbx");

    loadModel(staticModelPath);



    characterTexture = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_D.tga").c_str());



    characterNormalMap = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_bump.tga").c_str());



    characterMaskTexture = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_cc.tga").c_str());



    initializeOpenGL(window);



    while (!glfwWindowShouldClose(window)) {

        float currentFrame = static_cast<float>(glfwGetTime());

        deltaTime = currentFrame - lastFrame;

        lastFrame = currentFrame;



        processInput(window);



        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {

            glfwTerminate();

            return 0;

        }



        render(window);



        glfwSwapBuffers(window);

        glfwPollEvents();

    }



    glfwTerminate();

    return 0;

}



void initializeOpenGL(GLFWwindow* window) {

    glViewport(0, 0, WIDTH, HEIGHT);

    glEnable(GL_DEPTH_TEST);



    initializeShaders();

    initializeCharacterGridLayout();

    initializeSkybox();



    // Create and bind the character instance buffer

    glGenBuffers(1, &characterInstanceVBO);

    glGenBuffers(1, &characterInstanceColorVBO);

}



void renderScene(GLFWwindow* window) {

    projectionMatrix = camera.getProjectionMatrix(static_cast<float>(WIDTH) / static_cast<float>(HEIGHT));

    viewMatrix = camera.getViewMatrix();



    frustum.update(viewMatrix, projectionMatrix);



    // Render skybox

    glDepthMask(GL_FALSE);

    glDepthFunc(GL_LEQUAL);

    glUseProgram(skyboxShaderProgram);

    glm::mat4 viewMatrixSkybox = glm::mat4(glm::mat3(camera.getViewMatrix()));

    glUniformMatrix4fv(glGetUniformLocation(skyboxShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(viewMatrixSkybox));

    glUniformMatrix4fv(glGetUniformLocation(skyboxShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));



    glBindVertexArray(skyboxVAO);

    glActiveTexture(GL_TEXTURE0);

    glBindTexture(GL_TEXTURE_CUBE_MAP, waterCubeMapTexture);

    glUniform1i(glGetUniformLocation(skyboxShaderProgram, "skybox"), 0);

    glDrawArrays(GL_TRIANGLES, 0, 36);

    glBindVertexArray(0);

    glDepthFunc(GL_LESS);

    glDepthMask(GL_TRUE);



    // Perform frustum culling on character instances

    std::vector<glm::vec3> visibleCharacterPositions;

    std::vector<glm::vec3> visibleCharacterColors; // Add this to store visible colors



    for (const auto& position : characterPositions) {

        glm::mat4 modelMatrix = glm::mat4(1.0f);

        modelMatrix = glm::translate(modelMatrix, position);

        modelMatrix = glm::rotate(modelMatrix, staticNodeRotationAngle, staticNodeRotationAxis);

        modelMatrix = glm::rotate(modelMatrix, (float)glfwGetTime() * characterRotationSpeed, glm::vec3(0.0f, 0.0f, 1.0f));

        modelMatrix = glm::scale(modelMatrix, glm::vec3(0.025f));



        AABB transformedAABB = transformAABB(loadedModelAABB, modelMatrix);

        if (frustum.isAABBInFrustum(transformedAABB.min, transformedAABB.max)) {

            visibleCharacterPositions.push_back(position);

            visibleCharacterColors.push_back(characterInstanceColors[&position - &characterPositions[0]]); // Match color with position

        }

    }



    if (!visibleCharacterPositions.empty()) {

        glBindBuffer(GL_ARRAY_BUFFER, characterInstanceVBO);

        glBufferData(GL_ARRAY_BUFFER, visibleCharacterPositions.size() * sizeof(glm::vec3), &visibleCharacterPositions[0], GL_STATIC_DRAW);



        glBindBuffer(GL_ARRAY_BUFFER, characterInstanceColorVBO);

        glBufferData(GL_ARRAY_BUFFER, visibleCharacterColors.size() * sizeof(glm::vec3), &visibleCharacterColors[0], GL_STATIC_DRAW);

    }



    glUseProgram(characterShaderProgram);



    glm::vec3 lightDir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));

    glm::vec3 viewPos = camera.getPosition();

    glm::vec3 ambientColor = glm::vec3(0.4f, 0.4f, 0.4f);

    glm::vec3 diffuseColor = glm::vec3(1.0f, 1.0f, 1.0f);

    glm::vec3 specularColor = glm::vec3(0.4f, 0.4f, 0.4f);

    float shininess = 32.0f;

    float lightIntensity = 1.0f;



    glUniform3fv(glGetUniformLocation(characterShaderProgram, "lightDir"), 1, glm::value_ptr(lightDir));

    glUniform3fv(glGetUniformLocation(characterShaderProgram, "viewPos"), 1, glm::value_ptr(viewPos));

    glUniform3fv(glGetUniformLocation(characterShaderProgram, "ambientColor"), 1, glm::value_ptr(ambientColor));

    glUniform3fv(glGetUniformLocation(characterShaderProgram, "diffuseColor"), 1, glm::value_ptr(diffuseColor));

    glUniform3fv(glGetUniformLocation(characterShaderProgram, "specularColor"), 1, glm::value_ptr(specularColor));

    glUniform1f(glGetUniformLocation(characterShaderProgram, "shininess"), shininess);

    glUniform1f(glGetUniformLocation(characterShaderProgram, "lightIntensity"), lightIntensity);



    unsigned int modelLocChar = glGetUniformLocation(characterShaderProgram, "model");

    unsigned int viewLocChar = glGetUniformLocation(characterShaderProgram, "view");

    unsigned int projectionLocChar = glGetUniformLocation(characterShaderProgram, "projection");

    unsigned int textureLoc = glGetUniformLocation(characterShaderProgram, "texture_diffuse");

    unsigned int normalMapLoc = glGetUniformLocation(characterShaderProgram, "texture_normal");

    unsigned int cubemapLoc = glGetUniformLocation(characterShaderProgram, "cubemap");



    glUniformMatrix4fv(viewLocChar, 1, GL_FALSE, glm::value_ptr(viewMatrix));

    glUniformMatrix4fv(projectionLocChar, 1, GL_FALSE, glm::value_ptr(projectionMatrix));



    glActiveTexture(GL_TEXTURE0);

    glBindTexture(GL_TEXTURE_2D, characterTexture);

    glUniform1i(textureLoc, 0);



    glActiveTexture(GL_TEXTURE1);

    glBindTexture(GL_TEXTURE_2D, characterNormalMap);

    glUniform1i(normalMapLoc, 1);



    glActiveTexture(GL_TEXTURE2);

    glBindTexture(GL_TEXTURE_2D, characterMaskTexture);

    glUniform1i(glGetUniformLocation(characterShaderProgram, "texture_mask"), 2);



    glActiveTexture(GL_TEXTURE3);

    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);

    glUniform1i(cubemapLoc, 3);



    for (const auto& mesh : loadedMeshes) {

        glBindVertexArray(mesh.VAO);



        glBindBuffer(GL_ARRAY_BUFFER, characterInstanceVBO);

        glEnableVertexAttribArray(7);

        glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        glVertexAttribDivisor(7, 1);



        glBindBuffer(GL_ARRAY_BUFFER, characterInstanceColorVBO);

        glEnableVertexAttribArray(8);

        glVertexAttribPointer(8, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        glVertexAttribDivisor(8, 1);



        if (scene && scene->mNumAnimations > 0) {

            float currentTime = static_cast<float>(glfwGetTime());

            updateBoneTransforms(currentTime, scene);



            for (unsigned int i = 0; i < boneTransforms.size(); i++) {

                std::string uniformName = "boneTransforms[" + std::to_string(i) + "]";

                glUniformMatrix4fv(glGetUniformLocation(characterShaderProgram, uniformName.c_str()), 1, GL_FALSE, glm::value_ptr(boneTransforms[i]));

            }

        }

        else {

            std::cerr << "ERROR::SCENE:: No valid animations found in the scene." << std::endl;

        }



        for (const auto& position : visibleCharacterPositions) {

            glm::mat4 modelMatrix = glm::mat4(1.0f);

            modelMatrix = glm::translate(modelMatrix, position);

            modelMatrix = glm::rotate(modelMatrix, staticNodeRotationAngle, staticNodeRotationAxis);

            modelMatrix = glm::rotate(modelMatrix, (float)glfwGetTime() * characterRotationSpeed, glm::vec3(0.0f, 0.0f, 1.0f));

            modelMatrix = glm::scale(modelMatrix, glm::vec3(0.025f));

            glUniformMatrix4fv(modelLocChar, 1, GL_FALSE, glm::value_ptr(modelMatrix));



            glDrawElementsInstanced(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0, 1);

        }

    }



    std::string windowTitle = "Frustum Culling - Visible Objects: " + std::to_string(visibleCharacterPositions.size());

    glfwSetWindowTitle(window, windowTitle.c_str());

}



// Inside the render function, before the blur pass

void render(GLFWwindow* window) {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderScene(window);

}



void processInput(GLFWwindow* window) {

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)

        camera.processKeyboardInput(GLFW_KEY_W, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)

        camera.processKeyboardInput(GLFW_KEY_S, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)

        camera.processKeyboardInput(GLFW_KEY_A, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)

        camera.processKeyboardInput(GLFW_KEY_D, deltaTime);

}



void mouseCallback(GLFWwindow* window, double xpos, double ypos) {

    if (firstMouse) {

        lastX = xpos;

        lastY = ypos;

        firstMouse = false;

    }



    float xoffset = xpos - lastX;

    float yoffset = lastY - ypos;

    lastX = xpos;

    lastY = ypos;



    camera.processMouseMovement(xoffset, yoffset);

}



void framebufferSizeCallback(GLFWwindow* window, int width, int height) {

    glViewport(0, 0, width, height);

}



void initializeCharacterGridLayout() {

    int characterGridSize = 1; // Adjust this value as desired

    float characterSpacing = 5.0f; // Adjust the spacing between characters



    for (int i = -characterGridSize; i <= characterGridSize; i++) {

        for (int j = -characterGridSize; j <= characterGridSize; j++) {

            characterPositions.push_back(glm::vec3(i * characterSpacing, 0.0f, j * characterSpacing));

            characterInstanceColors.push_back(getRandomColor());

        }

    }



    // Generate and bind the VBO for instance positions

    glGenBuffers(1, &characterInstanceVBO);

    glBindBuffer(GL_ARRAY_BUFFER, characterInstanceVBO);

    glBufferData(GL_ARRAY_BUFFER, characterPositions.size() * sizeof(glm::vec3), &characterPositions[0], GL_STATIC_DRAW);



    // Generate and bind the VBO for instance colors

    glGenBuffers(1, &characterInstanceColorVBO);

    glBindBuffer(GL_ARRAY_BUFFER, characterInstanceColorVBO);

    glBufferData(GL_ARRAY_BUFFER, characterInstanceColors.size() * sizeof(glm::vec3), &characterInstanceColors[0], GL_STATIC_DRAW);

}



void initializeShaders() {

    const char* characterVertexShaderSource = R"(

    #version 430 core



    layout(location = 0) in vec3 aPos;

    layout(location = 1) in vec2 aTexCoord;

    layout(location = 2) in vec3 aNormal;

    layout(location = 3) in vec3 aTangent;

    layout(location = 4) in vec3 aBitangent;

    layout(location = 5) in ivec4 aBoneIDs;

    layout(location = 6) in vec4 aWeights;

    layout(location = 7) in vec3 aInstancePosition;

    layout(location = 8) in vec3 aInstanceColor;



    out vec2 TexCoord;

    out vec3 FragPos;

    out vec3 TangentLightDir;

    out vec3 TangentViewPos;

    out vec3 TangentFragPos;

    out vec3 ReflectDir;

    out vec3 InstanceColor;



    uniform mat4 model;

    uniform mat4 view;

    uniform mat4 projection;

    uniform mat4 boneTransforms[40];



    uniform vec3 lightDir;

    uniform vec3 viewPos;



    void main() {

        mat4 boneTransform = boneTransforms[aBoneIDs[0]] * aWeights[0];

        boneTransform += boneTransforms[aBoneIDs[1]] * aWeights[1];

        boneTransform += boneTransforms[aBoneIDs[2]] * aWeights[2];

        boneTransform += boneTransforms[aBoneIDs[3]] * aWeights[3];



        vec3 transformedPos = vec3(boneTransform * vec4(aPos, 1.0)) + aInstancePosition;



        vec4 worldPos = model * vec4(transformedPos, 1.0);

        gl_Position = projection * view * worldPos;



        FragPos = vec3(worldPos);



        TexCoord = aTexCoord;



        InstanceColor = aInstanceColor;



        // Transform the normal, tangent, and bitangent vectors using the bone transformations

        mat3 boneTransformIT = transpose(inverse(mat3(boneTransform)));

        vec3 transformedNormal = normalize(boneTransformIT * aNormal);

        vec3 transformedTangent = normalize(boneTransformIT * aTangent);

        vec3 transformedBitangent = normalize(boneTransformIT * aBitangent);



        mat3 normalMatrix = transpose(inverse(mat3(model)));

        vec3 T = normalize(normalMatrix * transformedTangent);

        vec3 N = normalize(normalMatrix * transformedNormal);

        T = normalize(T - dot(T, N) * N);

        vec3 B = cross(N, T);



        mat3 TBN = transpose(mat3(T, B, N));

        TangentLightDir = TBN * lightDir;

        TangentViewPos = TBN * viewPos;

        TangentFragPos = TBN * FragPos;



        vec3 I = normalize(viewPos - FragPos);

        ReflectDir = reflect(I, N);

    }

    )";



    const char* characterFragmentShaderSource = R"(

        #version 430 core



        out vec4 FragColor;



        in vec2 TexCoord;

        in vec3 TangentLightDir;

        in vec3 TangentViewPos;

        in vec3 TangentFragPos;

        in vec3 ReflectDir;

        in vec3 InstanceColor;



        uniform vec3 ambientColor;

        uniform vec3 diffuseColor;

        uniform vec3 specularColor;

        uniform float shininess;

        uniform vec3 changeColor;



        uniform sampler2D texture_diffuse;

        uniform sampler2D texture_normal;

        uniform sampler2D texture_mask;

        uniform samplerCube cubemap;

        uniform float lightIntensity;



        void main() {

            vec3 normal = texture(texture_normal, TexCoord).rgb;

            normal = normal * 2.0f - 1.0f;

            normal.y = -normal.y;

            normal = normalize(normal);



            vec4 diffuseTexture = texture(texture_diffuse, TexCoord);

            vec3 diffuseTexColor = diffuseTexture.rgb;

            float alphaValue = diffuseTexture.a;

            float blendFactor = 0.2f;



            vec3 maskValue = texture(texture_mask, TexCoord).rgb;

            vec3 blendedColor = mix(diffuseTexColor, diffuseTexColor * InstanceColor, maskValue);



            vec3 alphaBlendedColor = mix(blendedColor, blendedColor * alphaValue, blendFactor);



            float specularMask = diffuseTexture.a;



            vec3 ambient = ambientColor * alphaBlendedColor;



            vec3 lightDir = normalize(TangentLightDir);

            float diff = max(dot(normal, lightDir), 0.0f) * lightIntensity;

            vec3 diffuse = diffuseColor * diff * alphaBlendedColor;



            vec3 viewDir = normalize(TangentViewPos - TangentFragPos);

            vec3 halfwayDir = normalize(lightDir + viewDir);

            float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess) * lightIntensity;

            vec3 specular = specularColor * spec * specularMask;



            float fresnelBias = 0.1f;

            float fresnelScale = 1.0f;

            float fresnelPower = 1.0f;

            vec3 I = normalize(TangentFragPos - TangentViewPos);

            float fresnel = fresnelBias + fresnelScale * pow(1.0f - dot(I, normal), fresnelPower);

            specular *= fresnel;



            vec3 color = ambient + diffuse + specular;



            vec3 reflectedColor = texture(cubemap, ReflectDir).rgb;

            reflectedColor *= specularMask;

            color = mix(color, reflectedColor, 0.2f);



            FragColor = vec4(color, 1.0f);

        }

    )";



    // Compile and link character shader

    unsigned int characterVertexShader = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(characterVertexShader, 1, &characterVertexShaderSource, NULL);

    glCompileShader(characterVertexShader);



    unsigned int characterFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(characterFragmentShader, 1, &characterFragmentShaderSource, NULL);

    glCompileShader(characterFragmentShader);



    characterShaderProgram = glCreateProgram();

    glAttachShader(characterShaderProgram, characterVertexShader);

    glAttachShader(characterShaderProgram, characterFragmentShader);

    glLinkProgram(characterShaderProgram);



    glDeleteShader(characterVertexShader);

    glDeleteShader(characterFragmentShader);



    // Skybox shader

    const char* skyboxVertexShaderSource = R"(

    #version 430 core

    layout (location = 0) in vec3 aPos;

    out vec3 TexCoords;



    uniform mat4 view;

    uniform mat4 projection;



    void main()

    {

        TexCoords = aPos;

        mat4 rotView = mat4(mat3(view)); // remove translation part of the view matrix

        gl_Position = projection * rotView * vec4(aPos, 1.0);

    }

    )";



    const char* skyboxFragmentShaderSource = R"(

    #version 430 core

    out vec4 FragColor;

    in vec3 TexCoords;



    uniform samplerCube skybox;



    void main()

    {

        FragColor = texture(skybox, TexCoords);

    }

    )";



    unsigned int skyboxVertexShader = compileShader(GL_VERTEX_SHADER, skyboxVertexShaderSource);

    unsigned int skyboxFragmentShader = compileShader(GL_FRAGMENT_SHADER, skyboxFragmentShaderSource);



    skyboxShaderProgram = createShaderProgram(skyboxVertexShader, skyboxFragmentShader);



    glDeleteShader(skyboxVertexShader);

    glDeleteShader(skyboxFragmentShader);

}



unsigned int compileShader(unsigned int type, const char* source) {

    unsigned int shader = glCreateShader(type);

    glShaderSource(shader, 1, &source, nullptr);

    glCompileShader(shader);



    int success;

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {

        char infoLog[512];

        glGetShaderInfoLog(shader, 512, nullptr, infoLog);

        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;

    }

    return shader;

}



unsigned int createShaderProgram(unsigned int vertexShader, unsigned int fragmentShader) {

    unsigned int program = glCreateProgram();

    glAttachShader(program, vertexShader);

    glAttachShader(program, fragmentShader);

    glLinkProgram(program);



    int success;

    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {

        char infoLog[512];

        glGetProgramInfoLog(program, 512, nullptr, infoLog);

        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;

    }

    return program;

}



unsigned int loadTexture(const char* path) {

    unsigned int textureID;

    glGenTextures(1, &textureID);



    int width, height, nrComponents;

    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);

    if (data) {

        GLenum format;

        if (nrComponents == 1)

            format = GL_RED;

        else if (nrComponents == 3)

            format = GL_RGB;

        else if (nrComponents == 4)

            format = GL_RGBA;



        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

        glGenerateMipmap(GL_TEXTURE_2D);



        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_LINEAR);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_LINEAR);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);



        stbi_image_free(data);

    }

    else {

        std::cerr << "Texture failed to load at path: " << path << std::endl;

        stbi_image_free(data);

    }



    return textureID;

}



void loadModel(const std::string& path) {

    scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_JoinIdenticalVertices);



    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {

        std::cerr << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;

        return;

    }



    std::cout << "Model loaded successfully." << std::endl;



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



void storeMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices) {

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



    // Instance Positions

    glBindBuffer(GL_ARRAY_BUFFER, characterInstanceVBO);

    glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    glEnableVertexAttribArray(7);

    glVertexAttribDivisor(7, 1);



    // Instance Colors

    glBindBuffer(GL_ARRAY_BUFFER, characterInstanceColorVBO);

    glVertexAttribPointer(8, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    glEnableVertexAttribArray(8);

    glVertexAttribDivisor(8, 1);



    glBindVertexArray(0);



    mesh.indices = indices;

    loadedMeshes.push_back(mesh);

}



void processNode(aiNode* node, const aiScene* scene) {

    for (unsigned int i = 0; i < node->mNumMeshes; i++) {

        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        processMesh(mesh, scene, node->mTransformation);

    }



    for (unsigned int i = 0; i < node->mNumChildren; i++) {

        processNode(node->mChildren[i], scene);

    }

}



void processMesh(aiMesh* mesh, const aiScene* scene, const aiMatrix4x4& nodeTransformation) {

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



    // Aggregate vertices for AABB computation

    aggregatedVertices.insert(aggregatedVertices.end(), vertices.begin(), vertices.end());



    // Store the processed mesh

    storeMesh(vertices, indices);

}



// Function to compute AABB

AABB computeAABB(const std::vector<Vertex>& vertices) {

    glm::vec3 min = vertices[0].Position;

    glm::vec3 max = vertices[0].Position;



    for (const auto& vertex : vertices) {

        min = glm::min(min, vertex.Position);

        max = glm::max(max, vertex.Position);

    }



    return { min, max };

}



// Function to transform AABB

AABB transformAABB(const AABB& aabb, const glm::mat4& transform) {

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



// Function to generate a random float between 0.0 and 1.0

float randomFloat() {

    static std::default_random_engine engine(static_cast<unsigned int>(time(0)));

    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);

}



void initializeSkybox()

{

    float skyboxVertices[] = {

        // positions          

        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,

         1.0f, -1.0f, -1.0f,

         1.0f, -1.0f, -1.0f,

         1.0f,  1.0f, -1.0f,

        -1.0f,  1.0f, -1.0f,



        -1.0f, -1.0f,  1.0f,

        -1.0f, -1.0f, -1.0f,

        -1.0f,  1.0f, -1.0f,

        -1.0f,  1.0f, -1.0f,

        -1.0f,  1.0f,  1.0f,

        -1.0f, -1.0f,  1.0f,



         1.0f, -1.0f, -1.0f,

         1.0f, -1.0f,  1.0f,

         1.0f,  1.0f,  1.0f,

         1.0f,  1.0f,  1.0f,

         1.0f,  1.0f, -1.0f,

         1.0f, -1.0f, -1.0f,



        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f,  1.0f,

         1.0f,  1.0f,  1.0f,

         1.0f,  1.0f,  1.0f,

         1.0f, -1.0f,  1.0f,

        -1.0f, -1.0f,  1.0f,



        -1.0f,  1.0f, -1.0f,

         1.0f,  1.0f, -1.0f,

         1.0f,  1.0f,  1.0f,

         1.0f,  1.0f,  1.0f,

        -1.0f,  1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,



        -1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,

         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f,  1.0f

    };



    glGenVertexArrays(1, &skyboxVAO);

    glGenBuffers(1, &skyboxVBO);

    glBindVertexArray(skyboxVAO);

    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

}



unsigned int findScaling(float animationTime, const aiNodeAnim* nodeAnim) {

    for (unsigned int i = 0; i < nodeAnim->mNumScalingKeys - 1; i++) {

        if (animationTime < (float)nodeAnim->mScalingKeys[i + 1].mTime) {

            return i;

        }

    }

    return 0;

}



unsigned int findRotation(float animationTime, const aiNodeAnim* nodeAnim) {

    for (unsigned int i = 0; i < nodeAnim->mNumRotationKeys - 1; i++) {

        if (animationTime < (float)nodeAnim->mRotationKeys[i + 1].mTime) {

            return i;

        }

    }

    return 0;

}



unsigned int findPosition(float animationTime, const aiNodeAnim* nodeAnim) {

    for (unsigned int i = 0; i < nodeAnim->mNumPositionKeys - 1; i++) {

        if (animationTime < (float)nodeAnim->mPositionKeys[i + 1].mTime) {

            return i;

        }

    }

    return 0;

}



void updateBoneTransforms(float timeInSeconds, const aiScene* scene) {

    if (!scene || !scene->mAnimations || scene->mNumAnimations == 0) {

        std::cerr << "ERROR::ASSIMP:: No animations found in the model." << std::endl;

        return;

    }



    const aiAnimation* animation = scene->mAnimations[0];

    float ticksPerSecond = animation->mTicksPerSecond != 0 ? animation->mTicksPerSecond : 25.0f;

    float timeInTicks = timeInSeconds * ticksPerSecond;

    float animationTime = fmod(timeInTicks, animation->mDuration);



    glm::mat4 identity = glm::mat4(1.0f);

    readNodeHierarchy(animationTime, scene->mRootNode, identity);



    boneTransforms.resize(boneInfo.size());

    for (unsigned int i = 0; i < boneInfo.size(); i++) {

        boneTransforms[i] = boneInfo[i].FinalTransformation;

    }

}



void readNodeHierarchy(float animationTime, const aiNode* node, const glm::mat4& parentTransform) {

    std::string nodeName(node->mName.data);



    const aiAnimation* animation = scene->mAnimations[0];

    glm::mat4 nodeTransformation = glm::transpose(glm::make_mat4(&node->mTransformation.a1));



    const aiNodeAnim* nodeAnim = findNodeAnim(animation, nodeName);



    if (nodeAnim) {

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

        boneInfo[boneIndex].FinalTransformation = globalTransformation * boneInfo[boneIndex].BoneOffset;

    }



    for (unsigned int i = 0; i < node->mNumChildren; i++) {

        readNodeHierarchy(animationTime, node->mChildren[i], globalTransformation);

    }

}



const aiNodeAnim* findNodeAnim(const aiAnimation* animation, const std::string nodeName) {

    for (unsigned int i = 0; i < animation->mNumChannels; i++) {

        const aiNodeAnim* nodeAnim = animation->mChannels[i];

        if (std::string(nodeAnim->mNodeName.data) == nodeName) {

            return nodeAnim;

        }

    }

    return nullptr;

}



void calcInterpolatedScaling(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim) {

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



void calcInterpolatedRotation(aiQuaternion& out, float animationTime, const aiNodeAnim* nodeAnim) {

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



void calcInterpolatedPosition(aiVector3D& out, float animationTime, const aiNodeAnim* nodeAnim) {

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