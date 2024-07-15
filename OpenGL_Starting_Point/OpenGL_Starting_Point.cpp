#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include "Camera.h"
#include "FileSystemUtils.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <random>
#include <chrono>
#include <map>
#include <iostream>
#include <fstream>
#include "AnimatedModel.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <cmath>
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "BehaviorTrees.h"
#include "NPC.h"

// Constants and global variables
const int WIDTH = 2560;
const int HEIGHT = 1080;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame
double previousTime = 0.0;
int frameCount = 0;

// Single directional light
glm::vec3 lightDir = glm::normalize(glm::vec3(0.716f, 0.325f, 0.618f));
glm::vec3 originalLightDir = glm::vec3(0.3f, 1.0f, 0.5f);

// Add a flag to keep track of mouse control state
bool mouseControlEnabled = true;

// UBO structures
struct CameraData {
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 viewPos;
    float padding; // To ensure 16-byte alignment
};

struct LightData {
    glm::vec3 lightDir;
    float padding1; // To ensure 16-byte alignment
    glm::vec3 ambientColor;
    float padding2;
    glm::vec3 diffuseColor;
    float padding3;
    glm::vec3 specularColor;
    float lightIntensity;
    float shininess;
    float padding4[3]; // To ensure 16-byte alignment
};

GLuint cameraUBO, lightUBO;

Camera camera(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -180.0f, 0.0f, 6.0f, 0.1f, 45.0f);
ModelLoader modelLoader;
std::vector<Mesh> loadedMeshes;
AABB loadedModelAABB;
glm::mat4 projectionMatrix;
glm::mat4 viewMatrix;

unsigned int computeShaderProgram;
unsigned int characterShaderProgram;
unsigned int visorShaderProgram;
unsigned int characterTexture;
unsigned int characterNormalMap;
unsigned int cubemapTexture;
unsigned int visorCubemapTexture;
unsigned int characterMaskTexture;

glm::vec3 currentArmorColor;

int currentAnimationIndex = 0;
float animationTime = 0.0f;
std::vector<std::string> animationNames = { "combat_sword_idle", "combat_sword_move_front" };
float blendFactor = 0.0f; // 0.0 means fully "combat_sword_idle", 1.0 means fully "combat_sword_move_front"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> distIdle(0, 0); // Update to include the indices 0 and 2

// Add these variables at the top of your file
float idleAnimationChangeTimer = 0.0f;
const float idleAnimationChangeInterval = 2.0f; // Minimum interval between idle animation changes in seconds
bool idleAnimationSelected = false; // Add this flag at the top of your file

NPCManager npcManager;

// Plane geometry shit
unsigned int planeVAO, planeVBO;
unsigned int planeTexture;
unsigned int planeShaderProgram;

// Pathfinding shit
const int GRID_SIZE = 100; // This is the one you'll change when you want to adjust the grid
const float CELL_SIZE = 1.0f; // Size of each grid cell in world units
const float WORLD_SIZE = GRID_SIZE * CELL_SIZE; // Total world size
std::vector<std::vector<bool>> navigationGrid(GRID_SIZE, std::vector<bool>(GRID_SIZE, true));
std::vector<glm::vec3> currentPath;
int currentPathIndex = 0;
glm::vec3 currentDestination;

unsigned int debugVAO, debugVBO;
unsigned int debugShaderProgram;
std::vector<float> debugVertices;

struct GridNode {
    int x, y;
    float g, h, f;
    GridNode* parent;

    GridNode(int x, int y) : x(x), y(y), g(0), h(0), f(0), parent(nullptr) {}
};

// Method declarations
unsigned int loadCubemap(std::vector<std::string> faces);
unsigned int compileShader(unsigned int type, const char* source);
unsigned int createShaderProgram(unsigned int vertexShader, unsigned int fragmentShader);
unsigned int compileComputeShader(const char* source);
unsigned int createComputeShaderProgram(unsigned int computeShader);
unsigned int loadTexture(const char* path);
void initShaders();
void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void windowIconifyCallback(GLFWwindow* window, int iconified);
void createPlane(float width, float height, float textureTiling);
void loadPlaneTexture();
void initPlaneShaders();
void updateNPCAnimations(float deltaTime);
void initUBOs();
void updateUBOs(const glm::vec3& lightDir);
void setupImGui(GLFWwindow* window);
void initializeNPCs();
void dispatchComputeShader();

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

unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        int infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<char> infoLog(infoLogLength);
        glGetShaderInfoLog(shader, infoLogLength, nullptr, infoLog.data());
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED of type: " << (type == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT") << "\n" << infoLog.data() << std::endl;
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
        int infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<char> infoLog(infoLogLength);
        glGetProgramInfoLog(program, infoLogLength, nullptr, infoLog.data());
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog.data() << std::endl;
    }
    return program;
}

unsigned int compileComputeShader(const char* source) {
    unsigned int shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        int infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<char> infoLog(infoLogLength);
        glGetShaderInfoLog(shader, infoLogLength, nullptr, infoLog.data());
        std::cerr << "ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n" << infoLog.data() << std::endl;
    }
    return shader;
}

unsigned int createComputeShaderProgram(unsigned int computeShader) {
    unsigned int program = glCreateProgram();
    glAttachShader(program, computeShader);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        int infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<char> infoLog(infoLogLength);
        glGetProgramInfoLog(program, infoLogLength, nullptr, infoLog.data());
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog.data() << std::endl;
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
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
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

const char* boneTransformComputeShaderSource = R"(
    #version 430 core

    layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

    layout(std140, binding = 0) buffer BoneTransforms {
        mat4 boneTransforms[];
    };

    layout(std140, binding = 1) buffer AnimationMatrices {
        mat4 animationMatrices[];
    };

    uniform int numBones;
    uniform int numAnimations;
    uniform int numNPCs;

    void main() {
        uint npcIndex = gl_GlobalInvocationID.x / numBones;
        uint boneIndex = gl_GlobalInvocationID.x % numBones;
        uint matrixIndex = npcIndex * numBones + boneIndex;

        if (npcIndex >= numNPCs || boneIndex >= numBones) {
            return;
        }

        // Retrieve the bone transformation matrix
        mat4 boneTransform = boneTransforms[matrixIndex];

        // Update the animation matrices
        animationMatrices[matrixIndex] = boneTransform;
    }
)";

const char* characterVertexShaderSource = R"(
    #version 430 core

    layout(std140, binding = 0) uniform CameraData
    {
        mat4 view;
        mat4 projection;
        vec3 viewPos;
    };

    layout(std140, binding = 1) uniform LightData
    {
        vec3 lightDir;
        vec3 ambientColor;
        vec3 diffuseColor;
        vec3 specularColor;
        float lightIntensity;
        float shininess;
    };

    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;
    layout(location = 2) in vec3 aNormal;
    layout(location = 3) in vec3 aTangent;
    layout(location = 4) in vec3 aBitangent;
    layout(location = 5) in ivec2 aBoneIDs;  // Changed from ivec4
    layout(location = 6) in vec2 aWeights;   // Changed from vec4
    layout(location = 7) in mat4 instanceModel;
    layout(location = 11) in vec3 instanceColor;
    layout(location = 12) in float instanceAnimationTime;
    layout(location = 13) in float instanceStartFrame;
    layout(location = 14) in float instanceEndFrame;
    layout(location = 15) in int instanceID;

    out vec2 TexCoord;
    out vec3 FragPos;
    out vec3 TangentLightDir;
    out vec3 TangentViewPos;
    out vec3 TangentFragPos;
    out vec3 TangentViewDir;
    out vec3 InstanceColor;

    layout(std430, binding = 0) readonly buffer BoneTransforms {
        mat4 boneTransforms[];
    };

    #define NUM_BONES 31

    mat4 calculateBoneTransform(ivec2 boneIDs, vec2 weights, int instanceID) {
        mat4 boneTransform = mat4(0.0);
        int boneOffset = instanceID * NUM_BONES;

        for (int i = 0; i < 2; ++i) {  // Changed from 4 to 2
            if (weights[i] > 0.0) {
                int index = boneOffset + boneIDs[i];
                mat4 boneMatrix = boneTransforms[index];
                boneTransform += boneMatrix * weights[i];
            }
        }

        return boneTransform;
    }

    void main() {
        float animationDuration = instanceEndFrame - instanceStartFrame;
        float localAnimationTime = instanceStartFrame + mod(instanceAnimationTime - instanceStartFrame, animationDuration);

        // Use precomputed bone transform from compute shader
        mat4 finalBoneTransform = calculateBoneTransform(aBoneIDs, aWeights, instanceID);
        vec3 transformedPos = vec3(finalBoneTransform * vec4(aPos, 1.0));
        vec4 worldPos = instanceModel * vec4(transformedPos, 1.0);
        gl_Position = projection * view * worldPos;

        FragPos = vec3(worldPos);
        TexCoord = aTexCoord;
        InstanceColor = instanceColor;

        mat3 normalMatrix = transpose(inverse(mat3(instanceModel) * mat3(finalBoneTransform)));
        vec3 N = normalize(normalMatrix * aNormal);
        vec3 T = normalize(normalMatrix * aTangent);
        T = normalize(T - dot(T, N) * N);
        vec3 B = normalize(cross(N, T));

        mat3 TBN = transpose(mat3(T, B, N));
        TangentLightDir = TBN * lightDir;
        TangentViewPos = TBN * viewPos;
        TangentFragPos = TBN * FragPos;
        TangentViewDir = TBN * (viewPos - FragPos);
    }
)";

const char* characterFragmentShaderSource = R"(
    #version 430 core

    layout(std140, binding = 1) uniform LightData
    {
        vec3 lightDir;
        vec3 ambientColor;
        vec3 diffuseColor;
        vec3 specularColor;
        float lightIntensity;
        float shininess;
    };

    out vec4 FragColor;

    in vec2 TexCoord;
    in vec3 TangentLightDir;
    in vec3 TangentViewPos;
    in vec3 TangentFragPos;
    in vec3 TangentViewDir;
    in vec3 InstanceColor;

    layout(binding = 0) uniform sampler2D texture_diffuse;
    layout(binding = 1) uniform sampler2D texture_normal;
    layout(binding = 2) uniform sampler2D texture_mask;
    layout(binding = 3) uniform samplerCube cubemap;

    void main() {
        vec3 normal = texture(texture_normal, TexCoord).rgb * 2.0 - 1.0;
        normal.y = -normal.y;
        normal = normalize(normal);

        vec4 diffuseTexture = texture(texture_diffuse, TexCoord);
        vec3 diffuseTexColor = diffuseTexture.rgb;
        float alphaValue = diffuseTexture.a;

        vec3 maskValue = texture(texture_mask, TexCoord).rgb;
        vec3 blendedColor = mix(diffuseTexColor, diffuseTexColor * InstanceColor, maskValue);

        float specularMask = diffuseTexture.a;

        vec3 ambient = ambientColor * blendedColor;

        vec3 lightDir = normalize(TangentLightDir);
        float diff = max(dot(normal, lightDir), 0.0f) * lightIntensity;
        vec3 diffuse = diffuseColor * diff * blendedColor;

        vec3 viewDir = normalize(TangentViewDir);
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess) * lightIntensity;

        // Increase the influence of the specular component
        float specularInfluence = 2.5; // Adjust this value to increase/decrease specular influence
        vec3 specular = specularColor * spec * specularInfluence * specularMask;

        float fresnelBias = 0.2f; // Increase this value for a stronger base effect
        float fresnelScale = 2.0f; // Increase this value to enhance the contrast
        float fresnelPower = 0.3f; // Decrease this value for a smoother transition
        float fresnel = fresnelBias + fresnelScale * pow(1.0f - dot(viewDir, normal), fresnelPower);
        specular *= fresnel;

        vec3 color = ambient + diffuse + specular;

        vec3 reflectedDir = reflect(-viewDir, normal); // Reflect view direction in tangent space
        vec3 reflectedColor = texture(cubemap, reflectedDir).rgb;
        reflectedColor *= specularMask;
        color = mix(color, reflectedColor, 0.4f);

        FragColor = vec4(color, alphaValue);
    }
)";

const char* visorFragmentShaderSource = R"(
    #version 430 core

    layout(std140, binding = 1) uniform LightData
    {
        vec3 lightDir;
        vec3 ambientColor;
        vec3 diffuseColor;
        vec3 specularColor;
        float lightIntensity;
        float shininess;
    };

    out vec4 FragColor;

    in vec2 TexCoord;
    in vec3 TangentLightDir;
    in vec3 TangentViewPos;
    in vec3 TangentFragPos;
    in vec3 TangentViewDir;

    layout(binding = 0) uniform sampler2D texture_diffuse;
    layout(binding = 1) uniform sampler2D texture_normal;
    layout(binding = 3) uniform samplerCube visorCubemap;

    // uniform for the reflection color
    uniform vec3 reflectionColor = vec3(0.9372f, 0.7215f, 0.7215f);
    uniform float normalMapStrength = 0.4f; // Control the strength of the normal map

    void main() {
        vec3 normal = texture(texture_normal, TexCoord).rgb * 2.0 - 1.0;
        normal.y = -normal.y;
        normal.xy *= normalMapStrength; // Use the normal map strength uniform
        normal = normalize(normal);

        vec4 diffuseTexture = texture(texture_diffuse, TexCoord);
        vec3 diffuseTexColor = diffuseTexture.rgb;
        float alphaValue = diffuseTexture.a;
        float specularMask = diffuseTexture.a;

        vec3 ambient = ambientColor * diffuseTexColor;

        vec3 lightDir = normalize(TangentLightDir);
        float diff = max(dot(normal, lightDir), 0.0f);
        vec3 diffuse = diffuseColor * diff * diffuseTexColor;

        vec3 viewDir = normalize(TangentViewDir);
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
        vec3 specular = specularColor * spec * specularMask;

        // Increase the influence of the specular component
        float specularInfluence = 2.0; // Adjust this value to increase/decrease specular influence
        specular *= specularInfluence;

        float fresnelBias = 0.1f; // Increased bias for more reflection
        float fresnelScale = 1.0f;
        float fresnelPower = 2.0f; // Increased power for sharper effect
        float fresnel = fresnelBias + fresnelScale * pow(1.0f - dot(viewDir, normal), fresnelPower);
        specular *= fresnel;

        vec3 color = ambient + diffuse + specular;

        // Reflection calculation in tangent space
        vec3 reflectedDir = reflect(-viewDir, normal); // Reflect view direction in tangent space
        vec3 reflectedColor = texture(visorCubemap, reflectedDir).rgb;
        reflectedColor *= specularMask;

        // Apply the reflection color
        reflectedColor *= reflectionColor;

        // Adjusted Fresnel effect for reflections
        float reflectionFresnelFactor = pow(1.0 - max(dot(viewDir, normal), 0.0), 2.0); // Increased power for more noticeable reflections
        reflectionFresnelFactor = mix(0.65, 1.0, reflectionFresnelFactor); // Adjusted range for better visibility

        // Blend the original color and the reflected color
        color = mix(color, reflectedColor, reflectionFresnelFactor);

        // Balance the specular highlights for a more metallic look
        color += specular; // Adjusted influence
        FragColor = vec4(color, alphaValue);
    }
)";

const char* planeVertexShaderSource = R"(
    #version 430 core

    layout(std140, binding = 0) uniform CameraData
    {
        mat4 view;
        mat4 projection;
        vec3 viewPos;
    };

    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    uniform mat4 model;

    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        TexCoord = aTexCoord;
}
)";

const char* planeFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;

    in vec2 TexCoord;

    uniform sampler2D texture_diffuse;

    void main() {
        FragColor = texture(texture_diffuse, TexCoord);
    }
)";

const char* debugVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;

    out vec3 Color;

    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * view * vec4(aPos, 1.0);
        Color = aColor;
    }
)";

const char* debugFragmentShaderSource = R"(
    #version 430 core
    in vec3 Color;
    out vec4 FragColor;

    void main()
    {
        FragColor = vec4(Color, 1.0);
    }
)";

void initShaders() {
    // Compile character vertex shader
    unsigned int characterVertexShader = compileShader(GL_VERTEX_SHADER, characterVertexShaderSource);

    // Compile character fragment shader
    unsigned int characterFragmentShader = compileShader(GL_FRAGMENT_SHADER, characterFragmentShaderSource);
    characterShaderProgram = createShaderProgram(characterVertexShader, characterFragmentShader);

    // Compile visor fragment shader
    unsigned int visorFragmentShader = compileShader(GL_FRAGMENT_SHADER, visorFragmentShaderSource);
    visorShaderProgram = createShaderProgram(characterVertexShader, visorFragmentShader);

    // Now we can delete the character vertex shader after it's used in both programs
    glDeleteShader(characterVertexShader);
    glDeleteShader(characterFragmentShader);
    glDeleteShader(visorFragmentShader);

    // Compile and link plane shader
    unsigned int planeVertexShader = compileShader(GL_VERTEX_SHADER, planeVertexShaderSource);
    unsigned int planeFragmentShader = compileShader(GL_FRAGMENT_SHADER, planeFragmentShaderSource);
    planeShaderProgram = createShaderProgram(planeVertexShader, planeFragmentShader);

    glDeleteShader(planeVertexShader);
    glDeleteShader(planeFragmentShader);

    // Compile and link debug shader
    unsigned int debugVertexShader = compileShader(GL_VERTEX_SHADER, debugVertexShaderSource);
    unsigned int debugFragmentShader = compileShader(GL_FRAGMENT_SHADER, debugFragmentShaderSource);
    debugShaderProgram = createShaderProgram(debugVertexShader, debugFragmentShader);

    glDeleteShader(debugVertexShader);
    glDeleteShader(debugFragmentShader);

    // Compile and link compute shader
    unsigned int boneTransformComputeShader = compileComputeShader(boneTransformComputeShaderSource);
    computeShaderProgram = createComputeShaderProgram(boneTransformComputeShader);

    glDeleteShader(boneTransformComputeShader);
}

// Update the character's position based on forward direction and speed
void updateCharacterPosition(glm::vec3& position, glm::vec3& forwardDirection, float speed, float deltaTime) {
    position += forwardDirection * speed * deltaTime;
}

void updateCameraPosition(Camera& camera, const glm::vec3& characterPosition) {
    glm::vec3 cameraOffset = glm::vec3(3.0f, 1.5f, 2.0f); // Adjust as necessary for the desired follow distance
    camera.position = characterPosition + cameraOffset;
    camera.updateCameraVectors(); // Ensure the camera's vectors are updated based on the new position
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

    // Toggle mouse control with the "M" key
    static bool mKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS && !mKeyPressed) {
        mKeyPressed = true;
        mouseControlEnabled = !mouseControlEnabled; // Toggle mouse control state
        if (mouseControlEnabled) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_RELEASE) {
        mKeyPressed = false;
    }
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (!mouseControlEnabled) {
        return; // Do not process mouse movement if mouse control is disabled
    }

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

float lerpAngle(float start, float end, float t) {
    float difference = std::fmod(end - start + 180.0f, 360.0f) - 180.0f;
    return start + difference * t;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a GLFW window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL Basic Application", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync to cap frame rate to monitor's refresh rate
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // Set the window size and iconify callbacks
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetWindowIconifyCallback(window, windowIconifyCallback);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Define the viewport dimensions
    glViewport(0, 0, WIDTH, HEIGHT);

    setupImGui(window);

    glEnable(GL_DEPTH_TEST);

    // Initialize shaders
    initShaders();

    // Initialize UBOs
    initUBOs();

    // Initialize shaders for the plane
    initPlaneShaders();

    // Create the plane with specific size and texture tiling
    float planeWidth = WORLD_SIZE / 2;
    float planeHeight = WORLD_SIZE / 2;
    float textureTiling = WORLD_SIZE / 2; // Example value for tiling
    createPlane(planeWidth, planeHeight, textureTiling);

    // Load the plane texture
    loadPlaneTexture();

    // Load the model and textures
    std::string staticModelPath = FileSystemUtils::getAssetFilePath("models/masterchief.fbx");

    // Load the model
    modelLoader.loadModel(staticModelPath);
    loadedMeshes = modelLoader.getLoadedMeshes();
    loadedModelAABB = modelLoader.getLoadedModelAABB();

    modelLoader.processAnimations(); // Process animations after loading the model
    modelLoader.setCurrentAnimation(animationNames[currentAnimationIndex]);

    characterTexture = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_D.tga").c_str());
    characterNormalMap = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_bump.tga").c_str());
    characterMaskTexture = loadTexture(FileSystemUtils::getAssetFilePath("textures/masterchief_cc.tga").c_str());

    std::vector<std::string> faces{
        FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_right.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_left.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_top.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_down.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_front.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/armor_back.tga")
    };

    cubemapTexture = loadCubemap(faces);

    std::vector<std::string> visorfaces{
        FileSystemUtils::getAssetFilePath("textures/cubemaps/mirror_surface_right.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/mirror_surface_left.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/mirror_surface_up.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/mirror_surface_down.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/mirror_surface_front.tga"),
        FileSystemUtils::getAssetFilePath("textures/cubemaps/mirror_surface_back.tga")
    };

    visorCubemapTexture = loadCubemap(visorfaces);

    // Create buffers for instance data
    std::vector<glm::mat4> instanceModels(NPCManager::MAX_NPCS);
    std::vector<glm::vec3> instanceColors(NPCManager::MAX_NPCS);
    std::vector<float> instanceAnimationTimes(NPCManager::MAX_NPCS);
    std::vector<float> instanceStartFrames(NPCManager::MAX_NPCS);
    std::vector<float> instanceEndFrames(NPCManager::MAX_NPCS);
    std::vector<int> instanceIDs(NPCManager::MAX_NPCS);

    glm::mat4 originalModelMatrix = glm::mat4(1.0f);
    originalModelMatrix = glm::rotate(originalModelMatrix, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f)); // Original rotation
    originalModelMatrix = glm::scale(originalModelMatrix, glm::vec3(0.025f)); // Original scaling

    initializeNPCs();

    const auto& npcs = npcManager.getNPCs();
    for (size_t i = 0; i < npcs.size(); ++i) {
        instanceModels[i] = npcs[i].getModelMatrix();
        instanceColors[i] = npcs[i].getColor();
        instanceAnimationTimes[i] = npcs[i].getAnimationTime();
        instanceStartFrames[i] = npcs[i].getStartFrame();
        instanceEndFrames[i] = npcs[i].getEndFrame();
        instanceIDs[i] = i;
    }

    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

    // Calculate the total size of all instance data
    size_t totalSize = sizeof(glm::mat4) * NPCManager::MAX_NPCS +
        sizeof(glm::vec3) * NPCManager::MAX_NPCS +
        sizeof(float) * NPCManager::MAX_NPCS * 3 +
        sizeof(int) * NPCManager::MAX_NPCS;

    glBufferData(GL_ARRAY_BUFFER, totalSize, nullptr, GL_STATIC_DRAW);

    // Calculate offsets for each data type
    size_t modelOffset = 0;
    size_t colorOffset = sizeof(glm::mat4) * NPCManager::MAX_NPCS;
    size_t animTimeOffset = colorOffset + sizeof(glm::vec3) * NPCManager::MAX_NPCS;
    size_t startFrameOffset = animTimeOffset + sizeof(float) * NPCManager::MAX_NPCS;
    size_t endFrameOffset = startFrameOffset + sizeof(float) * NPCManager::MAX_NPCS;
    size_t idOffset = endFrameOffset + sizeof(float) * NPCManager::MAX_NPCS;

    glBufferSubData(GL_ARRAY_BUFFER, modelOffset, sizeof(glm::mat4)* NPCManager::MAX_NPCS, instanceModels.data());
    glBufferSubData(GL_ARRAY_BUFFER, colorOffset, sizeof(glm::vec3)* NPCManager::MAX_NPCS, instanceColors.data());
    glBufferSubData(GL_ARRAY_BUFFER, animTimeOffset, sizeof(float)* NPCManager::MAX_NPCS, instanceAnimationTimes.data());
    glBufferSubData(GL_ARRAY_BUFFER, startFrameOffset, sizeof(float)* NPCManager::MAX_NPCS, instanceStartFrames.data());
    glBufferSubData(GL_ARRAY_BUFFER, endFrameOffset, sizeof(float)* NPCManager::MAX_NPCS, instanceEndFrames.data());
    glBufferSubData(GL_ARRAY_BUFFER, idOffset, sizeof(int)* NPCManager::MAX_NPCS, instanceIDs.data());

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    for (const auto& mesh : loadedMeshes) {
        unsigned int VAO = mesh.VAO;
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

        // Set instance model matrix attributes
        for (unsigned int j = 0; j < 4; j++) {
            glEnableVertexAttribArray(7 + j);
            glVertexAttribPointer(7 + j, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * j));
            glVertexAttribDivisor(7 + j, 1);
        }

        // Set instance color attribute
        glEnableVertexAttribArray(11);
        glVertexAttribPointer(11, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)(sizeof(glm::mat4) * NPCManager::MAX_NPCS));
        glVertexAttribDivisor(11, 1);

        // Set instance animation attributes
        glEnableVertexAttribArray(12);
        glVertexAttribPointer(12, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(sizeof(glm::mat4) * NPCManager::MAX_NPCS + sizeof(glm::vec3) * NPCManager::MAX_NPCS));
        glVertexAttribDivisor(12, 1);

        glEnableVertexAttribArray(13);
        glVertexAttribPointer(13, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(sizeof(glm::mat4) * NPCManager::MAX_NPCS + sizeof(glm::vec3) * NPCManager::MAX_NPCS + sizeof(float) * NPCManager::MAX_NPCS));
        glVertexAttribDivisor(13, 1);

        glEnableVertexAttribArray(14);
        glVertexAttribPointer(14, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(sizeof(glm::mat4) * NPCManager::MAX_NPCS + sizeof(glm::vec3) * NPCManager::MAX_NPCS + sizeof(float) * NPCManager::MAX_NPCS * 2));
        glVertexAttribDivisor(14, 1);

        // Set up instance ID attribute
        glEnableVertexAttribArray(15);
        glVertexAttribIPointer(15, 1, GL_INT, 0, (void*)idOffset);
        glVertexAttribDivisor(15, 1);

        glBindVertexArray(0);
    }

    // Initialize Behavior Tree Factory
    BT::BehaviorTreeFactory factory;
    BT::registerNodes(factory);

    // Create a blackboard
    auto blackboard = BT::Blackboard::create();

    auto tree = factory.createTreeFromText(BT::getMainTreeXML(), blackboard);

    // Initialize SSBOs for bone transforms and vertex data
    modelLoader.initializeSSBOs();

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        double currentTime = glfwGetTime();
        double elapsedTime = currentTime - previousTime;
        frameCount++;

        // Update FPS every second
        if (elapsedTime >= 1.0) {
            double fps = frameCount / elapsedTime;
            std::string title = "OpenGL Basic Application - FPS: " + std::to_string(fps);
            glfwSetWindowTitle(window, title.c_str());

            // Reset for the next second
            frameCount = 0;
            previousTime = currentTime;
        }

        processInput(window);

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Set ImGui window position and size
        ImGui::SetNextWindowPos(ImVec2(0, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(150, 150), ImGuiCond_FirstUseEver); // Adjusted window size

        // ImGui controls for light direction
        ImGui::Begin("Light Controls");
        ImGui::Text("Original Light Direction: (%.3f, %.3f, %.3f)", originalLightDir.x, originalLightDir.y, originalLightDir.z);

        // Separate sliders for each component of lightDir
        static float lightDirX = lightDir.x;
        static float lightDirY = lightDir.y;
        static float lightDirZ = lightDir.z;

        ImGui::SliderFloat("Light Direction X", &lightDirX, -1.0f, 1.0f);
        ImGui::SliderFloat("Light Direction Y", &lightDirY, -1.0f, 1.0f);
        ImGui::SliderFloat("Light Direction Z", &lightDirZ, -1.0f, 1.0f);

        // Update lightDir with the new values from sliders
        lightDir = glm::normalize(glm::vec3(lightDirX, lightDirY, lightDirZ));

        ImGui::Text("Normalized Light Direction: (%.3f, %.3f, %.3f)", lightDir.x, lightDir.y, lightDir.z);
        ImGui::End();

        // Update UBOs
        updateUBOs(lightDir);

        for (auto& npc : npcManager.getNPCs()) {  // Use non-const reference
            blackboard->set("npc", &npc);  // Pass non-const pointer
            auto status = tree.tickOnce();
            npc.update(deltaTime);
        }

        // Update instance data for GPU
        for (size_t i = 0; i < npcs.size(); ++i) {
            instanceModels[i] = npcs[i].getModelMatrix();
            instanceColors[i] = npcs[i].getColor();
            instanceAnimationTimes[i] = npcs[i].getAnimationTime();
            instanceStartFrames[i] = npcs[i].getStartFrame();
            instanceEndFrames[i] = npcs[i].getEndFrame();
        }

        // Upload instance data to GPU
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, modelOffset, sizeof(glm::mat4) * NPCManager::MAX_NPCS, instanceModels.data());
        glBufferSubData(GL_ARRAY_BUFFER, colorOffset, sizeof(glm::vec3) * NPCManager::MAX_NPCS, instanceColors.data());
        glBufferSubData(GL_ARRAY_BUFFER, animTimeOffset, sizeof(float) * NPCManager::MAX_NPCS, instanceAnimationTimes.data());
        glBufferSubData(GL_ARRAY_BUFFER, startFrameOffset, sizeof(float) * NPCManager::MAX_NPCS, instanceStartFrames.data());
        glBufferSubData(GL_ARRAY_BUFFER, endFrameOffset, sizeof(float) * NPCManager::MAX_NPCS, instanceEndFrames.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Update bone transforms and upload to SSBO
        updateNPCAnimations(deltaTime);

        dispatchComputeShader();

        glUseProgram(characterShaderProgram);

        // Set up textures and draw the character
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, characterTexture);
        glUniform1i(glGetUniformLocation(characterShaderProgram, "texture_diffuse"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, characterNormalMap);
        glUniform1i(glGetUniformLocation(characterShaderProgram, "texture_normal"), 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, characterMaskTexture);
        glUniform1i(glGetUniformLocation(characterShaderProgram, "texture_mask"), 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glUniform1i(glGetUniformLocation(characterShaderProgram, "cubemap"), 3);

        for (const auto& mesh : loadedMeshes) {
            if (mesh.meshBufferIndex == 0) {
                glUseProgram(characterShaderProgram);
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
                glUniform1i(glGetUniformLocation(characterShaderProgram, "cubemap"), 3);
            }
            else if (mesh.meshBufferIndex == 1) {
                glUseProgram(visorShaderProgram); // Use the visor shader for the visor part
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_CUBE_MAP, visorCubemapTexture);
                glUniform1i(glGetUniformLocation(visorShaderProgram, "visorCubemap"), 3);
            }

            glBindVertexArray(mesh.VAO);
            glDrawElementsInstanced(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0, NPCManager::MAX_NPCS);
        }

        // Render the plane
        glUseProgram(planeShaderProgram);
        glm::mat4 planeModel = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(planeShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(planeModel));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, planeTexture);
        glUniform1i(glGetUniformLocation(planeShaderProgram, "texture_diffuse"), 0);

        glBindVertexArray(planeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Clean up
    for (const auto& mesh : loadedMeshes) {
        glDeleteVertexArrays(1, &mesh.VAO);
        glDeleteBuffers(1, &mesh.VBO);
        glDeleteBuffers(1, &mesh.EBO);
    }
    glDeleteProgram(characterShaderProgram);
    glDeleteProgram(visorShaderProgram);
    glDeleteVertexArrays(1, &debugVAO);
    glDeleteBuffers(1, &debugVBO);
    glDeleteProgram(debugShaderProgram);
    glDeleteBuffers(1, &cameraUBO);
    glDeleteBuffers(1, &lightUBO);

    glfwTerminate();
    return 0;
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    // You can also update the projection matrix here if needed
    projectionMatrix = camera.getProjectionMatrix(static_cast<float>(width) / static_cast<float>(height));
}

void windowIconifyCallback(GLFWwindow* window, int iconified) {
    if (iconified) {
        // Window was minimized
        std::cout << "Window minimized" << std::endl;
    }
    else {
        // Window was restored
        std::cout << "Window restored" << std::endl;
    }
}

void createPlane(float width, float height, float textureTiling) {
    float planeVertices[] = {
        // positions                        // texture coords
         WORLD_SIZE / 2,  0.0f,  WORLD_SIZE / 2,   textureTiling, 0.0f,
        -WORLD_SIZE / 2,  0.0f,  WORLD_SIZE / 2,   0.0f, 0.0f,
        -WORLD_SIZE / 2,  0.0f, -WORLD_SIZE / 2,   0.0f, textureTiling,

         WORLD_SIZE / 2,  0.0f,  WORLD_SIZE / 2,   textureTiling, 0.0f,
        -WORLD_SIZE / 2,  0.0f, -WORLD_SIZE / 2,   0.0f, textureTiling,
         WORLD_SIZE / 2,  0.0f, -WORLD_SIZE / 2,   textureTiling, textureTiling
    };

    glGenVertexArrays(1, &planeVAO);
    glGenBuffers(1, &planeVBO);
    glBindVertexArray(planeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
}

void loadPlaneTexture() {
    planeTexture = loadTexture(FileSystemUtils::getAssetFilePath("textures/metal plate floor ext.png").c_str());
}

void initPlaneShaders() {
    unsigned int planeVertexShader = compileShader(GL_VERTEX_SHADER, planeVertexShaderSource);
    unsigned int planeFragmentShader = compileShader(GL_FRAGMENT_SHADER, planeFragmentShaderSource);
    planeShaderProgram = createShaderProgram(planeVertexShader, planeFragmentShader);
    glDeleteShader(planeVertexShader);
    glDeleteShader(planeFragmentShader);
}

void updateNPCAnimations(float deltaTime) {
    size_t numBones = modelLoader.getNumBones();
    size_t numAnimations = animationNames.size();
    const auto& npcs = npcManager.getNPCs();

    // Resize allAnimationMatrices to fit all NPCs, all animations, and all bones
    std::vector<glm::mat4> allAnimationMatrices(npcs.size() * numAnimations * numBones, glm::mat4(1.0f));

    for (size_t npcIndex = 0; npcIndex < npcs.size(); ++npcIndex) {
        const auto& npc = npcs[npcIndex];
        for (size_t animIndex = 0; animIndex < numAnimations; ++animIndex) {
            std::vector<glm::mat4> npcBoneTransforms(numBones, glm::mat4(1.0f));

            modelLoader.updateBoneTransforms(
                npcIndex,  // Pass the NPC index
                npc.getAnimationTime(),
                animationNames[animIndex],
                npc.getBlendFactor(),
                npc.getStartFrame(),
                npc.getEndFrame(),
                npcBoneTransforms
            );

            // Copy the bone transforms for this NPC and animation to the correct position in allAnimationMatrices
            size_t startIndex = (npcIndex * numAnimations * numBones) + (animIndex * numBones);
            std::copy(npcBoneTransforms.begin(), npcBoneTransforms.end(), allAnimationMatrices.begin() + startIndex);
        }
    }

    // Upload all animation matrices to the SSBO
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, modelLoader.getAnimationMatricesSSBO());
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, allAnimationMatrices.size() * sizeof(glm::mat4), allAnimationMatrices.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void initUBOs() {
    // Create Camera UBO
    glGenBuffers(1, &cameraUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(CameraData), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, cameraUBO);

    // Create Light UBO
    glGenBuffers(1, &lightUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, lightUBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(LightData), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, lightUBO);
}

void updateUBOs(const glm::vec3& lightDir) {
    // Update Camera UBO
    CameraData cameraData;
    cameraData.view = camera.getViewMatrix();
    cameraData.projection = camera.getProjectionMatrix(static_cast<float>(WIDTH) / static_cast<float>(HEIGHT));
    cameraData.viewPos = camera.getPosition();

    glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(CameraData), &cameraData);

    // Update Light UBO
    LightData lightData;
    lightData.lightDir = lightDir;
    lightData.ambientColor = glm::vec3(0.45f, 0.45f, 0.45f);
    lightData.diffuseColor = glm::vec3(1.0f, 1.0f, 1.0f);
    lightData.specularColor = glm::vec3(0.6f, 0.6f, 0.6f);
    lightData.lightIntensity = 1.0f;
    lightData.shininess = 16.0f;

    glBindBuffer(GL_UNIFORM_BUFFER, lightUBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(LightData), &lightData);

    // Unbind the uniform buffer
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void setupImGui(GLFWwindow* window) {
    // Initialize ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    ImGui::StyleColorsDark();

    // Initialize ImGui for GLFW and OpenGL
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");
}

void initializeNPCs() {
    glm::mat4 originalModelMatrix = glm::mat4(1.0f);
    originalModelMatrix = glm::rotate(originalModelMatrix, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
    originalModelMatrix = glm::scale(originalModelMatrix, glm::vec3(0.025f));

    npcManager.initializeNPCs(WORLD_SIZE, NPCManager::MAX_NPCS, originalModelMatrix);
}

void dispatchComputeShader() {
    glUseProgram(computeShaderProgram);
    glUniform1i(glGetUniformLocation(computeShaderProgram, "numBones"), modelLoader.getNumBones());
    glUniform1i(glGetUniformLocation(computeShaderProgram, "numAnimations"), animationNames.size());
    glUniform1i(glGetUniformLocation(computeShaderProgram, "numNPCs"), NPCManager::MAX_NPCS);

    glDispatchCompute((modelLoader.getNumBones() * NPCManager::MAX_NPCS + 31) / 32, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    //// Read debug data
    //std::vector<glm::vec4> debugData(NPCManager::MAX_NPCS * modelLoader.getNumBones() * 4);
    //glBindBuffer(GL_SHADER_STORAGE_BUFFER, modelLoader.getDebugBufferSSBO());
    //glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, debugData.size() * sizeof(glm::vec4), debugData.data());

    //// Save to file using FileSystemUtils
    //std::string debugFilePath = FileSystemUtils::getAssetFilePath("debug_output.txt");
    //std::ofstream outFile(debugFilePath);
    //if (outFile.is_open()) {
    //    for (size_t i = 0; i < debugData.size(); i += 4) {
    //        outFile << "NPC: " << debugData[i + 3].x << ", Bone: " << debugData[i + 3].y
    //            << ", Index: " << (i / 4) << "\n";
    //        for (int row = 0; row < 4; ++row) {
    //            outFile << debugData[i + row].x << " "
    //                << debugData[i + row].y << " "
    //                << debugData[i + row].z << " "
    //                << debugData[i + row].w << "\n";
    //        }
    //        outFile << "\n";
    //    }
    //    outFile.close();
    //}
    //else {
    //    std::cerr << "Unable to open file for writing debug output." << std::endl;
    //}
}





