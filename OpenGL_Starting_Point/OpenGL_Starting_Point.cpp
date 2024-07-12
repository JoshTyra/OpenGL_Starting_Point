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
#include "AnimatedModel.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <cmath>
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

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

// Ai shit here
struct NPC {
    glm::vec3 position;
    glm::mat4 modelMatrix;
    glm::vec3 color;
    float idleTimer;
    std::vector<glm::vec3> currentPath;
    int currentPathIndex;
    glm::vec3 currentDestination;
    float currentRotationAngle;
    glm::mat4 currentRotationMatrix;
    float blendFactor;
    int currentAnimationIndex;
    float animationTime;
    float startFrame;
    float endFrame;
    bool isRunning; // Field to track if NPC is running

    // Default constructor initializes all member variables
    NPC() : position(0.0f), modelMatrix(1.0f), color(1.0f),
        idleTimer(0.0f), currentPathIndex(0), currentRotationAngle(0.0f),
        currentRotationMatrix(1.0f), blendFactor(0.0f),
        currentAnimationIndex(0), animationTime(0.0f),
        startFrame(0.0f), endFrame(58.0f), isRunning(false) {}
};

// Set numInstances to a cube e.g., 8, 27, 64, 125
const int numInstances = 12;
std::vector<NPC> npcs(numInstances);

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
unsigned int loadTexture(const char* path);
void initShaders();
void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void windowIconifyCallback(GLFWwindow* window, int iconified);
void createPlane(float width, float height, float textureTiling);
void loadPlaneTexture();
void initPlaneShaders();
void initTBO();
void updateNPCAnimations(float deltaTime);
void initUBOs();
void updateUBOs(const glm::vec3& lightDir);
void setupImGui(GLFWwindow* window);

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
    layout(location = 5) in ivec4 aBoneIDs;
    layout(location = 6) in vec4 aWeights;
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

    layout(binding = 4) uniform samplerBuffer boneTransformsTBO;

    #define NUM_BONES 31 // Adjust this to match the maximum required per mesh

    mat4 calculateBoneTransform(ivec4 boneIDs, vec4 weights, int instanceID) {
        mat4 boneTransform = mat4(0.0);
        int boneOffset = instanceID * NUM_BONES * 4;

        for (int i = 0; i < 4; ++i) {
            if (weights[i] > 0.0) {
                int index = boneOffset + boneIDs[i] * 4;
                mat4 boneMatrix = mat4(
                    texelFetch(boneTransformsTBO, index),
                    texelFetch(boneTransformsTBO, index + 1),
                    texelFetch(boneTransformsTBO, index + 2),
                    texelFetch(boneTransformsTBO, index + 3)
                );

                boneTransform += boneMatrix * weights[i];
            }
        }

        return boneTransform;
    }

    void main() {
        float animationDuration = instanceEndFrame - instanceStartFrame;
        float localAnimationTime = instanceStartFrame + mod(instanceAnimationTime - instanceStartFrame, animationDuration);

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
        TangentViewDir = TBN * (viewPos - FragPos); // Transform view direction to tangent space
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

    void main() {
        vec3 normal = texture(texture_normal, TexCoord).rgb * 2.0 - 1.0;
        normal.y = -normal.y;
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

        // Adjusted Fresnel effect for reflections
        float reflectionFresnelFactor = pow(1.0 - max(dot(viewDir, normal), 0.0), 2.0); // Increased power for more noticeable reflections
        reflectionFresnelFactor = mix(0.35, 1.0, reflectionFresnelFactor); // Adjusted range for better visibility

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
    "#707E71", // Halo ce multiplayer gray
    "#01FFFF", // Halo ce multiplayer cyan
    "#6493ED", // Halo ce multiplayer cobalt
    "#C69C6C", // Halo ce multiplayer tan
};

glm::vec3 getRandomColor() {
    static std::random_device rd;
    static std::mt19937 engine(rd());
    static std::uniform_int_distribution<int> distribution(0, colorCodes.size() - 1);
    return hexToRGB(colorCodes[distribution(engine)]);
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

    // Check for the "C" key press to change armor color
    static bool cKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !cKeyPressed) {
        cKeyPressed = true;
        currentArmorColor = getRandomColor(); // Change the armor color
    }
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE) {
        cKeyPressed = false;
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

    // Initialize TBO
    initTBO();

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

    // Set the initial random color once
    currentArmorColor = getRandomColor();

    float movementSpeed = 4.5f; // Adjust the speed as needed
    float rotationSpeed = 2.0f;
    float currentRotationAngle = 0.0f; // Current rotation angle
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-glm::radians(30.0f), glm::radians(30.0f)); // Random angle change

    camera.cameraLookAt(glm::vec3(-0.5f, 0.0f, 1.0f));

    const float IDLE_DURATION = 15.0f; // 15 seconds of idle time at destination
    const float PATH_COMPLETION_CHECK_THRESHOLD = 0.1f; // Distance threshold to consider a path point reached
    float idleTimer = 0.0f;

    // Define a blend speed multiplier
    const float blendSpeed = 5.0f; // Increase this value to make transitions faster
    glm::mat4 currentRotationMatrix = glm::mat4(1.0f);

    // Create buffers for instance data
    std::vector<glm::mat4> instanceModels(numInstances);
    std::vector<glm::vec3> instanceColors(numInstances);
    std::vector<float> instanceAnimationTimes(numInstances);
    std::vector<float> instanceStartFrames(numInstances);
    std::vector<float> instanceEndFrames(numInstances);
    std::vector<int> instanceIDs(numInstances);

    glm::mat4 originalModelMatrix = glm::mat4(1.0f);
    originalModelMatrix = glm::rotate(originalModelMatrix, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f)); // Original rotation
    originalModelMatrix = glm::scale(originalModelMatrix, glm::vec3(0.025f)); // Original scaling

    // Calculate safe spacing
    int gridSide = static_cast<int>(std::ceil(std::sqrt(numInstances)));
    float spacing = WORLD_SIZE / gridSide;

    for (int idx = 0; idx < numInstances; ++idx) {
        int row = idx / gridSide;
        int col = idx % gridSide;

        float x = -WORLD_SIZE / 2 + spacing * col + spacing / 2;
        float y = 0.0f;
        float z = -WORLD_SIZE / 2 + spacing * row + spacing / 2;

        npcs[idx].position = glm::vec3(x, y, z);
        npcs[idx].modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) * originalModelMatrix;
        npcs[idx].color = getRandomColor();
        npcs[idx].idleTimer = 0.0f;
        npcs[idx].currentPathIndex = 0;
        npcs[idx].currentRotationAngle = 0.0f;
        npcs[idx].currentRotationMatrix = glm::mat4(1.0f);
        npcs[idx].blendFactor = 0.0f;

        // Assign animations with validation
        npcs[idx].currentAnimationIndex = idx % animationNames.size();
        if (npcs[idx].currentAnimationIndex < 0 || npcs[idx].currentAnimationIndex >= animationNames.size()) {
            std::cerr << "Error: NPC " << idx << " has an invalid animation index: " << npcs[idx].currentAnimationIndex << std::endl;
            npcs[idx].currentAnimationIndex = 0; // Fallback to a valid index
        }

        if (npcs[idx].currentAnimationIndex == 0) {
            npcs[idx].startFrame = 0.0f;
            npcs[idx].endFrame = 58.0f;
        }
        else if (npcs[idx].currentAnimationIndex == 1) {
            npcs[idx].startFrame = 59.0f;
            npcs[idx].endFrame = 78.0f;
        }
    }

    // Ensure all NPCs are initialized
    for (int idx = 0; idx < numInstances; ++idx) {
        if (npcs[idx].currentAnimationIndex < 0 || npcs[idx].currentAnimationIndex >= animationNames.size()) {
            std::cerr << "Error: NPC " << idx << " has an invalid animation index: " << npcs[idx].currentAnimationIndex << std::endl;
            npcs[idx].currentAnimationIndex = 0; // Fallback to a valid index
        }
    }

    for (size_t i = 0; i < npcs.size(); ++i) {
        instanceModels[i] = npcs[i].modelMatrix;
        instanceColors[i] = npcs[i].color;
        instanceAnimationTimes[i] = npcs[i].animationTime;
        instanceStartFrames[i] = npcs[i].startFrame;
        instanceEndFrames[i] = npcs[i].endFrame;
        instanceIDs[i] = i;
    }

    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

    // Calculate the total size of all instance data
    size_t totalSize = sizeof(glm::mat4) * numInstances +
        sizeof(glm::vec3) * numInstances +
        sizeof(float) * numInstances * 3 +
        sizeof(int) * numInstances;

    glBufferData(GL_ARRAY_BUFFER, totalSize, nullptr, GL_STATIC_DRAW);

    // Calculate offsets for each data type
    size_t modelOffset = 0;
    size_t colorOffset = sizeof(glm::mat4) * numInstances;
    size_t animTimeOffset = colorOffset + sizeof(glm::vec3) * numInstances;
    size_t startFrameOffset = animTimeOffset + sizeof(float) * numInstances;
    size_t endFrameOffset = startFrameOffset + sizeof(float) * numInstances;
    size_t idOffset = endFrameOffset + sizeof(float) * numInstances;  // Add this line

    glBufferSubData(GL_ARRAY_BUFFER, modelOffset, sizeof(glm::mat4)* numInstances, instanceModels.data());
    glBufferSubData(GL_ARRAY_BUFFER, colorOffset, sizeof(glm::vec3)* numInstances, instanceColors.data());
    glBufferSubData(GL_ARRAY_BUFFER, animTimeOffset, sizeof(float)* numInstances, instanceAnimationTimes.data());
    glBufferSubData(GL_ARRAY_BUFFER, startFrameOffset, sizeof(float)* numInstances, instanceStartFrames.data());
    glBufferSubData(GL_ARRAY_BUFFER, endFrameOffset, sizeof(float)* numInstances, instanceEndFrames.data());
    glBufferSubData(GL_ARRAY_BUFFER, idOffset, sizeof(int)* numInstances, instanceIDs.data());  // Add this line

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
        glVertexAttribPointer(11, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)(sizeof(glm::mat4) * numInstances));
        glVertexAttribDivisor(11, 1);

        // Set instance animation attributes
        glEnableVertexAttribArray(12);
        glVertexAttribPointer(12, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(sizeof(glm::mat4) * numInstances + sizeof(glm::vec3) * numInstances));
        glVertexAttribDivisor(12, 1);

        glEnableVertexAttribArray(13);
        glVertexAttribPointer(13, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(sizeof(glm::mat4) * numInstances + sizeof(glm::vec3) * numInstances + sizeof(float) * numInstances));
        glVertexAttribDivisor(13, 1);

        glEnableVertexAttribArray(14);
        glVertexAttribPointer(14, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(sizeof(glm::mat4) * numInstances + sizeof(glm::vec3) * numInstances + sizeof(float) * numInstances * 2));
        glVertexAttribDivisor(14, 1);

        // Set up instance ID attribute
        glEnableVertexAttribArray(15);
        glVertexAttribIPointer(15, 1, GL_INT, 0, (void*)idOffset);
        glVertexAttribDivisor(15, 1);

        glBindVertexArray(0);
    }

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

        // Update instance data for GPU
        for (size_t i = 0; i < npcs.size(); ++i) {
            instanceModels[i] = npcs[i].modelMatrix;
            instanceColors[i] = npcs[i].color;
            instanceAnimationTimes[i] = npcs[i].animationTime;
            instanceStartFrames[i] = npcs[i].startFrame;
            instanceEndFrames[i] = npcs[i].endFrame;
        }

        // Upload instance data to GPU
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, modelOffset, sizeof(glm::mat4) * numInstances, instanceModels.data());
        glBufferSubData(GL_ARRAY_BUFFER, colorOffset, sizeof(glm::vec3) * numInstances, instanceColors.data());
        glBufferSubData(GL_ARRAY_BUFFER, animTimeOffset, sizeof(float) * numInstances, instanceAnimationTimes.data());
        glBufferSubData(GL_ARRAY_BUFFER, startFrameOffset, sizeof(float) * numInstances, instanceStartFrames.data());
        glBufferSubData(GL_ARRAY_BUFFER, endFrameOffset, sizeof(float) * numInstances, instanceEndFrames.data());
        // Note: You don't need to update instance IDs every frame as they don't change
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Update bone transforms and upload to TBO
        glBindBuffer(GL_TEXTURE_BUFFER, modelLoader.getBoneTransformsTBO());
        glBufferSubData(GL_TEXTURE_BUFFER, 0, modelLoader.getBoneTransforms().size() * sizeof(glm::mat4), modelLoader.getBoneTransforms().data());
        glBindBuffer(GL_TEXTURE_BUFFER, 0);

        glUseProgram(characterShaderProgram);

        // Update animation states for each NPC
        updateNPCAnimations(deltaTime);

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

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER, modelLoader.boneTransformsTBOTexture);

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
            glDrawElementsInstanced(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0, numInstances);
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

void initTBO() {
    int maxBonesPerInstance = 31; // Must be 31 as required by the shader
    glGenBuffers(1, &modelLoader.boneTransformsTBO);
    glBindBuffer(GL_TEXTURE_BUFFER, modelLoader.boneTransformsTBO);
    glBufferData(GL_TEXTURE_BUFFER, numInstances * maxBonesPerInstance * 16 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glGenTextures(1, &modelLoader.boneTransformsTBOTexture);
    glBindTexture(GL_TEXTURE_BUFFER, modelLoader.boneTransformsTBOTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, modelLoader.boneTransformsTBO);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

// Update NPC animations
void updateNPCAnimations(float deltaTime) {
    std::vector<glm::mat4> allBoneTransforms;

    for (NPC& npc : npcs) {
        npc.animationTime += deltaTime;

        // Check if the animation index is valid
        if (npc.currentAnimationIndex < 0 || npc.currentAnimationIndex >= animationNames.size()) {
            std::cerr << "Error: Invalid animation index for NPC!" << std::endl;
            continue;
        }

        std::vector<glm::mat4> npcBoneTransforms(modelLoader.getNumBones(), glm::mat4(1.0f));
        modelLoader.updateBoneTransforms(npc.animationTime, animationNames[npc.currentAnimationIndex], npc.blendFactor, npc.startFrame, npc.endFrame, npcBoneTransforms);

        if (npcBoneTransforms.size() != modelLoader.getNumBones()) {
            std::cerr << "Error: Bone transform size mismatch for NPC!" << std::endl;
        }

        allBoneTransforms.insert(allBoneTransforms.end(), npcBoneTransforms.begin(), npcBoneTransforms.end());
    }

    // Upload all bone transforms to the TBO
    glBindBuffer(GL_TEXTURE_BUFFER, modelLoader.getBoneTransformsTBO());
    if (allBoneTransforms.size() * sizeof(glm::mat4) > numInstances * 31 * 16 * sizeof(float)) {
        std::cerr << "Error: Exceeded allocated buffer size for bone transforms!" << std::endl;
    }
    glBufferData(GL_TEXTURE_BUFFER, allBoneTransforms.size() * sizeof(glm::mat4), allBoneTransforms.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
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
    lightData.ambientColor = glm::vec3(0.4f, 0.4f, 0.4f);
    lightData.diffuseColor = glm::vec3(1.0f, 1.0f, 1.0f);
    lightData.specularColor = glm::vec3(0.6f, 0.6f, 0.6f);
    lightData.lightIntensity = 1.25f;
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




