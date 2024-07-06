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
#include "AnimationStateMachine.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <cmath>

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

Camera camera(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -180.0f, 0.0f, 6.0f, 0.1f, 45.0f);
ModelLoader modelLoader;
std::vector<Mesh> loadedMeshes;
AABB loadedModelAABB;
glm::mat4 projectionMatrix;
glm::mat4 viewMatrix;

unsigned int characterShaderProgram;
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
    std::unique_ptr<AnimationStateMachine> stateMachine;
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

    NPC() : stateMachine(std::make_unique<AnimationStateMachine>()), animationTime(0.0f), startFrame(0.0f), endFrame(1.0f) {}
};

const int numInstances = 64;
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

    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;
    layout(location = 2) in vec3 aNormal;
    layout(location = 3) in vec3 aTangent;
    layout(location = 4) in vec3 aBitangent;
    layout(location = 5) in ivec4 aBoneIDs;
    layout(location = 6) in vec4 aWeights;
    // Instanced data
    layout(location = 7) in mat4 instanceModel;
    layout(location = 11) in vec3 instanceColor;
    layout(location = 12) in float instanceAnimationTime;
    layout(location = 13) in float instanceStartFrame;
    layout(location = 14) in float instanceEndFrame;

    out vec2 TexCoord;
    out vec3 FragPos;
    out vec3 TangentLightDir;
    out vec3 TangentViewPos;
    out vec3 TangentFragPos;
    out vec3 ReflectDir;
    out vec3 InstanceColor;

    uniform mat4 view;
    uniform mat4 projection;
    uniform vec3 lightDir;
    uniform vec3 viewPos;

    // TBO declaration
    layout(binding = 4) uniform samplerBuffer boneTransformsTBO;

    // Function to calculate the final bone transform
    mat4 calculateBoneTransform(ivec4 boneIDs, vec4 weights) {
        mat4 boneTransform = mat4(0.0);
        for (int i = 0; i < 4; ++i) {
            if (weights[i] > 0.0) {
                int index = boneIDs[i] * 4;
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
        // Compute the animation time within the specified frame range
        float animationDuration = instanceEndFrame - instanceStartFrame;
        float localAnimationTime = instanceStartFrame + mod(instanceAnimationTime - instanceStartFrame, animationDuration);

        // Calculate bone transformation
        mat4 finalBoneTransform = calculateBoneTransform(aBoneIDs, aWeights);

        // Apply bone transformation to vertex position
        vec3 transformedPos = vec3(finalBoneTransform * vec4(aPos, 1.0));

        // Apply instance model matrix
        vec4 worldPos = instanceModel * vec4(transformedPos, 1.0);
        gl_Position = projection * view * worldPos;

        FragPos = vec3(worldPos);
        TexCoord = aTexCoord;
        InstanceColor = instanceColor;

        // Transform normal, tangent, and bitangent
        mat3 normalMatrix = transpose(inverse(mat3(instanceModel) * mat3(finalBoneTransform)));
        vec3 N = normalize(normalMatrix * aNormal);
        vec3 T = normalize(normalMatrix * aTangent);
        T = normalize(T - dot(T, N) * N); // Ensure T is orthogonal to N
        vec3 B = normalize(cross(N, T));

        mat3 TBN = transpose(mat3(T, B, N));
        TangentLightDir = TBN * lightDir;
        TangentViewPos = TBN * viewPos;
        TangentFragPos = TBN * FragPos;

        vec3 viewDir = normalize(viewPos - FragPos);
        ReflectDir = reflect(-viewDir, N);
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

    layout(binding = 0) uniform sampler2D texture_diffuse;
    layout(binding = 1) uniform sampler2D texture_normal;
    layout(binding = 2) uniform sampler2D texture_mask;
    layout(binding = 3) uniform samplerCube cubemap;
    uniform float lightIntensity;

    void main() {
        vec3 normal = texture(texture_normal, TexCoord).rgb;
        normal = normal * 2.0f - 1.0f;
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
        color = mix(color, reflectedColor, 0.35f);

        FragColor = vec4(color, alphaValue);
    }
)";

const char* planeVertexShaderSource = R"(
    #version 430 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

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
    // Compile and link character shader
    unsigned int characterVertexShader = compileShader(GL_VERTEX_SHADER, characterVertexShaderSource);
    unsigned int characterFragmentShader = compileShader(GL_FRAGMENT_SHADER, characterFragmentShaderSource);
    characterShaderProgram = createShaderProgram(characterVertexShader, characterFragmentShader);

    glDeleteShader(characterVertexShader);
    glDeleteShader(characterFragmentShader);

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
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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

    // Initialize TBO
    initTBO();

    glEnable(GL_DEPTH_TEST);

    // Initialize shaders
    initShaders();

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

    glm::mat4 originalModelMatrix = glm::mat4(1.0f);
    originalModelMatrix = glm::rotate(originalModelMatrix, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f)); // Original rotation
    originalModelMatrix = glm::scale(originalModelMatrix, glm::vec3(0.025f)); // Original scaling

    // Calculate safe spacing
    int gridSide = static_cast<int>(std::sqrt(numInstances));
    float spacing = WORLD_SIZE / gridSide;

    for (int i = 0; i < gridSide; ++i) {
        for (int j = 0; j < gridSide; ++j) {
            int idx = i * gridSide + j;
            if (idx >= numInstances) break;

            float x = -WORLD_SIZE / 2 + spacing * i + spacing / 2;
            float y = 0.0f;
            float z = -WORLD_SIZE / 2 + spacing * j + spacing / 2;

            npcs[idx].position = glm::vec3(x, y, z);
            npcs[idx].modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) * originalModelMatrix;
            npcs[idx].color = getRandomColor();
            npcs[idx].stateMachine = std::make_unique<AnimationStateMachine>();
            npcs[idx].idleTimer = 0.0f;
            npcs[idx].currentPathIndex = 0;
            npcs[idx].currentRotationAngle = 0.0f;
            npcs[idx].currentRotationMatrix = glm::mat4(1.0f);
            npcs[idx].blendFactor = 0.0f;
            npcs[idx].currentAnimationIndex = 0;
        }
    }

    for (size_t i = 0; i < npcs.size(); ++i) {
        instanceModels[i] = npcs[i].modelMatrix;
        instanceColors[i] = npcs[i].color;
        instanceAnimationTimes[i] = npcs[i].animationTime;
        instanceStartFrames[i] = npcs[i].startFrame;
        instanceEndFrames[i] = npcs[i].endFrame;
    }

    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4)* numInstances + sizeof(glm::vec3) * numInstances + sizeof(float) * numInstances * 3, nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::mat4)* numInstances, instanceModels.data());
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4)* numInstances, sizeof(glm::vec3)* numInstances, instanceColors.data());
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4)* numInstances + sizeof(glm::vec3) * numInstances, sizeof(float)* numInstances, instanceAnimationTimes.data());
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4)* numInstances + sizeof(glm::vec3) * numInstances + sizeof(float) * numInstances, sizeof(float)* numInstances, instanceStartFrames.data());
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4)* numInstances + sizeof(glm::vec3) * numInstances + sizeof(float) * numInstances * 2, sizeof(float)* numInstances, instanceEndFrames.data());
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

        glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        projectionMatrix = camera.getProjectionMatrix(static_cast<float>(WIDTH) / static_cast<float>(HEIGHT));
        viewMatrix = camera.getViewMatrix();

        // Update instance data for GPU
        for (size_t i = 0; i < npcs.size(); ++i) {
            instanceModels[i] = npcs[i].modelMatrix;
            instanceAnimationTimes[i] = npcs[i].animationTime;
            instanceStartFrames[i] = npcs[i].stateMachine->startFrame;
            instanceEndFrames[i] = npcs[i].stateMachine->endFrame;
        }

        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::mat4) * numInstances, instanceModels.data());
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * numInstances, sizeof(glm::vec3) * numInstances, instanceColors.data());
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * numInstances + sizeof(glm::vec3) * numInstances, sizeof(float) * numInstances, instanceAnimationTimes.data());
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * numInstances + sizeof(glm::vec3) * numInstances + sizeof(float) * numInstances, sizeof(float) * numInstances, instanceStartFrames.data());
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * numInstances + sizeof(glm::vec3) * numInstances + sizeof(float) * numInstances * 2, sizeof(float) * numInstances, instanceEndFrames.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Update bone transforms and upload to TBO
        modelLoader.updateBoneTransforms(currentFrame, { "combat_sword_idle" }, 0.0f, 0.0f, 58.0f);
        glBindBuffer(GL_TEXTURE_BUFFER, modelLoader.getBoneTransformsTBO());
        glBufferSubData(GL_TEXTURE_BUFFER, 0, modelLoader.getBoneTransforms().size() * sizeof(glm::mat4), modelLoader.getBoneTransforms().data());
        glBindBuffer(GL_TEXTURE_BUFFER, 0);

        glUseProgram(characterShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(characterShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
        glUniformMatrix4fv(glGetUniformLocation(characterShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));

        // Set up lighting uniforms
        glm::vec3 lightDir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));
        glm::vec3 viewPos = camera.getPosition();
        glm::vec3 ambientColor = glm::vec3(0.45f, 0.45f, 0.45f);
        glm::vec3 diffuseColor = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 specularColor = glm::vec3(0.6f, 0.6f, 0.6f);
        float shininess = 16.0f;
        float lightIntensity = 1.25f;

        glUniform3fv(glGetUniformLocation(characterShaderProgram, "lightDir"), 1, glm::value_ptr(lightDir));
        glUniform3fv(glGetUniformLocation(characterShaderProgram, "viewPos"), 1, glm::value_ptr(viewPos));
        glUniform3fv(glGetUniformLocation(characterShaderProgram, "ambientColor"), 1, glm::value_ptr(ambientColor));
        glUniform3fv(glGetUniformLocation(characterShaderProgram, "diffuseColor"), 1, glm::value_ptr(diffuseColor));
        glUniform3fv(glGetUniformLocation(characterShaderProgram, "specularColor"), 1, glm::value_ptr(specularColor));
        glUniform1f(glGetUniformLocation(characterShaderProgram, "shininess"), shininess);
        glUniform1f(glGetUniformLocation(characterShaderProgram, "lightIntensity"), lightIntensity);
        glUniform3fv(glGetUniformLocation(characterShaderProgram, "changeColor"), 1, glm::value_ptr(currentArmorColor));

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
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
                glUniform1i(glGetUniformLocation(characterShaderProgram, "cubemap"), 3);
            }
            else if (mesh.meshBufferIndex == 1) {
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_CUBE_MAP, visorCubemapTexture);
                glUniform1i(glGetUniformLocation(characterShaderProgram, "cubemap"), 3);
            }

            glBindVertexArray(mesh.VAO);
            glDrawElementsInstanced(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0, numInstances);
        }

        // Render the plane
        glUseProgram(planeShaderProgram);
        glm::mat4 planeModel = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(planeShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(planeModel));
        glUniformMatrix4fv(glGetUniformLocation(planeShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
        glUniformMatrix4fv(glGetUniformLocation(planeShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, planeTexture);
        glUniform1i(glGetUniformLocation(planeShaderProgram, "texture_diffuse"), 0);

        glBindVertexArray(planeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    for (const auto& mesh : loadedMeshes) {
        glDeleteVertexArrays(1, &mesh.VAO);
        glDeleteBuffers(1, &mesh.VBO);
        glDeleteBuffers(1, &mesh.EBO);
    }
    glDeleteProgram(characterShaderProgram);
    glDeleteVertexArrays(1, &debugVAO);
    glDeleteBuffers(1, &debugVBO);
    glDeleteProgram(debugShaderProgram);

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
    GLuint boneTransformsTBO;
    glGenBuffers(1, &boneTransformsTBO);
    glBindBuffer(GL_TEXTURE_BUFFER, boneTransformsTBO);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(glm::mat4) * modelLoader.getBoneTransforms().size(), nullptr, GL_DYNAMIC_DRAW);

    GLuint boneTransformsTBOTexture;
    glGenTextures(1, &boneTransformsTBOTexture);
    glActiveTexture(GL_TEXTURE4);  // Use texture unit 4 for the TBO
    glBindTexture(GL_TEXTURE_BUFFER, boneTransformsTBOTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, boneTransformsTBO);

    modelLoader.setBoneTransformsTBO(boneTransformsTBO, boneTransformsTBOTexture);
}
