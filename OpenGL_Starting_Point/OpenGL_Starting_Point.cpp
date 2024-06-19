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

// Constants and global variables
const int WIDTH = 2560;
const int HEIGHT = 1080;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

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
AnimationStateMachine animationStateMachine;

// Plane geometry shit
unsigned int planeVAO, planeVBO;
unsigned int planeTexture;
unsigned int planeShaderProgram;


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

    out vec2 TexCoord;
    out vec3 FragPos;
    out vec3 TangentLightDir;
    out vec3 TangentViewPos;
    out vec3 TangentFragPos;
    out vec3 ReflectDir;

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

        vec3 transformedPos = vec3(boneTransform * vec4(aPos, 1.0));

        vec4 worldPos = model * vec4(transformedPos, 1.0);
        gl_Position = projection * view * worldPos;

        FragPos = vec3(worldPos);

        TexCoord = aTexCoord;

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

    uniform vec3 ambientColor;
    uniform vec3 diffuseColor;
    uniform vec3 specularColor;
    uniform float shininess;

    uniform sampler2D texture_diffuse;
    uniform sampler2D texture_normal;
    uniform sampler2D texture_mask;
    uniform samplerCube cubemap;
    uniform float lightIntensity;
    uniform vec3 changeColor;

    void main() {
        vec3 normal = texture(texture_normal, TexCoord).rgb;
        normal = normal * 2.0f - 1.0f;
        normal.y = -normal.y;
        normal = normalize(normal);  // Apply bump strength

        vec4 diffuseTexture = texture(texture_diffuse, TexCoord);
        vec3 diffuseTexColor = diffuseTexture.rgb;
        float alphaValue = diffuseTexture.a;

        vec3 maskValue = texture(texture_mask, TexCoord).rgb;
        vec3 blendedColor = mix(diffuseTexColor, diffuseTexColor * changeColor, maskValue);

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
        color = mix(color, reflectedColor, 0.45f);

        FragColor = vec4(color, 1.0f);
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

void initShaders() {
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
}

glm::vec3 hexToRGB(const std::string& hex) {
    int r = std::stoi(hex.substr(1, 2), nullptr, 16);
    int g = std::stoi(hex.substr(3, 2), nullptr, 16);
    int b = std::stoi(hex.substr(5, 2), nullptr, 16);
    return glm::vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}

std::vector<std::string> colorCodes = {
    "#993B3B", // Multiplayer Red
    "#3B4799", // Multiplayer Blue
    "#AF8B25", // Multiplayer Gold/Yellow
    "#32931B", // Multiplayer Green
    "#801B93", // Multiplayer Purple
    "#FFB732", // Multiplayer Orange
    "#582C01", // Multiplayer Brown
    "#CF3D81", // Multiplayer Pink
    "#BABABA", // Multiplayer White
    "#687746", // Campaign Color
    "#606060", // Halo ce multiplayer gray
    "#008888", // Halo ce multiplayer cyan
    "#2A7EF5", // Halo ce multiplayer cobalt
    "#806C52" // Halo ce multiplayer tan
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

    glEnable(GL_DEPTH_TEST);

    // Initialize shaders
    initShaders();

    // Initialize shaders for the plane
    initPlaneShaders();

    // Create the plane with specific size and texture tiling
    float planeWidth = 100.0f;
    float planeHeight = 100.0f;
    float textureTiling = 100.0f; // Example value for tiling
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

    glm::vec3 characterPosition(0.0f, 0.0f, 0.0f);
    float movementSpeed = 4.5f; // Adjust the speed as needed
    bool isMoving = false;
    float currentRotationAngle = 0.0f; // Current rotation angle
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-glm::radians(30.0f), glm::radians(30.0f)); // Random angle change

    camera.cameraLookAt(glm::vec3(-0.5f, 0.0f, 1.0f));

    float timeSinceLastChange = 0.0f;
    const float changeInterval = 2.0f; // Change direction every 2 seconds

    // Initialize the state machine
    animationStateMachine.initiate();

    float autonomousTimer = 0.0f;
    const float autonomousInterval = 5.0f; // Interval to change state

    // Define a blend speed multiplier
    const float blendSpeed = 5.0f; // Increase this value to make transitions faster

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        projectionMatrix = camera.getProjectionMatrix(static_cast<float>(WIDTH) / static_cast<float>(HEIGHT));
        viewMatrix = camera.getViewMatrix();

        // Autonomous behavior logic
        autonomousTimer += deltaTime;
        if (autonomousTimer >= autonomousInterval) {
            if (animationStateMachine.state_cast<const Idle*>() != nullptr) {
                animationStateMachine.process_event(StartWandering());
            }
            else if (animationStateMachine.state_cast<const Wandering*>() != nullptr) {
                animationStateMachine.process_event(StopWandering());
            }
            autonomousTimer = 0.0f; // Reset timer
        }

        // Update animation state based on the state machine
        if (animationStateMachine.state_cast<const Idle*>() != nullptr) {
            currentAnimationIndex = 0; // Idle animation
            blendFactor = glm::max(0.0f, blendFactor - blendSpeed * deltaTime); // Decrease blend factor smoothly
        }
        else if (animationStateMachine.state_cast<const Running*>() != nullptr) {
            currentAnimationIndex = 1; // Running animation
            blendFactor = glm::min(1.0f, blendFactor + blendSpeed * deltaTime); // Increase blend factor smoothly
        }
        else if (animationStateMachine.state_cast<const Wandering*>() != nullptr) {
            // Wandering state behavior
            currentAnimationIndex = 1; // Use running animation for wandering
            blendFactor = glm::min(1.0f, blendFactor + blendSpeed * deltaTime); // Increase blend factor smoothly
            isMoving = true;

            // Update rotation angle randomly at intervals
            timeSinceLastChange += deltaTime;
            if (timeSinceLastChange >= changeInterval) {
                currentRotationAngle += dis(gen); // Apply a random change in rotation angle
                timeSinceLastChange = 0.0f; // Reset timer
            }

            // Calculate the new forward direction based on the current rotation
            glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), currentRotationAngle, glm::vec3(0.0f, 1.0f, 0.0f));
            glm::vec3 forwardDirection = glm::normalize(glm::vec3(rotationMatrix * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));

            // Update character position
            updateCharacterPosition(characterPosition, forwardDirection, movementSpeed, deltaTime);
        }

        // Update camera position to follow the character
        //updateCameraPosition(camera, characterPosition);

        // Update animations with the current blend factor
        animationTime = glfwGetTime(); // Use the actual elapsed time for animation
        modelLoader.updateBoneTransforms(animationTime, animationNames, blendFactor);
        modelLoader.updateHeadRotation(deltaTime, animationNames, currentAnimationIndex); // Update head rotation with deltaTime

        glUseProgram(characterShaderProgram);

        // Set up model matrix
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        modelMatrix = glm::translate(modelMatrix, characterPosition);
        modelMatrix = glm::rotate(modelMatrix, currentRotationAngle, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate based on current angle
        modelMatrix = glm::rotate(modelMatrix, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f)); // Rotate to stand upright
        modelMatrix = glm::scale(modelMatrix, glm::vec3(0.025f)); // Scale the character

        glUniformMatrix4fv(glGetUniformLocation(characterShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
        glUniformMatrix4fv(glGetUniformLocation(characterShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
        glUniformMatrix4fv(glGetUniformLocation(characterShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));

        // Set up lighting uniforms
        glm::vec3 lightDir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));
        glm::vec3 viewPos = camera.getPosition();
        glm::vec3 ambientColor = glm::vec3(0.5f, 0.5f, 0.5f);
        glm::vec3 diffuseColor = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 specularColor = glm::vec3(0.5f, 0.5f, 0.5f);
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

        // Pass the bone transformations to the vertex shader
        for (unsigned int i = 0; i < modelLoader.getBoneTransforms().size(); i++) {
            std::string uniformName = "boneTransforms[" + std::to_string(i) + "]";
            glUniformMatrix4fv(glGetUniformLocation(characterShaderProgram, uniformName.c_str()), 1, GL_FALSE, glm::value_ptr(modelLoader.getBoneTransforms()[i]));
        }

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
            glDrawElements(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0);
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
         width,  0.0f,  height,   textureTiling, 0.0f,
        -width,  0.0f,  height,   0.0f, 0.0f,
        -width,  0.0f, -height,   0.0f, textureTiling,

         width,  0.0f,  height,   textureTiling, 0.0f,
        -width,  0.0f, -height,   0.0f, textureTiling,
         width,  0.0f, -height,   textureTiling, textureTiling
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
