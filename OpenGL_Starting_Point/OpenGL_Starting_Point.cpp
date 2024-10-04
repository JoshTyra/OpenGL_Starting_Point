#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "Camera.h"
#include "FileSystemUtils.h"

// Asset Importer
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

void APIENTRY MessageCallback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam)
{
    std::cerr << "GL CALLBACK: " << (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "")
        << " type = " << type
        << ", severity = " << severity
        << ", message = " << message << std::endl;
}

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

Camera camera(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -180.0f, 0.0f, 6.0f, 0.1f, 45.0f, 0.1f, 500.0f);

// Vertex Shader source code
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;

    out vec2 TexCoords;  // Pass to fragment shader

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
)";

// Fragment Shader source code
const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoords;

    uniform sampler2D diffuseTexture;
    uniform int isGlowing;

    void main() {
        // Sample the diffuse texture
        vec4 color = texture(diffuseTexture, TexCoords);

        // If the material is glowing, handle it differently
        if (isGlowing == 1) {
            FragColor = color;
        } else {
            // For non-glowing materials, just render the diffuse texture
            FragColor = color;
        }
    }
)";

const char* quadVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    }
)";

const char* quadFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;
    in vec2 TexCoords;

    uniform sampler2D colorMap;      // The color buffer (final scene color)
    uniform sampler2D normalMap;     // The world-space normal buffer
    uniform sampler2D depthMap;      // The depth buffer
    uniform sampler2D glowTexture;   // The glow buffer
    uniform int debugMode;           // Debug mode to switch between different G-buffer outputs

    uniform vec3 fogColor;           // Fog color
    uniform float near;              // Near plane
    uniform float far;               // Far plane
    uniform float fogDensity;        // Fog density for exponential fog
    uniform float fogStart;          // Start distance for fog
    uniform float fogEnd;            // End distance for fog

    float LinearizeDepth(float depth) {
        float z = depth * 2.0 - 1.0;  // Back to NDC
        return (2.0 * near * far) / (far + near - z * (far - near));
    }

    float CalculateExponentialFogFactor(float depth) {
        float fogFactor = exp(-depth * fogDensity); // Exponential decay based on depth
        return clamp(fogFactor, 0.0, 1.0);         // Ensure values are between 0 and 1
    }

    void main() {
       if (debugMode == 0) {
            // Sample color and depth from textures
            vec3 sceneColor = texture(colorMap, TexCoords).rgb;
            float depth = texture(depthMap, TexCoords).r;
    
            // Linearize the depth value and calculate fog factor
            float linearDepth = LinearizeDepth(depth); 

            // Exponential fog factor for smoother, more gradual fog
            float fogFactor = CalculateExponentialFogFactor(linearDepth);

            // Blend scene color with fog color based on fog factor
            vec3 finalColor = mix(fogColor, sceneColor, fogFactor);
    
            FragColor = vec4(finalColor, 1.0);
        } else if (debugMode == 1) {
            // Display the world-space normal buffer
            vec3 normal = texture(normalMap, TexCoords).rgb;
            FragColor = vec4(normal * 0.5 + 0.5, 1.0);  // Map normals from [-1, 1] to [0, 1] for display
        } else if (debugMode == 2) {
            float depth = texture(depthMap, TexCoords).r;
            float linearDepth = LinearizeDepth(depth);

            // Apply logarithmic scaling
            float depthValue = log2(linearDepth - near + 1.0) / log2(far - near + 1.0);

            // Clamp depthValue to [0, 1] to avoid artifacts
            depthValue = clamp(depthValue, 0.0, 1.0);

            FragColor = vec4(vec3(depthValue), 1.0);
        } else if (debugMode == 3) {
            // Render the glow buffer (glowTexture)
            FragColor = texture(glowTexture, TexCoords);
        }
    }
)";

const char* worldspaceNormalsVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;

    out vec3 FragPos;    // Pass world position to fragment shader
    out vec3 Normal;     // Pass world normal to fragment shader

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));  // Calculate world-space position
        Normal = mat3(transpose(inverse(model))) * aNormal;  // Transform normal to world space
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

const char* worldspaceNormalsFragmentShaderSource = R"(
    #version 430 core
    layout(location = 0) out vec3 NormalColor;    // For the world-space normal buffer

    in vec3 FragPos;
    in vec3 Normal;

    void main() {
        NormalColor = normalize(Normal);  // Output normalized world-space normal
    }
)";

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
    static bool firstMouse = true;
    static float lastX = WIDTH / 2.0f;
    static float lastY = HEIGHT / 2.0f;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    camera.processMouseMovement(xoffset, yoffset);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.processMouseScroll(static_cast<float>(yoffset));
}

// Utility function to load textures using stb_image or similar
GLuint loadTextureFromFile(const char* path, const std::string& directory);

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    mutable unsigned int VAO;  // Mark as mutable to allow modification in const functions
    GLuint diffuseTexture;  // Store diffuse texture ID
    bool isGlowing;  // New member for glowing flag

    // Updated constructor
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, GLuint diffuseTexture, bool isGlowing)
        : vertices(vertices), indices(indices), diffuseTexture(diffuseTexture), isGlowing(isGlowing) {
        setupMesh();
    }

    void setupMesh() const {
        // Set up the VAO, VBO, and EBO as before
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        unsigned int VBO, EBO;
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // Vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // Vertex Normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        // Vertex Texture Coords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));

        glBindVertexArray(0);
    }

    void Draw(GLuint shaderProgram, bool glowPass) const {
        // Bind diffuse texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "diffuseTexture"), 0);

        if (glowPass) {
            if (isGlowing) {
                glUniform1i(glGetUniformLocation(shaderProgram, "isGlowing"), 1);
            }
            else {
                // Skip non-glowing materials in the glow pass
                return;
            }
        }
        else {
            // Normal rendering pass: just bind the glow uniform
            glUniform1i(glGetUniformLocation(shaderProgram, "isGlowing"), isGlowing ? 1 : 0);
        }

        // Bind VAO and draw the mesh (remove glUseProgram call)
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};

std::vector<Mesh> loadModel(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate | aiProcess_FlipUVs |
        aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices |
        aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return {};
    }

    std::vector<Mesh> meshes;

    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        // Process vertices and indices
        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
            Vertex vertex;
            vertex.Position = glm::vec3(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
            vertex.Normal = glm::vec3(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);

            if (mesh->mTextureCoords[0]) {
                vertex.TexCoords = glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y);
            }
            else {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
            }

            vertices.push_back(vertex);
        }

        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++) {
                indices.push_back(face.mIndices[k]);
            }
        }

        // Load the material
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        GLuint diffuseTexture = 0;
        bool isGlowing = false;  // New flag for glowing

        if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
            aiString str;
            material->GetTexture(aiTextureType_DIFFUSE, 0, &str);
            std::string texturePath = FileSystemUtils::getAssetFilePath(std::string(str.C_Str()));
            diffuseTexture = loadTextureFromFile(texturePath.c_str(), "");
        }

        // Retrieve the emissive color 'Ke'
        aiColor3D emissive(0.0f, 0.0f, 0.0f);
        if (material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive) == AI_SUCCESS) {
            if (emissive.r > 0.0f || emissive.g > 0.0f || emissive.b > 0.0f) {
                isGlowing = true;
                std::cout << "Mesh " << i << " is glowing with emissive color: "
                    << emissive.r << ", " << emissive.g << ", " << emissive.b << std::endl;
            }
            else {
                std::cout << "Mesh " << i << " is not glowing." << std::endl;
            }
        }
        else {
            std::cout << "Mesh " << i << " does not have an emissive color property." << std::endl;
        }

        meshes.push_back(Mesh(vertices, indices, diffuseTexture, isGlowing));
    }

    return meshes;
}

GLuint loadTextureFromFile(const char* path, const std::string&) {
    GLuint textureID;
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

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a GLFW window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // Request OpenGL 4.3 or newer
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
    glfwSetScrollCallback(window, scrollCallback);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Clear any GLEW errors
    glGetError(); // Clear error flag set by GLEW

    // Enable OpenGL debugging if supported
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(MessageCallback, nullptr);

    // Optionally filter which types of messages you want to log
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

    // Define the viewport dimensions
    glViewport(0, 0, WIDTH, HEIGHT);

    glEnable(GL_DEPTH_TEST);

    // Load the model
    std::vector<Mesh> meshes = loadModel(FileSystemUtils::getAssetFilePath("models/tutorial_map.obj"));

    // Build and compile the shader program
    // Vertex Shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Fragment Shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link shaders
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Quad Vertex Shader
    GLuint quadVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(quadVertexShader, 1, &quadVertexShaderSource, NULL); // Using quadVertexShaderSource from before
    glCompileShader(quadVertexShader);

    // Check for vertex shader compile errors
    glGetShaderiv(quadVertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(quadVertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::QUAD_SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Quad Fragment Shader
    GLuint quadFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(quadFragmentShader, 1, &quadFragmentShaderSource, NULL); // Using quadFragmentShaderSource from before
    glCompileShader(quadFragmentShader);

    // Check for fragment shader compile errors
    glGetShaderiv(quadFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(quadFragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::QUAD_SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link the quad shaders into a shader program
    GLuint quadShaderProgram = glCreateProgram();
    glAttachShader(quadShaderProgram, quadVertexShader);
    glAttachShader(quadShaderProgram, quadFragmentShader);
    glLinkProgram(quadShaderProgram);

    // Check for linking errors
    glGetProgramiv(quadShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(quadShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::QUAD_SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // Clean up shaders as they are no longer necessary after linking
    glDeleteShader(quadVertexShader);
    glDeleteShader(quadFragmentShader);

    // Compile the world-space normals vertex shader
    GLuint worldspaceNormalsVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(worldspaceNormalsVertexShader, 1, &worldspaceNormalsVertexShaderSource, NULL);
    glCompileShader(worldspaceNormalsVertexShader);

    // Check for vertex shader compile errors
    glGetShaderiv(worldspaceNormalsVertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(worldspaceNormalsVertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::WORLSPACE_NORMALS::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Compile the world-space normals fragment shader
    GLuint worldspaceNormalsFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(worldspaceNormalsFragmentShader, 1, &worldspaceNormalsFragmentShaderSource, NULL);
    glCompileShader(worldspaceNormalsFragmentShader);

    // Check for fragment shader compile errors
    glGetShaderiv(worldspaceNormalsFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(worldspaceNormalsFragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::WORLSPACE_NORMALS::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link the world-space normals shaders into a shader program
    GLuint worldspaceNormalsShaderProgram = glCreateProgram();
    glAttachShader(worldspaceNormalsShaderProgram, worldspaceNormalsVertexShader);
    glAttachShader(worldspaceNormalsShaderProgram, worldspaceNormalsFragmentShader);
    glLinkProgram(worldspaceNormalsShaderProgram);

    // Check for linking errors
    glGetProgramiv(worldspaceNormalsShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(worldspaceNormalsShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::WORLSPACE_NORMALS::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // Clean up shaders as they are no longer necessary after linking
    glDeleteShader(worldspaceNormalsVertexShader);
    glDeleteShader(worldspaceNormalsFragmentShader);

    // Create a framebuffer for the color pass
    GLuint colorFBO;
    glGenFramebuffers(1, &colorFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, colorFBO);

    // Create a texture for the color attachment (color pass)
    GLuint colorMap;
    glGenTextures(1, &colorMap);
    glBindTexture(GL_TEXTURE_2D, colorMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Attach colorMap to colorFBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorMap, 0);

    // Create the shared depth texture
    GLuint depthMap;
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, WIDTH, HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Attach depthMap to colorFBO
    glBindFramebuffer(GL_FRAMEBUFFER, colorFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);

    // Create a framebuffer for the normal pass
    GLuint normalFBO;
    glGenFramebuffers(1, &normalFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, normalFBO);

    // Create a texture for the normal attachment
    GLuint normalMap;
    glGenTextures(1, &normalMap);
    glBindTexture(GL_TEXTURE_2D, normalMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);  // 16-bit float for normals
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, normalMap, 0);

    // Attach shared depthMap to normalFBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);

    // Check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Normal framebuffer is not complete!" << std::endl;

    // Create and set up glowFBO
    GLuint glowFBO, glowTexture;
    glGenFramebuffers(1, &glowFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, glowFBO);

    glGenTextures(1, &glowTexture);
    glBindTexture(GL_TEXTURE_2D, glowTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, glowTexture, 0);

    // Attach the shared depthMap to glowFBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Error setting up glow FBO!" << std::endl;

    // Set the list of draw buffers to use
    GLenum attachments[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, attachments);

    // Check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Glow framebuffer is not complete!" << std::endl;

    // Unbind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GLuint noiseTexture;
    glGenTextures(1, &noiseTexture);
    glBindTexture(GL_TEXTURE_2D, noiseTexture);
    // Load noise texture data here (this example assumes you have a function to load the image)
    int noiseWidth, noiseHeight, noiseChannels;
    unsigned char* noiseData = stbi_load(FileSystemUtils::getAssetFilePath("textures/noise.png").c_str(), &noiseWidth, &noiseHeight, &noiseChannels, 1); // Load as 1 channel (grayscale)

    if (noiseData) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, noiseWidth, noiseHeight, 0, GL_RED, GL_UNSIGNED_BYTE, noiseData);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    stbi_image_free(noiseData);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Set up the quad VAO
    GLuint quadVAO, quadVBO;
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    int debugMode = 0;  // Initialize outside the render loop
    bool keyPressed = false;  // Track key press state

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        // Input handling
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // ========== First Pass: Regular Forward Rendering ==========
        // Render the scene to the color FBO (Main Scene)
        glBindFramebuffer(GL_FRAMEBUFFER, colorFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use the regular shader program
        glUseProgram(shaderProgram);

        // Set up view and projection matrices
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = camera.getProjectionMatrix((float)WIDTH / (float)HEIGHT);

        // Pass view and projection matrices to the shader
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Render all objects (both glowing and non-glowing)
        for (const auto& mesh : meshes) {
            glm::mat4 model = glm::mat4(1.0f);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
            mesh.Draw(shaderProgram, false);  // Render all objects (non-glow and glow)
        }

        // ========== Second Pass: Glow Rendering ==========
        // Bind the glow FBO and render only the glowing objects
        glBindFramebuffer(GL_FRAMEBUFFER, glowFBO);
        glClear(GL_COLOR_BUFFER_BIT); // Do not clear depth buffer

        // Set depth function to GL_LEQUAL to allow fragments with depth values equal to the depth buffer to pass
        glDepthFunc(GL_LEQUAL);

        // Render only the glowing objects
        for (const auto& mesh : meshes) {
            glm::mat4 model = glm::mat4(1.0f);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
            mesh.Draw(shaderProgram, true);  // Glow pass
        }

        // Reset depth function to default (GL_LESS)
        glDepthFunc(GL_LESS);

        // Unbind the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // ========== Third Pass: World-Space Normals Rendering ==========
        glBindFramebuffer(GL_FRAMEBUFFER, normalFBO);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cerr << "Normal framebuffer is not complete!" << std::endl;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // Clear both color and depth buffers

        // Use the world-space normals shader program
        glUseProgram(worldspaceNormalsShaderProgram);

        // Set the same view, projection, and model matrices
        glUniformMatrix4fv(glGetUniformLocation(worldspaceNormalsShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(worldspaceNormalsShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Render the loaded meshes (normal buffer)
        for (const auto& mesh : meshes) {
            glm::mat4 model = glm::mat4(1.0f);
            GLuint modelLoc = glGetUniformLocation(worldspaceNormalsShaderProgram, "model");
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            mesh.Draw(worldspaceNormalsShaderProgram, false);
        }

        // Unbind the framebuffer after the second pass
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // ========== Fourth Pass: Post-Processing (e.g., SSAO, Debugging) ==========
        // Clear the screen and disable depth test for the quad
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        // Handle key inputs to toggle between modes
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS && !keyPressed) {
            debugMode = 0;  // Show color buffer (main scene)
            keyPressed = true;
        }
        else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS && !keyPressed) {
            debugMode = 1;  // Show normal buffer
            keyPressed = true;
        }
        else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS && !keyPressed) {
            debugMode = 2;  // Show depth buffer
            keyPressed = true;
        }
        else if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS && !keyPressed) {
            debugMode = 3;  // Show glow buffer
            keyPressed = true;
        }

        // Reset keyPressed state when the keys are released
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_2) == GLFW_RELEASE &&
            glfwGetKey(window, GLFW_KEY_3) == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_4) == GLFW_RELEASE) {
            keyPressed = false;
        }

        // Use the post-processing (quad) shader program
        glUseProgram(quadShaderProgram);
        glUniform1i(glGetUniformLocation(quadShaderProgram, "debugMode"), debugMode);

        // Set fog parameters
        glm::vec3 fogColor(0.337f, 0.349f, 0.435f); // Fog color
        float fogStart = 1.0f;
        float fogEnd = 20.0f;
        float fogDensity = 0.12f;
        glUniform3fv(glGetUniformLocation(quadShaderProgram, "fogColor"), 1, glm::value_ptr(fogColor));
        glUniform1f(glGetUniformLocation(quadShaderProgram, "fogDensity"), fogDensity);
        glUniform1f(glGetUniformLocation(quadShaderProgram, "fogStart"), fogStart);
        glUniform1f(glGetUniformLocation(quadShaderProgram, "fogEnd"), fogEnd);
        glUniform1f(glGetUniformLocation(quadShaderProgram, "near"), camera.getNearPlane());  // Near plane
        glUniform1f(glGetUniformLocation(quadShaderProgram, "far"), camera.getFarPlane());  // Far plane

        // Bind the textures for the quad (color, normal, depth)
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorMap);
        glUniform1i(glGetUniformLocation(quadShaderProgram, "colorMap"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalMap);  // Normal map from G-buffer
        glUniform1i(glGetUniformLocation(quadShaderProgram, "normalMap"), 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, depthMap);  // Depth map
        glUniform1i(glGetUniformLocation(quadShaderProgram, "depthMap"), 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, glowTexture);  // Glow buffer
        glUniform1i(glGetUniformLocation(quadShaderProgram, "glowTexture"), 3);

        // Render the quad (fullscreen post-processing pass)
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Re-enable depth testing
        glEnable(GL_DEPTH_TEST);

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
