#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "Camera.h"
#include "FileSystemUtils.h"
#include <random>

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

Camera camera(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -180.0f, 0.0f, 6.0f, 0.1f, 45.0f, 0.1f, 500.0f);

// Vertex Shader source code
const char* vertexShaderSource = R"(
    #version 430 core
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
    #version 430 core
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
    #version 430 core
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
        uniform sampler2D blurredGlowTexture; // The blurred glow buffer
        uniform sampler2D positionTexture; // The position buffer
        uniform sampler2D ssaoTexture;  // The ssao buffer 
        uniform int debugMode;           // Debug mode to switch between different G-buffer outputs

        uniform vec3 fogColor;           // Fog color
        uniform float near;              // Near plane
        uniform float far;               // Far plane
        uniform float fogDensity;        // Fog density for exponential fog
        uniform float fogStart;          // Start distance for fog
        uniform float fogEnd;            // End distance for fog
        uniform float bloomIntensity;    // Bloom intensity
        uniform int screenWidth;
        uniform int screenHeight;
        uniform int ssaoWidth;
        uniform int ssaoHeight;

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

            vec2 ssaoTexCoords = TexCoords * vec2(float(ssaoWidth) / float(screenWidth), float(ssaoHeight) / float(screenHeight));
            float ssao = texture(ssaoTexture, ssaoTexCoords).r;

            // Apply SSAO to the scene color
            vec3 occludedColor = sceneColor * ssao;

            // Blend occluded color with fog color based on fog factor
            vec3 finalColor = mix(fogColor, occludedColor, fogFactor);

            // Sample the blurred glow texture
            vec3 blurredGlow = texture(blurredGlowTexture, TexCoords).rgb;

            // Combine the scene color with the blurred glow
            finalColor += blurredGlow * bloomIntensity;

            FragColor = vec4(finalColor, 1.0);
        }else if (debugMode == 1) {
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
        }else if (debugMode == 4) {
            // Render the blurred glow buffer
            FragColor = vec4(texture(blurredGlowTexture, TexCoords).rgb, 1.0);
        }else if (debugMode == 5) {
            // Render the position buffer
            FragColor = vec4(texture(positionTexture, TexCoords).rgb, 1.0);
        } else if (debugMode == 6) {
            // Render the ssao buffer
            vec2 ssaoTexCoords = TexCoords * vec2(float(ssaoWidth) / float(screenWidth), float(ssaoHeight) / float(screenHeight));
            float ssao = texture(ssaoTexture, ssaoTexCoords).r;
            FragColor = vec4(vec3(ssao), 1.0);  // Convert the SSAO value to a grayscale color
        }
    }
)";

// World-Space Normals Vertex Shader Source Code
const char* worldspaceNormalsVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;

    out vec3 Normal;     // Pass view-space normal to fragment shader

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        mat3 normalMatrix = transpose(inverse(mat3(view * model)));
        Normal = normalize(normalMatrix * aNormal);  // Transform normal to view space
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
)";

// World-Space Normals Fragment Shader Source Code
const char* worldspaceNormalsFragmentShaderSource = R"(
    #version 430 core
    layout(location = 0) out vec3 NormalColor;

    in vec3 Normal;

    void main() {
        NormalColor = normalize(Normal); // Output normalized view-space normal
    }
)";

// Gaussian fragment Shader source code
const char* HorizontalblurfragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;
    in vec2 TexCoords;

    uniform sampler2D image;
    uniform float weight[5];  // Array of weights for the blur
    uniform float texelOffsetX; // The horizontal offset per texel

    void main() {
        vec2 texOffset = vec2(texelOffsetX, 0.0);  // Offset in the X direction
        vec4 result = texture(image, TexCoords) * weight[0];  // Center pixel

        for (int i = 1; i < 5; ++i) {
            result += texture(image, TexCoords + texOffset * float(i)) * weight[i];
            result += texture(image, TexCoords - texOffset * float(i)) * weight[i];
        }

        FragColor = result;
    }
)";

// Gaussian fragment Shader source code
const char* VerticalblurfragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;
    in vec2 TexCoords;

    uniform sampler2D image;
    uniform float weight[5];  // Array of weights for the blur
    uniform float texelOffsetY; // The vertical offset per texel

    void main() {
        vec2 texOffset = vec2(0.0, texelOffsetY);  // Offset in the Y direction
        vec4 result = texture(image, TexCoords) * weight[0];  // Center pixel

        for (int i = 1; i < 5; ++i) {
            result += texture(image, TexCoords + texOffset * float(i)) * weight[i];
            result += texture(image, TexCoords - texOffset * float(i)) * weight[i];
        }

        FragColor = result;
    }
)";

// Position Vertex Shader Source Code
const char* positionVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragPos;

    void main() {
        vec4 worldPos = model * vec4(aPos, 1.0);
        FragPos = vec3(view * worldPos); // Transform to view space
        gl_Position = projection * view * vec4(aPos, 1.0);
    }
)";

// Position Fragment Shader Source Code
const char* positionFragmentShaderSource = R"(
    #version 430 core
    layout (location = 0) out vec3 PositionColor;

    in vec3 FragPos;

    void main() {
        PositionColor = FragPos; // Store positions in view space
    }
)";

// SSAO Vertex Shader Source Code
const char* ssaoVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
)";

// SSAO Fragment Shader Source Code
const char* ssaoFragmentShaderSource = R"(
    #version 430 core
    layout(location = 0) out float FragColor;

    in vec2 TexCoords;

    uniform sampler2D positionTexture;
    uniform sampler2D normalTexture;
    uniform sampler2D noiseTexture;
    uniform vec3 samples[32];
    uniform vec2 noiseScale;
    uniform mat4 projection;

    uniform float radius;
    uniform float bias;
    uniform int screenWidth;
    uniform int screenHeight;
    uniform int ssaoWidth;
    uniform int ssaoHeight;

    void main() {
        vec2 fullResTexCoords = TexCoords * vec2(float(screenWidth) / float(ssaoWidth), float(screenHeight) / float(ssaoHeight));
        vec3 fragPos = texture(positionTexture, fullResTexCoords).rgb;
        vec3 normal = normalize(texture(normalTexture, fullResTexCoords).rgb);
        vec3 randomVec = normalize(texture(noiseTexture, TexCoords * noiseScale).xyz);

        // Create TBN matrix
        vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
        vec3 bitangent = cross(normal, tangent);
        mat3 TBN = mat3(tangent, bitangent, normal);

        // Accumulate occlusion
        float occlusion = 0.0;
        for (int i = 0; i < 32; ++i) {
            // Sample position in view space
            vec3 samplePos = fragPos + TBN * samples[i] * radius;

            // Project sample position (only projection matrix since positions are in view space)
            vec4 offset = projection * vec4(samplePos, 1.0);
            offset.xyz /= offset.w;
            offset.xyz = offset.xyz * 0.5 + 0.5;

            // Get depth value of sample position
            float sampleDepth = texture(positionTexture, offset.xy).z;

            // Range check and accumulate occlusion
            float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
            occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
        }
        occlusion = 1.0 - (occlusion / 32.0);
        FragColor = occlusion;
    }
)";

const char* ssaoHorizontalBlurFragmentShaderSource = R"(
    #version 430 core
    out float FragColor;
    in vec2 TexCoords;

    uniform sampler2D image;
    uniform float weight[5];
    uniform float texelOffsetX;

    void main() {
        vec2 texOffset = vec2(texelOffsetX, 0.0);
        float result = texture(image, TexCoords).r * weight[0];

        for (int i = 1; i < 5; ++i) {
            result += texture(image, TexCoords + texOffset * float(i)).r * weight[i];
            result += texture(image, TexCoords - texOffset * float(i)).r * weight[i];
        }

        FragColor = result;
    }
)";

const char* ssaoVerticalBlurFragmentShaderSource = R"(
    #version 430 core
    out float FragColor;
    in vec2 TexCoords;

    uniform sampler2D image;
    uniform float weight[5];
    uniform float texelOffsetY;

    void main() {
        vec2 texOffset = vec2(0.0, texelOffsetY);
        float result = texture(image, TexCoords).r * weight[0];

        for (int i = 1; i < 5; ++i) {
            result += texture(image, TexCoords + texOffset * float(i)).r * weight[i];
            result += texture(image, TexCoords - texOffset * float(i)).r * weight[i];
        }

        FragColor = result;
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
        GLint diffuseLoc = glGetUniformLocation(shaderProgram, "diffuseTexture");
        if (diffuseLoc != -1) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, diffuseTexture);
            glUniform1i(diffuseLoc, 0);
        }

        GLint isGlowingLoc = glGetUniformLocation(shaderProgram, "isGlowing");
        if (glowPass) {
            if (isGlowing) {
                if (isGlowingLoc != -1) glUniform1i(isGlowingLoc, 1);
            }
            else {
                // Skip non-glowing materials in the glow pass
                return;
            }
        }
        else {
            if (isGlowingLoc != -1) glUniform1i(isGlowingLoc, isGlowing ? 1 : 0);
        }

        // Bind VAO and draw the mesh
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

void attachSharedDepthBuffer(GLuint framebuffer, GLuint depthMap) {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // Unbind framebuffer after attaching depth
}

GLuint createFramebuffer(GLuint& colorTexture, GLenum internalFormat, GLenum format, GLenum type, int width, int height) {
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glGenTextures(1, &colorTexture);
    glBindTexture(GL_TEXTURE_2D, colorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);

    GLenum attachments[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, attachments);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0); // Unbind framebuffer after setup
    return framebuffer;
}

struct Framebuffer {
    GLuint framebuffer;
    GLuint colorTexture;

    Framebuffer(GLenum internalFormat, GLenum format, GLenum type, int width, int height) {
        framebuffer = createFramebuffer(colorTexture, internalFormat, format, type, width, height);
    }

    void attachDepthBuffer(GLuint depthMap) {
        attachSharedDepthBuffer(framebuffer, depthMap);
    }
};


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

    glCullFace(GL_BACK); // Cull back faces (default)

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

    /* Blur shaders section */
    // Horizontal Blur Fragment Shader
    GLuint horizontalBlurFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(horizontalBlurFragmentShader, 1, &HorizontalblurfragmentShaderSource, NULL);
    glCompileShader(horizontalBlurFragmentShader);

    // Check for shader compile errors
    glGetShaderiv(horizontalBlurFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(horizontalBlurFragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::HORIZONTAL_BLUR_SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link Horizontal Blur Shader into a Program
    GLuint horizontalBlurShaderProgram = glCreateProgram();
    glAttachShader(horizontalBlurShaderProgram, quadVertexShader); // Use the quad vertex shader
    glAttachShader(horizontalBlurShaderProgram, horizontalBlurFragmentShader);
    glLinkProgram(horizontalBlurShaderProgram);

    // Check for linking errors
    glGetProgramiv(horizontalBlurShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(horizontalBlurShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::HORIZONTAL_BLUR_SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // Clean up shaders as they are no longer necessary after linking
    glDeleteShader(horizontalBlurFragmentShader);

    // Vertical Blur Fragment Shader
    GLuint verticalBlurFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(verticalBlurFragmentShader, 1, &VerticalblurfragmentShaderSource, NULL);
    glCompileShader(verticalBlurFragmentShader);

    // Check for shader compile errors
    glGetShaderiv(verticalBlurFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(verticalBlurFragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::VERTICAL_BLUR_SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link Vertical Blur Shader into a Program
    GLuint verticalBlurShaderProgram = glCreateProgram();
    glAttachShader(verticalBlurShaderProgram, quadVertexShader); // Use the quad vertex shader
    glAttachShader(verticalBlurShaderProgram, verticalBlurFragmentShader);
    glLinkProgram(verticalBlurShaderProgram);

    // Check for linking errors
    glGetProgramiv(verticalBlurShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(verticalBlurShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::VERTICAL_BLUR_SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // Clean up shaders as they are no longer necessary after linking
    glDeleteShader(verticalBlurFragmentShader);

    GLuint positionVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(positionVertexShader, 1, &positionVertexShaderSource, NULL);
    glCompileShader(positionVertexShader);

    glGetShaderiv(positionVertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(positionVertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::POSITION_VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint positionFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(positionFragmentShader, 1, &positionFragmentShaderSource, NULL);
    glCompileShader(positionFragmentShader);

    glGetShaderiv(positionFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(positionFragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::POSITION_FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint positionShaderProgram = glCreateProgram();
    glAttachShader(positionShaderProgram, positionVertexShader);
    glAttachShader(positionShaderProgram, positionFragmentShader);
    glLinkProgram(positionShaderProgram);

    glGetProgramiv(positionShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(positionShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::POSITION_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(positionVertexShader);
    glDeleteShader(positionFragmentShader);

    // SSAO Shader Program
    GLuint ssaoVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(ssaoVertexShader, 1, &ssaoVertexShaderSource, NULL);
    glCompileShader(ssaoVertexShader);

    glGetShaderiv(ssaoVertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(ssaoVertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::SSAO_VERTEX::COMPILATION_FAILED" << infoLog << std::endl;
    }

    GLuint ssaoFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(ssaoFragmentShader, 1, &ssaoFragmentShaderSource, NULL);
    glCompileShader(ssaoFragmentShader);

    glGetShaderiv(ssaoFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(ssaoFragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::SSAO_FRAGMENT::COMPILATION_FAILED" << infoLog << std::endl;
    }
    GLuint ssaoShaderProgram = glCreateProgram();
    glAttachShader(ssaoShaderProgram, ssaoVertexShader);
    glAttachShader(ssaoShaderProgram, ssaoFragmentShader);
    glLinkProgram(ssaoShaderProgram);
    glGetProgramiv(ssaoShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(ssaoShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::PROGRAM::SSAO_SHADER::LINKING_FAILED" << infoLog << std::endl;
    }

    glDeleteShader(ssaoVertexShader);
    glDeleteShader(ssaoFragmentShader);

    // Compile SSAO horizontal blur fragment shader
    GLuint ssaoHorizontalBlurFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(ssaoHorizontalBlurFragmentShader, 1, &ssaoHorizontalBlurFragmentShaderSource, NULL);
    glCompileShader(ssaoHorizontalBlurFragmentShader);

    // Check for shader compile errors
    glGetShaderiv(ssaoHorizontalBlurFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(ssaoHorizontalBlurFragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SSAO_HORIZONTAL_BLUR_SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link SSAO horizontal blur shader into a program
    GLuint ssaoHorizontalBlurShaderProgram = glCreateProgram();
    glAttachShader(ssaoHorizontalBlurShaderProgram, quadVertexShader); // Use the quad vertex shader
    glAttachShader(ssaoHorizontalBlurShaderProgram, ssaoHorizontalBlurFragmentShader);
    glLinkProgram(ssaoHorizontalBlurShaderProgram);

    // Check for linking errors
    glGetProgramiv(ssaoHorizontalBlurShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(ssaoHorizontalBlurShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SSAO_HORIZONTAL_BLUR_SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // Clean up shaders
    glDeleteShader(ssaoHorizontalBlurFragmentShader);

    // Compile and link SSAO vertical blur shader similarly
    GLuint ssaoVerticalBlurFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(ssaoVerticalBlurFragmentShader, 1, &ssaoVerticalBlurFragmentShaderSource, NULL);
    glCompileShader(ssaoVerticalBlurFragmentShader);
    // Check for errors...
    GLuint ssaoVerticalBlurShaderProgram = glCreateProgram();
    glAttachShader(ssaoVerticalBlurShaderProgram, quadVertexShader);
    glAttachShader(ssaoVerticalBlurShaderProgram, ssaoVerticalBlurFragmentShader);
    glLinkProgram(ssaoVerticalBlurShaderProgram);
    // Check for errors...
    glDeleteShader(ssaoVerticalBlurFragmentShader);
    // Delete quadVertex shader after it's been used by the other shaders above
    glDeleteShader(quadVertexShader);

    /* FBO setup */

    // Create depth texture
    GLuint depthMap;
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, WIDTH, HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Main Color Pass (Scene)
    Framebuffer colorPass(GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, WIDTH, HEIGHT);

    // Normal Pass
    Framebuffer normalPass(GL_RGB16F, GL_RGB, GL_FLOAT, WIDTH, HEIGHT);

    // Glow Pass
    Framebuffer glowPass(GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, WIDTH, HEIGHT);

    // Position Pass
    Framebuffer positionPass(GL_RGB32F, GL_RGB, GL_FLOAT, WIDTH, HEIGHT);

    // Create a half-resolution framebuffer for SSAO
    int ssaoWidth = WIDTH / 2;
    int ssaoHeight = HEIGHT / 2;
    Framebuffer ssaoPass(GL_R16F, GL_RED, GL_FLOAT, ssaoWidth, ssaoHeight);

    // Set texture parameters for the SSAO texture
    glBindTexture(GL_TEXTURE_2D, ssaoPass.colorTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // For minification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // For magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Prevent wrapping artifacts
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    Framebuffer ssaoBlurPass1(GL_R16F, GL_RED, GL_FLOAT, ssaoWidth, ssaoHeight);
    Framebuffer ssaoBlurPass2(GL_R16F, GL_RED, GL_FLOAT, ssaoWidth, ssaoHeight);

    // Set texture parameters for the SSAO blur textures
    glBindTexture(GL_TEXTURE_2D, ssaoBlurPass1.colorTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // For minification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // For magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Prevent wrapping artifacts
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, ssaoBlurPass2.colorTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // For minification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // For magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Prevent wrapping artifacts
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Blur Passes
    Framebuffer blurPass1(GL_RGB16F, GL_RGB, GL_FLOAT, WIDTH, HEIGHT);
    Framebuffer blurPass2(GL_RGB16F, GL_RGB, GL_FLOAT, WIDTH, HEIGHT);


    // Attach shared depth buffer to each framebuffer
    colorPass.attachDepthBuffer(depthMap);
    normalPass.attachDepthBuffer(depthMap);
    glowPass.attachDepthBuffer(depthMap);
    positionPass.attachDepthBuffer(depthMap);

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
    // Gaussian weights for a 9-tap kernel (example values)
    float weight[5] = { 0.2270270270f, 0.1945945946f, 0.1216216216f, 0.0540540541f, 0.0162162162f };

    // Generate sample kernel
    std::vector<glm::vec3> ssaoKernel;
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
    std::default_random_engine generator;

    // Generate a hemisphere of samples in the positive z direction
    for (unsigned int i = 0; i < 32; ++i) {
        glm::vec3 sample = glm::vec3(
            (randomFloats(generator) * 2.0 - 1.0),
            (randomFloats(generator) * 2.0 - 1.0),
            randomFloats(generator)
        );
        sample = normalize(sample);
        sample *= randomFloats(generator);
        float scale = float(i) / 32.0;
        scale = glm::mix(0.1f, 1.0f, scale * scale);
        sample *= scale;
        ssaoKernel.push_back(sample);
    }


    // Generate noise texture
    std::vector<glm::vec3> ssaoNoise;
    for (unsigned int i = 0; i < 16; i++) {
        glm::vec3 noise(
            randomFloats(generator) * 2.0 - 1.0,
            randomFloats(generator) * 2.0 - 1.0,
            0.0f); // Rotate around z-axis (in tangent space)
        ssaoNoise.push_back(noise);
    }

    GLuint ssaonoiseTexture;
    glGenTextures(1, &ssaonoiseTexture);
    glBindTexture(GL_TEXTURE_2D, ssaonoiseTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

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
        glBindFramebuffer(GL_FRAMEBUFFER, colorPass.framebuffer);
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
        glBindFramebuffer(GL_FRAMEBUFFER, glowPass.framebuffer);
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

        // Render to position pass framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, positionPass.framebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(positionShaderProgram);

        glUniformMatrix4fv(glGetUniformLocation(positionShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(positionShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Render all objects using the position shader program
        for (const auto& mesh : meshes) {
            glm::mat4 model = glm::mat4(1.0f);
            glUniformMatrix4fv(glGetUniformLocation(positionShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
            mesh.Draw(positionShaderProgram, false);  // Use positionShaderProgram here
        }

        // Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Blur pass - horizontal and vertical blur
        bool horizontal = true, first_iteration = true;
        unsigned int amount = 10; // Number of blur passes

        for (unsigned int i = 0; i < amount; i++) {
            glBindFramebuffer(GL_FRAMEBUFFER, horizontal ? blurPass1.framebuffer : blurPass2.framebuffer);
            glClear(GL_COLOR_BUFFER_BIT);

            // Switch shaders and set uniforms
            if (horizontal) {
                glUseProgram(horizontalBlurShaderProgram);
                // Set uniforms for horizontal blur shader
                glUniform1fv(glGetUniformLocation(horizontalBlurShaderProgram, "weight"), 5, weight);
                glUniform1f(glGetUniformLocation(horizontalBlurShaderProgram, "texelOffsetX"), 1.0f / WIDTH);
                glUniform1i(glGetUniformLocation(horizontalBlurShaderProgram, "image"), 0);
            }
            else {
                glUseProgram(verticalBlurShaderProgram);
                // Set uniforms for vertical blur shader
                glUniform1fv(glGetUniformLocation(verticalBlurShaderProgram, "weight"), 5, weight);
                glUniform1f(glGetUniformLocation(verticalBlurShaderProgram, "texelOffsetY"), 1.0f / HEIGHT);
                glUniform1i(glGetUniformLocation(verticalBlurShaderProgram, "image"), 0);
            }

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, first_iteration ? glowPass.colorTexture : (horizontal ? blurPass2.colorTexture : blurPass1.colorTexture));

            glBindVertexArray(quadVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);

            horizontal = !horizontal;
            if (first_iteration)
                first_iteration = false;
        }

        // Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // ========== Third Pass: World-Space Normals Rendering ==========
        glBindFramebuffer(GL_FRAMEBUFFER, normalPass.framebuffer);
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


        // SSAO Pass
        glViewport(0, 0, ssaoWidth, ssaoHeight);
        glBindFramebuffer(GL_FRAMEBUFFER, ssaoPass.framebuffer);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(ssaoShaderProgram);

        // Set kernel samples
        for (unsigned int i = 0; i < 32; ++i) {
            std::string uniformName = "samples[" + std::to_string(i) + "]";
            glUniform3fv(glGetUniformLocation(ssaoShaderProgram, uniformName.c_str()), 1, glm::value_ptr(ssaoKernel[i]));
        }

        // Set SSAO parameters
        float radius = 0.5f;
        float bias = 0.025f;
        glUniform1f(glGetUniformLocation(ssaoShaderProgram, "radius"), radius);
        glUniform1f(glGetUniformLocation(ssaoShaderProgram, "bias"), bias);

        // Set uniforms and bind G-buffer textures
        glUniform1i(glGetUniformLocation(ssaoShaderProgram, "positionTexture"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, positionPass.colorTexture);

        glUniform1i(glGetUniformLocation(ssaoShaderProgram, "normalTexture"), 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalPass.colorTexture);

        glUniform1i(glGetUniformLocation(ssaoShaderProgram, "noiseTexture"), 2);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, ssaonoiseTexture);

        // Set projection matrix
        glUniformMatrix4fv(glGetUniformLocation(ssaoShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Set noise scale
        glm::vec2 noiseScale = glm::vec2((float)ssaoWidth / 4.0f, (float)ssaoHeight / 4.0f);
        glUniform2fv(glGetUniformLocation(ssaoShaderProgram, "noiseScale"), 1, glm::value_ptr(noiseScale));

        glUniform1i(glGetUniformLocation(ssaoShaderProgram, "screenWidth"), WIDTH);
        glUniform1i(glGetUniformLocation(ssaoShaderProgram, "screenHeight"), HEIGHT);
        glUniform1i(glGetUniformLocation(ssaoShaderProgram, "ssaoWidth"), ssaoWidth);
        glUniform1i(glGetUniformLocation(ssaoShaderProgram, "ssaoHeight"), ssaoHeight);

        // Render the full-screen quad
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Blur SSAO texture - horizontal and vertical blur
        bool ssaoHorizontal = true, ssaoFirst_iteration = true;
        unsigned int ssaoBlurAmount = 2; // Number of blur passes for SSAO

        // Set viewport for SSAO blur passes
        glViewport(0, 0, ssaoWidth, ssaoHeight);

        for (unsigned int i = 0; i < ssaoBlurAmount; i++) {
            glBindFramebuffer(GL_FRAMEBUFFER, ssaoHorizontal ? ssaoBlurPass1.framebuffer : ssaoBlurPass2.framebuffer);
            glClear(GL_COLOR_BUFFER_BIT);

            // Switch shaders and set uniforms
            if (ssaoHorizontal) {
                glUseProgram(ssaoHorizontalBlurShaderProgram);
                glUniform1fv(glGetUniformLocation(ssaoHorizontalBlurShaderProgram, "weight"), 5, weight);
                glUniform1f(glGetUniformLocation(ssaoHorizontalBlurShaderProgram, "texelOffsetX"), 1.0f / ssaoWidth);
                glUniform1i(glGetUniformLocation(ssaoHorizontalBlurShaderProgram, "image"), 0);
            }
            else {
                glUseProgram(ssaoVerticalBlurShaderProgram);
                glUniform1fv(glGetUniformLocation(ssaoVerticalBlurShaderProgram, "weight"), 5, weight);
                glUniform1f(glGetUniformLocation(ssaoVerticalBlurShaderProgram, "texelOffsetY"), 1.0f / ssaoHeight);
                glUniform1i(glGetUniformLocation(ssaoVerticalBlurShaderProgram, "image"), 0);
            }

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, ssaoFirst_iteration ? ssaoPass.colorTexture : (ssaoHorizontal ? ssaoBlurPass2.colorTexture : ssaoBlurPass1.colorTexture));

            glBindVertexArray(quadVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);

            ssaoHorizontal = !ssaoHorizontal;
            if (ssaoFirst_iteration)
                ssaoFirst_iteration = false;
        }

        // Unbind the framebuffer after the blur passes
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Reset viewport to full screen
        glViewport(0, 0, WIDTH, HEIGHT);

        // Determine the final blurred SSAO texture
        GLuint ssaoBlurredTexture = ssaoHorizontal ? ssaoBlurPass2.colorTexture : ssaoBlurPass1.colorTexture;

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
        else if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS && !keyPressed) {
            debugMode = 4;  // Show glow buffer
            keyPressed = true;
        }
        else if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS && !keyPressed) {
            debugMode = 5;  // Show position buffer
            keyPressed = true;
        }
        else if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS && !keyPressed) {
            debugMode = 6;  // Show ssao buffer
            keyPressed = true;
        }

        // Reset keyPressed state when the keys are released
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_2) == GLFW_RELEASE &&
            glfwGetKey(window, GLFW_KEY_3) == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_4) == GLFW_RELEASE && 
            glfwGetKey(window, GLFW_KEY_5) == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_6) == GLFW_RELEASE && 
            glfwGetKey(window, GLFW_KEY_7) == GLFW_RELEASE) {
            keyPressed = false;
        }

        // Use the post-processing (quad) shader program
        glUseProgram(quadShaderProgram);

        // Set fog parameters
        glm::vec3 fogColor(0.337f, 0.349f, 0.435f); // Fog color
        float fogStart = 1.0f;
        float fogEnd = 20.0f;
        float fogDensity = 0.12f;
        glUniform1i(glGetUniformLocation(quadShaderProgram, "debugMode"), debugMode);
        glUniform1i(glGetUniformLocation(quadShaderProgram, "screenWidth"), WIDTH);
        glUniform1i(glGetUniformLocation(quadShaderProgram, "screenHeight"), HEIGHT);
        glUniform1i(glGetUniformLocation(quadShaderProgram, "ssaoWidth"), ssaoWidth);
        glUniform1i(glGetUniformLocation(quadShaderProgram, "ssaoHeight"), ssaoHeight);
        glUniform3fv(glGetUniformLocation(quadShaderProgram, "fogColor"), 1, glm::value_ptr(fogColor));
        glUniform1f(glGetUniformLocation(quadShaderProgram, "fogDensity"), fogDensity);
        glUniform1f(glGetUniformLocation(quadShaderProgram, "fogStart"), fogStart);
        glUniform1f(glGetUniformLocation(quadShaderProgram, "fogEnd"), fogEnd);
        glUniform1f(glGetUniformLocation(quadShaderProgram, "near"), camera.getNearPlane());
        glUniform1f(glGetUniformLocation(quadShaderProgram, "far"), camera.getFarPlane());
        glUniform1f(glGetUniformLocation(quadShaderProgram, "bloomIntensity"), 2.0f);

        // Bind the textures for the quad (color, normal, depth, glow, blurred glow, position)
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorPass.colorTexture);
        GLint colorMapLoc = glGetUniformLocation(quadShaderProgram, "colorMap");
        if (colorMapLoc != -1) {
            glUniform1i(colorMapLoc, 0);
        }
        else {
            std::cerr << "Uniform 'colorMap' not found in shader program." << std::endl;
        }

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalPass.colorTexture);
        GLint normalMapLoc = glGetUniformLocation(quadShaderProgram, "normalMap");
        if (normalMapLoc != -1) {
            glUniform1i(normalMapLoc, 1);
        }
        else {
            std::cerr << "Uniform 'normalMap' not found in shader program." << std::endl;
        }

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        GLint depthMapLoc = glGetUniformLocation(quadShaderProgram, "depthMap");
        if (depthMapLoc != -1) {
            glUniform1i(depthMapLoc, 2);
        }
        else {
            std::cerr << "Uniform 'depthMap' not found in shader program." << std::endl;
        }

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, glowPass.colorTexture);
        GLint glowTextureLoc = glGetUniformLocation(quadShaderProgram, "glowTexture");
        if (glowTextureLoc != -1) {
            glUniform1i(glowTextureLoc, 3);
        }
        else {
            std::cerr << "Uniform 'glowTexture' not found in shader program." << std::endl;
        }

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, blurPass2.colorTexture);
        GLint blurredGlowLoc = glGetUniformLocation(quadShaderProgram, "blurredGlowTexture");
        if (blurredGlowLoc != -1) {
            glUniform1i(blurredGlowLoc, 4);
        }
        else {
            std::cerr << "Uniform 'blurredGlowTexture' not found in shader program." << std::endl;
        }

        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, positionPass.colorTexture);
        GLint positionTextureLoc = glGetUniformLocation(quadShaderProgram, "positionTexture");
        if (positionTextureLoc != -1) {
            glUniform1i(positionTextureLoc, 5);
        }
        else {
            std::cerr << "Uniform 'positionTexture' not found in shader program." << std::endl;
        }

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, ssaoBlurredTexture);
        GLint ssaoTextureLoc = glGetUniformLocation(quadShaderProgram, "ssaoTexture");
        if (ssaoTextureLoc != -1) {
            glUniform1i(ssaoTextureLoc, 6);
        }
        else {
            std::cerr << "Uniform 'ssaoTexture' not found in shader program." << std::endl;
        }

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
    glDeleteProgram(quadShaderProgram);
    glDeleteProgram(worldspaceNormalsShaderProgram);
    glDeleteProgram(horizontalBlurShaderProgram);
    glDeleteProgram(verticalBlurShaderProgram);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteFramebuffers(1, &depthMap);

    glfwTerminate();
    return 0;
}
