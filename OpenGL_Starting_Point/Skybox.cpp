#include "Skybox.h"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <ktx.h>

const char* skyboxVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;

    out vec3 TexCoords;

    uniform mat4 projection;
    uniform mat4 view;

    void main()
    {
        TexCoords = aPos;
        vec4 pos = projection * view * vec4(aPos, 1.0);
        gl_Position = pos.xyww;
    }
)";

const char* skyboxFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;

    in vec3 TexCoords;

    uniform samplerCube skybox;

    vec3 SRGBToLinear(vec3 srgbColor) {
        return pow(srgbColor, vec3(0.5));  // Correct gamma conversion
    }

    void main()
    {    
        vec4 texColor = texture(skybox, TexCoords);
        vec3 linearColor = SRGBToLinear(texColor.rgb);
        FragColor = vec4(linearColor, texColor.a);
    }
)";

Skybox::Skybox(const std::string& ktxFilePath) : m_isValid(false), m_textureWidth(0), m_textureHeight(0) {
    cubemapTexture = loadCubemap(ktxFilePath);
    if (cubemapTexture == 0) {
        m_isValid = false;
        return;
    }

    setupMesh();

    // Compile and link shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &skyboxVertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkGLError("Vertex Shader Compilation");

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &skyboxFragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkGLError("Fragment Shader Compilation");

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    checkGLError("Shader Program Linking");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    m_isValid = true;
}

Skybox::~Skybox() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteTextures(1, &cubemapTexture);
    glDeleteProgram(shaderProgram);
}

unsigned int Skybox::loadCubemap(const std::string& ktxFilePath) {
    unsigned int textureID = 0;
    ktxTexture* kTexture;
    KTX_error_code result = ktxTexture_CreateFromNamedFile(ktxFilePath.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &kTexture);
    if (result != KTX_SUCCESS) {
        std::cerr << "Failed to load KTX texture: " << ktxFilePath << std::endl;
        m_lastError = "Failed to load KTX texture";
        return 0;
    }

    // Upload the texture to OpenGL
    GLenum target;
    result = ktxTexture_GLUpload(kTexture, &textureID, &target, nullptr);
    if (result != KTX_SUCCESS) {
        std::cerr << "Failed to upload KTX texture to OpenGL" << std::endl;
        m_lastError = "Failed to upload KTX texture to OpenGL";
        ktxTexture_Destroy(kTexture);
        return 0;
    }

    // Ensure the texture is a cubemap
    if (target != GL_TEXTURE_CUBE_MAP) {
        std::cerr << "Loaded KTX texture is not a cubemap" << std::endl;
        m_lastError = "Loaded KTX texture is not a cubemap";
        ktxTexture_Destroy(kTexture);
        return 0;
    }

    // Set texture parameters
    glBindTexture(target, textureID);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(target, 0);

    checkGLError("Cubemap Texture Loading");

    ktxTexture_Destroy(kTexture);
    return textureID;
}

void Skybox::setupMesh() {
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

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    checkGLError("Skybox Mesh Setup");
}

void Skybox::draw(const glm::mat4& view, const glm::mat4& projection) {
    if (!m_isValid) {
        std::cerr << "Cannot draw invalid Skybox." << std::endl;
        return;
    }

    glDepthFunc(GL_LEQUAL);
    glUseProgram(shaderProgram);

    glm::mat4 skyboxView = glm::mat4(glm::mat3(view)); // Remove translation from the view matrix
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(skyboxView));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    glBindVertexArray(VAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
    glUniform1i(glGetUniformLocation(shaderProgram, "skybox"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    glDepthFunc(GL_LESS);
    checkGLError("Skybox Draw");
}

void Skybox::checkGLError(const char* operation) {
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL error after " << operation << ": " << std::hex << error << std::dec << std::endl;
        m_lastError = std::string(operation) + ": GL error " + std::to_string(error);
    }
}

void Skybox::printDebugInfo() const {
    std::cout << "Skybox Debug Information:" << std::endl;
    std::cout << "  Is Valid: " << (m_isValid ? "Yes" : "No") << std::endl;
    std::cout << "  Texture Dimensions: " << m_textureWidth << "x" << m_textureHeight << std::endl;
    std::cout << "  VAO: " << VAO << std::endl;
    std::cout << "  VBO: " << VBO << std::endl;
    std::cout << "  Cubemap Texture ID: " << cubemapTexture << std::endl;
    std::cout << "  Shader Program ID: " << shaderProgram << std::endl;
    std::cout << "  Last Error: " << (m_lastError.empty() ? "None" : m_lastError) << std::endl;
}
