// Skybox.cpp
#include "Skybox.h"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <stb_image.h>

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

    void main()
    {    
        FragColor = texture(skybox, TexCoords);
    }
)";

Skybox::Skybox(const std::vector<std::string>& faces) : m_isValid(false), m_textureWidth(0), m_textureHeight(0) {
    cubemapTexture = loadCubemap(faces);
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
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    glDepthFunc(GL_LESS);
    checkGLError("Skybox Draw");
}

unsigned int Skybox::loadCubemap(const std::vector<std::string>& faces) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++) {
        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            GLenum format;
            GLenum internalFormat;
            if (nrChannels == 4) {
                format = GL_RGBA;
                internalFormat = GL_RGBA8;
            }
            else if (nrChannels == 3) {
                format = GL_RGB;
                internalFormat = GL_RGB8;
            }
            else {
                std::cerr << "Unexpected number of channels (" << nrChannels
                    << ") in cubemap texture: " << faces[i] << std::endl;
                stbi_image_free(data);
                continue;
            }

            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, data);

            stbi_image_free(data);
            if (i == 0) {
                m_textureWidth = width;
                m_textureHeight = height;
            }
        }
        else {
            std::cerr << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
            m_lastError = "Failed to load texture: " + faces[i];
            return 0;
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    checkGLError("Cubemap Texture Loading");
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

void Skybox::checkGLError(const char* operation) {
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL error after " << operation << ": " << error << std::endl;
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