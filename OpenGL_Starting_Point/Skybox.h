#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <string>
#include <iostream>

class Skybox {
public:
    Skybox(const std::string& ktxFilePath);
    ~Skybox();

    void draw(const glm::mat4& view, const glm::mat4& projection);

    // Debug methods
    void printDebugInfo() const;
    bool isValid() const { return m_isValid; }

private:
    unsigned int loadCubemap(const std::string& ktxFilePath);
    void setupMesh();
    void checkGLError(const char* operation);

    unsigned int VAO, VBO;
    unsigned int cubemapTexture;
    unsigned int shaderProgram;
    bool m_isValid;

    // Debug info
    int m_textureWidth, m_textureHeight;
    std::string m_lastError;
};
