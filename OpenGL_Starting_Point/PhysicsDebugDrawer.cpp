// PhysicsDebugDrawer.cpp
#include "PhysicsDebugDrawer.h"
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

const char* vertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;
    out vec3 Color;
    uniform mat4 viewProjection;
    void main() {
        gl_Position = viewProjection * vec4(aPos, 1.0);
        Color = aColor;
    }
)";

const char* fragmentShaderSource = R"(
    #version 430 core
    in vec3 Color;
    out vec4 FragColor;
    void main() {
        FragColor = vec4(Color, 1.0);
    }
)";

PhysicsDebugDrawer::PhysicsDebugDrawer() : m_debugMode(btIDebugDraw::DBG_DrawWireframe) {}

PhysicsDebugDrawer::~PhysicsDebugDrawer() {
    glDeleteProgram(m_shaderProgram);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
}

void PhysicsDebugDrawer::init() {
    compileShaders();
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
}

void PhysicsDebugDrawer::compileShaders() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vertexShader);
    glAttachShader(m_shaderProgram, fragmentShader);
    glLinkProgram(m_shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void PhysicsDebugDrawer::drawLine(const btVector3& from, const btVector3& to, const btVector3& color) {
    m_lineVertices.insert(m_lineVertices.end(), { from.x(), from.y(), from.z(), color.x(), color.y(), color.z() });
    m_lineVertices.insert(m_lineVertices.end(), { to.x(), to.y(), to.z(), color.x(), color.y(), color.z() });
}

void PhysicsDebugDrawer::drawContactPoint(const btVector3& pointOnB, const btVector3& normalOnB, btScalar distance, int lifeTime, const btVector3& color) {
    drawLine(pointOnB, pointOnB + normalOnB * distance, color);
}

void PhysicsDebugDrawer::reportErrorWarning(const char* warningString) {
    std::cout << "Bullet Physics Warning: " << warningString << std::endl;
}

void PhysicsDebugDrawer::draw3dText(const btVector3& location, const char* textString) {
    // Implement if needed
}

void PhysicsDebugDrawer::setDebugMode(int debugMode) {
    m_debugMode = debugMode;
}

int PhysicsDebugDrawer::getDebugMode() const {
    return m_debugMode;
}

void PhysicsDebugDrawer::render(const glm::mat4& viewProjection) {
    glUseProgram(m_shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "viewProjection"), 1, GL_FALSE, glm::value_ptr(viewProjection));

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_lineVertices.size() * sizeof(float), m_lineVertices.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    glDrawArrays(GL_LINES, 0, m_lineVertices.size() / 6);

    glBindVertexArray(0);
    glUseProgram(0);

    m_lineVertices.clear();
}