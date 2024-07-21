// PhysicsDebugDrawer.h
#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <btBulletDynamicsCommon.h>
#include <vector>

class PhysicsDebugDrawer : public btIDebugDraw {
public:
    PhysicsDebugDrawer();
    ~PhysicsDebugDrawer();

    void drawLine(const btVector3& from, const btVector3& to, const btVector3& color) override;
    void drawContactPoint(const btVector3& pointOnB, const btVector3& normalOnB, btScalar distance, int lifeTime, const btVector3& color) override;
    void reportErrorWarning(const char* warningString) override;
    void draw3dText(const btVector3& location, const char* textString) override;
    void setDebugMode(int debugMode) override;
    int getDebugMode() const override;

    void init();
    void render(const glm::mat4& viewProjection);

private:
    int m_debugMode;
    GLuint m_shaderProgram;
    GLuint m_vao;
    GLuint m_vbo;
    std::vector<float> m_lineVertices;

    void compileShaders();
};