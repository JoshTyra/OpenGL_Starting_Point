// PhysicsWorld.h
#pragma once

#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>
#include <vector>
#include "PhysicsDebugDrawer.h"

class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    void initialize();
    void update(float deltaTime);
    void addGroundPlane(float y = 0.0f);
    int addRigidBody(const glm::vec3& position, const glm::vec3& size, float mass);
    int addCapsuleRigidBody(const glm::vec3& position, float radius, float height, float mass, float yOffset = 0.0f);
    glm::mat4 getTransform(int index) const;
    void removeRigidBody(int index);
    size_t getNumBodies() const;
    void applyForce(int bodyIndex, const glm::vec3& force);
    void applyImpulse(int bodyIndex, const glm::vec3& impulse);

    void setDebugDrawer(PhysicsDebugDrawer* debugDrawer);
    void debugDraw(const glm::mat4& viewProjection);

private:
    btDefaultCollisionConfiguration* collisionConfiguration;
    btCollisionDispatcher* dispatcher;
    btBroadphaseInterface* overlappingPairCache;
    btSequentialImpulseConstraintSolver* solver;
    btDiscreteDynamicsWorld* dynamicsWorld;
    PhysicsDebugDrawer* m_debugDrawer;

    std::vector<btRigidBody*> rigidBodies;

    btRigidBody* createRigidBody(const glm::vec3& position, const glm::vec3& size, float mass);
};