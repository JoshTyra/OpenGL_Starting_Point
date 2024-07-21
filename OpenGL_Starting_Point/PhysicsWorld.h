// PhysicsWorld.h
#pragma once

#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>
#include <vector>

class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    void initialize();
    void update(float deltaTime);
    void addGroundPlane(float y = 0.0f);
    int addRigidBody(const glm::vec3& position, const glm::vec3& size, float mass);
    glm::mat4 getTransform(int index) const;
    void removeRigidBody(int index);
    size_t getNumBodies() const;
    void applyForce(int bodyIndex, const glm::vec3& force);
    void applyImpulse(int bodyIndex, const glm::vec3& impulse);

private:
    btDefaultCollisionConfiguration* collisionConfiguration;
    btCollisionDispatcher* dispatcher;
    btBroadphaseInterface* overlappingPairCache;
    btSequentialImpulseConstraintSolver* solver;
    btDiscreteDynamicsWorld* dynamicsWorld;

    std::vector<btRigidBody*> rigidBodies;

    btRigidBody* createRigidBody(const glm::vec3& position, const glm::vec3& size, float mass);
};