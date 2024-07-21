// PhysicsWorld.cpp
#include "PhysicsWorld.h"
#include <glm/gtc/type_ptr.hpp>

PhysicsWorld::PhysicsWorld() : collisionConfiguration(nullptr), dispatcher(nullptr),
overlappingPairCache(nullptr), solver(nullptr), dynamicsWorld(nullptr) {}

PhysicsWorld::~PhysicsWorld() {
    for (auto body : rigidBodies) {
        dynamicsWorld->removeRigidBody(body);
        delete body->getMotionState();
        delete body->getCollisionShape();
        delete body;
    }

    delete dynamicsWorld;
    delete solver;
    delete overlappingPairCache;
    delete dispatcher;
    delete collisionConfiguration;
}

void PhysicsWorld::initialize() {
    collisionConfiguration = new btDefaultCollisionConfiguration();
    dispatcher = new btCollisionDispatcher(collisionConfiguration);
    overlappingPairCache = new btDbvtBroadphase();
    solver = new btSequentialImpulseConstraintSolver;
    dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, overlappingPairCache, solver, collisionConfiguration);

    dynamicsWorld->setGravity(btVector3(0, -9.81f, 0));
}

void PhysicsWorld::update(float deltaTime) {
    dynamicsWorld->stepSimulation(deltaTime, 10);
}

void PhysicsWorld::addGroundPlane(float y) {
    btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), y);
    btDefaultMotionState* groundMotionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, y, 0)));
    btRigidBody::btRigidBodyConstructionInfo groundRigidBodyCI(0, groundMotionState, groundShape, btVector3(0, 0, 0));
    btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
    dynamicsWorld->addRigidBody(groundRigidBody);
    rigidBodies.push_back(groundRigidBody);
}

int PhysicsWorld::addRigidBody(const glm::vec3& position, const glm::vec3& size, float mass) {
    btRigidBody* body = createRigidBody(position, size, mass);
    dynamicsWorld->addRigidBody(body);
    rigidBodies.push_back(body);
    return rigidBodies.size() - 1;  // Return the index of the new body
}

int PhysicsWorld::addCapsuleRigidBody(const glm::vec3& position, float radius, float height, float mass, float yOffset) {
    btCollisionShape* capsuleShape = new btCapsuleShape(radius, height);

    btTransform transform;
    transform.setIdentity();
    transform.setOrigin(btVector3(position.x, position.y, position.z));

    // Create a child transform for the capsule shape
    btTransform localTransform;
    localTransform.setIdentity();
    // Move the capsule up by yOffset
    localTransform.setOrigin(btVector3(0, yOffset, 0));

    // Create a compound shape
    btCompoundShape* compoundShape = new btCompoundShape();
    compoundShape->addChildShape(localTransform, capsuleShape);

    btVector3 localInertia(0, 0, 0);
    if (mass != 0.0f) compoundShape->calculateLocalInertia(mass, localInertia);

    btDefaultMotionState* motionState = new btDefaultMotionState(transform);
    btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, motionState, compoundShape, localInertia);
    btRigidBody* body = new btRigidBody(rbInfo);

    dynamicsWorld->addRigidBody(body);
    rigidBodies.push_back(body);
    return rigidBodies.size() - 1;
}

glm::mat4 PhysicsWorld::getTransform(int index) const {
    if (index < 0 || index >= rigidBodies.size()) return glm::mat4(1.0f);

    btTransform transform;
    rigidBodies[index]->getMotionState()->getWorldTransform(transform);

    glm::mat4 glmTransform;
    transform.getOpenGLMatrix(glm::value_ptr(glmTransform));

    return glmTransform;
}

btRigidBody* PhysicsWorld::createRigidBody(const glm::vec3& position, const glm::vec3& size, float mass) {
    btCollisionShape* shape = new btBoxShape(btVector3(size.x / 2, size.y / 2, size.z / 2));

    btTransform transform;
    transform.setIdentity();
    transform.setOrigin(btVector3(position.x, position.y, position.z));

    btVector3 localInertia(0, 0, 0);
    if (mass != 0.0f) shape->calculateLocalInertia(mass, localInertia);

    btDefaultMotionState* motionState = new btDefaultMotionState(transform);
    btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, motionState, shape, localInertia);
    return new btRigidBody(rbInfo);
}

void PhysicsWorld::removeRigidBody(int index) {
    if (index >= 0 && index < rigidBodies.size()) {
        dynamicsWorld->removeRigidBody(rigidBodies[index]);
        delete rigidBodies[index]->getMotionState();
        delete rigidBodies[index]->getCollisionShape();
        delete rigidBodies[index];
        rigidBodies.erase(rigidBodies.begin() + index);
    }
}

size_t PhysicsWorld::getNumBodies() const {
    return rigidBodies.size();
}

void PhysicsWorld::applyForce(int bodyIndex, const glm::vec3& force) {
    if (bodyIndex >= 0 && bodyIndex < rigidBodies.size()) {
        rigidBodies[bodyIndex]->applyCentralForce(btVector3(force.x, force.y, force.z));
    }
}

void PhysicsWorld::applyImpulse(int bodyIndex, const glm::vec3& impulse) {
    if (bodyIndex >= 0 && bodyIndex < rigidBodies.size()) {
        rigidBodies[bodyIndex]->applyCentralImpulse(btVector3(impulse.x, impulse.y, impulse.z));
    }
}

void PhysicsWorld::setDebugDrawer(PhysicsDebugDrawer* debugDrawer) {
    m_debugDrawer = debugDrawer;
    dynamicsWorld->setDebugDrawer(m_debugDrawer);
}

void PhysicsWorld::debugDraw(const glm::mat4& viewProjection) {
    if (m_debugDrawer) {
        dynamicsWorld->debugDrawWorld();
        m_debugDrawer->render(viewProjection);
    }
}