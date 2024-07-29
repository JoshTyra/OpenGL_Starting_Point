// NPCManager.cpp
#include "NPCManager.h"

NPCManager::NPCManager(size_t maxNPCs, PhysicsWorld& physicsWorld)
    : m_physicsWorld(physicsWorld), m_maxNPCs(maxNPCs), m_nextID(0) {
    m_npcs.reserve(maxNPCs);
}

NPCManager::~NPCManager() {
    for (auto& npc : m_npcs) {
        if (npc->getRigidBody()) {
            m_physicsWorld.removeRigidBody(npc->getRigidBody());
        }
    }
}

void NPCManager::update(float deltaTime) {
    for (auto& npc : m_npcs) {
        if (npc->isActive()) {
            npc->update(deltaTime);
        }
    }
}

int NPCManager::addNPC(const glm::vec3& position) {
    if (m_npcs.size() >= m_maxNPCs) {
        return -1; // No more space for new NPCs
    }

    int id = m_nextID++;
    auto npc = std::make_unique<NPC>(id, position);

    // Create physics body for the NPC
    btCollisionShape* shape = new btCapsuleShape(0.5f, 1.0f);
    btTransform transform;
    transform.setIdentity();
    transform.setOrigin(btVector3(position.x, position.y, position.z));
    btScalar mass = 1.0f;
    btVector3 localInertia(0, 0, 0);
    shape->calculateLocalInertia(mass, localInertia);
    btDefaultMotionState* motionState = new btDefaultMotionState(transform);
    btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, motionState, shape, localInertia);
    btRigidBody* body = new btRigidBody(rbInfo);

    npc->setRigidBody(body);
    m_physicsWorld.addRigidBody(body);

    m_npcs.push_back(std::move(npc));
    return id;
}

void NPCManager::removeNPC(int id) {
    auto it = std::find_if(m_npcs.begin(), m_npcs.end(),
        [id](const std::unique_ptr<NPC>& npc) { return npc->getID() == id; });

    if (it != m_npcs.end()) {
        if ((*it)->getRigidBody()) {
            m_physicsWorld.removeRigidBody((*it)->getRigidBody());
        }
        (*it)->setActive(false);
    }
}

NPC* NPCManager::getNPC(int id) {
    auto it = std::find_if(m_npcs.begin(), m_npcs.end(),
        [id](const std::unique_ptr<NPC>& npc) { return npc->getID() == id; });

    return it != m_npcs.end() ? it->get() : nullptr;
}

void NPCManager::updateNPCs(float deltaTime) {
    for (auto& npc : m_npcs) {
        npc->update(deltaTime);
    }
}

void NPCManager::initializeNPCs(float worldSize, const glm::mat4& originalModelMatrix) {
    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-worldSize / 2, worldSize / 2);
    std::uniform_real_distribution<> colorDis(0.0, 1.0);
    std::uniform_real_distribution<> animDis(0.0f, 1.0f);

    for (size_t i = 0; i < m_maxNPCs; ++i) {
        // Generate random x and z coordinates within the world bounds
        float x = dis(gen);
        float z = dis(gen);
        // Set y to a small positive value to ensure NPCs start above the ground
        float y = 1.0f;
        glm::vec3 position(x, y, z);
        int id = addNPC(position);

        if (id != -1) {
            NPC* npc = getNPC(id);
            if (npc) {
                // Set a random color for the NPC
                glm::vec3 color(colorDis(gen), colorDis(gen), colorDis(gen));
                npc->setColor(color);

                // Initialize animation data
                NPCAnimation& anim = npc->getAnimation();
                anim.animationTime = animDis(gen) * 58.0f; // Random start time
                anim.currentAnimationIndex = 0; // Start with idle animation
                anim.startFrame = 0.0f;
                anim.endFrame = 58.0f; // Adjust as needed for your idle animation
                anim.blendFactor = 0.0f;

                // Apply the original model matrix
                btTransform transform = npc->getRigidBody()->getWorldTransform();
                btMatrix3x3 rotationMatrix;
                rotationMatrix.setFromOpenGLSubMatrix(glm::value_ptr(originalModelMatrix));
                transform.setBasis(rotationMatrix);
                npc->getRigidBody()->setWorldTransform(transform);

                // Set initial velocity (optional)
                glm::vec3 initialVelocity(dis(gen) * 0.1f, 0.0f, dis(gen) * 0.1f);
                npc->getRigidBody()->setLinearVelocity(btVector3(initialVelocity.x, initialVelocity.y, initialVelocity.z));

                // Activate the rigid body
                npc->getRigidBody()->activate(true);
            }
        }
    }
}

void NPCManager::setupBehaviorTrees(BT::BehaviorTreeFactory& factory) {
    for (auto& npc : m_npcs) {
        // Create a behavior tree for each NPC
        auto tree = factory.createTreeFromText(BT::getMainTreeXML());

        // Assume we've added a setupBehaviorTree method to NPC
        npc->setupBehaviorTree(std::move(tree));

        // Set the NPC pointer in the tree's blackboard
        auto blackboard = npc->getBehaviorTree().rootBlackboard();
        blackboard->set("npc", npc.get());
    }
}

void NPCManager::checkAndRemoveFallenNPCs(float threshold) {
    std::vector<int> npcToRemove;
    for (const auto& npc : m_npcs) {
        glm::vec3 position = npc->getPosition();
        if (position.y < threshold) {
            npcToRemove.push_back(npc->getID());
        }
    }

    for (int id : npcToRemove) {
        removeNPC(id);
        std::cout << "Removed NPC with ID " << id << " (fell below threshold)" << std::endl;
    }
}

void NPCManager::debugPrintNPCs() const {
    std::cout << "Current NPCs:" << std::endl;
    for (const auto& npc : m_npcs) {
        glm::vec3 position = npc->getPosition();
        std::cout << "NPC ID: " << npc->getID()
            << ", Position: (" << position.x << ", " << position.y << ", " << position.z << ")"
            << ", Active: " << (npc->isActive() ? "Yes" : "No")
            << std::endl;
    }
    std::cout << "Total NPCs: " << m_npcs.size() << std::endl;
}