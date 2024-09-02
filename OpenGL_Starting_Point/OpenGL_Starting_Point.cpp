#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "Camera.h"
#include "FileSystemUtils.h"
#include <chrono>

// Asset Importer
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Path-finding
#include <numeric>
#include <queue>
#include <unordered_map>
#include <functional>
#include <cstdlib>
#include <ctime>   // for time()

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

Camera camera(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -180.0f, 0.0f, 12.0f, 0.1f, 45.0f);

// Booleans for debug data rendering
bool renderGridLines = false;
bool renderCellCenters = false;
bool renderRaycasts = false;
float debugLineSize = 1.0f;  // Default line size
bool prevKey1State = false;
bool prevKey2State = false;
bool prevKey3State = false;

// Vertex Shader source code
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    fragColor = aColor;
}
)";

// Fragment Shader source code
const char* fragmentShaderSource = R"(
#version 330 core
in vec3 fragColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(fragColor, 1.0);
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

    // Toggle rendering options
    bool currentKey1State = glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS;
    bool currentKey2State = glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS;
    bool currentKey3State = glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS;

    if (currentKey1State && !prevKey1State) {
        renderGridLines = !renderGridLines;
    }
    if (currentKey2State && !prevKey2State) {
        renderCellCenters = !renderCellCenters;
    }
    if (currentKey3State && !prevKey3State) {
        renderRaycasts = !renderRaycasts;
    }

    // Update the previous key states
    prevKey1State = currentKey1State;
    prevKey2State = currentKey2State;
    prevKey3State = currentKey3State;

    // Adjust debug line size
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        debugLineSize += 0.1f;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        debugLineSize = std::max(0.1f, debugLineSize - 0.1f); // Ensure line size doesn't go below 0.1
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

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    mutable unsigned int VAO;  // Mark as mutable to allow modification in const functions

    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
        : vertices(vertices), indices(indices) {
        setupMesh();
    }

    void setupMesh() const {  // Mark as const
        unsigned int VBO, EBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
            &indices[0], GL_STATIC_DRAW);

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

    void Draw(GLuint shaderProgram) const {
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        if (!indices.empty()) {
            glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        }
        else {
            glDrawArrays(GL_TRIANGLES, 0, vertices.size());
        }
        glBindVertexArray(0);
    }
};

struct Model {
    std::vector<Mesh> meshes;
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
};

class NavigationGrid {
public:
    // Debug rendering color scheme
    static const glm::vec3 COLOR_WALKABLE;       // Green: Flat, walkable surfaces
    static const glm::vec3 COLOR_NONWALKABLE;    // Red: Steep, non-walkable surfaces
    static const glm::vec3 COLOR_VERTICAL;       // Blue: Near-vertical surfaces (walls)
    static const glm::vec3 COLOR_NOHIT;          // Orange: Areas where raycasts didn't hit anything
    static const glm::vec3 COLOR_GRID;           // Cyan: Base grid structure
    static const glm::vec3 COLOR_RAYCAST;        // Magenta: Debug raycasts

    NavigationGrid(const Model& model, float cellSize)
        : m_cellSize(cellSize), m_model(&model) {
        createGrid(model);
    }

    struct Cell {
        glm::vec3 center;
        bool walkable;
        glm::vec3 color;
    };

    int getGridWidth() const { return m_gridWidth; }
    int getGridDepth() const { return m_gridDepth; }

    glm::vec3 getGridCellCenter(int x, int z) const {
        int index = z * m_gridWidth + x;
        return m_grid[index].center;
    }

    float getGridCellHeight(int x, int z) const {
        return getGridCellCenter(x, z).y;
    }

    float getCellSize() const { return m_cellSize; }

    glm::ivec2 getGridPosition(float x, float z) const;

    void createGrid(const Model& model);
    void render(GLuint shaderProgram) const;
    bool isWalkable(int x, int z) const;
    void forceWalkable(int x, int z);
    std::vector<glm::ivec2> getWalkableCells() const;

private:
    const Model* m_model;
    std::vector<Cell> m_grid;
    float m_cellSize;
    int m_gridWidth;
    int m_gridDepth;
    glm::vec3 m_minBounds;
    glm::vec3 m_maxBounds;

    void determineWalkableAreas();
    bool isPointAboveSurface(const glm::vec3& point) const;

    bool rayTriangleIntersect(
        const glm::vec3& rayOrigin, const glm::vec3& rayVector,
        const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
        float& t, float& u, float& v) const;
};

// Define the colors outside the class
const glm::vec3 NavigationGrid::COLOR_WALKABLE = glm::vec3(0.0f, 1.0f, 0.0f);
const glm::vec3 NavigationGrid::COLOR_NONWALKABLE = glm::vec3(1.0f, 0.0f, 0.0f);
const glm::vec3 NavigationGrid::COLOR_VERTICAL = glm::vec3(0.0f, 0.0f, 1.0f);
const glm::vec3 NavigationGrid::COLOR_NOHIT = glm::vec3(1.0f, 0.5f, 0.0f);
const glm::vec3 NavigationGrid::COLOR_GRID = glm::vec3(0.0f, 1.0f, 1.0f);
const glm::vec3 NavigationGrid::COLOR_RAYCAST = glm::vec3(1.0f, 0.0f, 1.0f);

bool NavigationGrid::rayTriangleIntersect(
    const glm::vec3& rayOrigin, const glm::vec3& rayVector,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    float& t, float& u, float& v) const
{
    const float EPSILON = 0.0000001f;
    glm::vec3 edge1, edge2, h, s, q;
    float a, f;

    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = glm::cross(rayVector, edge2);
    a = glm::dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) {
        return false;  // Ray is parallel to the triangle
    }

    f = 1.0f / a;
    s = rayOrigin - v0;
    u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    q = glm::cross(s, edge1);
    v = f * glm::dot(rayVector, q);

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    t = f * glm::dot(edge2, q);

    if (t > EPSILON) {
        return true;
    }

    return false;
}

void NavigationGrid::createGrid(const Model& model) {
    // Set class member variables for bounds
    m_minBounds = model.aabbMin;
    m_maxBounds = model.aabbMax;

    // Calculate grid dimensions
    m_gridWidth = static_cast<int>((m_maxBounds.x - m_minBounds.x) / m_cellSize);
    m_gridDepth = static_cast<int>((m_maxBounds.z - m_minBounds.z) / m_cellSize);

    // Adjust m_maxBounds to fit the grid exactly
    m_maxBounds.x = m_minBounds.x + m_gridWidth * m_cellSize;
    m_maxBounds.z = m_minBounds.z + m_gridDepth * m_cellSize;

    std::cout << "Grid dimensions: " << m_gridWidth << " x " << m_gridDepth << std::endl;
    std::cout << "Mesh bounds: Min(" << m_minBounds.x << ", " << m_minBounds.y << ", " << m_minBounds.z
        << ") Max(" << m_maxBounds.x << ", " << m_maxBounds.y << ", " << m_maxBounds.z << ")" << std::endl;

    // Initialize grid
    m_grid.resize(m_gridWidth * m_gridDepth);

    // Fill grid
    float initialY = m_minBounds.y + 0.1f; // Start slightly above the lowest point
    for (int x = 0; x < m_gridWidth; ++x) {
        for (int z = 0; z < m_gridDepth; ++z) {
            int index = z * m_gridWidth + x;
            m_grid[index].center = glm::vec3(
                m_minBounds.x + (x + 0.5f) * m_cellSize,
                initialY,
                m_minBounds.z + (z + 0.5f) * m_cellSize
            );
            m_grid[index].walkable = false;  // Assume not walkable by default
        }
    }

    determineWalkableAreas();
}

void NavigationGrid::determineWalkableAreas() {
    const float rayLength = m_model->aabbMax.y - m_model->aabbMin.y;
    const float maxWalkableSlope = glm::radians(45.0f);
    const float nearVerticalSlope = glm::radians(80.0f);
    const float rayStartOffset = 1.0f;

    int hitCount = 0;
    int totalCells = m_gridWidth * m_gridDepth;

    for (int x = 0; x < m_gridWidth; ++x) {
        for (int z = 0; z < m_gridDepth; ++z) {
            int index = z * m_gridWidth + x;
            glm::vec3 rayOrigin = m_grid[index].center;
            rayOrigin.y = m_model->aabbMax.y + rayStartOffset;
            glm::vec3 rayDirection(0.0f, -1.0f, 0.0f);

            std::vector<float> slopes;
            glm::vec3 weightedHitPoint(0.0f);
            glm::vec3 avgNormal(0.0f);
            float closestT = std::numeric_limits<float>::max();
            bool hit = false;

            for (const auto& mesh : m_model->meshes) {
                for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                    const glm::vec3& v0 = mesh.vertices[mesh.indices[i]].Position;
                    const glm::vec3& v1 = mesh.vertices[mesh.indices[i + 1]].Position;
                    const glm::vec3& v2 = mesh.vertices[mesh.indices[i + 2]].Position;

                    float t, u, v;
                    if (rayTriangleIntersect(rayOrigin, rayDirection, v0, v1, v2, t, u, v)) {
                        glm::vec3 hitPoint = rayOrigin + t * rayDirection;
                        glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

                        // Simplified version to just use the closest hit point
                        if (t < closestT) {
                            closestT = t;
                            m_grid[index].center.y = hitPoint.y;
                            avgNormal = normal;
                            slopes.push_back(glm::acos(glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f))));
                            hit = true;
                        }

                        // Render the intersection point for debugging
                        glm::vec3 debugHitColor(1.0f, 1.0f, 0.0f); // Yellow color for debug hit points
                        glBegin(GL_POINTS);
                        glColor3f(debugHitColor.r, debugHitColor.g, debugHitColor.b);
                        glVertex3f(hitPoint.x, hitPoint.y, hitPoint.z);
                        glEnd();
                    }
                }
            }

            if (hit) {
                hitCount++;
                // Check the slope of the hit surface and log if it's marked as non-walkable
                if (slopes.size() > 0 && *std::max_element(slopes.begin(), slopes.end()) > maxWalkableSlope) {
                    std::cout << "Cell at (" << x << ", " << z << ") is too steep to walk on." << std::endl;
                }
                else {
                    m_grid[index].walkable = true;
                    m_grid[index].color = COLOR_WALKABLE;
                }
            }
            else {
                m_grid[index].walkable = false;
                m_grid[index].color = COLOR_NOHIT;
                std::cout << "Raycast didn't hit anything at cell (" << x << ", " << z << ")." << std::endl;
            }
        }
    }

    std::cout << "Raycast hits: " << hitCount << " / " << totalCells << " cells" << std::endl;
}

bool NavigationGrid::isPointAboveSurface(const glm::vec3& point) const {
    static int callCount = 0;
    callCount++;

    if (callCount % 1000 == 0) {
        std::cout << "isPointAboveSurface called " << callCount << " times" << std::endl;
    }

    for (const auto& mesh : m_model->meshes) {
        // For a flat plane, we can simplify the check
        if (!mesh.vertices.empty()) {
            float planeHeight = mesh.vertices[0].Position.y;

            // Add a small epsilon to account for floating-point imprecision
            const float epsilon = 1e-5f;

            if (point.y > planeHeight + epsilon) {
                return true;
            }
            else {
                return false;
            }
        }
    }

    // If we get here, the point is not above any surface
    return true;
}

void NavigationGrid::forceWalkable(int x, int z) {
    int index = z * m_gridWidth + x;
    m_grid[index].walkable = true;
    m_grid[index].color = COLOR_WALKABLE;  // Update color for visual debugging
}


bool NavigationGrid::isWalkable(int x, int z) const {
    int index = z * m_gridWidth + x;
    return m_grid[index].walkable;
}

glm::ivec2 NavigationGrid::getGridPosition(float x, float z) const {
    int gridX = static_cast<int>((x - m_minBounds.x) / m_cellSize);
    int gridZ = static_cast<int>((z - m_minBounds.z) / m_cellSize);
    return glm::ivec2(gridX, gridZ);
}

std::vector<glm::ivec2> NavigationGrid::getWalkableCells() const {
    std::vector<glm::ivec2> walkableCells;
    for (int x = 0; x < m_gridWidth; ++x) {
        for (int z = 0; z < m_gridDepth; ++z) {
            if (isWalkable(x, z)) {
                walkableCells.push_back(glm::ivec2(x, z));
            }
        }
    }
    return walkableCells;
}

void NavigationGrid::render(GLuint shaderProgram) const {
    // Create and bind a VAO
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    std::vector<float> vertexData;
    auto safeGetCell = [this](int x, int z) -> const Cell& {
        if (x < 0 || x >= m_gridWidth || z < 0 || z >= m_gridDepth) {
            static const Cell defaultCell{ {0, 0, 0}, false, COLOR_NOHIT };
            return defaultCell;
        }
        return m_grid[z * m_gridWidth + x];
        };

    // Grid lines
    if (renderGridLines) {
        for (int x = 0; x <= m_gridWidth; ++x) {
            const auto& startCell = safeGetCell(x, 0);
            const auto& endCell = safeGetCell(x, m_gridDepth - 1);
            vertexData.insert(vertexData.end(), { startCell.center.x, m_model->aabbMin.y, startCell.center.z });
            vertexData.insert(vertexData.end(), glm::value_ptr(COLOR_GRID), glm::value_ptr(COLOR_GRID) + 3);
            vertexData.insert(vertexData.end(), { endCell.center.x, m_model->aabbMin.y, endCell.center.z });
            vertexData.insert(vertexData.end(), glm::value_ptr(COLOR_GRID), glm::value_ptr(COLOR_GRID) + 3);
        }
        for (int z = 0; z <= m_gridDepth; ++z) {
            const auto& startCell = safeGetCell(0, z);
            const auto& endCell = safeGetCell(m_gridWidth - 1, z);
            vertexData.insert(vertexData.end(), { startCell.center.x, m_model->aabbMin.y, startCell.center.z });
            vertexData.insert(vertexData.end(), glm::value_ptr(COLOR_GRID), glm::value_ptr(COLOR_GRID) + 3);
            vertexData.insert(vertexData.end(), { endCell.center.x, m_model->aabbMin.y, endCell.center.z });
            vertexData.insert(vertexData.end(), glm::value_ptr(COLOR_GRID), glm::value_ptr(COLOR_GRID) + 3);
        }
    }

    // Cell centers
    if (renderCellCenters) {
        for (int z = 0; z < m_gridDepth; ++z) {
            for (int x = 0; x < m_gridWidth; ++x) {
                const auto& cell = safeGetCell(x, z);
                vertexData.insert(vertexData.end(), { cell.center.x, cell.center.y, cell.center.z });
                vertexData.insert(vertexData.end(), glm::value_ptr(cell.color), glm::value_ptr(cell.color) + 3);
            }
        }
    }

    // Debug raycasts
    if (renderRaycasts) {
        for (int z = 0; z < m_gridDepth; ++z) {
            for (int x = 0; x < m_gridWidth; ++x) {
                const auto& cell = safeGetCell(x, z);
                vertexData.insert(vertexData.end(), { cell.center.x, m_model->aabbMax.y + 1.0f, cell.center.z });
                vertexData.insert(vertexData.end(), glm::value_ptr(COLOR_RAYCAST), glm::value_ptr(COLOR_RAYCAST) + 3);
                vertexData.insert(vertexData.end(), { cell.center.x, cell.center.y, cell.center.z });
                vertexData.insert(vertexData.end(), glm::value_ptr(COLOR_RAYCAST), glm::value_ptr(COLOR_RAYCAST) + 3);
            }
        }
    }

    // Buffer the data
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

    // Set up vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Set line width
    glLineWidth(debugLineSize);

    // Use the shader program
    glUseProgram(shaderProgram);

    // Draw grid lines
    if (renderGridLines) {
        glDrawArrays(GL_LINES, 0, (m_gridWidth + m_gridDepth + 2) * 2);
    }

    // Draw cell centers
    if (renderCellCenters) {
        glPointSize(2.8f);
        glDrawArrays(GL_POINTS, (m_gridWidth + m_gridDepth + 2) * 2, m_gridWidth * m_gridDepth);
    }

    // Draw debug raycasts
    if (renderRaycasts) {
        glDrawArrays(GL_LINES, (m_gridWidth + m_gridDepth + 2) * 2 + m_gridWidth * m_gridDepth, m_gridWidth * m_gridDepth * 2);
    }

    // Clean up
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

struct Cube {
    glm::vec3 position;  // The position of the cube in the 3D world
    glm::vec3 size;      // The size of the cube
    glm::vec3 color;     // The color of the cube
    float yaw;           // Yaw angle for rotation around the Y-axis (horizontal rotation)
    float pitch;         // Pitch angle for rotation around the X-axis (vertical rotation)

    Cube(const glm::vec3& pos, const glm::vec3& sz, const glm::vec3& col)
        : position(pos), size(sz), color(col), yaw(0.0f), pitch(0.0f) {}

    void updateRotation(const glm::vec3& surfaceNormal) {
        // Calculate pitch (rotation around X-axis)
        // Pitch is based on the angle between the surface normal and the Z-axis (up vector in world space)
        float dotProductZ = glm::dot(surfaceNormal, glm::vec3(0.0f, 0.0f, 1.0f));
        pitch = acos(dotProductZ);

        // Calculate yaw (rotation around Y-axis)
        // Yaw is based on the angle between the surface normal and the Y-axis (forward vector in world space)
        float dotProductY = glm::dot(surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        yaw = atan2(surfaceNormal.y, surfaceNormal.x);
    }

    void Draw(GLuint shaderProgram) const {
        // Build the transformation matrix
        // Apply translation to position the cube, then apply rotations (pitch and yaw)
        glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
        model = glm::rotate(model, pitch, glm::vec3(1.0f, 0.0f, 0.0f));  // Rotate around the X-axis (pitch)
        model = glm::rotate(model, yaw, glm::vec3(0.0f, 1.0f, 0.0f));    // Rotate around the Y-axis (yaw)
        model = glm::scale(model, size);  // Scale the cube to its specified size

        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

        // Simple cube vertex data
        float vertices[] = {
            // positions          // colors
            -0.5f, -0.5f, -0.5f,  color.r, color.g, color.b,
             0.5f, -0.5f, -0.5f,  color.r, color.g, color.b,
             0.5f,  0.5f, -0.5f,  color.r, color.g, color.b,
            -0.5f,  0.5f, -0.5f,  color.r, color.g, color.b,

            -0.5f, -0.5f,  0.5f,  color.r, color.g, color.b,
             0.5f, -0.5f,  0.5f,  color.r, color.g, color.b,
             0.5f,  0.5f,  0.5f,  color.r, color.g, color.b,
            -0.5f,  0.5f,  0.5f,  color.r, color.g, color.b
        };

        unsigned int indices[] = {
            0, 1, 3, 1, 2, 3,  // back face
            4, 5, 7, 5, 6, 7,  // front face
            0, 1, 4, 1, 5, 4,  // bottom face
            2, 3, 6, 3, 7, 6,  // top face
            0, 3, 4, 3, 7, 4,  // left face
            1, 2, 5, 2, 6, 5   // right face
        };

        // Set up vertex buffer and array objects
        unsigned int VBO, VAO, EBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // Vertex positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // Vertex colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Draw the cube
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

        // Clean up
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteVertexArrays(1, &VAO);
    }
};

// Node structure for A* algorithm
struct Node {
    int x, z;
    float gCost;  // Cost from start to current node
    float hCost;  // Heuristic cost (estimated cost to target)
    float fCost;  // Total cost (gCost + hCost)
    Node* parent; // Pointer to the parent node

    Node() : x(0), z(0), gCost(0.0f), hCost(0.0f), fCost(0.0f), parent(nullptr) {} // Default constructor

    Node(int x, int z, float gCost, float hCost, Node* parent = nullptr)
        : x(x), z(z), gCost(gCost), hCost(hCost), fCost(gCost + hCost), parent(parent) {}

    bool operator>(const Node& other) const {
        return fCost > other.fCost;
    }
};

std::vector<glm::vec3> findPath(const glm::vec3& startPos, const glm::vec3& targetPos, NavigationGrid& grid) {
    // A* setup
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openSet;
    std::unordered_map<int, Node> allNodes;
    int gridWidth = grid.getGridWidth();
    int gridDepth = grid.getGridDepth();

    auto heuristic = [](const glm::vec3& a, const glm::vec3& b) -> float {
        return glm::length(a - b);
        };

    auto getIndex = [gridWidth](int x, int z) -> int {
        return z * gridWidth + x;
        };

    // Initialize the start node
    int startX = static_cast<int>((startPos.x - grid.getGridCellCenter(0, 0).x) / grid.getCellSize());
    int startZ = static_cast<int>((startPos.z - grid.getGridCellCenter(0, 0).z) / grid.getCellSize());
    int targetX = static_cast<int>((targetPos.x - grid.getGridCellCenter(0, 0).x) / grid.getCellSize());
    int targetZ = static_cast<int>((targetPos.z - grid.getGridCellCenter(0, 0).z) / grid.getCellSize());

    Node startNode(startX, startZ, 0.0f, heuristic(startPos, targetPos));
    openSet.push(startNode);
    allNodes[getIndex(startX, startZ)] = startNode;

    // Directions for neighbor cells (up, down, left, right)
    std::vector<std::pair<int, int>> directions = {
        { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 }
    };

    while (!openSet.empty()) {
        Node current = openSet.top();
        openSet.pop();

        // Check if we've reached the target
        if (current.x == targetX && current.z == targetZ) {
            // Reconstruct the path
            std::vector<glm::vec3> path;
            Node* node = &allNodes[getIndex(current.x, current.z)];
            while (node != nullptr) {
                path.push_back(grid.getGridCellCenter(node->x, node->z));
                node = node->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        // Explore neighbors
        for (const auto& direction : directions) {
            int neighborX = current.x + direction.first;
            int neighborZ = current.z + direction.second;

            // Check bounds and walkability
            if (neighborX < 0 || neighborX >= gridWidth || neighborZ < 0 || neighborZ >= gridDepth ||
                !grid.isWalkable(neighborX, neighborZ)) {
                continue;
            }

            float newGCost = current.gCost + 1.0f;  // Assume all movements cost 1
            float hCost = heuristic(grid.getGridCellCenter(neighborX, neighborZ), targetPos);
            int neighborIndex = getIndex(neighborX, neighborZ);

            if (allNodes.find(neighborIndex) == allNodes.end() || newGCost < allNodes[neighborIndex].gCost) {
                Node neighborNode(neighborX, neighborZ, newGCost, hCost, &allNodes[getIndex(current.x, current.z)]);
                openSet.push(neighborNode);
                allNodes[neighborIndex] = neighborNode;
            }
        }
    }

    // If we get here, there's no valid path
    return {};
}

glm::vec3 findFirstWalkableCell(const NavigationGrid& navGrid) {
    for (int z = 0; z < navGrid.getGridDepth(); ++z) {
        for (int x = 0; x < navGrid.getGridWidth(); ++x) {
            if (navGrid.isWalkable(x, z)) {
                // Return the center of the first walkable cell found
                return navGrid.getGridCellCenter(x, z);
            }
        }
    }
    // If no walkable cell is found, return a default value (e.g., (0, 0, 0))
    return glm::vec3(0.0f, 0.0f, 0.0f);
}

glm::vec3 findRandomWalkableCell(const NavigationGrid& navGrid) {
    std::vector<glm::vec3> walkableCells;

    for (int z = 0; z < navGrid.getGridDepth(); ++z) {
        for (int x = 0; x < navGrid.getGridWidth(); ++x) {
            if (navGrid.isWalkable(x, z)) {
                walkableCells.push_back(navGrid.getGridCellCenter(x, z));
            }
        }
    }

    if (!walkableCells.empty()) {
        // Pick a random walkable cell
        int randomIndex = rand() % walkableCells.size();
        return walkableCells[randomIndex];
    }

    // If no walkable cell is found, return a default value (e.g., (0, 0, 0))
    return glm::vec3(0.0f, 0.0f, 0.0f);
}

Model loadModel(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return {};
    }

    Model model;
    model.aabbMin = glm::vec3(std::numeric_limits<float>::max());
    model.aabbMax = glm::vec3(std::numeric_limits<float>::lowest());

    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        // Process vertices
        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
            Vertex vertex;
            vertex.Position = glm::vec3(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);

            // Update global AABB
            model.aabbMin = glm::min(model.aabbMin, vertex.Position);
            model.aabbMax = glm::max(model.aabbMax, vertex.Position);

            if (mesh->HasNormals()) {
                vertex.Normal = glm::vec3(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);
            }

            if (mesh->mTextureCoords[0]) {
                vertex.TexCoords = glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y);
            }
            else {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
            }

            vertices.push_back(vertex);
        }

        // Process indices
        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++) {
                indices.push_back(face.mIndices[k]);
            }
        }

        model.meshes.push_back(Mesh(vertices, indices));
    }

    return model;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a GLFW window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
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

    // Define the viewport dimensions
    glViewport(0, 0, WIDTH, HEIGHT);

    glEnable(GL_DEPTH_TEST);

    // Used for random destination for ai agent
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Load the model
    Model model = loadModel(FileSystemUtils::getAssetFilePath("models/nav_test_tutorial_map.obj"));

    std::cout << "Creating navigation grid..." << std::endl;

    // Controls how dense the navigation grid will be
    float cellSize = 1.0f; 
    NavigationGrid navGrid(model, cellSize);
    std::cout << "Navigation grid created." << std::endl;

    // Used for debug rendering of the ai agent path
    GLuint pathVAO, pathVBO;
    glGenVertexArrays(1, &pathVAO);
    glGenBuffers(1, &pathVBO);

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

    float movementSpeed = 3.5f; // Adjust this value to control how fast the AI cube moves

    // Create a test AI cube
    Cube aiCube(glm::vec3(-3.0f, 0.0f, -3.0f), glm::vec3(0.5f), glm::vec3(1.0f, 0.0f, 0.0f));

    // Find the first walkable cell for the start position
    glm::vec3 startPosition = findFirstWalkableCell(navGrid);
    aiCube.position = startPosition;

    float halfCubeHeight = aiCube.size.y / 2.0f;

    std::vector<glm::vec3> path;

    // Get the list of walkable cells
    std::vector<glm::ivec2> walkableCells = navGrid.getWalkableCells();

    if (!walkableCells.empty()) {
        // Pick the first walkable cell as the start
        glm::ivec2 startCell = walkableCells.front();

        // Randomly select a target cell from the walkable cells
        int randomIndex = std::rand() % walkableCells.size();
        glm::ivec2 targetCell = walkableCells[randomIndex];

        // Convert to world positions
        glm::vec3 startPosition = navGrid.getGridCellCenter(startCell.x, startCell.y);
        glm::vec3 targetPosition = navGrid.getGridCellCenter(targetCell.x, targetCell.y);

        // Update AI cube position
        aiCube.position = startPosition;

        // Perform pathfinding
        path = findPath(startPosition, targetPosition, navGrid);

        if (path.empty()) {
            std::cout << "No path found between the selected walkable cells." << std::endl;
        }
        else {
            std::cout << "Path found from (" << startCell.x << ", " << startCell.y << ") to ("
                << targetCell.x << ", " << targetCell.y << ")" << std::endl;
        }
    }
    else {
        std::cout << "No walkable cells found in the grid." << std::endl;
    }

    int currentPathIndex = 0;

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        double currentTime = glfwGetTime();
        double elapsedTime = currentTime - previousTime;
        frameCount++;

        processInput(window);

        // Input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use the shader program
        glUseProgram(shaderProgram);

        // Set up view and projection matrices
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = camera.getProjectionMatrix((float)WIDTH / (float)HEIGHT);

        // Pass view and projection matrices to the shader
        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // Render the loaded meshes
        for (const auto& mesh : model.meshes) {
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));
            mesh.Draw(shaderProgram);
        }

        // Set up the grid model matrix
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));

        if (!path.empty() && currentPathIndex < path.size()) {
            glm::ivec2 currentCell = navGrid.getGridPosition(aiCube.position.x, aiCube.position.z);
            glm::vec3 gridCellCenter = navGrid.getGridCellCenter(currentCell.x, currentCell.y);

            glm::vec3 targetPoint = path[currentPathIndex];

            // Adjust the targetPoint to keep the cube above the ground
            targetPoint.y = gridCellCenter.y + halfCubeHeight;

            if (glm::length(aiCube.position - targetPoint) < 0.1f) {
                currentPathIndex++;
            }
            else {
                glm::vec3 direction = glm::normalize(targetPoint - aiCube.position);
                aiCube.updateRotation(direction);
                aiCube.position += direction * movementSpeed * deltaTime;
            }

            // Update path data in the VBO
            glBindVertexArray(pathVAO);
            glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
            glBufferData(GL_ARRAY_BUFFER, path.size() * sizeof(glm::vec3), path.data(), GL_STATIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);

            // Render the path
            glBindVertexArray(pathVAO);
            glDrawArrays(GL_LINE_STRIP, 0, path.size());
            glBindVertexArray(0);
        }

        // Render the navigation grid
        navGrid.render(shaderProgram);

        // Render the AI cube
        aiCube.Draw(shaderProgram);

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteProgram(shaderProgram);
    glDeleteVertexArrays(1, &pathVAO);
    glDeleteBuffers(1, &pathVBO);

    glfwTerminate();
    return 0;
}
