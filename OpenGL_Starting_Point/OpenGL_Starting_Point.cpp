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

// Vertex Shader source code
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 color;

out vec3 fragColor;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    fragColor = color;
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
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        unsigned int VBO, EBO;
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

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

    void Draw(GLuint shaderProgram) const {  // Mark as const
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};

class NavigationGrid {
public:
    struct Cell {
        glm::vec3 center;
        bool walkable;
    };

    NavigationGrid(const std::vector<Mesh>& meshes, float cellSize);
    void createGrid(const std::vector<Mesh>& meshes);
    void render(GLuint shaderProgram) const;

private:
    std::vector<Cell> m_grid;
    float m_cellSize;
    int m_gridWidth;
    int m_gridDepth;

    void determineWalkableAreas(const std::vector<Mesh>& meshes);
    bool isPointAboveSurface(const glm::vec3& point, const std::vector<Mesh>& meshes) const;

    // Add this new private method
    bool rayTriangleIntersect(
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
};

NavigationGrid::NavigationGrid(const std::vector<Mesh>& meshes, float cellSize)
    : m_cellSize(cellSize) {
    createGrid(meshes);
}

void NavigationGrid::createGrid(const std::vector<Mesh>& meshes) {
    // Find the bounds of the mesh
    glm::vec3 minBound(std::numeric_limits<float>::max());
    glm::vec3 maxBound(std::numeric_limits<float>::lowest());

    for (const auto& mesh : meshes) {
        for (const auto& vertex : mesh.vertices) {
            minBound = glm::min(minBound, vertex.Position);
            maxBound = glm::max(maxBound, vertex.Position);
        }
    }

    // Calculate grid dimensions
    m_gridWidth = static_cast<int>((maxBound.x - minBound.x) / m_cellSize);
    m_gridDepth = static_cast<int>((maxBound.z - minBound.z) / m_cellSize);

    // Adjust maxBound to fit the grid exactly
    maxBound.x = minBound.x + m_gridWidth * m_cellSize;
    maxBound.z = minBound.z + m_gridDepth * m_cellSize;

    std::cout << "Grid dimensions: " << m_gridWidth << " x " << m_gridDepth << std::endl;
    std::cout << "Adjusted mesh bounds: Min(" << minBound.x << ", " << minBound.y << ", " << minBound.z
        << ") Max(" << maxBound.x << ", " << maxBound.y << ", " << maxBound.z << ")" << std::endl;

    // Initialize grid
    m_grid.resize(m_gridWidth * m_gridDepth);

    // Fill grid
    for (int x = 0; x < m_gridWidth; ++x) {
        for (int z = 0; z < m_gridDepth; ++z) {
            int index = z * m_gridWidth + x;
            m_grid[index].center = glm::vec3(
                minBound.x + (x + 0.5f) * m_cellSize,
                (minBound.y + maxBound.y) / 2,  // Set to average height initially
                minBound.z + (z + 0.5f) * m_cellSize
            );
            m_grid[index].walkable = true;  // Assume walkable by default
        }
    }

    determineWalkableAreas(meshes);
}

void NavigationGrid::determineWalkableAreas(const std::vector<Mesh>& meshes) {
    if (meshes.empty() || meshes[0].vertices.empty()) {
        std::cout << "No valid mesh data to process." << std::endl;
        return;
    }

    float planeHeight = meshes[0].vertices[0].Position.y;
    std::cout << "Plane height: " << planeHeight << std::endl;

    for (int x = 0; x < m_gridWidth; ++x) {
        for (int z = 0; z < m_gridDepth; ++z) {
            int index = z * m_gridWidth + x;
            m_grid[index].center.y = planeHeight;
            m_grid[index].walkable = true;

            std::cout << "Cell (" << x << ", " << z << ") is walkable at height " << m_grid[index].center.y << std::endl;
        }

        if ((x + 1) % 10 == 0 || x == m_gridWidth - 1) {
            int processedCells = (x + 1) * m_gridDepth;
            int totalCells = m_gridWidth * m_gridDepth;
            float percentage = (float)processedCells / totalCells * 100.0f;
            std::cout << "Processed " << processedCells << " of " << totalCells << " cells ("
                << percentage << "%)" << std::endl;
        }
    }
}

bool NavigationGrid::isPointAboveSurface(const glm::vec3& point, const std::vector<Mesh>& meshes) const {
    static int callCount = 0;
    callCount++;

    if (callCount % 1000 == 0) {
        std::cout << "isPointAboveSurface called " << callCount << " times" << std::endl;
    }

    for (const auto& mesh : meshes) {
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

void NavigationGrid::render(GLuint shaderProgram) const {
    // Generate and bind a VAO for the grid lines
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Prepare line vertices
    std::vector<glm::vec3> lineVertices;
    glm::vec3 gridOrigin = m_grid[0].center - glm::vec3(m_cellSize / 2, 0, m_cellSize / 2);
    for (int x = 0; x <= m_gridWidth; ++x) {
        lineVertices.push_back(gridOrigin + glm::vec3(x * m_cellSize, 0, 0));
        lineVertices.push_back(gridOrigin + glm::vec3(x * m_cellSize, 0, m_gridDepth * m_cellSize));
    }
    for (int z = 0; z <= m_gridDepth; ++z) {
        lineVertices.push_back(gridOrigin + glm::vec3(0, 0, z * m_cellSize));
        lineVertices.push_back(gridOrigin + glm::vec3(m_gridWidth * m_cellSize, 0, z * m_cellSize));
    }

    // Buffer the line data
    glBufferData(GL_ARRAY_BUFFER, lineVertices.size() * sizeof(glm::vec3), lineVertices.data(), GL_STATIC_DRAW);

    // Set up vertex attributes
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    // Use the shader program
    glUseProgram(shaderProgram);

    // Set line color to white
    GLint colorLoc = glGetUniformLocation(shaderProgram, "color");
    glUniform3f(colorLoc, 1.0f, 1.0f, 1.0f);

    // Draw grid lines
    glDrawArrays(GL_LINES, 0, lineVertices.size());

    // Prepare cell center points
    std::vector<glm::vec3> cellCenters;
    std::vector<glm::vec3> cellColors;

    for (const auto& cell : m_grid) {
        cellCenters.push_back(cell.center);
        cellColors.push_back(cell.walkable ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f));
    }

    // Buffer the cell center data
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, cellCenters.size() * sizeof(glm::vec3), cellCenters.data(), GL_STATIC_DRAW);

    // Draw cell centers
    glPointSize(5.0f);
    for (size_t i = 0; i < cellCenters.size(); ++i) {
        glUniform3fv(colorLoc, 1, glm::value_ptr(cellColors[i]));
        glDrawArrays(GL_POINTS, i, 1);
    }

    // Clean up
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

std::vector<Mesh> loadModel(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return {};
    }

    std::vector<Mesh> meshes;

    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
            Vertex vertex;
            vertex.Position = glm::vec3(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
            vertex.Normal = glm::vec3(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);

            if (mesh->mTextureCoords[0]) {
                vertex.TexCoords = glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y);
            }
            else {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
            }

            vertices.push_back(vertex);
        }

        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++) {
                indices.push_back(face.mIndices[k]);
            }
        }

        meshes.push_back(Mesh(vertices, indices));
    }

    return meshes;
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

    // Load the model
    std::vector<Mesh> meshes = loadModel(FileSystemUtils::getAssetFilePath("models/test_plane.obj"));

    std::cout << "Creating navigation grid..." << std::endl;
    float cellSize = 5.0f; // Adjust this value based on your needs
    NavigationGrid navGrid(meshes, cellSize);
    std::cout << "Navigation grid created." << std::endl;

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
        glm::mat4 model = glm::mat4(1.0f);
        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = camera.getProjectionMatrix((float)WIDTH / (float)HEIGHT);

        // Pass view and projection matrices to the shader
        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // Render the loaded meshes
        for (const auto& mesh : meshes) {
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

            // Set a default color for the mesh (e.g., light gray)
            GLint colorLoc = glGetUniformLocation(shaderProgram, "color");
            glUniform3f(colorLoc, 0.3f, 0.3f, 0.3f);

            mesh.Draw(shaderProgram);
        }

        // Set up the grid model matrix
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

        // Render the navigation grid
        navGrid.render(shaderProgram);

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
