#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "Camera.h"
#include "FileSystemUtils.h"
#include <fstream>

// Asset Importer
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// RecastDetour 
#include "DetourNavMesh.h"
#include "DetourNavMeshQuery.h"
#include <cstdio>

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

Camera camera(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -180.0f, 0.0f, 6.0f, 0.1f, 45.0f);

const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
    )";

const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    uniform vec4 color;

    void main()
    {
        FragColor = color;
    }
    )";

// Navmesh vertex shader
const char* navmeshVertexShaderSource = R"(
    // navmesh_vertex_shader.glsl
    #version 330 core
    layout (location = 0) in vec3 aPos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        vec4 worldPos = model * vec4(aPos, 1.0);
        gl_Position = projection * view * worldPos;
    }
    )";

// Navmesh fragment shader
const char* navmeshFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    uniform vec4 color;

    void main() {
        FragColor = color;
    }
    )";

// Function to compile shader from source
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    // Check for shader compile errors
    GLint success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shader;
}

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
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    mutable unsigned int VAO;  // Mark as mutable to allow modification in const functions

    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
        : vertices(vertices), indices(indices)
    {
        setupMesh();
    }

    void setupMesh() const
    {
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

        glBindVertexArray(0);
    }

    void Draw(GLuint shaderProgram) const
    {
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);

        // Set the color uniform (e.g., white color)
        GLuint colorLoc = glGetUniformLocation(shaderProgram, "color");
        glUniform4f(colorLoc, 1.0f, 1.0f, 1.0f, 1.0f);

        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};

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
            vertex.Position = glm::vec3(
                mesh->mVertices[j].x,
                mesh->mVertices[j].y,
                mesh->mVertices[j].z
            );

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

#define NAVMESHSET_MAGIC 'MSET'
#define NAVMESHSET_VERSION 1

std::vector<float> offMeshConnectionVertices;

struct NavMeshSetHeader {
    int magic;
    int version;
    int numTiles;
    dtNavMeshParams params;
};

struct NavMeshTileHeader {
    dtTileRef tileRef;
    int dataSize;
};

struct NavMeshRenderData {
    GLuint VAO;
    GLuint VBO;
    size_t vertexCount;
    GLuint offMeshVAO;
    GLuint offMeshVBO;
    size_t offMeshVertexCount;
};

dtNavMesh* loadNavMeshFromFile(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        printf("Failed to open file %s\n", path);
        return nullptr;
    }

    // Read header.
    NavMeshSetHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(NavMeshSetHeader));
    if (!file || header.magic != NAVMESHSET_MAGIC) {
        printf("Bad magic number or failed to read header in navmesh file %s\n", path);
        return nullptr;
    }
    if (header.version != NAVMESHSET_VERSION) {
        printf("Wrong version in navmesh file %s\n", path);
        return nullptr;
    }

    dtNavMesh* mesh = dtAllocNavMesh();
    if (!mesh) {
        printf("Failed to allocate navmesh\n");
        return nullptr;
    }

    dtStatus status = mesh->init(&header.params);
    if (dtStatusFailed(status)) {
        printf("Failed to initialize navmesh\n");
        dtFreeNavMesh(mesh);
        return nullptr;
    }

    // Read tiles.
    for (int i = 0; i < header.numTiles; ++i) {
        NavMeshTileHeader tileHeader;
        file.read(reinterpret_cast<char*>(&tileHeader), sizeof(NavMeshTileHeader));
        if (!file || !tileHeader.tileRef || !tileHeader.dataSize)
            break;

        unsigned char* data = (unsigned char*)dtAlloc(tileHeader.dataSize, DT_ALLOC_PERM);
        if (!data) break;
        memset(data, 0, tileHeader.dataSize);
        file.read(reinterpret_cast<char*>(data), tileHeader.dataSize);
        if (!file) {
            dtFree(data);
            break;
        }

        mesh->addTile(data, tileHeader.dataSize, DT_TILE_FREE_DATA, tileHeader.tileRef, 0);
    }

    return mesh;
}

void renderNavMesh(const dtNavMesh* navMesh) {
    const int maxTiles = navMesh->getMaxTiles();
    std::vector<float> vertices;

    for (int i = 0; i < maxTiles; i++) {
        const dtMeshTile* tile = navMesh->getTile(i);
        if (!tile || !tile->header) continue;

        const dtPoly* polys = tile->polys;
        const dtPolyDetail* detailPolys = tile->detailMeshes;
        const float* verts = tile->verts; // Changed from dtVert* to float*
        const dtLink* links = tile->links;

        for (int j = 0; j < tile->header->polyCount; j++) {
            const dtPoly* poly = &polys[j];
            const dtPolyDetail* detail = &detailPolys[j];

            // Iterate over triangle detail polygons
            for (int k = 0; k < detail->triCount; k++) {
                const unsigned char* tri = &tile->detailTris[(detail->triBase + k) * 4];
                for (int m = 0; m < 3; m++) {
                    unsigned short vertIndex = tri[m];
                    const float* v = &verts[vertIndex * 3];
                    vertices.push_back(v[0]);
                    vertices.push_back(v[1]);
                    vertices.push_back(v[2]);
                }
            }
        }
    }

    // Send vertices to OpenGL (the rendering function)
    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // Rendering
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
    glBindVertexArray(0);

    // Clean up
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

NavMeshRenderData createNavMeshRenderData(const dtNavMesh* navMesh) {
    NavMeshRenderData renderData = { 0 };

    std::vector<float> vertices;
    std::vector<float> offMeshConnectionVertices;

    for (int i = 0; i < navMesh->getMaxTiles(); i++) {
        const dtMeshTile* tile = navMesh->getTile(i);
        if (!tile || !tile->header) continue;

        const dtPoly* polys = tile->polys;
        const float* verts = tile->verts;
        const dtPolyDetail* detailMeshes = tile->detailMeshes;
        const float* detailVerts = tile->detailVerts;
        const unsigned char* detailTris = tile->detailTris;

        for (int j = 0; j < tile->header->polyCount; j++) {
            const dtPoly* poly = &polys[j];
            const dtPolyDetail* pd = &detailMeshes[j];

            if (poly->getType() == DT_POLYTYPE_OFFMESH_CONNECTION) {
                // Process off-mesh connections
                unsigned int idx0 = poly->verts[0];
                unsigned int idx1 = poly->verts[1];

                const float* v0 = &verts[idx0 * 3];
                const float* v1 = &verts[idx1 * 3];

                // Store the two endpoints to render later
                offMeshConnectionVertices.push_back(v0[0]);
                offMeshConnectionVertices.push_back(v0[1]);
                offMeshConnectionVertices.push_back(v0[2]);

                offMeshConnectionVertices.push_back(v1[0]);
                offMeshConnectionVertices.push_back(v1[1]);
                offMeshConnectionVertices.push_back(v1[2]);

                continue;
            }

            // Existing code for normal polygons
            for (int k = 0; k < pd->triCount; ++k) {
                const unsigned char* t = &detailTris[(pd->triBase + k) * 4];

                for (int m = 0; m < 3; ++m) {
                    int vertIndex;
                    if (t[m] < poly->vertCount) {
                        vertIndex = poly->verts[t[m]];
                        vertices.push_back(verts[vertIndex * 3]);
                        vertices.push_back(verts[vertIndex * 3 + 1]);
                        vertices.push_back(verts[vertIndex * 3 + 2]);
                    }
                    else {
                        vertIndex = t[m] - poly->vertCount;
                        vertices.push_back(detailVerts[(pd->vertBase + vertIndex) * 3]);
                        vertices.push_back(detailVerts[(pd->vertBase + vertIndex) * 3 + 1]);
                        vertices.push_back(detailVerts[(pd->vertBase + vertIndex) * 3 + 2]);
                    }
                }
            }
        }
    }

    // Create VAO and VBO for navmesh
    renderData.vertexCount = vertices.size() / 3;

    glGenVertexArrays(1, &renderData.VAO);
    glGenBuffers(1, &renderData.VBO);

    glBindVertexArray(renderData.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, renderData.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // Create VAO and VBO for off-mesh connections
    glGenVertexArrays(1, &renderData.offMeshVAO);
    glGenBuffers(1, &renderData.offMeshVBO);

    glBindVertexArray(renderData.offMeshVAO);
    glBindBuffer(GL_ARRAY_BUFFER, renderData.offMeshVBO);
    glBufferData(GL_ARRAY_BUFFER, offMeshConnectionVertices.size() * sizeof(float), offMeshConnectionVertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    renderData.offMeshVertexCount = offMeshConnectionVertices.size() / 3;

    glBindVertexArray(0);

    return renderData;
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
    std::vector<Mesh> meshes = loadModel(FileSystemUtils::getAssetFilePath("models/nav_test_tutorial_map.obj"));

    // Build and compile the shader program using compileShader()
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    // Link shaders into a shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    GLint success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::MODEL_SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Compile and link navmesh shader program
    GLuint navmeshVertexShader = compileShader(GL_VERTEX_SHADER, navmeshVertexShaderSource);
    GLuint navmeshFragmentShader = compileShader(GL_FRAGMENT_SHADER, navmeshFragmentShaderSource);

    GLuint navmeshShaderProgram = glCreateProgram();
    glAttachShader(navmeshShaderProgram, navmeshVertexShader);
    glAttachShader(navmeshShaderProgram, navmeshFragmentShader);
    glLinkProgram(navmeshShaderProgram);

    // Check for linking errors
    glGetProgramiv(navmeshShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(navmeshShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::NAVMESH_SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(navmeshVertexShader);
    glDeleteShader(navmeshFragmentShader);

    // Load the navmesh
    dtNavMesh* navMesh = loadNavMeshFromFile(FileSystemUtils::getAssetFilePath("models/tutorial_navmesh.bin").c_str());
    if (!navMesh) {
        std::cerr << "Failed to load navmesh!" << std::endl;
        return -1;
    }

    // Create the navmesh render data
    NavMeshRenderData navMeshRenderData = createNavMeshRenderData(navMesh);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        // Time calculations
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwTerminate();
            return 0;
        }

        // Render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set up view and projection matrices
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = camera.getProjectionMatrix((float)WIDTH / (float)HEIGHT);

        // Render the model
        glUseProgram(shaderProgram);
        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        for (const auto& mesh : meshes) {
            glm::mat4 model = glm::mat4(1.0f);
            GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            mesh.Draw(shaderProgram);
        }

        // Set wireframe mode for navmesh
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        // Render the navmesh
        glUseProgram(navmeshShaderProgram);
        GLuint navModelLoc = glGetUniformLocation(navmeshShaderProgram, "model");
        GLuint navViewLoc = glGetUniformLocation(navmeshShaderProgram, "view");
        GLuint navProjLoc = glGetUniformLocation(navmeshShaderProgram, "projection");
        GLuint colorLoc = glGetUniformLocation(navmeshShaderProgram, "color");

        glUniformMatrix4fv(navModelLoc, 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
        glUniformMatrix4fv(navViewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(navProjLoc, 1, GL_FALSE, glm::value_ptr(projection));
        glUniform4f(colorLoc, 0.0f, 1.0f, 0.0f, 1.0f); // Set the navmesh color to green

        glBindVertexArray(navMeshRenderData.VAO);
        glDrawArrays(GL_TRIANGLES, 0, navMeshRenderData.vertexCount);
        glBindVertexArray(0);

        // Reset to fill mode after rendering navmesh
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // Set the color for off-mesh connections (e.g., red)
        glUniform4f(colorLoc, 1.0f, 0.0f, 0.0f, 1.0f); // Red color

        // Render off-mesh connections as lines
        glBindVertexArray(navMeshRenderData.offMeshVAO);
        glDrawArrays(GL_LINES, 0, navMeshRenderData.offMeshVertexCount);
        glBindVertexArray(0);

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteProgram(shaderProgram);
    glDeleteProgram(navmeshShaderProgram);
    glDeleteVertexArrays(1, &navMeshRenderData.VAO);
    glDeleteBuffers(1, &navMeshRenderData.VBO);
    glDeleteVertexArrays(1, &navMeshRenderData.offMeshVAO);
    glDeleteBuffers(1, &navMeshRenderData.offMeshVBO);
    dtFreeNavMesh(navMesh);

    glfwTerminate();
    return 0;
}
