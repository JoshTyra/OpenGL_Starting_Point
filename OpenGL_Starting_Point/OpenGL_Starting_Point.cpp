#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include "Camera.h"
#include "FileSystemUtils.h"
#include "Skybox.h"

// Asset Importer
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// RecastDetour 
#include "DetourNavMesh.h"
#include "DetourNavMeshQuery.h"
#include "DetourCommon.h"
#include <cstdio>

void APIENTRY MessageCallback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam)
{
    std::cerr << "GL CALLBACK: " << (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "")
        << " type = " << type
        << ", severity = " << severity
        << ", message = " << message << std::endl;
}

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

GLuint pathVAO, pathVBO;

dtNavMeshQuery* navQuery;
dtQueryFilter queryFilter;

// Max number of polygons in the path
const int MAX_POLYS = 256;
dtPolyRef polys[MAX_POLYS];
int numPolys;

// Max number of points in the smooth path
const int MAX_SMOOTH_PATH = 2048;
float smoothPath[MAX_SMOOTH_PATH * 3];
int smoothPathCount;

const char* vertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragPos; // Pass the world position to the fragment shader

    void main()
    {
        vec4 worldPos = model * vec4(aPos, 1.0);
        FragPos = worldPos.xyz;
        gl_Position = projection * view * worldPos;
    }
    )";

const char* fragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;

    in vec3 FragPos; // Received from vertex shader

    uniform vec4 color;
    uniform vec3 cameraPos;        // Camera position in world space
    uniform vec3 fogColor;         // Color of the fog (e.g., vec3(0.5, 0.5, 0.5))
    uniform float fogDensity;      // Controls how quickly the fog becomes dense
    uniform float fogStartHeight;  // Height at which fog starts
    uniform float fogHeightFalloff; // Controls the vertical density distribution

    void main()
    {
        // Calculate distance from the camera to the fragment
        float distance = length(FragPos - cameraPos);

        // Calculate the height factor
        float heightFactor = clamp((FragPos.y - fogStartHeight) * fogHeightFalloff, 0.0, 1.0);

        // Exponential fog formula
        float fogFactor = 1.0 - exp(-distance * fogDensity * heightFactor);

        // Clamp the fog factor
        fogFactor = clamp(fogFactor, 0.0, 1.0);

        // Blend the original color with the fog color
        FragColor = mix(color, vec4(fogColor, 1.0), fogFactor);
    }
    )";

// Navmesh vertex shader
const char* navmeshVertexShaderSource = R"(
    // navmesh_vertex_shader.glsl
    #version 430 core
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
    #version 430 core
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

        // Set the color uniform
        GLuint colorLoc = glGetUniformLocation(shaderProgram, "color");
        glUniform4f(colorLoc, 0.3f, 0.3f, 0.3f, 1.0f);

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

struct AICube {
    glm::vec3 position;
    glm::vec3 forwardDirection;  // Forward direction for rotation
    float speed;
    int currentTargetIndex;
    float rotation;      // Angle of rotation for the cube (in radians)
    float rotationSpeed; // Speed at which the cube can rotate (radians per second)

    AICube(const glm::vec3& startPos, float spd)
        : position(startPos), forwardDirection(1.0f, 0.0f, 0.0f),
        speed(spd), currentTargetIndex(0), rotation(0.0f), rotationSpeed(2.0f) {}

    void moveTo(const float* targetPos) {
        glm::vec3 target = glm::vec3(targetPos[0], targetPos[1], targetPos[2]);
        glm::vec3 desiredDirection = glm::normalize(target - position);

        // **Move towards the target**
        position += desiredDirection * speed * deltaTime;

        // **Rotate smoothly towards the desired direction**
        float desiredRotation = std::atan2(desiredDirection.z, desiredDirection.x);

        // Calculate the smallest angle between current and desired rotation
        float deltaAngle = desiredRotation - rotation;
        // Wrap the angle between -? and ?
        deltaAngle = std::fmod(deltaAngle + glm::pi<float>(), glm::two_pi<float>()) - glm::pi<float>();

        // Limit the rotation to the rotation speed
        float maxRotation = rotationSpeed * deltaTime;
        deltaAngle = glm::clamp(deltaAngle, -maxRotation, maxRotation);
        rotation += deltaAngle;

        // Update forwardDirection based on the new rotation
        forwardDirection = glm::vec3(std::cos(rotation), 0.0f, std::sin(rotation));

        // Check if the cube reached the target
        if (glm::distance(position, target) < 0.1f) {
            currentTargetIndex++;
        }
    }
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

int fixupCorridor(dtPolyRef* path, int npath, int maxPath,
    const dtPolyRef* visited, int nvisited) {
    int furthestPath = -1;
    int furthestVisited = -1;

    // Find furthest common polygon
    for (int i = npath - 1; i >= 0; --i) {
        bool found = false;
        for (int j = nvisited - 1; j >= 0; --j) {
            if (path[i] == visited[j]) {
                furthestPath = i;
                furthestVisited = j;
                found = true;
                break;
            }
        }
        if (found)
            break;
    }

    // If no intersection found, return current path
    if (furthestPath == -1 || furthestVisited == -1)
        return npath;

    // Adjust the path
    int req = nvisited - furthestVisited;
    int orig = furthestPath + 1 < npath ? npath - (furthestPath + 1) : 0;
    int size = dtMin(req + orig, maxPath);

    // Rearrange path
    if (size > 0) {
        dtPolyRef tempPath[MAX_POLYS];
        memcpy(tempPath, &visited[furthestVisited], sizeof(dtPolyRef) * req);
        if (orig > 0)
            memcpy(&tempPath[req], &path[furthestPath + 1], sizeof(dtPolyRef) * orig);
        memcpy(path, tempPath, sizeof(dtPolyRef) * size);
    }

    return size;
}

bool getSteerTarget(dtNavMeshQuery* navQuery, const float* startPos, const float* endPos, float minTargetDist,
    const dtPolyRef* path, int pathSize, float* steerPos, unsigned char& steerPosFlag, dtPolyRef& steerPosRef) {
    // Find steer target
    static const int MAX_STEER_POINTS = 3;
    float steerPath[MAX_STEER_POINTS * 3];
    unsigned char steerPathFlags[MAX_STEER_POINTS];
    dtPolyRef steerPathPolys[MAX_STEER_POINTS];
    int nsteerPath = 0;
    navQuery->findStraightPath(startPos, endPos, path, pathSize,
        steerPath, steerPathFlags, steerPathPolys, &nsteerPath, MAX_STEER_POINTS);

    if (nsteerPath == 0)
        return false;

    // Find the point to steer towards
    int ns = 0;
    while (ns < nsteerPath) {
        // Stop at the first point that is further than minTargetDist
        float dist = dtVdist2D(&steerPath[ns * 3], startPos);
        if (dist > minTargetDist)
            break;
        ns++;
    }

    if (ns >= nsteerPath)
        return false;

    memcpy(steerPos, &steerPath[ns * 3], sizeof(float) * 3);
    steerPos[1] = startPos[1]; // Keep same height
    steerPosFlag = steerPathFlags[ns];
    steerPosRef = steerPathPolys[ns];

    return true;
}

void getSmoothPath(const float* startPos, const float* endPos, dtPolyRef* polys, int numPolys) {
    float iterPos[3], targetPos[3];
    navQuery->closestPointOnPolyBoundary(polys[0], startPos, iterPos);
    navQuery->closestPointOnPolyBoundary(polys[numPolys - 1], endPos, targetPos);

    smoothPathCount = 0;

    static const float STEP_SIZE = 0.5f;
    static const float SLOP = 0.01f;

    memcpy(&smoothPath[smoothPathCount * 3], iterPos, sizeof(float) * 3);
    smoothPathCount++;

    while (smoothPathCount < MAX_SMOOTH_PATH) {
        // Find the current polygon's neighbors and the straight path to the next corner
        float steerPos[3];
        unsigned char steerPosFlag;
        dtPolyRef steerPosRef;

        if (!getSteerTarget(navQuery, iterPos, targetPos, SLOP, polys, numPolys, steerPos, steerPosFlag, steerPosRef))
            break;

        // Find movement delta
        float delta[3], len;
        dtVsub(delta, steerPos, iterPos);
        len = dtMathSqrtf(dtVdot(delta, delta));

        // If steerPos is within slop radius, advance to next steer target
        if (len < STEP_SIZE) {
            len = 1.0f;
        }
        else {
            len = STEP_SIZE / len;
        }

        float moveTgt[3];
        dtVmad(moveTgt, iterPos, delta, len);

        // Move
        float result[3];
        dtPolyRef visited[16];
        int nvisited = 0;
        navQuery->moveAlongSurface(polys[0], iterPos, moveTgt, &queryFilter, result, visited, &nvisited, 16);

        numPolys = fixupCorridor(polys, numPolys, MAX_POLYS, visited, nvisited);
        navQuery->getPolyHeight(polys[0], result, &result[1]);

        memcpy(iterPos, result, sizeof(float) * 3);

        // Store the result
        memcpy(&smoothPath[smoothPathCount * 3], iterPos, sizeof(float) * 3);
        smoothPathCount++;

        // Check if reached the target
        float endDelta[3];
        dtVsub(endDelta, iterPos, targetPos);
        if (dtVdot(endDelta, endDelta) < SLOP * SLOP)
            break;
    }
}

void performPathfinding(const glm::vec3& startPos, const glm::vec3& endPos) {
    // Convert glm::vec3 to float arrays
    float startPosArray[3] = { startPos.x, startPos.y, startPos.z };
    float endPosArray[3] = { endPos.x, endPos.y, endPos.z };

    // Variables to store the results
    dtPolyRef startRef, endRef;
    float nearestStartPos[3], nearestEndPos[3];

    // Search extents
    float extents[3] = { 2.0f, 4.0f, 2.0f };

    // Find nearest polygons to the start and end points
    navQuery->findNearestPoly(startPosArray, extents, &queryFilter, &startRef, nearestStartPos);
    navQuery->findNearestPoly(endPosArray, extents, &queryFilter, &endRef, nearestEndPos);

    // Find the path corridor
    navQuery->findPath(startRef, endRef, nearestStartPos, nearestEndPos, &queryFilter, polys, &numPolys, MAX_POLYS);

    // Generate the smooth path
    getSmoothPath(nearestStartPos, nearestEndPos, polys, numPolys);

    // Update the path VBO with smooth path
    glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
    glBufferData(GL_ARRAY_BUFFER, smoothPathCount * 3 * sizeof(float), smoothPath, GL_STATIC_DRAW);
}

void processInput(GLFWwindow* window, AICube& aiCube) {
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_W, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_S, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_A, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_D, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        std::cout << "Camera Position X: " << std::endl << camera.getPosition().x <<
            "Camera Position Y: " << std::endl << camera.getPosition().y <<
            "Camera Position Z: " << std::endl << camera.getPosition().z << std::endl;
    }

    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
        // Perform pathfinding when 'P' is pressed
        glm::vec3 startPos = aiCube.position;
        glm::vec3 endPos(2.60156f, 2.62101f, -24.6509f);  // Example destination
        performPathfinding(startPos, endPos);

        // Reset the cube's target index
        aiCube.currentTargetIndex = 0;
    }
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a GLFW window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // Request OpenGL 4.3 or newer
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Request an sRGB-capable framebuffer
    //glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

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

    // Clear any GLEW errors
    glGetError(); // Clear error flag set by GLEW

    // Enable OpenGL debugging if supported
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(MessageCallback, nullptr);

    // Optionally filter which types of messages you want to log
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

    // Define the viewport dimensions
    glViewport(0, 0, WIDTH, HEIGHT);

    glEnable(GL_DEPTH_TEST);

    // Enable sRGB framebuffer gamma correction
    //glEnable(GL_FRAMEBUFFER_SRGB);


    // Create Skybox using KTX2 file
    Skybox skybox(FileSystemUtils::getAssetFilePath("textures/cubemaps/night_sky.ktx2"));

    if (!skybox.isValid()) {
        std::cerr << "Skybox initialization failed!" << std::endl;
    }

    skybox.printDebugInfo();

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

    // Initialize the navigation query
    navQuery = new dtNavMeshQuery();
    dtStatus status = navQuery->init(navMesh, 2048); // 2048 is the maximum number of nodes (you can adjust this)
    if (dtStatusFailed(status)) {
        std::cerr << "Failed to initialize nav mesh query" << std::endl;
        return -1;
    }

    // Initialize the query filter (set default values or customize as needed)
    queryFilter.setIncludeFlags(0xffff);
    queryFilter.setExcludeFlags(0);

    // Define start and end positions
    glm::vec3 startPos(62.826f, 1.25716f, 45.6817f);
    glm::vec3 endPos(-28.7087f, 1.3362f, -42.3046f);

    // Path rendering data
    glGenVertexArrays(1, &pathVAO);
    glGenBuffers(1, &pathVBO);

    performPathfinding(startPos, endPos);

    // Convert glm::vec3 to float arrays
    float startPosArray[3] = { startPos.x, startPos.y, startPos.z };
    float endPosArray[3] = { endPos.x, endPos.y, endPos.z };

    // Variables to store the results
    dtPolyRef startRef, endRef;
    float nearestStartPos[3], nearestEndPos[3];

    // Search extents (tolerance for finding nearest polygon)
    float extents[3] = { 2.0f, 4.0f, 2.0f }; // [x, y, z] extents

    // Find nearest polygons to the start and end points
    navQuery->findNearestPoly(startPosArray, extents, &queryFilter, &startRef, nearestStartPos);
    navQuery->findNearestPoly(endPosArray, extents, &queryFilter, &endRef, nearestEndPos);

    // Max number of polygons in the path
    const int MAX_POLYS = 256;
    dtPolyRef polys[MAX_POLYS];
    int numPolys;

    // Find the path corridor
    navQuery->findPath(startRef, endRef, nearestStartPos, nearestEndPos, &queryFilter, polys, &numPolys, MAX_POLYS);

    // Generate the smooth path
    getSmoothPath(nearestStartPos, nearestEndPos, polys, numPolys);

    // Create the navmesh render data
    NavMeshRenderData navMeshRenderData = createNavMeshRenderData(navMesh);

    // Define start position for AI cube
    glm::vec3 cubeStartPosition(62.826f, 1.25716f, 45.6817f);  // Example start position for the cube

    // Create the AI Cube object
    AICube aiCube(cubeStartPosition, 5.0f);  // Set speed to 2.0 (can be adjusted)
    aiCube.rotationSpeed = 3.0f;

    performPathfinding(cubeStartPosition, endPos);

    glBindVertexArray(pathVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
    glBufferData(GL_ARRAY_BUFFER, smoothPathCount * 3 * sizeof(float), smoothPath, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // Vertices for ai cube
    float cubeVertices[] = {
        // positions         
        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,

        -0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,

         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,

        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f,  0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f, -0.5f
    };

    GLuint cubeVAO, cubeVBO;
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

    // Define the position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        // Time calculations
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window, aiCube);  // Pass the AI cube to the input handler

        // Check if there is a path and move the AI cube along it
        if (smoothPathCount > 0 && aiCube.currentTargetIndex < smoothPathCount) {
            const float* targetPos = &smoothPath[aiCube.currentTargetIndex * 3];
            aiCube.moveTo(targetPos);
        }

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

        // Draw skybox here
        skybox.draw(view, projection);

        // Render the model
        glUseProgram(shaderProgram);

        // Set uniform matrices
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Set fog uniforms
        glUniform3fv(glGetUniformLocation(shaderProgram, "cameraPos"), 1, glm::value_ptr(camera.getPosition()));
        glUniform3f(glGetUniformLocation(shaderProgram, "fogColor"), 0.3f, 0.3f, 0.6f);
        glUniform1f(glGetUniformLocation(shaderProgram, "fogDensity"), 0.05f); // Less dense fog
        glUniform1f(glGetUniformLocation(shaderProgram, "fogStartHeight"), -5.0f); // Start fog below ground level
        glUniform1f(glGetUniformLocation(shaderProgram, "fogHeightFalloff"), 50.0f); // Adjust vertical distribution

        for (const auto& mesh : meshes) {
            glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
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
        glUniform4f(colorLoc, 1.0f, 1.0f, 0.0f, 1.0f); // Red color

        // Render off-mesh connections as lines
        glBindVertexArray(navMeshRenderData.offMeshVAO);
        glDrawArrays(GL_LINES, 0, navMeshRenderData.offMeshVertexCount);
        glBindVertexArray(0);

        // Set the color for the path (e.g., blue)
        glUniform4f(colorLoc, 0.0f, 0.0f, 1.0f, 1.0f); // Blue color

        // Render the path as a line strip
        glBindVertexArray(pathVAO);
        glDrawArrays(GL_LINE_STRIP, 0, smoothPathCount);
        glBindVertexArray(0);

        // Render the AI cube
        glUseProgram(navmeshShaderProgram);
        glBindVertexArray(cubeVAO);

        glm::mat4 model = glm::mat4(1.0f);

        // Move cube to AI position
        model = glm::translate(model, aiCube.position);

        // Apply rotation
        model = glm::rotate(model, aiCube.rotation, glm::vec3(0.0f, 1.0f, 0.0f));  // Rotate around Y-axis

        // Move cube up by 0.5 units to align bottom at position
        model = glm::translate(model, glm::vec3(0.0f, 0.5f, 0.0f));

        GLuint modelLoc = glGetUniformLocation(navmeshShaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(navViewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(navProjLoc, 1, GL_FALSE, glm::value_ptr(projection));
        glUniform4f(colorLoc, 1.0f, 1.0f, 0.0f, 1.0f);  // Cube color

        glDrawArrays(GL_TRIANGLES, 0, 36);

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

    // Delete path rendering data
    glDeleteVertexArrays(1, &pathVAO);
    glDeleteBuffers(1, &pathVBO);

    dtFreeNavMesh(navMesh);
    delete navQuery;

    glfwTerminate();
    return 0;
}
