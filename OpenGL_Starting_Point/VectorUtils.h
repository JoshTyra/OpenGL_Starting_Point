// VectorUtils.h
#pragma once

#include <glm/glm.hpp>
#include <string>
#include <sstream>

inline std::string vec3ToString(const glm::vec3& v) {
    std::stringstream ss;
    ss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return ss.str();
}