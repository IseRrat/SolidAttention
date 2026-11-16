#pragma once

#include <filesystem>
#include <string>

namespace solid {
std::string sha256_file(const std::filesystem::path & path);
}
