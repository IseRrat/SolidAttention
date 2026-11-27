// 计算文件 SHA-256，用于标识模型、输入和可执行文件。
#pragma once

#include <filesystem>
#include <string>

namespace solid {
std::string sha256_file(const std::filesystem::path & path);
}
