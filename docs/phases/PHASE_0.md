# 阶段 0：上游与原版基线

代码完成于 2026-11-17，阶段提交 `44acc89`。

## 目标与产物

固定 llama.cpp 与首个模型的来源，建立统一构建、原版运行器和结果记录入口，为 SSD attention 接入提供可追溯的参考。

| 产物 | 代码或文档入口 | 核查要点 |
| --- | --- | --- |
| 固定上游 | `third_party/llama.cpp`、`cmake/Upstream.cmake` | 提交 `ef6bc6f367082338006c8379bc7bcab3e1584766`；配置阶段检查 |
| 固定模型 | `configs/models.lock.json` | Qwen2.5-7B-Instruct Q4_K_M 两分片、revision、大小与 SHA256 |
| 构建入口 | `CMakePresets.json`、`scripts/check.sh` | Linux/CUDA 与纯单元检查入口 |
| 原版运行器 | `src/baseline.cpp` | 原生 tokenizer、分批 prefill、greedy decode、同步计时 |
| 配置与文件校验 | `src/config.cpp`、`src/filehash.cpp` | 字段、范围、分片完整性、SHA256 |
| 运行来源 | `cmake/BuildInfo.cmake`、`scripts/run_baseline.py` | 源码指纹、干净代码树、二进制及输入哈希、配置快照 |
| 结果与资源记录 | `scripts/run_baseline.py` | 结果、逐 token 轨迹、进程内存采样和后端日志 |
| 接口定位 | [UPSTREAM_MAP](../UPSTREAM_MAP.md) | 模型加载、KV、投影、位置、attention、后端与 microbatch |

## 检查

阶段完成时通过 CUDA 编译、4 项 CTest、5 项 Python 检查及源码指纹核对。检查覆盖配置、哈希、命令行、结果写入和来源一致性，记录见 [CHECKS](../CHECKS.md)。当前完整检查入口为：

```bash
bash scripts/check.sh --cuda
```

## 实验环境核查

按 [DEVELOPMENT](../DEVELOPMENT.md) 设置模型与输入路径，运行原版参考，保存实际 token、GGUF 哈希、环境、逐 token 轨迹、prefill/decode 时间及 DRAM/VRAM 峰值。后续阶段沿用同一组模型与输入。
