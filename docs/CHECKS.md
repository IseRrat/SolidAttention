# 开发检查记录

## 2026-11-17：上游接入与原版运行器

| 项目 | 检查结果 |
| --- | --- |
| 上游依赖 | submodule 固定到 `ef6bc6f367082338006c8379bc7bcab3e1584766`，Windows/Linux 读取一致 |
| 构建 | Ubuntu 24.04 / GCC 13.3.0 / CUDA Toolkit 12.6.85，CUDA 版本编译与链接通过 |
| 配置 | 类型、数值范围、预算关系、未知字段、重复字段、路径和 GGUF 分片声明检查通过 |
| 哈希 | SHA256 标准向量与跨读取分块的百万字节向量通过 |
| 命令行 | `--help`、`--build-info`、锁定模型配置解析通过 |
| 构建来源 | 编译时源码哈希与当前文件逐项一致，陈旧二进制识别测试通过 |
| 结果文件 | 原子 JSON 写入、非有限值拒绝和已有结果保留测试通过 |
| 模型分片 | 分片遗漏与内容变更检测通过，官方 Q4_K_M 两分片信息已锁定 |
| 文档 | 本地链接、代码块和差异格式检查通过 |

最终检查入口：

```bash
bash scripts/check.sh --cuda
```

结果为 4 项 CTest 和 5 项 Python 单元检查通过。检查使用配置文件、临时文本和合成数据。模型推理、性能及准确率使用 [DEVELOPMENT](DEVELOPMENT.md) 中的实验环境入口执行。

接口定位见 [UPSTREAM_MAP](UPSTREAM_MAP.md)。

## 2026-11-17：同步 SSD 与稀疏路径

| 检查范围 | 证据 |
| --- | --- |
| FP16 转换 | 全部有限 half 位模式 round-trip、subnormal 和 ties-to-even 边界 |
| 同步文件 | 独立与交错布局、多层多头、尾块、短读写重试、脏数据转发 |
| 错误处理 | EOF、写失败、在用文件独占创建、临时文件清理、计算图错误传递 |
| 内存 | SSD 写缓存容量随上下文保持固定；内存参考按上下文容量预留空间 |
| 代表与选择 | 后续窗口定稿、GQA、正负相关度、init/local 去重及 token 预算 |
| Prefill | 分批与连续处理的一致性、causal mask、独立 dense softmax 公式对照 |
| 图接入 | 非连续 Q/K/V 视图、FP16 收集、多头 mask、padding、CPU Flash Attention 数值对照 |
| 回放工具 | 固定实际输入 token、参考上下文继承、checksum、非有限值及首次超差定位 |
| 上游适配 | 构建目录生成文件，原始源文件字节保持一致，固定提交检查通过 |

完整检查入口为 `bash scripts/check.sh --cuda`，包含 CUDA 编译、11 项 CTest、10 项 Python 单元检查和源码指纹核对。计算使用合成张量与临时文件。实验环境的原版/内存/SSD 数值对照命令见 [SSD_PATH](SSD_PATH.md)。
