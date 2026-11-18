# Decode 当前 KV 异步归档

论文图 11 与 §6.2 将当前 KV 安装和回写拆成两条分支。`overlap` 和 `overlap+sync_reuse` 的设备输入路径现在先在显存安装当前行，再单独下载供 `DRAM → SSD` 回写的副本。

```text
设备 Q 就绪 → 选择 → 历史块 READ/H2D ─┐
设备 KV 就绪 ────────────────────────┴→ 安装当前行 ──→ Flash Attention
                                              ├──────→ D2H → 主机追加 → 异步 WRITE
                                              └──────→ 下一层 SSD 预取
```

## 源码与槽位顺序

1. [设备算子](../../include/solid/device_ops.cuh) 以 4 字节 `kv_token` 建立 KV consolidator 到 CPU 控制节点的依赖。主机读取令牌，代表环仍从设备张量读取 K。原始 KV 分配保留到当前行 D2D 完成。
2. [预取器](../../src/prefetch.cpp) 在历史选择中保留当前块已有行的快照。位于块首的当前 token 直接获得新槽位；块内追加时，旧版本原槽保留，通过新 generation 安装当前行。补传范围只有缺失的历史前缀。正式选择的 token 集合与 mask 使用原规则。[槽位代码](../../src/device_cache.cpp)
3. [GGML 协调器](../../src/ggml_prefetch.cpp) 先把安装完成事件交给计算流，再按各 KV head 的实际物理槽位排入 D2H。下载在同一复制流中排在安装之后，使用单个固定的 pinned token 工作区与独立完成事件。同步复用只等待当前安装，归档事件走回写分支。
4. 安装完成后，新一层可以启动 SSD 预测读取。槽位保留到归档事件完成且本层 attention 读者释放。单层模型的后继轮额外等待主机追加，以便读到同层最新版本。错误处理排空设备访问并回收历史快照租约。
5. [会话](../../src/session.cpp) 验证归档值和 layer/position 后追加 FP16 行。满缓冲区沿 [异步回写](ASYNC_WRITEBACK.md) 交给 liburing。串行模式沿用原主机前沿。

## 完成证据

层图用 `kv_device_ready` 确认 FP16 生产、用 `kv_archive_copy_complete` 确认 D2H。`attention_until_boundary` 和 `kv_d2h_ready` 都依赖当前行安装，追加和 SSD 回写只沿归档分支推进。`decode_kv_download_bytes` 维持实际归档字节口径。

现有 CUDA 构建、CPU 合成图、liburing 临时文件、错误排空、轨迹和源码指纹检查通过。图检查核对 4 字节前沿、逐层归档、任意槽位数值及主机写入字节。设备执行留在实验环境。
