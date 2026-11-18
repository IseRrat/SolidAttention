# 聚合 KV 异步回写

论文图 11、§6.2 和 §7 将当前层的聚合回写和下一层预取放在同一调度窗口。`overlap` 与 `overlap+sync_reuse` 的 SSD 路径通过现有 liburing 所有者提交 `WRITE`，协调线程继续处理预取与设备完成。

## 写入与版本

- [kv_store.cpp](../../src/kv_store.cpp) 为每层保留原有追加缓冲区，默认 32 KiB。满层缓冲区按块和 KV head 打包，沿用交错文件布局与分段偏移。所有层共用最多一批的打包工作区。
- [prefetch.cpp](../../src/prefetch.cpp) 每层最多登记一批回写。未提交的批次保留在原层缓冲区，先排到的批次借用打包工作区。历史读取继续从原层缓冲区转发脏行，磁盘段只覆盖已提交的前缀。
- [io_worker.cpp](../../src/io_worker.cpp) 与 [async_reader.cpp](../../src/async_reader.cpp) 共用读写准入和完成通道。正式缺块与预测读优先于排队回写。准入保留一个读取位置，内核队列最多一半在途请求用于写入。短写提交剩余区间，全部原始 WRITE 完成后才推进 `committed_` 并释放源和工作区。取消确认与原始 CQE 分开处理。
- 同层下一次追加如果仍占用自己的满缓冲区，就收割该层写入完成。其他层继续推进。结束请求时先完成在途批次，再同步写出最后的部分缓冲区。写失败导致会话终止，原操作排空后回收工作区。

## 调度与核查

[ggml_prefetch.cpp](../../src/ggml_prefetch.cpp) 用 `original_write_complete` 结束层图的 `buffered_ssd_store` 任务。上一层 attention 完成即可释放显存槽位并启动后继上传，层图在回写完成后汇总。结果的 `async_write_requests` 和 `async_write_bytes` 分别记录实际提交与完成字节；汇总写入字段同时包含同步路径。

主机输入的同步复用模式在当前复制批次提交后即可派发回写。写缓冲区与上传使用的主机槽位分离，所以下一层读取沿用复制完成前沿，当前层回写可以提前执行。设备输入模式从显存独立归档，回写在归档后的追加前沿派发。见 [当前 KV 归档](KV_ARCHIVE.md)。

已有两层 GGML 合成图使用实际 liburing 临时文件，核对异步写入字节、注意力数值与层图闭合。`naive` 保留同步参照。CUDA build 编译设备路径，本机执行 CPU 合成检查。
