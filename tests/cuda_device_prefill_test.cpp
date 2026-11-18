#include "device_prefill_graph_cases.hpp"
#include "ggml-cuda.h"
#include <iostream>
#include <string>
int main(int argc,char ** argv) {
    if (argc<2 || std::string(argv[1])!="--run-device-check") { std::cout<<"Use --run-device-check [device] on the experiment machine\n"; return 0; }
    try {
        if (argc>3) throw std::invalid_argument("unexpected device check argument");
        const int device=argc==3?std::stoi(argv[2]):0;
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> gpu(ggml_backend_cuda_init(device),ggml_backend_free),cpu(ggml_backend_cpu_init(),ggml_backend_free);
        test_prefill_graph::require(gpu && cpu,"CUDA prefill check backends");
        const auto prefill=[&](auto g,auto qh,auto queries,auto tile) { return solid::make_cuda_tiled_prefill(g,qh,queries,tile,device); };
        const auto ranker=[&](auto g,auto qh,auto b) { return solid::make_cuda_block_ranker(g,qh,b.representatives,b.total,device,b.local); };
        for (bool sparse:{false,true}) for (bool packed:{false,true}) for (bool disk:{false,true})
            test_prefill_graph::run(gpu.get(),cpu.get(),solid::DeviceMemory::cuda,prefill,ranker,sparse,packed,disk);
        test_prefill_graph::run(gpu.get(),cpu.get(),solid::DeviceMemory::cuda,prefill,ranker,true,true,false,true);
        test_prefill_graph::run(gpu.get(),cpu.get(),solid::DeviceMemory::cuda,prefill,ranker,true,false,false,false,true);
        std::cout<<"CUDA prefill borrowed Q/KV, representative batches and direct output checks passed\n"; return 0;
    } catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
