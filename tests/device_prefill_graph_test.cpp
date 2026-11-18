#include "device_prefill_graph_cases.hpp"
#include "tiled_prefill_cases.hpp"
#include "representative_builder_cases.hpp"
#include <iostream>
int main() {
    try {
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
        const auto prefill=[](auto g,auto qh,auto queries,auto tile) { return std::make_unique<test_tiled::Emulator>(g,qh,queries,tile,true); };
        const auto ranker=[](auto g,auto qh,auto b) { return std::make_unique<test_builder::Emulator>(g,qh,b,true); };
        for (bool sparse:{false,true}) for (bool packed:{false,true}) for (bool disk:{false,true})
            test_prefill_graph::run(cpu.get(),cpu.get(),solid::DeviceMemory::host_emulator,prefill,ranker,sparse,packed,disk);
        test_prefill_graph::run(cpu.get(),cpu.get(),solid::DeviceMemory::host_emulator,prefill,ranker,true,true,false,true);
        test_prefill_graph::run(cpu.get(),cpu.get(),solid::DeviceMemory::host_emulator,prefill,ranker,true,false,false,false,true);
        std::cout<<"device prefill graph, direct output, repeated allocation and failure checks passed on CPU\n"; return 0;
    } catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
