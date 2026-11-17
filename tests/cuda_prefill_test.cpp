#include "tiled_prefill_cases.hpp"
#include <iostream>
#include <string>
int main(int argc,char ** argv) {
    if (argc<2 || std::string(argv[1])!="--run-device-check") {
        std::cout<<"Usage: solid-cuda-prefill-test --run-device-check [device ordinal]\n"; return argc==1?0:2;
    }
    try {
        if (argc>3) throw std::invalid_argument("unexpected device-check argument");
        const auto device=argc==3?std::stoi(argv[2]):0;
        test_tiled::run_cases([&](auto g,auto qh,auto queries,auto tile) { return solid::make_cuda_tiled_prefill(g,qh,queries,tile,device); });
        std::cout<<"CUDA tiled prefill device checks passed\n"; return 0;
    } catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
