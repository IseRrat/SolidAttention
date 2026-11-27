// 真实 CUDA stream、图复用和 attention 完成通知检查。
#include "solid/attention_events.hpp"
#include "solid/cuda_completion.cuh"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void check(cudaError_t status,const char * operation) {
    if (status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
struct Stream {
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    void * data=nullptr;
    ~Stream() {
        if (stream) (void) cudaStreamSynchronize(stream);
        if (executable) (void) cudaGraphExecDestroy(executable);
        if (graph) (void) cudaGraphDestroy(graph);
        if (data) (void) cudaFree(data);
        if (stream) (void) cudaStreamDestroy(stream);
    }
};
void run(int device) {
    check(cudaSetDevice(device),"select device");
    auto events=solid::make_cuda_attention_events(1,device);
    auto * token=events->token(0);
    Stream owner; check(cudaStreamCreateWithFlags(&owner.stream,cudaStreamNonBlocking),"create check stream");
    check(cudaMalloc(&owner.data,4096),"allocate synthetic work");
    check(cudaStreamBeginCapture(owner.stream,cudaStreamCaptureModeRelaxed),"begin graph capture");
    check(solid::cuda_attention_begin(token,owner.stream),"capture start event");
    check(cudaMemsetAsync(owner.data,17,4096,owner.stream),"capture synthetic operation");
    check(solid::cuda_attention_end(token,owner.stream),"capture completion event and notification");
    check(cudaStreamEndCapture(owner.stream,&owner.graph),"end graph capture");
    check(cudaGraphInstantiate(&owner.executable,owner.graph,nullptr,nullptr,0),"instantiate graph");
    for (std::uint64_t epoch=1;epoch<=3;++epoch) {
        events->arm(0,epoch);
        if (events->poll(0,epoch)) throw std::runtime_error("previous replay satisfied a new epoch");
        check(cudaGraphLaunch(owner.executable,owner.stream),"replay graph");
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        std::optional<solid::AttentionReceipt> receipt;
        while (!(receipt=events->poll(0,epoch))) {
            if (std::chrono::steady_clock::now()>deadline) throw std::runtime_error("captured host notification did not replay");
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        if (receipt->proof!=solid::AttentionProof::device_signal || events->poll(0,epoch)) throw std::runtime_error("replay proof or duplicate receipt");
        check(cudaStreamSynchronize(owner.stream),"retire replay callback");
    }
}
}
int main(int argc,char ** argv) {
    if (argc<2 || std::string(argv[1])!="--run-device-check") { std::cout<<"Use --run-device-check [device] on the experiment machine\n"; return 0; }
    try { run(argc>2?std::stoi(argv[2]):0); std::cout<<"CUDA external events and captured completion notifications replayed across epochs\n"; return 0; }
    catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
