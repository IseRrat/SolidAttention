#pragma once
#include "solid/stream_completion.hpp"
#include <cuda_runtime.h>

namespace solid {
inline cudaError_t cuda_record_completion_event(void * event,cudaStream_t stream) {
    if (!event) return cudaErrorInvalidResourceHandle;
    cudaStreamCaptureStatus capture;
    const auto status=cudaStreamIsCapturing(stream,&capture);
    if (status!=cudaSuccess) return status;
    return cudaEventRecordWithFlags(static_cast<cudaEvent_t>(event),stream,
        capture==cudaStreamCaptureStatusNone?cudaEventRecordDefault:cudaEventRecordExternal);
}
inline void CUDART_CB cuda_publish_completion(void * userdata) {
    auto * token=static_cast<StreamCompletionToken *>(userdata);
    token->notify(token->userdata);
}
inline cudaError_t cuda_attention_begin(StreamCompletionToken * token,cudaStream_t stream) {
    return cuda_record_completion_event(token->begin_event,stream);
}
inline cudaError_t cuda_attention_end(StreamCompletionToken * token,cudaStream_t stream) {
    const auto status=cuda_record_completion_event(token->end_event,stream);
    if (status!=cudaSuccess) return status;
    return cudaLaunchHostFunc(stream,cuda_publish_completion,token);
}
} // namespace solid
