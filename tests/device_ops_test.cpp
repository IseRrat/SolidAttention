#include "solid/device_ops.hpp"
#include "solid/kv_store.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value,const char * message) { if (!value) throw std::runtime_error(message); }
void run() {
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({1024*1024,nullptr,true}),ggml_free);
    constexpr int width=17,heads=3;
    auto * packed=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,width*2+5,heads,1);
    auto * k=ggml_view_3d(context.get(),packed,width,heads,1,packed->nb[1],packed->nb[2],0);
    auto * v=ggml_view_3d(context.get(),packed,width,heads,1,packed->nb[1],packed->nb[2],width*sizeof(float));
    auto * q=ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,width,heads*2,1);
    auto * early=ggml_new_tensor_1d(context.get(),GGML_TYPE_F32,1);
    auto * token=solid::device_query_token(context.get(),q);
    auto * kv=solid::device_consolidate_kv(context.get(),k,v,q,early);
    auto * mask=ggml_new_tensor_1d(context.get(),GGML_TYPE_F16,83);
    auto * release=solid::device_release_mask(context.get(),mask,kv);
    require(!solid::device_op_supported(q) && solid::device_op_supported(token) && solid::device_op_supported(kv) && solid::device_op_supported(release),"tagged operation support");
    const auto saved=token->op_params[solid::device_op_offset/sizeof(std::int32_t)];
    token->op_params[solid::device_op_offset/sizeof(std::int32_t)]=0;
    require(!solid::device_op_supported(token),"unknown CPU custom operation claimed by device backend");
    token->op_params[solid::device_op_offset/sizeof(std::int32_t)]=saved;
    const auto stride=k->nb[0]; k->nb[0]=8;
    require(!solid::device_op_supported(kv),"unsupported element stride accepted"); k->nb[0]=stride;
    auto * graph=ggml_new_graph_custom(context.get(),128,false);
    ggml_build_forward_expand(graph,token); ggml_build_forward_expand(graph,release);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(),backend.get()),ggml_backend_buffer_free);
    require(buffer!=nullptr,"device operation CPU allocation");
    const float edges[]{0,-0.0f,0x1p-25f,0x1.8p-24f,0x1.ffcp-15f,1.00048828125f,1.00146484375f,65504,65520,-65520};
    std::vector<float> values(ggml_nelements(packed),91),query(ggml_nelements(q),1);
    std::vector<solid::Half> expected(83),actual_mask(83),actual(ggml_nelements(kv));
    for (std::size_t i=0;i<expected.size();++i) expected[i]=static_cast<solid::Half>(i*811);
    ggml_backend_tensor_set(q,query.data(),0,query.size()*sizeof(float));
    ggml_backend_tensor_set(mask,expected.data(),0,expected.size()*sizeof(solid::Half));
    const float zero=0; ggml_backend_tensor_set(early,&zero,0,sizeof(zero));
    for (int epoch=0;epoch<3;++epoch) {
        for (int h=0;h<heads;++h) for (int plane=0;plane<2;++plane) for (int d=0;d<width;++d)
            values[h*(2*width+5)+plane*width+d]=edges[(h+plane+d+epoch)%10];
        ggml_backend_tensor_set(packed,values.data(),0,values.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"pure CPU operation graph");
        ggml_backend_tensor_get(kv,actual.data(),0,actual.size()*sizeof(solid::Half));
        ggml_backend_tensor_get(release,actual_mask.data(),0,actual_mask.size()*sizeof(solid::Half));
        float marker=1; ggml_backend_tensor_get(token,&marker,0,sizeof(marker));
        require(marker==0 && actual_mask==expected,"token or mask bits changed");
        for (int h=0;h<heads;++h) for (int plane=0;plane<2;++plane) for (int d=0;d<width;++d)
            require(actual[(h*2+plane)*width+d]==solid::to_half(values[h*(2*width+5)+plane*width+d]),"strided FP16 consolidator layout or bits");
    }
}
}
int main() {
    try { run(); std::cout<<"pure device graph operations, FP16 bits and strided layout passed on CPU\n"; return 0; }
    catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }
}
