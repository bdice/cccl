#include <cuda/atomic>

__global__ void cas_device_relaxed_non_volatile(int* data, int* out, int n)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  ref.compare_exchange_strong(*out, n, cuda::std::memory_order_relaxed);
}

// clang-format off
/*

; SM8X-LABEL: .target sm_80
; SM8X:      .visible .entry [[FUNCTION:_.*cas_device_relaxed_non_volatile.*]](
; SM8X-DAG:  ld.param.u64 %rd[[#ATOM:]], {{.*}}[[FUNCTION]]_param_0{{.*}}
; SM8X-DAG:  ld.param.u64 %rd[[#EXPECTED:]], {{.*}}[[FUNCTION]]_param_1{{.*}}
; SM8X-DAG:  ld.param.u32 %r[[#INPUT:]], {{.*}}[[FUNCTION]]_param_2{{.*}}
; SM8X-DAG:  cvta.to.global.u64 %rd[[#GOUT:]], %rd[[#EXPECTED]];
; SM8X-DAG:  ld.global.u32 %r[[#LOCALEXP:]], [%rd[[#INPUT]]];
; SM8X-NEXT: {{/*[[:space:]] *}}atom.cas.relaxed.gpu.b32 %r[[#DEST:]],[%rd[[#ATOM]]],%r[[#LOCALEXP]],%r[[#INPUT]];{{[[:space:]]/*}}
; SM8X-NEXT: st.global.u32 [%rd[[#GOUT]]], %r[[#DEST]];
; SM8X-NEXT: ret;

*/
