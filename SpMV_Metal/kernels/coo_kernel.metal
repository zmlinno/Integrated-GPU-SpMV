//
//  coo_kernel.metal
//  SpMV_Metal
//
//  Created by 张木林 on 4/8/25.
//

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void coo_spmv(
    device const uint* row_idx       [[ buffer(0) ]],
    device const uint* col_idx       [[ buffer(1) ]],
    device const float* values       [[ buffer(2) ]],
    device const float* x            [[ buffer(3) ]],
    device atomic_float* y           [[ buffer(4) ]],
    uint tid                         [[ thread_position_in_grid ]]
) {
    uint row = row_idx[tid];
    uint col = col_idx[tid];
    float val = values[tid];
    float prod = val * x[col];

    // 原子加到 y[row]
    atomic_fetch_add_explicit(&y[row], prod, memory_order_relaxed);
}


