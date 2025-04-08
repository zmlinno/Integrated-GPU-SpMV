//
//  csc_kernel.metal
//  SpMV_Metal
//
//  Created by 张木林 on 4/8/25.
//

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void csc_spmv(
    device const uint* col_ptrs       [[ buffer(0) ]],
    device const uint* row_indices    [[ buffer(1) ]],
    device const float* values        [[ buffer(2) ]],
    device const float* x             [[ buffer(3) ]],
    device atomic_float* y            [[ buffer(4) ]],
    constant uint& num_cols           [[ buffer(5) ]],
    uint tid                          [[ thread_position_in_grid ]]
) {
    if (tid >= num_cols) return;

    uint col_start = col_ptrs[tid];
    uint col_end = col_ptrs[tid + 1];

    float x_val = x[tid];

    for (uint i = col_start; i < col_end; ++i) {
        uint row = row_indices[i];
        float val = values[i];
        float contrib = val * x_val;

        atomic_fetch_add_explicit(&y[row], contrib, memory_order_relaxed);
    }
}


