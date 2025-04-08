//
//  hyb_kernel.metal
//  SpMV_Metal
//
//  Created by 张木林 on 4/8/25.
//

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void hyb_spmv(
    device const uint* ell_indices       [[ buffer(0) ]],
    device const float* ell_values       [[ buffer(1) ]],
    device const uint* coo_rows          [[ buffer(2) ]],
    device const uint* coo_cols          [[ buffer(3) ]],
    device const float* coo_vals         [[ buffer(4) ]],
    device const float* x                [[ buffer(5) ]],
    device atomic_float* y               [[ buffer(6) ]],
    constant uint& num_rows              [[ buffer(7) ]],
    constant uint& num_cols_per_row      [[ buffer(8) ]],
    constant uint& coo_nnz               [[ buffer(9) ]],
    constant uint& total_threads         [[ buffer(10)]],
    uint tid                             [[ thread_position_in_grid ]]
) {
    if (tid >= num_rows) return;

    // ========= ELL 部分 =========
    float sum = 0.0f;
    for (uint k = 0; k < num_cols_per_row; ++k) {
        uint idx = tid * num_cols_per_row + k;
        uint col = ell_indices[idx];
        float val = ell_values[idx];

        if (col != UINT_MAX && col < num_rows) {
            sum += val * x[col];
        }
    }
    atomic_fetch_add_explicit(&(y[tid]), sum, memory_order_relaxed);

    // ========= COO 部分 =========
    for (uint i = tid; i < coo_nnz; i += total_threads) {
        uint row = coo_rows[i];
        uint col = coo_cols[i];
        float val = coo_vals[i];

        if (row < num_rows && col < num_rows) {
            float contrib = val * x[col];
            atomic_fetch_add_explicit(&y[row], contrib, memory_order_relaxed);
        }
    }
}

