//
//  ell_kernel.metal
//  SpMV_Metal
//
//  Created by 张木林 on 4/8/25.
//
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void ell_spmv(
    device const uint* indices      [[ buffer(0) ]],   // [rows × K]
    device const float* values      [[ buffer(1) ]],   // [rows × K]
    device const float* x           [[ buffer(2) ]],   // input vector
    device float* y                 [[ buffer(3) ]],   // output
    constant uint& num_cols_per_row[[ buffer(4) ]],    // 即 K
    constant uint& num_rows        [[ buffer(5) ]],    // 矩阵行数
    uint tid                        [[ thread_position_in_grid ]]
) {
    if (tid >= num_rows) return;

    float sum = 0.0f;
    for (uint k = 0; k < num_cols_per_row; ++k) {
        uint idx = tid * num_cols_per_row + k;
        uint col = indices[idx];
        float val = values[idx];

        // 通常 padding 部分会设置为 col == UINT_MAX（表示无效项）
        if (col != UINT_MAX) {
            sum += val * x[col];
        }
    }
    y[tid] = sum;
}


