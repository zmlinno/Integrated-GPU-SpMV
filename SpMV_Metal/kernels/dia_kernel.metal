//
//  dia_kernel.metal
//  SpMV_Metal
//
//  Created by 张木林 on 4/8/25.
//

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void dia_spmv(
    device const float* values       [[ buffer(0) ]],   // [num_diagonals × num_rows]
    device const int* offsets        [[ buffer(1) ]],   // 对角线偏移量
    device const float* x            [[ buffer(2) ]],   // 输入向量
    device float* y                  [[ buffer(3) ]],   // 输出向量
    constant uint& num_diagonals     [[ buffer(4) ]],
    constant uint& num_rows          [[ buffer(5) ]],
    uint tid                         [[ thread_position_in_grid ]]
) {
    if (tid >= num_rows) return;

    float sum = 0.0f;
    for (uint d = 0; d < num_diagonals; ++d) {
        int col = int(tid) + offsets[d];
        if (col >= 0 && col < int(num_rows)) {
            uint idx = d * num_rows + tid;
            sum += values[idx] * x[col];
        }
    }
    y[tid] = sum;
}


