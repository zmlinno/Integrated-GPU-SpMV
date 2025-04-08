//
//  csr_kernel.metal
//  SpMV_Metal
//
//  Created by 张木林 on 4/8/25.
//

#include <metal_stdlib>
using namespace metal;

// CSR arrays
// row_ptr: 每行的开始索引
// col_ind: 非零值对应的列索引
// values: 非零值数组
// x: 输入向量
// y: 输出结果向量

kernel void csr_spmv(
    device const uint* row_ptr       [[ buffer(0) ]],
    device const uint* col_ind       [[ buffer(1) ]],
    device const float* values       [[ buffer(2) ]],
    device const float* x            [[ buffer(3) ]],
    device float* y                  [[ buffer(4) ]],
    uint tid                         [[ thread_position_in_grid ]]
) {
    // 每个线程处理一行
    uint row_start = row_ptr[tid];
    uint row_end = row_ptr[tid + 1];

    float sum = 0.0f;
    for (uint i = row_start; i < row_end; ++i) {
        uint col = col_ind[i];
        sum += values[i] * x[col];
    }
    y[tid] = sum;
}

