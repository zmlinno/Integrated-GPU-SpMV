//
//  COO.metal
//  SPMVGpu
//
//  Created by 张木林 on 1/10/25.
//

#include <metal_stdlib>
using namespace metal;


#define BLOCK_SIZE 32
 // 共享内存优化
//kernel void spmv_coo(
//    device const float* __restrict values [[buffer(0)]],
//    device const int* __restrict rowIndices [[buffer(1)]],
//    device const int* __restrict columnIndices [[buffer(2)]],
//    device const float* __restrict x [[buffer(3)]],
//    device float* __restrict y [[buffer(4)]],
//    uint gid [[thread_position_in_grid]]
//) {
//    if (gid >= rowIndices[0]) return;
//
//    int row = rowIndices[gid];
//    int col = columnIndices[gid];
//    atomic_fetch_add_explicit((device atomic_float *)&y[row], values[gid] * x[col], memory_order_relaxed);
//}
//kernel void spmv_coo(device const float* values,
//                     device const int* rowIndices,
//                     device const int* columnIndices,
//                     device const float* x,
//                     device float* y,
//                     constant int& nnz [[buffer(5)]], // 传递 nnz
//                     uint tid [[thread_position_in_grid]]) {
//    
//    uint index = tid;
//    if (index < nnz) {
//        atomic_fetch_add_explicit((device atomic_float*)&y[rowIndices[index]],
//                                  values[index] * x[columnIndices[index]],
//                                  memory_order_relaxed);
//    }
//}
