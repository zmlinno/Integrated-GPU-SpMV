//
//  CSC.metal
//  SPMVGpu
//
//  Created by 张木林 on 1/10/25.
//

#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 32

//kernel void spmv_csc(
//    device const float* __restrict values [[buffer(0)]],
//    device const int* __restrict rowIndices [[buffer(1)]],
//    device const int* __restrict columnPointers [[buffer(2)]],
//    device const float* __restrict x [[buffer(3)]],
//    device float* __restrict y [[buffer(4)]],
//    uint gid [[thread_position_in_grid]]
//) {
//    if (gid >= columnPointers[0]) return;
//
//    int col_start = columnPointers[gid];
//    int col_end = columnPointers[gid + 1];
//
//    for (int i = col_start; i < col_end; i++) {
//        int row = rowIndices[i];
//        atomic_fetch_add_explicit((device atomic_float *)&y[row], values[i] * x[gid], memory_order_relaxed);
//    }
//}
//
//kernel void spmv_csc(device const float* values,
//                     device const int* rowIndices,
//                     device const int* colPointers,
//                     device const float* x,
//                     device float* y,
//                     constant int& numCols [[buffer(5)]],  // ✅ 传递 numCols
//                     uint tid [[thread_position_in_grid]]) {
//
//    uint col = tid;
//    if (col < numCols) {
//        for (int j = colPointers[col]; j < colPointers[col + 1]; j++) {
//            atomic_fetch_add_explicit((device atomic_float*)&y[rowIndices[j]],
//                                      values[j] * x[col],
//                                      memory_order_relaxed);
//        }
//    }
//}
