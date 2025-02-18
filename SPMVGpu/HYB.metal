//
//  HYB.metal
//  SPMVGpu
//
//  Created by 张木林 on 1/10/25.
//

#include <metal_stdlib>
using namespace metal;


#define BLOCK_SIZE 32
 // 共享内存优化
//kernel void spmv_hyb(
//    device const float* __restrict csr_values [[buffer(0)]],
//    device const int* __restrict csr_columnIndices [[buffer(1)]],
//    device const int* __restrict csr_rowPointers [[buffer(2)]],
//    device const float* __restrict ell_values [[buffer(3)]],
//    device const int* __restrict ell_columnIndices [[buffer(4)]],
//    device const float* __restrict x [[buffer(5)]],
//    device float* __restrict y [[buffer(6)]],
//    uint gid [[thread_position_in_grid]]
//) {
//    if (gid >= csr_rowPointers[0]) return;
//
//    // Process CSR part
//    int csr_row_start = csr_rowPointers[gid];
//    int csr_row_end = csr_rowPointers[gid + 1];
//
//    float sum = 0.0f;
//    for (int i = csr_row_start; i < csr_row_end; i++) {
//        sum += csr_values[i] * x[csr_columnIndices[i]];
//    }
//
//    // Process ELL part
//    int ell_max_cols = csr_rowPointers[1]; // Assume this value is known
//    for (int i = 0; i < ell_max_cols; i++) {
//        int index = gid * ell_max_cols + i;
//        if (ell_columnIndices[index] >= 0) {
//            sum += ell_values[index] * x[ell_columnIndices[index]];
//        }
//    }
//
//    y[gid] = sum;
//}


//kernel void spmv_hyb(device const float* csrValues,
//                     device const int* csrColumnIndices,
//                     device const int* csrRowPointers,
//                     device const float* ellValues,
//                     device const int* ellColumnIndices,
//                     device const float* x,
//                     device float* y,
//                     constant int& numRows [[buffer(6)]],  // ✅ 传递 numRows
//                     constant int& numCols [[buffer(7)]],  // ✅ 传递 numCols
//                     constant int& ellWidth [[buffer(8)]], // ✅ 传递 ellWidth
//                     uint tid [[thread_position_in_grid]]) {
//
//    uint row = tid;
//    if (row < numRows) {
//        float sum = 0.0;
//
//        // **ELL 部分计算**
//        for (int i = 0; i < ellWidth; i++) {
//            int col = ellColumnIndices[row * ellWidth + i];
//            if (col >= 0) {  // -1 代表无效索引
//                sum += ellValues[row * ellWidth + i] * x[col];
//            }
//        }
//
//        // **CSR 部分计算**
//        for (int j = csrRowPointers[row]; j < csrRowPointers[row + 1]; j++) {
//            sum += csrValues[j] * x[csrColumnIndices[j]];
//        }
//
//        y[row] = sum;
//    }
//}
