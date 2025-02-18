//
//  Shaders.metal
//  SPMVGpu
//
//  Created by 张木林 on 1/10/25.
//

#include <metal_stdlib>

#define BLOCK_SIZE 32

using namespace metal;

// Kernel function for SpMV using CSR format
//kernel void spmv_csr(
//    device const float* values [[buffer(0)]],
//    device const int* columnIndices [[buffer(1)]],
//    device const int* rowPointers [[buffer(2)]],
//    device const float* x [[buffer(3)]],
//    device float* y [[buffer(4)]],
//    uint gid [[thread_position_in_grid]]
//) {
//    if (gid >= rowPointers[3]) return;  // 防止访问越界
//
//    float sum = 0.0;
//    int row_start = rowPointers[gid];
//    int row_end = rowPointers[gid + 1];
//
//    for (int i = row_start; i < row_end; i++) {
//        sum += values[i] * x[columnIndices[i]];
//    }
//
//    y[gid] = sum;
//}
//kernel void spmv_csr(device const float* values,
//                     device const int* columnIndices,
//                     device const int* rowPointers,
//                     device const float* x,
//                     device float* y,
//                     constant int& numRows [[buffer(5)]], // 传递 numRows
//                     uint tid [[thread_position_in_grid]],
//                     threadgroup float* shared_x [[threadgroup(0)]]) {
//    
//    uint row = tid;
//    if (row < numRows) {
//        float sum = 0.0;
//        int rowStart = rowPointers[row];
//        int rowEnd = rowPointers[row + 1];
//
//        for (int j = rowStart; j < rowEnd; j++) {
//            int col = columnIndices[j];
//            sum += values[j] * x[col];
//        }
//        
//        y[row] = sum;
//    }
//}


// **CSR Kernel**
kernel void spmv_csr(device const float* values,
                     device const int* columnIndices,
                     device const int* rowPointers,
                     device const float* x,
                     device float* y,
                     constant int& numRows [[buffer(5)]],  // ✅ 传递 numRows
                     uint tid [[thread_position_in_grid]]) {

    uint row = tid;
    if (row < numRows) {
        float sum = 0.0;
        for (int j = rowPointers[row]; j < rowPointers[row + 1]; j++) {
            sum += values[j] * x[columnIndices[j]];
        }
        y[row] = sum;
    }
}

// **COO Kernel**
kernel void spmv_coo(device const float* values,
                     device const int* rowIndices,
                     device const int* colIndices,
                     device const float* x,
                     device float* y,
                     constant int& nnz [[buffer(5)]],  // ✅ 传递 nnz
                     uint tid [[thread_position_in_grid]]) {

    uint index = tid;
    if (index < nnz) {
        atomic_fetch_add_explicit((device atomic_float*)&y[rowIndices[index]],
                                  values[index] * x[colIndices[index]],
                                  memory_order_relaxed);
    }
}

// **CSC Kernel**
kernel void spmv_csc(device const float* values,
                     device const int* rowIndices,
                     device const int* colPointers,
                     device const float* x,
                     device float* y,
                     constant int& numCols [[buffer(5)]],  // ✅ 传递 numCols
                     uint tid [[thread_position_in_grid]]) {

    uint col = tid;
    if (col < numCols) {
        for (int j = colPointers[col]; j < colPointers[col + 1]; j++) {
            atomic_fetch_add_explicit((device atomic_float*)&y[rowIndices[j]],
                                      values[j] * x[col],
                                      memory_order_relaxed);
        }
    }
}

// **HYB Kernel (CSR + ELL)**
kernel void spmv_hyb(device const float* csrValues,
                     device const int* csrColumnIndices,
                     device const int* csrRowPointers,
                     device const float* ellValues,
                     device const int* ellColumnIndices,
                     device const float* x,
                     device float* y,
                     constant int& numRows [[buffer(6)]],  // ✅ 传递 numRows
                     constant int& numCols [[buffer(7)]],  // ✅ 传递 numCols
                     constant int& ellWidth [[buffer(8)]], // ✅ 传递 ellWidth
                     uint tid [[thread_position_in_grid]]) {

    uint row = tid;
    if (row < numRows) {
        float sum = 0.0;

        // **ELL 部分计算**
        for (int i = 0; i < ellWidth; i++) {
            int col = ellColumnIndices[row * ellWidth + i];
            if (col >= 0) {  // -1 代表无效索引
                sum += ellValues[row * ellWidth + i] * x[col];
            }
        }

        // **CSR 部分计算**
        for (int j = csrRowPointers[row]; j < csrRowPointers[row + 1]; j++) {
            sum += csrValues[j] * x[csrColumnIndices[j]];
        }

        y[row] = sum;
    }
}
