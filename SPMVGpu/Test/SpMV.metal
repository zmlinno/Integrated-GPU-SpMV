//
//  SpMV.metal
//  Test
//
//  Created by 张木林 on 2/18/25.
//

#include <metal_stdlib>
using namespace metal;



#define BLOCK_SIZE 32











// ========================================
// CSR Kernel Function
// ========================================
kernel void spmv_csr(
    device const float* __restrict values [[buffer(0)]],
    device const int* __restrict columnIndices [[buffer(1)]],
    device const int* __restrict rowPointers [[buffer(2)]],
    device const float* __restrict x [[buffer(3)]],
    device float* __restrict y [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rowPointers[0]) return;

    float sum = 0.0f;
    int row_start = rowPointers[gid];
    int row_end = rowPointers[gid + 1];

    for (int i = row_start; i < row_end; i++) {
        sum += values[i] * x[columnIndices[i]];
    }
    y[gid] = sum;
}

// ========================================
// COO Kernel Function
// ========================================
kernel void spmv_coo(
    device const float* __restrict values [[buffer(0)]],
    device const int* __restrict rowIndices [[buffer(1)]],
    device const int* __restrict columnIndices [[buffer(2)]],
    device const float* __restrict x [[buffer(3)]],
    device float* __restrict y [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rowIndices[0]) return;

    int row = rowIndices[gid];
    int col = columnIndices[gid];
    atomic_fetch_add_explicit((device atomic_float*)&y[row], values[gid] * x[col], memory_order_relaxed);
}

// ========================================
// CSC Kernel Function
// ========================================
kernel void spmv_csc(
    device const float* __restrict values [[buffer(0)]],
    device const int* __restrict rowIndices [[buffer(1)]],
    device const int* __restrict columnPointers [[buffer(2)]],
    device const float* __restrict x [[buffer(3)]],
    device float* __restrict y [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= columnPointers[0]) return;

    int col_start = columnPointers[gid];
    int col_end = columnPointers[gid + 1];

    for (int i = col_start; i < col_end; i++) {
        int row = rowIndices[i];
        atomic_fetch_add_explicit((device atomic_float*)&y[row], values[i] * x[gid], memory_order_relaxed);
    }
}

// ========================================
// HYB Kernel Function
// ========================================
kernel void spmv_hyb(
    device const float* __restrict csr_values [[buffer(0)]],
    device const int* __restrict csr_columnIndices [[buffer(1)]],
    device const int* __restrict csr_rowPointers [[buffer(2)]],
    device const float* __restrict ell_values [[buffer(3)]],
    device const int* __restrict ell_columnIndices [[buffer(4)]],
    device const float* __restrict x [[buffer(5)]],
    device float* __restrict y [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= csr_rowPointers[0]) return;

    // Process CSR part
    float sum = 0.0f;
    int row_start = csr_rowPointers[gid];
    int row_end = csr_rowPointers[gid + 1];

    for (int i = row_start; i < row_end; i++) {
        sum += csr_values[i] * x[csr_columnIndices[i]];
    }

    // Process ELL part
    int ellWidth = 32; // 设定固定的 ELL 列宽
    for (int i = 0; i < ellWidth; i++) {
        int col = ell_columnIndices[gid * ellWidth + i];
        if (col >= 0) {
            sum += ell_values[gid * ellWidth + i] * x[col];
        }
    }

    y[gid] = sum;
}
