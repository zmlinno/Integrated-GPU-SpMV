//
//  NSObject+main.m
//  Test
//
//  Created by 张木林 on 2/18/25.
//


#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <time.h>
#include<QuartzCore/QuartzCore.h>
#include<MetalPerformanceShaders/MetalPerformanceShaders.h>
#include<stdlib.h>
#include<string.h>
#include<stdio.h>


//第四处改动
//优化 VSync
void disableVSync() {
    CVDisplayLinkRef displayLink;
    CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
    CVDisplayLinkStop(displayLink); // 停止 VSync 触发
    CVDisplayLinkRelease(displayLink); // 释放资源
}


// 函数声明：加载 Matrix Market 格式的矩阵数据
void loadMatrixMarket(const char* filePath,
                      NSMutableArray<NSNumber*> *values,
                      NSMutableArray<NSNumber*> *rowIndices,
                      NSMutableArray<NSNumber*> *colIndices,
                      int* numRows,
                      int* numCols);

auto executeKernelWithTiming = ^(id<MTLComputePipelineState> pipeline,
                                 NSArray<id<MTLBuffer>> *buffers,
                                 id<MTLCommandQueue> commandQueue,
                                 int numRows,
                                 double *cpuSetupTime,
                                 double *gpuExecutionTime) {
    // 1. 测量 CPU 设置时间
    clock_t cpuStart = clock();
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    for (int i = 0; i < buffers.count; i++) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    
    //NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
    //这里是第一处改动
    NSUInteger optimalThreadGroupSize = MIN(256,pipeline.maxTotalThreadsPerThreadgroup);
    MTLSize gridSize = MTLSizeMake(numRows, 1, 1);
    MTLSize threadGroupSize3D = MTLSizeMake(optimalThreadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize3D];
    [encoder endEncoding];
    clock_t cpuEnd = clock();

    // 2. 提交命令并等待完成，测量 GPU 执行时间
    clock_t gpuStart = clock();
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    clock_t gpuEnd = clock();

    // 3. 计算 CPU 和 GPU 的耗时
    *cpuSetupTime = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
    *gpuExecutionTime = (double)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;
    
    //在这里是第二处改动
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cmdBuffer)
     {
        clock_t gpuEnd = clock();
        *gpuExecutionTime = (double)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;
        NSLog(@"GPU Execution Time: %f seconds", *gpuExecutionTime);;
    }];
    *cpuSetupTime = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
};

int main() {
    @autoreleasepool {
        
        //这是一起的修改
        //禁用VSync
        disableVSync();
        // 初始化 Metal
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // ========================
        // 加载数据集
        // ========================
        NSMutableArray<NSNumber*> *values = [NSMutableArray array];
        NSMutableArray<NSNumber*> *rowIndices = [NSMutableArray array];
        NSMutableArray<NSNumber*> *colIndices = [NSMutableArray array];
        int numRows = 0, numCols = 0;

        // 替换为您的文件路径
        const char* filePath = "/Users/zhangmulin/Downloads/kmer_V2a/kmer_V2a.mtx";
        loadMatrixMarket(filePath, values, rowIndices, colIndices, &numRows, &numCols);

        // 将 NSMutableArray 转换为原始 C 数组
        int nnz = (int)values.count;
        float* c_values = (float*)malloc(nnz * sizeof(float));
        int* c_rowIndices = (int*)malloc(nnz * sizeof(int));
        int* c_colIndices = (int*)malloc(nnz * sizeof(int));

        for (int i = 0; i < nnz; i++) {
            c_values[i] = [values[i] floatValue];
            c_rowIndices[i] = [rowIndices[i] intValue];
            c_colIndices[i] = [colIndices[i] intValue];
        }

        // 转换为 CSR 格式（计算行指针）
        int* c_rowPointers = (int*)malloc((numRows + 1) * sizeof(int));
        memset(c_rowPointers, 0, (numRows + 1) * sizeof(int));
        for (int i = 0; i < nnz; i++) {
            c_rowPointers[c_rowIndices[i] + 1]++;
        }
        for (int i = 1; i <= numRows; i++) {
            c_rowPointers[i] += c_rowPointers[i - 1];
        }

        // 转换为 CSC 格式
        int* c_cscColPointers = (int*)malloc((numCols + 1) * sizeof(int));
        int* c_cscRowIndices = (int*)malloc(nnz * sizeof(int));
        float* c_cscValues = (float*)malloc(nnz * sizeof(float));
        memset(c_cscColPointers, 0, (numCols + 1) * sizeof(int));
        for (int i = 0; i < nnz; i++) {
            c_cscColPointers[c_colIndices[i] + 1]++;
        }
        for (int i = 1; i <= numCols; i++) {
            c_cscColPointers[i] += c_cscColPointers[i - 1];
        }
        for (int i = 0; i < nnz; i++) {
            int col = c_colIndices[i];
            int dest = c_cscColPointers[col]++;
            c_cscRowIndices[dest] = c_rowIndices[i];
            c_cscValues[dest] = c_values[i];
        }
        for (int i = numCols; i > 0; i--) {
            c_cscColPointers[i] = c_cscColPointers[i - 1];
        }
        c_cscColPointers[0] = 0;

        // HYB 格式（CSR + ELL）
        int ellWidth = 0;
        for (int row = 0; row < numRows; row++) {
            int rowStart = c_rowPointers[row];
            int rowEnd = c_rowPointers[row + 1];
            ellWidth = MAX(ellWidth, rowEnd - rowStart);
        }
        float* ellValues = (float*)malloc(numRows * ellWidth * sizeof(float));
        int* ellColumnIndices = (int*)malloc(numRows * ellWidth * sizeof(int));
        memset(ellValues, 0, numRows * ellWidth * sizeof(float));
        memset(ellColumnIndices, -1, numRows * ellWidth * sizeof(int));
        for (int row = 0; row < numRows; row++) {
            int rowStart = c_rowPointers[row];
            int rowEnd = c_rowPointers[row + 1];
            for (int i = rowStart, col = 0; i < rowEnd; i++, col++) {
                ellValues[row * ellWidth + col] = c_values[i];
                ellColumnIndices[row * ellWidth + col] = c_colIndices[i];
            }
        }

        // 初始化向量 x 和 y
        float* x = (float*)malloc(numCols * sizeof(float));
        float* y = (float*)malloc(numRows * sizeof(float));
        for (int i = 0; i < numCols; i++) x[i] = 1.0f;
        memset(y, 0, numRows * sizeof(float));

        // ========================
        // 创建 Metal 缓冲区
        // ========================
        id<MTLBuffer> xBuffer = [device newBufferWithBytes:x length:numCols * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> yBuffer = [device newBufferWithBytes:y length:numRows * sizeof(float) options:MTLResourceStorageModeShared];
        //在这里是第三处改动
        // 初始化 Metal 资源（优化存储模式）
        MTLResourceOptions sharedOptions = MTLResourceStorageModeShared;
        MTLResourceOptions privateOptions = MTLResourceStorageModePrivate;

        // 1️⃣ **创建 CPU 可访问的临时缓冲区**
        id<MTLBuffer> tempCsrValuesBuffer = [device newBufferWithBytes:c_values
                                                                length:nnz * sizeof(float)
                                                               options:sharedOptions];

        id<MTLBuffer> tempCsrColumnIndicesBuffer = [device newBufferWithBytes:c_colIndices
                                                                        length:nnz * sizeof(int)
                                                                       options:sharedOptions];

        id<MTLBuffer> tempCsrRowPointersBuffer = [device newBufferWithBytes:c_rowPointers
                                                                      length:(numRows + 1) * sizeof(int)
                                                                     options:sharedOptions];

        id<MTLBuffer> tempCooRowIndicesBuffer = [device newBufferWithBytes:c_rowIndices
                                                                     length:nnz * sizeof(int)
                                                                    options:sharedOptions];

        id<MTLBuffer> tempCscValuesBuffer = [device newBufferWithBytes:c_cscValues
                                                                 length:nnz * sizeof(float)
                                                                options:sharedOptions];

        id<MTLBuffer> tempCscRowIndicesBuffer = [device newBufferWithBytes:c_cscRowIndices
                                                                     length:nnz * sizeof(int)
                                                                    options:sharedOptions];

        id<MTLBuffer> tempCscColPointersBuffer = [device newBufferWithBytes:c_cscColPointers
                                                                      length:(numCols + 1) * sizeof(int)
                                                                     options:sharedOptions];

        id<MTLBuffer> tempHybEllValuesBuffer = [device newBufferWithBytes:ellValues
                                                                   length:(numRows * ellWidth) * sizeof(float)
                                                                  options:sharedOptions];

        id<MTLBuffer> tempHybEllColumnIndicesBuffer = [device newBufferWithBytes:ellColumnIndices
                                                                           length:(numRows * ellWidth) * sizeof(int)
                                                                          options:sharedOptions];

        // 2️⃣ **创建 GPU 端的 Private Buffer**
        id<MTLBuffer> csrValuesBuffer = [device newBufferWithLength:nnz * sizeof(float) options:privateOptions];
        id<MTLBuffer> csrColumnIndicesBuffer = [device newBufferWithLength:nnz * sizeof(int) options:privateOptions];
        id<MTLBuffer> csrRowPointersBuffer = [device newBufferWithLength:(numRows + 1) * sizeof(int) options:privateOptions];

        id<MTLBuffer> cooRowIndicesBuffer = [device newBufferWithLength:nnz * sizeof(int) options:privateOptions];

        id<MTLBuffer> cscValuesBuffer = [device newBufferWithLength:nnz * sizeof(float) options:privateOptions];
        id<MTLBuffer> cscRowIndicesBuffer = [device newBufferWithLength:nnz * sizeof(int) options:privateOptions];
        id<MTLBuffer> cscColPointersBuffer = [device newBufferWithLength:(numCols + 1) * sizeof(int) options:privateOptions];

        id<MTLBuffer> hybEllValuesBuffer = [device newBufferWithLength:(numRows * ellWidth) * sizeof(float) options:privateOptions];
        id<MTLBuffer> hybEllColumnIndicesBuffer = [device newBufferWithLength:(numRows * ellWidth) * sizeof(int) options:privateOptions];

        // 3️⃣ **使用 Blit Command 复制数据**
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

        [blitEncoder copyFromBuffer:tempCsrValuesBuffer sourceOffset:0 toBuffer:csrValuesBuffer destinationOffset:0 size:nnz * sizeof(float)];
        [blitEncoder copyFromBuffer:tempCsrColumnIndicesBuffer sourceOffset:0 toBuffer:csrColumnIndicesBuffer destinationOffset:0 size:nnz * sizeof(int)];
        [blitEncoder copyFromBuffer:tempCsrRowPointersBuffer sourceOffset:0 toBuffer:csrRowPointersBuffer destinationOffset:0 size:(numRows + 1) * sizeof(int)];

        [blitEncoder copyFromBuffer:tempCooRowIndicesBuffer sourceOffset:0 toBuffer:cooRowIndicesBuffer destinationOffset:0 size:nnz * sizeof(int)];

        [blitEncoder copyFromBuffer:tempCscValuesBuffer sourceOffset:0 toBuffer:cscValuesBuffer destinationOffset:0 size:nnz * sizeof(float)];
        [blitEncoder copyFromBuffer:tempCscRowIndicesBuffer sourceOffset:0 toBuffer:cscRowIndicesBuffer destinationOffset:0 size:nnz * sizeof(int)];
        [blitEncoder copyFromBuffer:tempCscColPointersBuffer sourceOffset:0 toBuffer:cscColPointersBuffer destinationOffset:0 size:(numCols + 1) * sizeof(int)];

        [blitEncoder copyFromBuffer:tempHybEllValuesBuffer sourceOffset:0 toBuffer:hybEllValuesBuffer destinationOffset:0 size:(numRows * ellWidth) * sizeof(float)];
        [blitEncoder copyFromBuffer:tempHybEllColumnIndicesBuffer sourceOffset:0 toBuffer:hybEllColumnIndicesBuffer destinationOffset:0 size:(numRows * ellWidth) * sizeof(int)];

        [blitEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted]; // 确保拷贝完成

        // ✅ **释放临时缓冲区**
        tempCsrValuesBuffer = nil;
        tempCsrColumnIndicesBuffer = nil;
        tempCsrRowPointersBuffer = nil;
        tempCooRowIndicesBuffer = nil;
        tempCscValuesBuffer = nil;
        tempCscRowIndicesBuffer = nil;
        tempCscColPointersBuffer = nil;
        tempHybEllValuesBuffer = nil;
        tempHybEllColumnIndicesBuffer = nil;
        

        // ========================
        // 加载 Metal 内核
        // ========================
        NSError *error = nil;
        NSString *libraryPath = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
        id<MTLLibrary> library = [device newLibraryWithFile:libraryPath error:&error];
        id<MTLFunction> csrKernel = [library newFunctionWithName:@"spmv_csr"];
        id<MTLFunction> cooKernel = [library newFunctionWithName:@"spmv_coo"];
        id<MTLFunction> cscKernel = [library newFunctionWithName:@"spmv_csc"];
        id<MTLFunction> hybKernel = [library newFunctionWithName:@"spmv_hyb"];
        id<MTLComputePipelineState> csrPipeline = [device newComputePipelineStateWithFunction:csrKernel error:&error];
        id<MTLComputePipelineState> cooPipeline = [device newComputePipelineStateWithFunction:cooKernel error:&error];
        id<MTLComputePipelineState> cscPipeline = [device newComputePipelineStateWithFunction:cscKernel error:&error];
        id<MTLComputePipelineState> hybPipeline = [device newComputePipelineStateWithFunction:hybKernel error:&error];

        // ========================
        // 执行计算并记录时间
        // ========================
        double cpuSetupTime, gpuExecutionTime;
        executeKernelWithTiming(csrPipeline,
                                @[csrValuesBuffer, csrColumnIndicesBuffer, csrRowPointersBuffer, xBuffer, yBuffer],
                                commandQueue, numRows, &cpuSetupTime, &gpuExecutionTime);
        NSLog(@"CSR CPU Setup Time: %f seconds", cpuSetupTime);
        NSLog(@"CSR GPU Execution Time: %f seconds", gpuExecutionTime);

        executeKernelWithTiming(cooPipeline,
                                @[csrValuesBuffer, cooRowIndicesBuffer, csrColumnIndicesBuffer, xBuffer, yBuffer],
                                commandQueue, numRows, &cpuSetupTime, &gpuExecutionTime);
        NSLog(@"COO CPU Setup Time: %f seconds", cpuSetupTime);
        NSLog(@"COO GPU Execution Time: %f seconds", gpuExecutionTime);

        executeKernelWithTiming(cscPipeline,
                                @[cscValuesBuffer, cscRowIndicesBuffer, cscColPointersBuffer, xBuffer, yBuffer],
                                commandQueue, numRows, &cpuSetupTime, &gpuExecutionTime);
        NSLog(@"CSC CPU Setup Time: %f seconds", cpuSetupTime);
        NSLog(@"CSC GPU Execution Time: %f seconds", gpuExecutionTime);

        executeKernelWithTiming(hybPipeline,
                                @[csrValuesBuffer, csrColumnIndicesBuffer, csrRowPointersBuffer,
                                  hybEllValuesBuffer, hybEllColumnIndicesBuffer, xBuffer, yBuffer],
                                commandQueue, numRows, &cpuSetupTime, &gpuExecutionTime);
        NSLog(@"HYB CPU Setup Time: %f seconds", cpuSetupTime);
        NSLog(@"HYB GPU Execution Time: %f seconds", gpuExecutionTime);

        // 释放内存
        free(c_values);
        free(c_rowIndices);
        free(c_colIndices);
        free(c_rowPointers);
        free(c_cscColPointers);
        free(c_cscRowIndices);
        free(c_cscValues);
        free(ellValues);
        free(ellColumnIndices);
        free(x);
        free(y);

        return 0;
    }
}




// ========================
// Matrix Market 数据加载函数
// ========================
void loadMatrixMarket(const char* filePath,
                      NSMutableArray<NSNumber*> *values,
                      NSMutableArray<NSNumber*> *rowIndices,
                      NSMutableArray<NSNumber*> *colIndices,
                      int* numRows,
                      int* numCols) {
    FILE* file = fopen(filePath, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    char line[1024];
    *numRows = 0;
    *numCols = 0;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '%') continue; // 忽略注释行

        int row, col;
        float value;
        if (sscanf(line, "%d %d %f", &row, &col, &value) == 3) {
            [rowIndices addObject:@(row - 1)]; // 将 Matrix Market 索引从 1-based 转为 0-based
            [colIndices addObject:@(col - 1)];
            [values addObject:@(value)];
            if (row > *numRows) *numRows = row;
            if (col > *numCols) *numCols = col;
        }
    }

    fclose(file);
}



// // 函数声明：加载 Matrix Market 格式的矩阵数据
// void loadMatrixMarket(const char* filePath,
//                       NSMutableArray<NSNumber*> *values,
//                       NSMutableArray<NSNumber*> *rowIndices,
//                       NSMutableArray<NSNumber*> *colIndices,
//                       int* numRows,
//                       int* numCols);

// auto executeKernelWithTiming = ^(id<MTLComputePipelineState> pipeline,
//                                  NSArray<id<MTLBuffer>> *buffers,
//                                  id<MTLCommandQueue> commandQueue,
//                                  int numRows,
//                                  double *cpuSetupTime,
//                                  double *gpuExecutionTime) {
//     // 1. 测量 CPU 设置时间
//     clock_t cpuStart = clock();
//     id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//     id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
//     [encoder setComputePipelineState:pipeline];
//     for (int i = 0; i < buffers.count; i++) {
//         [encoder setBuffer:buffers[i] offset:0 atIndex:i];
//     }
//     MTLSize gridSize = MTLSizeMake(numRows, 1, 1);
//     NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
//     MTLSize threadGroupSize3D = MTLSizeMake(threadGroupSize, 1, 1);
//     [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize3D];
//     [encoder endEncoding];
//     clock_t cpuEnd = clock();

//     // 2. 提交命令并等待完成，测量 GPU 执行时间
//     clock_t gpuStart = clock();
//     [commandBuffer commit];
//     [commandBuffer waitUntilCompleted];
//     clock_t gpuEnd = clock();

//     // 3. 计算 CPU 和 GPU 的耗时
//     *cpuSetupTime = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
//     *gpuExecutionTime = (double)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;
// };

// int main() {
//     @autoreleasepool {
//         // 初始化 Metal
//         id<MTLDevice> device = MTLCreateSystemDefaultDevice();
//         id<MTLCommandQueue> commandQueue = [device newCommandQueue];

//         // ========================
//         // 加载数据集
//         // ========================
//         NSMutableArray<NSNumber*> *values = [NSMutableArray array];
//         NSMutableArray<NSNumber*> *rowIndices = [NSMutableArray array];
//         NSMutableArray<NSNumber*> *colIndices = [NSMutableArray array];
//         int numRows = 0, numCols = 0;

//         // 替换为您的文件路径
//         const char* filePath = "/Users/zhangmulin/Downloads/kmer_V2a/kmer_V2a.mtx";
//         loadMatrixMarket(filePath, values, rowIndices, colIndices, &numRows, &numCols);

//         // 将 NSMutableArray 转换为原始 C 数组
//         int nnz = (int)values.count;
//         float* c_values = (float*)malloc(nnz * sizeof(float));
//         int* c_rowIndices = (int*)malloc(nnz * sizeof(int));
//         int* c_colIndices = (int*)malloc(nnz * sizeof(int));

//         for (int i = 0; i < nnz; i++) {
//             c_values[i] = [values[i] floatValue];
//             c_rowIndices[i] = [rowIndices[i] intValue];
//             c_colIndices[i] = [colIndices[i] intValue];
//         }

//         // 转换为 CSR 格式（计算行指针）
//         int* c_rowPointers = (int*)malloc((numRows + 1) * sizeof(int));
//         memset(c_rowPointers, 0, (numRows + 1) * sizeof(int));
//         for (int i = 0; i < nnz; i++) {
//             c_rowPointers[c_rowIndices[i] + 1]++;
//         }
//         for (int i = 1; i <= numRows; i++) {
//             c_rowPointers[i] += c_rowPointers[i - 1];
//         }

//         // 转换为 CSC 格式
//         int* c_cscColPointers = (int*)malloc((numCols + 1) * sizeof(int));
//         int* c_cscRowIndices = (int*)malloc(nnz * sizeof(int));
//         float* c_cscValues = (float*)malloc(nnz * sizeof(float));
//         memset(c_cscColPointers, 0, (numCols + 1) * sizeof(int));
//         for (int i = 0; i < nnz; i++) {
//             c_cscColPointers[c_colIndices[i] + 1]++;
//         }
//         for (int i = 1; i <= numCols; i++) {
//             c_cscColPointers[i] += c_cscColPointers[i - 1];
//         }
//         for (int i = 0; i < nnz; i++) {
//             int col = c_colIndices[i];
//             int dest = c_cscColPointers[col]++;
//             c_cscRowIndices[dest] = c_rowIndices[i];
//             c_cscValues[dest] = c_values[i];
//         }
//         for (int i = numCols; i > 0; i--) {
//             c_cscColPointers[i] = c_cscColPointers[i - 1];
//         }
//         c_cscColPointers[0] = 0;

//         // HYB 格式（CSR + ELL）
//         int ellWidth = 0;
//         for (int row = 0; row < numRows; row++) {
//             int rowStart = c_rowPointers[row];
//             int rowEnd = c_rowPointers[row + 1];
//             ellWidth = MAX(ellWidth, rowEnd - rowStart);
//         }
//         float* ellValues = (float*)malloc(numRows * ellWidth * sizeof(float));
//         int* ellColumnIndices = (int*)malloc(numRows * ellWidth * sizeof(int));
//         memset(ellValues, 0, numRows * ellWidth * sizeof(float));
//         memset(ellColumnIndices, -1, numRows * ellWidth * sizeof(int));
//         for (int row = 0; row < numRows; row++) {
//             int rowStart = c_rowPointers[row];
//             int rowEnd = c_rowPointers[row + 1];
//             for (int i = rowStart, col = 0; i < rowEnd; i++, col++) {
//                 ellValues[row * ellWidth + col] = c_values[i];
//                 ellColumnIndices[row * ellWidth + col] = c_colIndices[i];
//             }
//         }

//         // 初始化向量 x 和 y
//         float* x = (float*)malloc(numCols * sizeof(float));
//         float* y = (float*)malloc(numRows * sizeof(float));
//         for (int i = 0; i < numCols; i++) x[i] = 1.0f;
//         memset(y, 0, numRows * sizeof(float));

//         // ========================
//         // 创建 Metal 缓冲区
//         // ========================
//         id<MTLBuffer> xBuffer = [device newBufferWithBytes:x length:numCols * sizeof(float) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> yBuffer = [device newBufferWithBytes:y length:numRows * sizeof(float) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> csrValuesBuffer = [device newBufferWithBytes:c_values length:nnz * sizeof(float) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> csrColumnIndicesBuffer = [device newBufferWithBytes:c_colIndices length:nnz * sizeof(int) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> csrRowPointersBuffer = [device newBufferWithBytes:c_rowPointers length:(numRows + 1) * sizeof(int) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> cooRowIndicesBuffer = [device newBufferWithBytes:c_rowIndices length:nnz * sizeof(int) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> cscValuesBuffer = [device newBufferWithBytes:c_cscValues length:nnz * sizeof(float) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> cscRowIndicesBuffer = [device newBufferWithBytes:c_cscRowIndices length:nnz * sizeof(int) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> cscColPointersBuffer = [device newBufferWithBytes:c_cscColPointers length:(numCols + 1) * sizeof(int) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> hybEllValuesBuffer = [device newBufferWithBytes:ellValues length:(numRows * ellWidth) * sizeof(float) options:MTLResourceStorageModeShared];
//         id<MTLBuffer> hybEllColumnIndicesBuffer = [device newBufferWithBytes:ellColumnIndices length:(numRows * ellWidth) * sizeof(int) options:MTLResourceStorageModeShared];

//         // ========================
//         // 加载 Metal 内核
//         // ========================
//         NSError *error = nil;
//         NSString *libraryPath = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
//         id<MTLLibrary> library = [device newLibraryWithFile:libraryPath error:&error];
//         id<MTLFunction> csrKernel = [library newFunctionWithName:@"spmv_csr"];
//         id<MTLFunction> cooKernel = [library newFunctionWithName:@"spmv_coo"];
//         id<MTLFunction> cscKernel = [library newFunctionWithName:@"spmv_csc"];
//         id<MTLFunction> hybKernel = [library newFunctionWithName:@"spmv_hyb"];
//         id<MTLComputePipelineState> csrPipeline = [device newComputePipelineStateWithFunction:csrKernel error:&error];
//         id<MTLComputePipelineState> cooPipeline = [device newComputePipelineStateWithFunction:cooKernel error:&error];
//         id<MTLComputePipelineState> cscPipeline = [device newComputePipelineStateWithFunction:cscKernel error:&error];
//         id<MTLComputePipelineState> hybPipeline = [device newComputePipelineStateWithFunction:hybKernel error:&error];

//         // ========================
//         // 执行计算并记录时间
//         // ========================
//         double cpuSetupTime, gpuExecutionTime;
//         executeKernelWithTiming(csrPipeline,
//                                 @[csrValuesBuffer, csrColumnIndicesBuffer, csrRowPointersBuffer, xBuffer, yBuffer],
//                                 commandQueue, numRows, &cpuSetupTime, &gpuExecutionTime);
//         NSLog(@"CSR CPU Setup Time: %f seconds", cpuSetupTime);
//         NSLog(@"CSR GPU Execution Time: %f seconds", gpuExecutionTime);

//         executeKernelWithTiming(cooPipeline,
//                                 @[csrValuesBuffer, cooRowIndicesBuffer, csrColumnIndicesBuffer, xBuffer, yBuffer],
//                                 commandQueue, numRows, &cpuSetupTime, &gpuExecutionTime);
//         NSLog(@"COO CPU Setup Time: %f seconds", cpuSetupTime);
//         NSLog(@"COO GPU Execution Time: %f seconds", gpuExecutionTime);

//         executeKernelWithTiming(cscPipeline,
//                                 @[cscValuesBuffer, cscRowIndicesBuffer, cscColPointersBuffer, xBuffer, yBuffer],
//                                 commandQueue, numRows, &cpuSetupTime, &gpuExecutionTime);
//         NSLog(@"CSC CPU Setup Time: %f seconds", cpuSetupTime);
//         NSLog(@"CSC GPU Execution Time: %f seconds", gpuExecutionTime);

//         executeKernelWithTiming(hybPipeline,
//                                 @[csrValuesBuffer, csrColumnIndicesBuffer, csrRowPointersBuffer,
//                                   hybEllValuesBuffer, hybEllColumnIndicesBuffer, xBuffer, yBuffer],
//                                 commandQueue, numRows, &cpuSetupTime, &gpuExecutionTime);
//         NSLog(@"HYB CPU Setup Time: %f seconds", cpuSetupTime);
//         NSLog(@"HYB GPU Execution Time: %f seconds", gpuExecutionTime);

//         // 释放内存
//         free(c_values);
//         free(c_rowIndices);
//         free(c_colIndices);
//         free(c_rowPointers);
//         free(c_cscColPointers);
//         free(c_cscRowIndices);
//         free(c_cscValues);
//         free(ellValues);
//         free(ellColumnIndices);
//         free(x);
//         free(y);

//         return 0;
//     }
// }

// // ========================
// // Matrix Market 数据加载函数
// // ========================
// void loadMatrixMarket(const char* filePath,
//                       NSMutableArray<NSNumber*> *values,
//                       NSMutableArray<NSNumber*> *rowIndices,
//                       NSMutableArray<NSNumber*> *colIndices,
//                       int* numRows,
//                       int* numCols) {
//     FILE* file = fopen(filePath, "r");
//     if (!file) {
//         perror("Error opening file");
//         return;
//     }

//     char line[1024];
//     *numRows = 0;
//     *numCols = 0;

//     while (fgets(line, sizeof(line), file)) {
//         if (line[0] == '%') continue; // 忽略注释行

//         int row, col;
//         float value;
//         if (sscanf(line, "%d %d %f", &row, &col, &value) == 3) {
//             [rowIndices addObject:@(row - 1)]; // 将 Matrix Market 索引从 1-based 转为 0-based
//             [colIndices addObject:@(col - 1)];
//             [values addObject:@(value)];
//             if (row > *numRows) *numRows = row;
//             if (col > *numCols) *numCols = col;
//         }
//     }

//     fclose(file);
// }
