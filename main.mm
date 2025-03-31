//
//  NSObject+main.m
//  SPMVGpu
//
//  Created by 张木林 on 1/10/25.
//改一下

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <time.h>
#include<QuartzCore/QuartzCore.h>
#include<MetalPerformanceShaders/MetalPerformanceShaders.h>
#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#define CSC_NUMCOLS_IDX 4
#define HYB_ELLWIDTH_IDX 9


double measurePowerUsage(id<MTLDevice> device) {
    if (![device supportsFamily:MTLGPUFamilyApple1]) {
        NSLog(@"Device does not support Metal Performance Counters.");
        return -1.0;
    }

    // Metal Performance Counters (MPS) 功耗测量
    MPSMatrixMultiplication *mpsOp = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                           transposeLeft:false
                                                                          transposeRight:false
                                                                          resultRows:1
                                                                       resultColumns:1
                                                                   interiorColumns:1
                                                                             alpha:1.0
                                                                              beta:1.0];

    if (!mpsOp) {
        NSLog(@"Failed to initialize MPS operation.");
        return -1.0;
    }

    return 1.0;  // 这里只是示例，实际功耗测量需要结合 Profiling 工具
}


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
            [rowIndices addObject:@(row - 1)]; // Matrix Market 是 1-based
            [colIndices addObject:@(col - 1)];
            [values addObject:@(value)];
            if (row > *numRows) *numRows = row;
            if (col > *numCols) *numCols = col;
        }
    }

    fclose(file);
}


// **批量执行 SpMV 内核**
void executeKernels(id<MTLCommandQueue> commandQueue,
                    NSArray<id<MTLComputePipelineState>> *pipelines,
                    NSArray<NSArray<id<MTLBuffer>> *> *allBuffers,
                    int numRows,
                    int numCols,
                    id<MTLBuffer>numColsBuffer,
                    id<MTLBuffer>ellWidthBuffer,
                    double executionTimes[4]) {
    
    //在这里添加两处新的代码，可以删除
    for (int i = 0; i < pipelines.count; i++) {
        // **为每个 Kernel 任务创建独立的 CommandBuffer**
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // **绑定计算管线**
        [encoder setComputePipelineState:pipelines[i]];

        // **绑定 Buffer**
        NSArray<id<MTLBuffer>> *buffers = allBuffers[i];
        for (int j = 0; j < buffers.count; j++) {
            [encoder setBuffer:buffers[j] offset:0 atIndex:j];
        }

        // **额外绑定 CSC 和 HYB 相关的 Buffer**
        if (i == 2) { // CSC
            [encoder setBuffer:numColsBuffer offset:0 atIndex:CSC_NUMCOLS_IDX];
        } else if (i == 3) { // HYB
            [encoder setBuffer:ellWidthBuffer offset:0 atIndex:HYB_ELLWIDTH_IDX];
        }

        // **优化线程调度**
        NSUInteger threadGroupSizeX = MIN(1024, pipelines[i].maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(numRows, 1, 1);  // 2D 计算，提高 GPU 并行性
        MTLSize threadGroupSize3D = MTLSizeMake(threadGroupSizeX, 1, 1);
        MTLSize threadGroups = MTLSizeMake((numRows + threadGroupSizeX - 1)/threadGroupSizeX,1,1);

        // **记录 GPU 执行时间**
        CFTimeInterval start = CACurrentMediaTime();
        //[encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize3D];
        [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize3D];
        CFTimeInterval end = CACurrentMediaTime();
        executionTimes[i] = end - start;

        // **提交任务**
        [encoder endEncoding];
        
        //3.这是第三个
        [commandBuffer encodeSignalEvent:sharedEvent value:i+1];
        [commandBuffer commit];  // ✅ 立即提交，避免阻塞

        // ❌ 不再使用 `waitUntilCompleted`，避免 CPU 被 GPU 阻塞
        // [commandBuffer waitUntilCompleted];
    }
}

int main() {
    @autoreleasepool {
        // **初始化 Metal**
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        //1.新添加的办法1
        //添加MTLSharedEvent进行CPU-GPU精确同步
        id<MTLSharedEvent> sharedEvent = [device newSharedEvent];
        


        // **加载数据集**
        NSMutableArray<NSNumber*> *values = [NSMutableArray array];
        NSMutableArray<NSNumber*> *rowIndices = [NSMutableArray array];
        NSMutableArray<NSNumber*> *colIndices = [NSMutableArray array];

        int numRows = 0, numCols = 0;
      // const char* filePath = "/Users/zhangmulin/Downloads/12month1/12month1.mtx";
      // const char* filePath = "/Users/zhangmulin/Downloads/hugebubbles-00020/hugebubbles-00020.mtx";
        //const char* filePath = "/Users/zhangmulin/Downloads/mawi_201512020000 2/mawi_201512020000.mtx";
        const char* filePath = "/Users/zhangmulin/Downloads/kmer_V2a/kmer_V2a.mtx";
        loadMatrixMarket(filePath, values, rowIndices, colIndices, &numRows, &numCols);

        int nnz = (int)values.count;
        float* c_values = (float*)malloc(nnz * sizeof(float));
        int* c_rowIndices = (int*)malloc(nnz * sizeof(int));
        int* c_colIndices = (int*)malloc(nnz * sizeof(int));

        for (int i = 0; i < nnz; i++) {
            c_values[i] = [values[i] floatValue];
            c_rowIndices[i] = [rowIndices[i] intValue];
            c_colIndices[i] = [colIndices[i] intValue];
        }
        
        //在这里新添加的代码是 计算MTLHeap需要的大小
        // **计算 MTLHeap 需要的大小**
        size_t totalRequiredMemory =
            (nnz * sizeof(float)) + (nnz * sizeof(int)) + ((numRows + 1) * sizeof(int)) +
            (nnz * sizeof(float)) + (nnz * sizeof(int)) + (nnz * sizeof(int)) +
            (nnz * sizeof(float)) + (nnz * sizeof(int)) + ((numCols + 1) * sizeof(int)) +
            ((numRows * 32) * sizeof(float)) + ((numRows * 32) * sizeof(int));
        
        size_t heapSize = totalRequiredMemory * 1.5;  // 额外加 20% 预留空间

        if (heapSize > 512 * 1024 * 1024) {  // 限制单个 Heap 大小 ≤ 256MB
            heapSize = 512 * 1024 * 1024;
        }

        // **创建 Metal Heap**
        MTLHeapDescriptor *heapDescriptor = [[MTLHeapDescriptor alloc] init];
        heapDescriptor.size = heapSize;
        heapDescriptor.storageMode = MTLStorageModePrivate;
        id<MTLHeap> heap = [device newHeapWithDescriptor:heapDescriptor];
        
        

        // **转换为 CSR 格式**
        int* c_rowPointers = (int*)malloc((numRows + 1) * sizeof(int));
        memset(c_rowPointers, 0, (numRows + 1) * sizeof(int));

        for (int i = 0; i < nnz; i++) {
            c_rowPointers[c_rowIndices[i] + 1]++;
        }

        for (int i = 1; i <= numRows; i++) {
            c_rowPointers[i] += c_rowPointers[i - 1];
        }

        // **转换为 CSC 格式**
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

        // **转换为 COO 格式**
        int* c_cooRowIndices = (int*)malloc(nnz * sizeof(int));
        int* c_cooColumnIndices = (int*)malloc(nnz * sizeof(int));
        float* c_cooValues = (float*)malloc(nnz * sizeof(float));

        for (int i = 0; i < nnz; i++) {
            c_cooRowIndices[i] = c_rowIndices[i];
            c_cooColumnIndices[i] = c_colIndices[i];
            c_cooValues[i] = c_values[i];
        }

        // **转换为 HYB 格式（ELL + CSR 结合）**
        int ellWidth = 0;
        for (int row = 0; row < numRows; row++) {
            int rowStart = c_rowPointers[row];
            int rowEnd = c_rowPointers[row + 1];
            ellWidth = (rowEnd - rowStart) > ellWidth ? (rowEnd - rowStart) : ellWidth;
        }

        float* c_ellValues = (float*)malloc(numRows * ellWidth * sizeof(float));
        int* c_ellColumnIndices = (int*)malloc(numRows * ellWidth * sizeof(int));
        memset(c_ellValues, 0, numRows * ellWidth * sizeof(float));
        memset(c_ellColumnIndices, -1, numRows * ellWidth * sizeof(int)); // -1 代表无效索引

        for (int row = 0; row < numRows; row++) {
            int rowStart = c_rowPointers[row];
            int rowEnd = c_rowPointers[row + 1];
            for (int i = rowStart, col = 0; i < rowEnd; i++, col++) {
                c_ellValues[row * ellWidth + col] = c_values[i];
                c_ellColumnIndices[row * ellWidth + col] = c_colIndices[i];
            }
        }
        //这里是新加的，分配xhey向量
        // **分配 x 和 y 向量**
        float* x = (float*)malloc(numCols * sizeof(float));
        float* y = (float*)malloc(numRows * sizeof(float));

        // 初始化 x 为全 1，y 设为 0
        for (int i = 0; i < numCols; i++) x[i] = 1.0f;
        memset(y, 0, numRows * sizeof(float));

        
        
        //下面这个代码新加，也是必要时可以删
        // **CSR 格式 Buffers**
        id<MTLBuffer> csrValuesBuffer = [heap newBufferWithLength:nnz * sizeof(float) options:MTLResourceStorageModePrivate];
        if (!csrValuesBuffer) {
            NSLog(@"csrValuesBuffer 分配失败，回退到 Shared 模式");
            csrValuesBuffer = [device newBufferWithLength:nnz * sizeof(float) options:MTLResourceStorageModeShared];
        }
     
        id<MTLBuffer> csrColumnIndicesBuffer = [heap newBufferWithLength:nnz * sizeof(int) options:MTLResourceStorageModePrivate];
        if (!csrColumnIndicesBuffer) csrColumnIndicesBuffer = [device newBufferWithLength:nnz * sizeof(int) options:MTLResourceStorageModeShared];
        
        id<MTLBuffer>csrRowPointersBuffer = [heap newBufferWithLength:(numRows+1)*sizeof(int)options:MTLResourceStorageModePrivate];
        if (!csrRowPointersBuffer) csrRowPointersBuffer = [device newBufferWithLength:(numRows + 1) * sizeof(int) options:MTLResourceStorageModeShared];

        // **COO 格式 Buffers**
        id<MTLBuffer> cooValuesBuffer = [heap newBufferWithLength:nnz * sizeof(float) options:MTLResourceStorageModePrivate];
        if (!cooValuesBuffer) cooValuesBuffer = [device newBufferWithLength:nnz * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLBuffer> cooRowIndicesBuffer = [heap newBufferWithLength:nnz * sizeof(int) options:MTLResourceStorageModePrivate];
        if (!cooRowIndicesBuffer) cooRowIndicesBuffer = [device newBufferWithLength:nnz * sizeof(int) options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> cooColumnIndicesBuffer = [heap newBufferWithLength:nnz * sizeof(int) options:MTLResourceStorageModePrivate];
        if (!cooColumnIndicesBuffer) cooColumnIndicesBuffer = [device newBufferWithLength:nnz * sizeof(int) options:MTLResourceStorageModeShared];
        

        // **CSC 格式 Buffers**
        id<MTLBuffer> cscValuesBuffer = [heap newBufferWithLength:nnz * sizeof(float) options:MTLResourceStorageModePrivate];
        if (!cscValuesBuffer) cscValuesBuffer = [device newBufferWithLength:nnz * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cscRowIndicesBuffer = [heap newBufferWithLength:nnz * sizeof(int) options:MTLResourceStorageModePrivate];
        if (!cscRowIndicesBuffer) cscRowIndicesBuffer = [device newBufferWithLength:nnz * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cscColPointersBuffer = [heap newBufferWithLength:(numCols + 1) * sizeof(int) options:MTLResourceStorageModePrivate];
        if (!cscColPointersBuffer) cscColPointersBuffer = [device newBufferWithLength:(numCols + 1) * sizeof(int) options:MTLResourceStorageModeShared];

        // **HYB 格式 Buffers（ELL + CSR）**
        id<MTLBuffer> hybCsrValuesBuffer = csrValuesBuffer; // 直接复用 CSR 数据
        if(!hybCsrValuesBuffer)hybCsrValuesBuffer = [device newBufferWithLength:nnz*sizeof(float)options:MTLResourceStorageModePrivate];
        id<MTLBuffer> hybCsrColumnIndicesBuffer = csrColumnIndicesBuffer;
        if (!hybCsrColumnIndicesBuffer) hybCsrColumnIndicesBuffer = [device newBufferWithLength:nnz * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> hybCsrRowPointersBuffer = csrRowPointersBuffer;
        if (!hybCsrRowPointersBuffer) hybCsrRowPointersBuffer = [device newBufferWithLength:(numRows + 1) * sizeof(int) options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> hybEllValuesBuffer = [heap newBufferWithLength:(numRows * ellWidth) * sizeof(float) options:MTLResourceStorageModePrivate];
        if (!hybEllValuesBuffer) hybEllValuesBuffer = [device newBufferWithLength:(numRows * ellWidth) * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> hybEllColumnIndicesBuffer = [heap newBufferWithLength:(numRows * ellWidth) * sizeof(int) options:MTLResourceStorageModePrivate];
        if (!hybEllColumnIndicesBuffer) hybEllColumnIndicesBuffer = [device newBufferWithLength:(numRows * ellWidth) * sizeof(int) options:MTLResourceStorageModeShared];


        id<MTLBuffer> numRowsBuffer = [device newBufferWithBytes:&numRows length:sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> numColsBuffer = [device newBufferWithBytes:&numCols length:sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nnzBuffer = [device newBufferWithBytes:&nnz length:sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> xBuffer = [device newBufferWithBytes:x length:numCols * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> yBuffer = [device newBufferWithBytes:y length:numRows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>ellWidthBuffer = [device newBufferWithBytes:&ellWidth length:sizeof(int) options:MTLResourceStorageModeShared];
        

        // **加载 Metal 计算着色器**
        NSError *error = nil;
        NSString *libraryPath = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
        id<MTLLibrary> library = [device newLibraryWithFile:libraryPath error:&error];

        id<MTLComputePipelineState> csrPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"spmv_csr"] error:&error];
        id<MTLComputePipelineState> cscPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"spmv_csc"] error:&error];
        id<MTLComputePipelineState>cooPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"spmv_coo"] error:&error];
        id<MTLComputePipelineState>hybPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"spmv_hyb"] error:&error];

        NSArray<NSArray<id<MTLBuffer>> *> *allBuffers = @[
            @[csrValuesBuffer, csrColumnIndicesBuffer, csrRowPointersBuffer, xBuffer, yBuffer, numRowsBuffer],
            @[cooValuesBuffer, cooRowIndicesBuffer, cooColumnIndicesBuffer, xBuffer, yBuffer, nnzBuffer],
            @[cscValuesBuffer, cscRowIndicesBuffer, cscColPointersBuffer, xBuffer, yBuffer, numColsBuffer],
            @[hybCsrValuesBuffer, hybCsrColumnIndicesBuffer, hybCsrRowPointersBuffer, hybEllValuesBuffer, hybEllColumnIndicesBuffer, xBuffer, yBuffer, numColsBuffer, ellWidthBuffer]
        ];

        NSArray<id<MTLComputePipelineState>> *pipelines = @[csrPipeline, cscPipeline,cooPipeline,hybPipeline];

        // **执行 SpMV 计算**
        double executionTimes[4];
        executeKernels(commandQueue, pipelines, allBuffers, numRows,numCols,numColsBuffer, ellWidthBuffer, executionTimes);
        
        
        //2.这是第二个添加的办法完成GPU计算完成后，CPU等待同步
        [sharedEvent waitUntilCompleted];

        NSLog(@"CSR Execution Time: %f seconds", executionTimes[0]);
        NSLog(@"COO Execution Time: %f seconds", executionTimes[1]);
        NSLog(@"CSC Execution Time: %f seconds", executionTimes[2]);
        NSLog(@"HYB Execution Time: %f seconds", executionTimes[3]);

        return 0;
    }
}

