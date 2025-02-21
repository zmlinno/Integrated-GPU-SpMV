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
    MTLSize gridSize = MTLSizeMake(numRows, 1, 1);
    NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
    MTLSize threadGroupSize3D = MTLSizeMake(threadGroupSize, 1, 1);
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
};

int main() {
    @autoreleasepool {
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
        const char* filePath = "/Users/zhangmulin/Downloads/hugebubbles-00020/hugebubbles-00020.mtx"; // 替换为实际路径
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
        id<MTLBuffer> csrValuesBuffer = [device newBufferWithBytes:c_values length:nnz * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> csrColumnIndicesBuffer = [device newBufferWithBytes:c_colIndices length:nnz * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> csrRowPointersBuffer = [device newBufferWithBytes:c_rowPointers length:(numRows + 1) * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cooRowIndicesBuffer = [device newBufferWithBytes:c_rowIndices length:nnz * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cscValuesBuffer = [device newBufferWithBytes:c_cscValues length:nnz * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cscRowIndicesBuffer = [device newBufferWithBytes:c_cscRowIndices length:nnz * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cscColPointersBuffer = [device newBufferWithBytes:c_cscColPointers length:(numCols + 1) * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> hybEllValuesBuffer = [device newBufferWithBytes:ellValues length:(numRows * ellWidth) * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> hybEllColumnIndicesBuffer = [device newBufferWithBytes:ellColumnIndices length:(numRows * ellWidth) * sizeof(int) options:MTLResourceStorageModeShared];

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

























/// 禁用 VSync 以减少 CPU-GPU 延迟
//void disableVSync() {
//    CVDisplayLinkRef displayLink;
//    CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
//    CVDisplayLinkStop(displayLink);
//    CVDisplayLinkRelease(displayLink);
//}
//
//// 读取 Matrix Market 数据
//void loadMatrixMarket(const char* filePath, std::vector<float>& values,
//                      std::vector<int>& rowIndices, std::vector<int>& colIndices,
//                      int& numRows, int& numCols) {
//    FILE* file = fopen(filePath, "r");
//    if (!file) {
//        perror("Error opening file");
//        return;
//    }
//
//    char line[1024];
//    numRows = 0;
//    numCols = 0;
//
//    while (fgets(line, sizeof(line), file)) {
//        if (line[0] == '%') continue; // 忽略注释行
//
//        int row, col;
//        float value;
//        if (sscanf(line, "%d %d %f", &row, &col, &value) == 3) {
//            rowIndices.push_back(row - 1);
//            colIndices.push_back(col - 1);
//            values.push_back(value);
//
//            if (row > numRows) numRows = row;
//            if (col > numCols) numCols = col;
//        }
//    }
//    fclose(file);
//}
//// **优化的 Kernel 执行**
//auto executeKernelWithTiming = ^(NSString* kernelName,
//                                 NSArray<id<MTLBuffer>> *buffers,
//                                 id<MTLCommandQueue> commandQueue,
//                                 int numRows) {
//    //id<MTLFunction> function = [commandQueue.device newDefaultLibrary].newFunctionWithName(kernelName);
//    id<MTLLibrary>library = [commandQueue.device newDefaultLibrary];
//    id<MTLFunction>function = [library newFunctionWithName:kernelName];
//    NSError *error = nil;
//    id<MTLComputePipelineState> pipeline = [commandQueue.device newComputePipelineStateWithFunction:function error:&error];
//    
//    if (!pipeline) {
//        NSLog(@"[ERROR] Compute pipeline not found: %@", kernelName);
//        return;
//    }
//
//    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
//    [encoder setComputePipelineState:pipeline];
//
//    for (int i = 0; i < buffers.count; i++) {
//        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
//    }
//
//    // ✅ 让 threadgroupSize 自适应矩阵大小
//    NSUInteger threadsPerThreadgroup = pipeline.maxTotalThreadsPerThreadgroup / 4; // 避免超载
//    MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
//    MTLSize threadgroups = MTLSizeMake((numRows + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1);
//    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroupSize];
//
//    [encoder endEncoding];
//    [commandBuffer commit];
//    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
//        NSLog(@"%@ Execution Completed", kernelName);
//    }];
//};
//
//
//
//
//
//
//int main() {
//    @autoreleasepool {
//        // ✅ 读取矩阵数据
//        const char* filePath = "/Users/zhangmulin/Downloads/FastLoad-main/Data/1138_bus.mtx";
//        std::vector<float> values;
//        std::vector<int> rowIndices;
//        std::vector<int> colIndices;
//        int numRows, numCols;
//
//        loadMatrixMarket(filePath, values, rowIndices, colIndices, numRows, numCols);
//        int nnz = static_cast<int>(values.size());  // 防止数据精度丢失
//
//        // ✅ Metal 设备初始化
//        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
//        if (!device) {
//            std::cerr << "Metal 设备不可用！" << std::endl;
//            return -1;
//        }
//
//        // ✅ 计算 Heap 需要的大小
//        size_t totalRequiredSize = (nnz * sizeof(float)) * 4 + (numRows + 1) * sizeof(int) + (numCols + 1) * sizeof(int);
//        size_t heapMaxSize = 1024 * 1024 * 512;
//        size_t heapSize = std::min(heapMaxSize, totalRequiredSize);
//
//        // ✅ Metal Heap 申请
//        MTLHeapDescriptor* heapDescriptor = [[MTLHeapDescriptor alloc] init];
//        heapDescriptor.size = 1024 * 1024 * 512;
//        heapDescriptor.storageMode = MTLStorageModeShared;
//       // heapDescriptor.size = heapSize;
//        id<MTLHeap> heap = [device newHeapWithDescriptor:heapDescriptor];
//
//        bool useHeap = (heap != nil);
//        if (!useHeap) {
//            std::cout << "⚠️ Heap 申请失败，使用普通 MTLBuffer" << std::endl;
//        }
//
//        // ✅ CSR 格式转换
//        id<MTLBuffer> csrValuesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(nnz * sizeof(float))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(nnz * sizeof(float))
//                                options:MTLResourceStorageModeShared];
//
//        id<MTLBuffer> csrRowPointersBuffer = useHeap && heap ?
//            [heap newBufferWithLength:((numRows + 1) * sizeof(int))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:((numRows + 1) * sizeof(int))
//                                options:MTLResourceStorageModeShared];
//
//        id<MTLBuffer> csrColumnIndicesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(nnz * sizeof(int))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(nnz * sizeof(int))
//                                options:MTLResourceStorageModeShared];
//
//        // ✅ CSC 格式转换
//        id<MTLBuffer> cscValuesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(nnz * sizeof(float))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(nnz * sizeof(float))
//                                options:MTLResourceStorageModeShared];
//
//        id<MTLBuffer> cscColPointersBuffer = useHeap && heap ?
//            [heap newBufferWithLength:((numCols + 1) * sizeof(int))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:((numCols + 1) * sizeof(int))
//                                options:MTLResourceStorageModeShared];
//
//        id<MTLBuffer> cscRowIndicesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(nnz * sizeof(int))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(nnz * sizeof(int))
//                                options:MTLResourceStorageModeShared];
//
//        // ✅ COO 格式转换
//        id<MTLBuffer> cooValuesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(nnz * sizeof(float))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(nnz * sizeof(float))
//                                options:MTLResourceStorageModeShared];
//
//        id<MTLBuffer> cooRowIndicesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(nnz * sizeof(int))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(nnz * sizeof(int))
//                                options:MTLResourceStorageModeShared];
//
//        id<MTLBuffer> cooColIndicesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(nnz * sizeof(int))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(nnz * sizeof(int))
//                                options:MTLResourceStorageModeShared];
//
//        // ✅ HYB 格式转换
//        int ellWidth = 0;
//        for (int row = 0; row < numRows; row++) {
//            int rowStart = rowIndices[row];
//            int rowEnd = rowIndices[row + 1];
//            ellWidth = std::max(ellWidth, rowEnd - rowStart);
//        }
//        ellWidth = std::min(ellWidth, 128); // 限制最大宽度
//
//        // ELL 部分
//        id<MTLBuffer> hybEllValuesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(numRows * ellWidth * sizeof(float))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(numRows * ellWidth * sizeof(float))
//                                options:MTLResourceStorageModeShared];
//
//        id<MTLBuffer> hybEllColIndicesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(numRows * ellWidth * sizeof(int))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(numRows * ellWidth * sizeof(int))
//                                options:MTLResourceStorageModeShared];
//
//        // CSR 部分
//        id<MTLBuffer> hybCsrValuesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(nnz * sizeof(float))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(nnz * sizeof(float))
//                                options:MTLResourceStorageModeShared];
//
//        id<MTLBuffer> hybCsrColumnIndicesBuffer = useHeap && heap ?
//            [heap newBufferWithLength:(nnz * sizeof(int))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:(nnz * sizeof(int))
//                                options:MTLResourceStorageModeShared];
//
//        id<MTLBuffer> hybCsrRowPointersBuffer = useHeap && heap ?
//            [heap newBufferWithLength:((numRows + 1) * sizeof(int))
//                              options:MTLResourceStorageModeShared] :
//            [device newBufferWithLength:((numRows + 1) * sizeof(int))
//                                options:MTLResourceStorageModeShared];
//
//        // ✅ Metal 计算管线
//        NSError* error = nil;
//        id<MTLLibrary> library = [device newDefaultLibrary];
//        id<MTLFunction> function = [library newFunctionWithName:@"spmv_csr"];
//        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
//
//        if (!pipeline) {
//            std::cerr << "⚠️ 计算管线创建失败：" << error.localizedDescription.UTF8String << std::endl;
//            return -1;
//        }
//
//        // ✅ Metal Command Queue & Command Buffer
//        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//
//        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
//        [encoder setComputePipelineState:pipeline];
//
//        // 绑定 Metal Buffer
//        [computeEncoder setBuffer:csrValuesBuffer offset:0 atIndex:0];
//        [computeEncoder setBuffer:csrColumnIndicesBuffer offset:0 atIndex:1];
//        [computeEncoder setBuffer:csrRowPointersBuffer offset:0 atIndex:2];
//        [computeEncoder setBuffer:xBuffer offset:0 atIndex:3]; // 🚀 确保这里绑定了 x
//        [computeEncoder setBuffer:yBuffer offset:0 atIndex:4];
//
//        // 启动 Kernel
//        MTLSize gridSize = MTLSizeMake(numRows, 1, 1);
//        MTLSize threadGroupSize = MTLSizeMake(128, 1, 1);
//        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
//
//        [encoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//
//        return 0;
//    }
//}
