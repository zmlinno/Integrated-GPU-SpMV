//
//  main.swift
//  SpMV_Metal
//
//  Created by 张木林 on 4/8/25.
//

import Foundation
import Metal
import MetalKit

struct CSRMatrix: Codable {
    let format: String
    let row_ptr: [UInt32]
    let col_ind: [UInt32]
    let data: [Float]
    let num_rows: Int
    let num_cols: Int
    let nnz: Int
}


struct COOMatrix:Codable
{
    let format:String
    let row:[UInt32]
    let col:[UInt32]
    let data:[Float]
    let num_rows:Int
    let num_cols:Int
    let nnz:Int
}


struct ELLMatrix:Codable
{
    let format:String
    let indices:[UInt32]
    let values:[Float]
    let num_rows:Int
    let num_cols:Int
    let num_cols_per_row:Int
}



struct HYBMatrix: Codable {
    let format: String
    let num_rows: Int
    let num_cols: Int
    let ell: ELL
    let coo_tail: COOTail

    enum CodingKeys: String, CodingKey {
        case format
        case num_rows
        case num_cols
        case ell = "ELL"
        case coo_tail = "COO_tail"
    }

    struct ELL: Codable {
        let indices: [UInt32]
        let values: [Float]
        let num_cols_per_row: Int
    }

    struct COOTail: Codable {
        let row: [UInt32]
        let col: [UInt32]
        let data: [Float]
    }
}



struct DIAMatrix: Codable {
    let format: String
    let num_rows: Int
    let num_cols: Int
    let num_diagonals: Int
    let values: [Float]
    let offsets: [Int]
}




struct CSCMatrix: Codable {
    let format: String
    let num_rows: Int
    let num_cols: Int
    let col_ptrs: [UInt32]
    let row_indices: [UInt32]
    let values: [Float]

    enum CodingKeys: String, CodingKey {
        case format
        case num_rows
        case num_cols
        case col_ptrs = "col_ptr"
        case row_indices = "row_idx"
        case values = "data"
    }
}



func loadCSRMatrix(from filePath: String) -> CSRMatrix? {
    let url = URL(fileURLWithPath: filePath)
    guard let data = try? Data(contentsOf: url) else {
        print("❌ 无法读取 JSON 文件")
        return nil
    }

    let decoder = JSONDecoder()
    return try? decoder.decode(CSRMatrix.self, from: data)
}

func runCSRSpMV(matrix: CSRMatrix) {
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue(),
          let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "csr_spmv") else {
        print("❌ Metal 初始化失败")
        return
    }

    let pipeline = try! device.makeComputePipelineState(function: function)

    // 输入向量 x，长度 = num_cols
    let x: [Float] = Array(repeating: 1.0, count: matrix.num_cols)
    var y: [Float] = Array(repeating: 0.0, count: matrix.num_rows)

    // 创建 Metal Buffers
    let rowPtrBuffer = device.makeBuffer(bytes: matrix.row_ptr, length: MemoryLayout<UInt32>.size * matrix.row_ptr.count, options: [])
    let colIndBuffer = device.makeBuffer(bytes: matrix.col_ind, length: MemoryLayout<UInt32>.size * matrix.col_ind.count, options: [])
    let valBuffer = device.makeBuffer(bytes: matrix.data, length: MemoryLayout<Float>.size * matrix.data.count, options: [])
    let xBuffer = device.makeBuffer(bytes: x, length: MemoryLayout<Float>.size * x.count, options: [])
    let yBuffer = device.makeBuffer(bytes: y, length: MemoryLayout<Float>.size * y.count, options: [])

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let computeEncoder = commandBuffer.makeComputeCommandEncoder()!

    computeEncoder.setComputePipelineState(pipeline)
    computeEncoder.setBuffer(rowPtrBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(colIndBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(valBuffer, offset: 0, index: 2)
    computeEncoder.setBuffer(xBuffer, offset: 0, index: 3)
    computeEncoder.setBuffer(yBuffer, offset: 0, index: 4)

    // Launch 1 thread per row
    let threadCount = MTLSize(width: matrix.num_rows, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, matrix.num_rows), height: 1, depth: 1)

    computeEncoder.dispatchThreads(threadCount, threadsPerThreadgroup: threadGroupSize)

    computeEncoder.endEncoding()

    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let endTime = CFAbsoluteTimeGetCurrent()

    // 拷贝结果
    let yResult = yBuffer?.contents().bindMemory(to: Float.self, capacity: y.count)
    let output = UnsafeBufferPointer(start: yResult, count: y.count)

    print("✅ 执行完毕，耗时: \((endTime - startTime) * 1000) ms")
    print("📊 输出向量前10项：")
    for i in 0..<min(10, output.count) {
        print("y[\(i)] = \(output[i])")
    }
}




func loadCOOMatrix(from filePath:String)->COOMatrix?
{
    let url = URL(fileURLWithPath:filePath)
    guard let data = try?Data(contentsOf: url)else{return nil}
    return try?JSONDecoder().decode(COOMatrix.self,from:data)
}
func runCOOSpMV(matrix: COOMatrix) {
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue(),
          let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "coo_spmv") else {
        print("❌ Metal 初始化失败")
        return
    }

    let pipeline = try! device.makeComputePipelineState(function: function)

    let x: [Float] = Array(repeating: 1.0, count: matrix.num_cols)
    var y: [Float] = Array(repeating: 0.0, count: matrix.num_rows)

    let rowBuffer = device.makeBuffer(bytes: matrix.row, length: MemoryLayout<UInt32>.size * matrix.row.count, options: [])
    let colBuffer = device.makeBuffer(bytes: matrix.col, length: MemoryLayout<UInt32>.size * matrix.col.count, options: [])
    let valBuffer = device.makeBuffer(bytes: matrix.data, length: MemoryLayout<Float>.size * matrix.data.count, options: [])
    let xBuffer = device.makeBuffer(bytes: x, length: MemoryLayout<Float>.size * x.count, options: [])
    let yBuffer = device.makeBuffer(bytes: y, length: MemoryLayout<Float>.size * y.count, options: [])

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(rowBuffer, offset: 0, index: 0)
    encoder.setBuffer(colBuffer, offset: 0, index: 1)
    encoder.setBuffer(valBuffer, offset: 0, index: 2)
    encoder.setBuffer(xBuffer, offset: 0, index: 3)
    encoder.setBuffer(yBuffer, offset: 0, index: 4)
    encoder.setBytes([UInt32(matrix.nnz)], length: MemoryLayout<UInt32>.size, index: 5)

    let threadCount = MTLSize(width: matrix.nnz, height: 1, depth: 1)
    let threadsPerGroup = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(threadCount, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()

    let startTime = CFAbsoluteTimeGetCurrent()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let endTime = CFAbsoluteTimeGetCurrent()

    let yResult = yBuffer?.contents().bindMemory(to: Float.self, capacity: y.count)
    let output = UnsafeBufferPointer(start: yResult, count: y.count)

    print("✅ COO 执行完毕，耗时: \((endTime - startTime) * 1000) ms")
    print("📊 输出向量前10项：")
    for i in 0..<min(10, output.count) {
        print("y[\(i)] = \(output[i])")
    }
}





//ELL的JSON加载函数
func loadELLMatrix(from filePath: String) -> ELLMatrix? {
    let url = URL(fileURLWithPath: filePath)
    guard let data = try? Data(contentsOf: url) else { return nil }
    return try? JSONDecoder().decode(ELLMatrix.self, from: data)
}
func runELLSpMV(matrix: ELLMatrix) {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue(),
          let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "ell_spmv") else {
        print("❌ Metal 初始化失败")
        return
    }

    let pipeline = try! device.makeComputePipelineState(function: function)

    let x = [Float](repeating: 1.0, count: matrix.num_cols)
    var y = [Float](repeating: 0.0, count: matrix.num_rows)

    let indexBuffer = device.makeBuffer(bytes: matrix.indices, length: matrix.indices.count * MemoryLayout<UInt32>.size, options: [])
    let valBuffer = device.makeBuffer(bytes: matrix.values, length: matrix.values.count * MemoryLayout<Float>.size, options: [])
    let xBuffer = device.makeBuffer(bytes: x, length: x.count * MemoryLayout<Float>.size, options: [])
    let yBuffer = device.makeBuffer(bytes: y, length: y.count * MemoryLayout<Float>.size, options: [])

    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(indexBuffer, offset: 0, index: 0)
    encoder.setBuffer(valBuffer, offset: 0, index: 1)
    encoder.setBuffer(xBuffer, offset: 0, index: 2)
    encoder.setBuffer(yBuffer, offset: 0, index: 3)
    encoder.setBytes([UInt32(matrix.num_rows)], length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes([UInt32(matrix.num_cols_per_row)], length: MemoryLayout<UInt32>.size, index: 5)

    let threads = MTLSize(width: matrix.num_rows, height: 1, depth: 1)
    let threadgroup = MTLSize(width: 32, height: 1, depth: 1)

    encoder.dispatchThreads(threads, threadsPerThreadgroup: threadgroup)
    encoder.endEncoding()

    let start = CFAbsoluteTimeGetCurrent()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = CFAbsoluteTimeGetCurrent()

    let yResult = yBuffer?.contents().bindMemory(to: Float.self, capacity: y.count)
    let output = UnsafeBufferPointer(start: yResult, count: y.count)

    print("✅ ELL 执行完毕，耗时: \((end - start) * 1000) ms")
    for i in 0..<min(10, output.count) {
        print("y[\(i)] = \(output[i])")
    }
}






//HYB加载的JSON
func loadHYBMatrix(from filePath: String) -> HYBMatrix? {
    let url = URL(fileURLWithPath: filePath)
    guard let data = try? Data(contentsOf: url) else { return nil }
    return try? JSONDecoder().decode(HYBMatrix.self, from: data)
}
//HYB
func runHYBSpMV(matrix: HYBMatrix) {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue(),
          let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "hyb_spmv") else {
        print("❌ Metal 初始化失败")
        return
    }

    let pipeline = try! device.makeComputePipelineState(function: function)

    let x = [Float](repeating: 1.0, count: matrix.num_cols)
    var y = [Float](repeating: 0.0, count: matrix.num_rows)

    let ell_indices = matrix.ell.indices
    let ell_values = matrix.ell.values
    let coo_row = matrix.coo_tail.row
    let coo_col = matrix.coo_tail.col
    let coo_data = matrix.coo_tail.data

    let ell_index_buf = device.makeBuffer(bytes: ell_indices, length: ell_indices.count * 4, options: [])
    let ell_value_buf = device.makeBuffer(bytes: ell_values, length: ell_values.count * 4, options: [])
    let coo_row_buf = device.makeBuffer(bytes: coo_row, length: coo_row.count * 4, options: [])
    let coo_col_buf = device.makeBuffer(bytes: coo_col, length: coo_col.count * 4, options: [])
    let coo_val_buf = device.makeBuffer(bytes: coo_data, length: coo_data.count * 4, options: [])
    let x_buf = device.makeBuffer(bytes: x, length: x.count * 4, options: [])
    let y_buf = device.makeBuffer(bytes: y, length: y.count * 4, options: [])

    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)

//    encoder.setBuffer(ell_index_buf, offset: 0, index: 0)
//    encoder.setBuffer(ell_value_buf, offset: 0, index: 1)
//    encoder.setBuffer(coo_row_buf, offset: 0, index: 2)
//    encoder.setBuffer(coo_col_buf, offset: 0, index: 3)
//    encoder.setBuffer(coo_val_buf, offset: 0, index: 4)
//    encoder.setBuffer(x_buf, offset: 0, index: 5)
//    encoder.setBuffer(y_buf, offset: 0, index: 6)
//    encoder.setBytes([UInt32(matrix.num_rows)], length: 4, index: 7)
//    encoder.setBytes([UInt32(matrix.ell.num_cols_per_row)], length: 4, index: 8)
//    encoder.setBytes([UInt32(coo_row.count)], length: 4, index: 9)
    encoder.setBuffer(ell_index_buf, offset: 0, index: 0)
    encoder.setBuffer(ell_value_buf, offset: 0, index: 1)
    encoder.setBuffer(coo_row_buf, offset: 0, index: 2)
    encoder.setBuffer(coo_col_buf, offset: 0, index: 3)
    encoder.setBuffer(coo_val_buf, offset: 0, index: 4)
    encoder.setBuffer(x_buf, offset: 0, index: 5)
    encoder.setBuffer(y_buf, offset: 0, index: 6)
    encoder.setBytes([UInt32(matrix.num_rows)], length: 4, index: 7)
    encoder.setBytes([UInt32(matrix.ell.num_cols_per_row)], length: 4, index: 8)
    encoder.setBytes([UInt32(coo_row.count)], length: 4, index: 9)
    encoder.setBytes([UInt32(matrix.num_rows)], length: 4, index: 10)

    encoder.dispatchThreads(MTLSize(width: matrix.num_rows, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()

    let start = CFAbsoluteTimeGetCurrent()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = CFAbsoluteTimeGetCurrent()

    let y_result = y_buf?.contents().bindMemory(to: Float.self, capacity: y.count)
    let output = UnsafeBufferPointer(start: y_result, count: y.count)

    print("✅ HYB 执行完毕，耗时: \((end - start) * 1000) ms")
    for i in 0..<min(10, output.count) {
        print("y[\(i)] = \(output[i])")
    }
}





//关于DIA的
func loadDIAMatrix(from filePath: String) -> DIAMatrix? {
    let url = URL(fileURLWithPath: filePath)
    guard let data = try? Data(contentsOf: url) else {
        print("❌ 无法读取 JSON 文件")
        return nil
    }

    let decoder = JSONDecoder()
    return try? decoder.decode(DIAMatrix.self, from: data)
}
func runDIASpMV(matrix: DIAMatrix) {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue(),
          let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "dia_spmv") else {
        print("❌ Metal 初始化失败")
        return
    }

    let pipeline = try! device.makeComputePipelineState(function: function)

    let x = [Float](repeating: 1.0, count: matrix.num_cols)
    var y = [Float](repeating: 0.0, count: matrix.num_rows)

    let values_buf = device.makeBuffer(bytes: matrix.values, length: matrix.values.count * MemoryLayout<Float>.size, options: [])
    let offsets_buf = device.makeBuffer(bytes: matrix.offsets, length: matrix.offsets.count * MemoryLayout<Int>.size, options: [])
    let x_buf = device.makeBuffer(bytes: x, length: x.count * 4, options: [])
    let y_buf = device.makeBuffer(bytes: y, length: y.count * 4, options: [])

    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)

    encoder.setBuffer(values_buf, offset: 0, index: 0)
    encoder.setBuffer(offsets_buf, offset: 0, index: 1)
    encoder.setBuffer(x_buf, offset: 0, index: 2)
    encoder.setBuffer(y_buf, offset: 0, index: 3)
    encoder.setBytes([UInt32(matrix.num_rows)], length: 4, index: 4)
    encoder.setBytes([UInt32(matrix.num_cols)], length: 4, index: 5)
    encoder.setBytes([UInt32(matrix.num_diagonals)], length: 4, index: 6)

    let start = CFAbsoluteTimeGetCurrent()
    encoder.dispatchThreads(MTLSize(width: matrix.num_rows, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = CFAbsoluteTimeGetCurrent()

    let yResult = y_buf?.contents().bindMemory(to: Float.self, capacity: y.count)
    let output = UnsafeBufferPointer(start: yResult, count: y.count)

    print("✅ DIA 执行完毕，耗时: \((end - start) * 1000) ms")
    for i in 0..<min(10, output.count) {
        print("y[\(i)] = \(output[i])")
    }
}




func loadCSCMatrix(from filePath: String) -> CSCMatrix? {
    let url = URL(fileURLWithPath: filePath)
    guard let data = try? Data(contentsOf: url) else {
        print("❌ 无法读取 CSC JSON 文件")
        return nil
    }

    let decoder = JSONDecoder()
    do
    {
        return try decoder.decode(CSCMatrix.self,from:data)
    }
    catch
    {
        print("❌ CSC 解析失败: \(error.localizedDescription)")
        return nil
    }
}
func runCSCSpMV(matrix: CSCMatrix) {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue(),
          let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "csc_spmv") else {
        print("❌ Metal 初始化失败")
        return
    }

    let pipeline = try! device.makeComputePipelineState(function: function)

    let x = [Float](repeating: 1.0, count: matrix.num_cols)
    var y = [Float](repeating: 0.0, count: matrix.num_rows)

    let col_ptrs = matrix.col_ptrs
    let row_indices = matrix.row_indices
    let values = matrix.values

    let col_ptrs_buf = device.makeBuffer(bytes: col_ptrs, length: col_ptrs.count * 4, options: [])
    let row_indices_buf = device.makeBuffer(bytes: row_indices, length: row_indices.count * 4, options: [])
    let values_buf = device.makeBuffer(bytes: values, length: values.count * 4, options: [])
    let x_buf = device.makeBuffer(bytes: x, length: x.count * 4, options: [])
    let y_buf = device.makeBuffer(bytes: y, length: y.count * 4, options: [])

    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)

    encoder.setBuffer(col_ptrs_buf, offset: 0, index: 0)
    encoder.setBuffer(row_indices_buf, offset: 0, index: 1)
    encoder.setBuffer(values_buf, offset: 0, index: 2)
    encoder.setBuffer(x_buf, offset: 0, index: 3)
    encoder.setBuffer(y_buf, offset: 0, index: 4)
    encoder.setBytes([UInt32(matrix.num_cols)], length: 4, index: 5)

    let start = CFAbsoluteTimeGetCurrent()

    encoder.dispatchThreads(
        MTLSize(width: matrix.num_cols, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )

    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = CFAbsoluteTimeGetCurrent()

    let yResult = y_buf?.contents().bindMemory(to: Float.self, capacity: y.count)
    let output = UnsafeBufferPointer(start: yResult, count: y.count)

    print("✅ CSC 执行完毕，耗时: \((end - start) * 1000) ms")
    for i in 0..<min(10, output.count) {
        print("y[\(i)] = \(output[i])")
    }
}



let path = "/Users/zhangmulin/Desktop/CS/Paper/SpMV/Script/tols4000/tols4000.mtx/tols4000_ell.json"

//
//if FileManager.default.fileExists(atPath: path) {
//    print("✅ 路径存在")
//} else {
//    print("❌ 路径不存在")
//}
//
//do {
//    let data = try Data(contentsOf: url)
//    print("✅ 成功读取 JSON，大小：\(data.count) bytes")
//
//    let decoder = JSONDecoder()
//    let matrix = try decoder.decode(HYBMatrix.self, from: data)
//
//    print("✅ 成功解码 JSON，准备执行 SpMV")
//    runHYBSpMV(matrix: matrix)
//
//} catch {
//    print("❌ 解码或读取失败：\(error.localizedDescription)")
//}

//
//if let matrix = loadCSRMatrix(from: path) {
//    runCSRSpMV(matrix: matrix)
//} else {
//    print("❌ 解析 JSON 失败")
//}


//if let matrix = loadCOOMatrix(from: path)
//{
//    runCOOSpMV(matrix: matrix) //这里用的COO的kernel
//}
//else
//{
//    print("解析失败")
//}


//这是ELL
if let matrix = loadELLMatrix(from: path)
{
    runELLSpMV(matrix: matrix)
}



//
//let url = URL(fileURLWithPath: path)
//do {
//   let data = try Data(contentsOf: url)
//   print("✅ 成功读取 JSON, 大小: \(data.count) bytes")
//
//   let decoder = JSONDecoder()
//   let matrix = try decoder.decode(HYBMatrix.self, from: data)
//
//   print("✅ 成功解码 JSON, 准备执行 SpMV")
//   runHYBSpMV(matrix: matrix)
//} catch {
//   print("❌ 解码或读取失败: \(error.localizedDescription)")
//}






//这是关于DIA的
//let diaPath = "/Users/zhangmulin/Desktop/CS/Paper/SpMV/Script/tols4000/tols4000.mtx/tols4000_dia.json"
//if let matrix = loadDIAMatrix(from: diaPath)
//{
//    runDIASpMV(matrix: matrix)
//}




//这是关于CSC的
//let cscPath = "/Users/zhangmulin/Desktop/CS/Paper/SpMV/Script/tols4000/tols4000.mtx/tols4000_csc.json"
//
//if let matrix = loadCSCMatrix(from: cscPath) {
//    runCSCSpMV(matrix: matrix)
//}
//else
//{
//    print("CSC解析失败")
//}
