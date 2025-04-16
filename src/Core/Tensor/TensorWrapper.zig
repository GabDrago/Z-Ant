//! This file contains the definition of the types of tensors that can be used in the library.
//! The tensorWrapper class is a wrapper around the tensor class so that it can be manipulated in the same way regardless of the true nature of the tensor itself.
//! - Tensor     -> used when weights ridden from the onnx file are not pre-processed in any way, regardless of the type of the weights.
//! - QuantTensor   -> used when weights ridden from the onnx file are quantized.
//! - ClustTensor   -> used when weights ridden from the onnx file are clustered.

const std = @import("std");
const zant = @import("../../zant.zig");

pub var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;

pub fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_function = func;
}

pub const TensorType = enum {
    Tensor,
    QuantTensor,
    ClusterTensor,
    null,
};

//------------------------------------------------------------------------------------------------------
/// TENSOR WRAPPER
///
/// TensorWrapper() is the superclass for all the possible implementation of a tensor (Tensor, QuantTensor, ClustTensor).
///
/// @param T:comptime type of the values in the tensor
pub fn TensorWrapper(comptime tensor: anytype, comptime dataType: type) type {

    return struct {
        //const Self: type = @ptrCast(@alignCast(ptr)); // puntatore a tensore
        // const Self = @This();
        
        // tensor_type: TensorType,    // Type of the tensor

        // Interface fields
        const tensorType = @TypeOf(tensor);                 // Type of the actual tensor implementation
        const tensorTypeInfo = @typeInfo(tensorType);       // Type info of the actual tensor implementation
        ptr: *anyopaque,                                    // Pointer to the actual tensor implementation

        // Interface functions
        initFn: *const fn (ctx: *anyopaque, allocator: *const std.mem.Allocator) !tensorType,
        deinitFn: *const fn (ctx: *anyopaque) void,
        fromArrayFn: *const fn (ctx: *anyopaque, allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) anyerror!tensorType,
        toArrayFn: *const fn (ctx: *anyopaque, comptime dimension: usize) !MagicalReturnType(dataType, dimension),
        copyFn: *const fn (ctx: *anyopaque) !tensorType(dataType),
        fromShapeFn: *const fn (allocator: *const std.mem.Allocator, shape: []usize) !tensorType,
        fromConstBufferFn: *const fn (allocator: *const std.mem.Allocator, shape: []usize) !tensorType,
        fillFn: *const fn (ctx: *anyopaque, inputArray: anytype, shape: []usize) !void,

        // GRANDE MURAGLIA CINESE

        infoMetalFn: *const fn (ctx: *anyopaque) void,
        ensure4DShapeFn: *const fn (shape: []const usize) ![]usize,
        sliceONNXFn: *const fn (ctx: *anyopaque, starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !tensorType,


        /// Function used to create a new tensor wrapper for the provided tensor.
        fn createWrapper(ptr: anytype) TensorWrapper(tensor, dataType) {
            // const T = @TypeOf(ptr);
            // const ptr_info = @typeInfo(T);

            const gen = struct {
                pub fn init(ctx: *anyopaque, allocator: *const std.mem.Allocator) !tensorType {
                    const self: tensorType = @ptrCast(@alignCast(ctx));
                    return ptr_info.Pointer.child.init(self, allocator);
                }

                pub fn deinit(ctx: *anyopaque) void {
                    const self: tensorType = @ptrCast(@alignCast(ctx));
                    return ptr_info.Pointer.child.deinit(self);
                }

                pub fn fromArray(ctx: *anyopaque, allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !tensorType {
                    const self: tensorType = @ptrCast(@alignCast(ctx));
                    return ptr_info.Pointer.child.fromArray(allocator, inputArray, shape);
                }

                pub fn toArray(ctx: *anyopaque, comptime dimension: usize) !MagicalReturnType(dataType, dimension) {
                    const self: tensorType = @ptrCast(@alignCast(ctx));
                    return ptr_info.Pointer.child.toArray(self, dimension);
                }
                
                pub fn copy(ctx: *anyopaque) !tensorType(dataType) {
                    const self: tensorType = @ptrCast(@alignCast(ctx));
                    return ptr_info.Pointer.child.copy(self);
                }
                
                pub fn fromShape(allocator: *const std.mem.Allocator, shape: []usize) !tensorType {
                    return ptr_info.Pointer.child.fromShape(allocator, shape);
                }

                pub fn fromConstBuffer(allocator: *const std.mem.Allocator, data: []const T, shape: []const usize) tensorType {
                    return ptr_info.Pointer.child.fromConstBuffer(allocator, shape);
                }

                pub fn fill(ctx: *anyopaque, inputArray: anytype, shape: []usize) !void {
                    const self: tensorType = @ptrCast(@alignCast(ctx));
                    return ptr_info.Pointer.child.fill(self, inputArray, shape);
                }

                // GRANDE BARRIERA CORALLINA

                pub fn infoMetal(ctx: *anyopaque) void {
                    const self: tensorType = @ptrCast(@alignCast(ctx));
                    return ptr_info.Pointer.child.infoMetal(self);
                }

                pub inline fn ensure4DShape(shape: []const usize) ![]usize {
                    return ptr_info.Pointer.child.ensure4DShape(shape);
                }

                pub fn sliceONNX(ctx: *anyopaque, starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !tensorType {
                    const self: tensorType = @ptrCast(@alignCast(ctx));
                    return ptr_info.Pointer.child.sliceONNX(self, starts, ends, axes, steps);
                }

            };

            return .{
                .ptr = ptr,
                .initFn = gen.init,
                .deinitFn = gen.deinit,
                .fromArrayFn = gen.fromArray,
                .toArrayFn = gen.toArray,
                .copyFn = gen.copy,
                .fromShapeFn = gen.fromShape,
                .fromConstBufferFn = gen.fromConstBuffer,
                .fillFn = gen.fill,

                // MURO DI TRUMP

                .infoMetalFn = gen.infoMetal,
                .ensure4DShapeFn = gen.ensure4DShape,
                .sliceONNXFn = gen.sliceONNX,
            };
        }

        pub fn init(self: @This(), allocator: *const std.mem.Allocator) !Tensor {
            return self.initFn(self.ptr, data);
        }

        pub fn deinit(self: @This()) void {
            return self.deinitFn(self.ptr);
        }

        pub fn fromArray(self: @This(), allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !@This() {
            return self.fromArrayFn(self.ptr, allocator, inputArray, shape);
        }

        pub fn toArray(self: @This(), comptime dimension: usize) !MagicalReturnType(T, dimension) {
            return self.toArrayFn(self.ptr, dimension);
        }
        
        pub fn copy(self: @This()) !tensorType(dataType) {
            return self.copyFn(self.ptr);
        }

        pub fn fromShape(self: @This(), allocator: *const std.mem.Allocator, shape: []usize) !tensorType {
            return self.fromShapeFn(allocator, shape);
        }

        pub fn fromConstBuffer(self: @This(), allocator: *const std.mem.Allocator, data: []const T, shape: []const usize) !tensorType {
            return self.fromConstBufferFn(allocator, data, shape);
        }

        pub fn fill(self: @This(), inputArray: anytype, shape: []usize) !void {
            return self.fillFn(self.ptr, inputArray, shape);
        }

        // MURO DI BERLINO

        pub fn infoMetal(self: @This()) void {
            return self.infoMetalFn(self.ptr);
        }

        pub fn ensure4DShape(self: @This(), shape: []const usize) ![]usize {
            return self.ensure4DShapeFn(shape);
        }

        pub fn slice_onnx(self: @This(), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !tensorType {
            return self.sliceONNXFn(self.ptr, starts, ends, axes, steps);
        }


        // FUNZIONI ORIGINALI DI TENSOR
        
        pub fn deinit(self: *Self) void
        pub fn fromArray(allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !Self
        pub fn toArray(self: Self, comptime dimension: usize) !MagicalReturnType(T, dimension)
        //fn setAllocator(tensor: *Tensor(T), alloc: *const std.mem.Allocator) void
        pub fn copy(self: *Self) !Tensor(T)
        pub fn fromShape(allocator: *const std.mem.Allocator, shape: []usize) !Self
        pub fn fromConstBuffer(allocator: *const std.mem.Allocator, data: []const T, shape: []const usize) Self
        pub fn fill(self: *Self, inputArray: anytype, shape: []usize) !void
        pub fn getSize(self: *Self) usize
        pub fn get(self: *const Self, idx: usize) !T
        pub fn set(self: *Self, idx: usize, value: T) !void
        pub fn get_at(self: *const Self, indices: []const usize) !T
        pub fn set_at(self: *Self, indices: []const usize, value: T) !void
        fn constructMultidimensionalArray(allocator: *const std.mem.Allocator, comptime ElementType: type, data: []ElementType, shape: []usize, comptime depth: usize, comptime dimension: usize) !MagicalReturnType(ElementType, dimension - depth)
        fn MagicalReturnType(comptime DataType: type, comptime dim_count: usize) type
        fn calculateProduct(slices: []usize) usize
        pub fn flatten_index(self: *const Self, indices: []const usize) !usize
        pub fn flatten_index_original(self: *const Self, indices: []const usize) !usize
        pub fn benchmark_flatten_index(self: *const Self, iterations: usize) struct { optimized: u64, original: u64 }
        pub fn slice(self: *Tensor(T), start_indices: []usize, slice_shape: []usize) !Tensor(T)
        fn copy_data_recursive(self: *Tensor(T), new_data: []T, new_data_index: *usize, start_indices: []usize, slice_shape: []usize, indices: []usize, dim: usize) !void
        fn get_flat_index(self: *Tensor(T), indices: []usize) !usize
        pub fn getStrides(self: *Tensor(T)) ![]usize
        pub fn info(self: *Self) void
        pub fn print(self: *Self) void
        pub fn printMultidim(self: *Self) void
        fn _printMultidimHelper(self: *Self, offset: usize, idx: usize) void
        pub fn setToZero(self: *Self) !void
        pub fn slice_onnx(self: *Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !Tensor(T)
        pub inline fn ensure_4D_shape(shape: []const usize) ![]usize
        pub fn info_metal(self: *Self) void

        pub fn init(self: *@This(), allocator: *const std.mem.Allocator) !@This() {
            return self.initFn(self.ptr, allocator);
        }
        pub fn deinit(self: *@This()) void {
            return self.deinitFn(self.ptr);
        }
        pub fn fromArray(self: *@This(), allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !@This() {
            return self.fromArrayFn(self.ptr, allocator, inputArray, shape);
        }
        pub fn toArray(self: @This(), comptime dimension: usize) !MagicalReturnType(T, dimension) {
            return self.toArrayFn(self.ptr, dimension);
        }
        fn setAllocator(self: *Tensor(T), tensor: *Tensor(T), alloc: *const std.mem.Allocator) void {
            return self.setAllocator(tensor, alloc);
        }
        pub fn copy(self: *@This()) !Tensor(T) {
            return self.copy();
        }
        pub fn fromShape(self: *@This(), allocator: *const std.mem.Allocator, shape: []usize) !@This() {
            return self.fromShape(allocator, shape);
        }
        pub fn fromConstBuffer(self: *@This(), allocator: *const std.mem.Allocator, data: []const T, shape: []const usize) @This() {
            return self.fromConstBuffer(allocator, data, shape);
        }
        pub fn fill(self: *@This(), inputArray: anytype, shape: []usize) !void {
            return self.fill(inputArray, shape);
        }
        pub fn getSize(self: *@This()) usize {
            return self.getSize();
        }
        pub fn get(self: *const @This(), idx: usize) !T {
            return self.get(idx);
        }
        pub fn set(self: *@This(), idx: usize, value: T) !void {
            return self.set(idx, value);
        }
        pub fn get_at(self: *const @This(), indices: []const usize) !T {
            return self.get_at(indices);
        }
        pub fn set_at(self: *@This(), indices: []const usize, value: T) !void {
            return self.set_at(indices, value);
        }
        fn constructMultidimensionalArray(self: *@This(), allocator: *const std.mem.Allocator, comptime ElementType: type, data: []ElementType, shape: []usize, comptime depth: usize, comptime dimension: usize) !MagicalReturnType(ElementType, dimension - depth) {
            return self.constructMultidimensionalArray(allocator, ElementType, data, shape, depth, dimension);
        }
        fn MagicalReturnType(self: *@This(), comptime DataType: type, comptime dim_count: usize) type {
            return self.MagicalReturnType(DataType, dim_count);
        }
        fn calculateProduct(self: *@This(), slices: []usize) usize {
            return self.calculateProduct(slices);
        }
        pub fn flatten_index(self: *const @This(), indices: []const usize) !usize {
            return self.flatten_index(indices);
        }
        pub fn flatten_index_original(self: *const @This(), indices: []const usize) !usize {
            return self.flatten_index_original(indices);
        }
        pub fn benchmark_flatten_index(self: *const @This(), iterations: usize) struct { optimized: u64, original: u64 } {
            return self.benchmark_flatten_index(iterations);
        }
        pub fn slice(self: *Tensor(T), start_indices: []usize, slice_shape: []usize) !Tensor(T) {
            return self.slice(start_indices, slice_shape);
        }
        fn copy_data_recursive(self: *Tensor(T), new_data: []T, new_data_index: *usize, start_indices: []usize, slice_shape: []usize, indices: []usize, dim: usize) !void {
            return self.copy_data_recursive(new_data, new_data_index, start_indices, slice_shape, indices, dim);
        }
        fn get_flat_index(self: *Tensor(T), indices: []usize) !usize {
            return self.get_flat_index(indices);
        }
        pub fn getStrides(self: *Tensor(T)) ![]usize {
            return self.getStrides();
        }
        pub fn info(self: *@This()) void {
            return self.info();
        }
        pub fn print(self: *@This()) void {
            return self.print();
        }
        pub fn printMultidim(self: *@This()) void {
            return self.printMultidim();
        }
        fn _printMultidimHelper(self: *@This(), offset: usize, idx: usize) void {
            return self._printMultidimHelper(offset, idx);
        }
        pub fn setToZero(self: *@This()) !void {
            return self.setToZero();
        }
        pub fn slice_onnx(self: *Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !Tensor(T) {
            return self.slice_onnx(starts, ends, axes, steps);
        }
        pub inline fn ensure_4D_shape(self: *@This(), shape: []const usize) ![]usize {
            return self.ensure_4D_shape(shape);
        }
        pub fn info_metal(self: *@This()) void {
            return self.info_metal();
        }
        
    };
}
