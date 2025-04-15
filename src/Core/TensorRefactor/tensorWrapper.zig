//! This file contains the definition of the types of tensors that can be used in the library.
//! The tensorWrapper class is a wrapper around the tensor class so that it can be manipulated in the same way regardless of the true nature of the tensor itself.
//! - classicTensor     -> used when weights ridden from the onnx file are not pre-processed in any way, regardless of the type of the weights.
//! - quantizedTensor   -> used when weights ridden from the onnx file are quantized.
//! - clusteredTensor   -> used when weights ridden from the onnx file are clustered.

const std = @import("std");
const zant = @import("../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const ClassicTensor = zant.core.classicTensor.ClassicTensor;
const QuantizedTensor = zant.core.quantizedTensor.QuantizedTensor;
const ClusteredTensor = zant.core.clusteredTensor.ClusteredTensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkgAllocator = zant.utils.allocator.allocator;
const LayerError = zant.utils.error_handler.LayerError;

// Enum containing the possible types of tensors
pub const TensorType = enum {
    ClassicTensor,
    QuantizedTensor,
    ClusteredTensor,
    null,
};

/// Interface methods to be implemented
pub const TayerInterface = struct {
    init: *const fn (ctx: *anyopaque, allocator: *const std.mem.Allocator, args: *anyopaque) anyerror!void,
    deinit: *const fn (ctx: *anyopaque) void,
    forward: *const fn (ctx: *anyopaque, input: *Tensor(T)) anyerror!Tensor(T),
    backward: *const fn (ctx: *anyopaque, dValues: *Tensor(T)) anyerror!Tensor(T),
    printLayer: *const fn (ctx: *anyopaque, choice: u8) void,
    get_n_inputs: *const fn (ctx: *anyopaque) usize,
    get_n_neurons: *const fn (ctx: *anyopaque) usize,
    get_input: *const fn (ctx: *anyopaque) *const Tensor(T),
    get_output: *const fn (ctx: *anyopaque) *Tensor(T),
};

//------------------------------------------------------------------------------------------------------
/// TENSOR WRAPPER
///
/// TensorWrapper() is the superclass for all the possible implementation of a tensor (ClassicTensor, QuantizedTensor, ClusteredTensor).
///
/// @param T:comptime type of the values in the tensor
pub fn Layer(comptime T: type) type {
    return struct {
        tensor_type: TensorType,
        tensor_ptr: *anyopaque,
        tensor_int: *const TensorInterface,

        const Self = @This();

        pub fn init(self: Self, alloc: *const std.mem.Allocator, args: *anyopaque) anyerror!void {
            return self.layer_impl.init(self.layer_ptr, alloc, args);
        }

        /// When deinit() pay attention to:
        /// - Double-freeing memory.
        /// - Using uninitialized or already-deallocated pointers.
        /// - Incorrect allocation or deallocation logic.
        ///
        pub fn deinit(self: Self) void {
            return self.layer_impl.deinit(self.layer_ptr);
        }
        pub fn forward(self: Self, input: *Tensor(T)) !Tensor(T) {
            return self.layer_impl.forward(self.layer_ptr, input);
        }
        pub fn backward(self: Self, dValues: *Tensor(T)) !Tensor(T) {
            return self.layer_impl.backward(self.layer_ptr, dValues);
        }
        pub fn printLayer(self: Self, choice: u8) void {
            return self.layer_impl.printLayer(self.layer_ptr, choice);
        }
        pub fn get_n_inputs(self: Self) usize {
            return self.layer_impl.get_n_inputs(self.layer_ptr);
        }
        pub fn get_n_neurons(self: Self) usize {
            return self.layer_impl.get_n_neurons(self.layer_ptr);
        }
        pub fn get_input(self: Self) *const Tensor(T) {
            return self.layer_impl.get_input(self.layer_ptr);
        }
        pub fn get_output(self: Self) *Tensor(T) {
            return self.layer_impl.get_output(self.layer_ptr);
        }
    };
}
