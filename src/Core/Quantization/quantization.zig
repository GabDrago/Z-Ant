const std = @import("std");
const zant = @import("../../zant.zig");
const Tensor = zant.core.tensor.Tensor;

pub const quantScheme = enum {
    SYMM,
    ASYM,
};

// ========== auxiliary functions
pub fn clamp(comptime T: type, comptime U: type, value: T, scale: T, zero: U, minInt: U, maxInt: U) U {
    const roundedVal: T = @round(value / scale + zero);

    if (roundedVal <= @as(T, @floatFromInt(minInt)))
        return minInt;
    if (roundedVal >= @as(T, @floatFromInt(maxInt)))
        return maxInt;

    const roundedValInt: U = @as(U, @intFromFloat(roundedVal));

    return roundedValInt;
}

pub inline fn get_scale_factor(comptime T: type, comptime U: type, minFloat: T, maxFloat: T) T {
    const num: T = maxFloat - minFloat;

    const num_elements = (1 << @bitSizeOf(U)) - 1; // 2^b - 1 values
    const denom: T = @as(T, @floatFromInt(num_elements));

    return num/denom;
}

pub inline fn get_zero_point(comptime T: type, comptime U: type, scale: T, minFloat: T) U {
    const zeroPointFloat: T = -minFloat / scale;

    return zeroPointFloat;
}

// ========== quantization

/// This function quantizes the input tensor, using the given parameters:
/// scale factor, zero point, minInt/maxInt (aka the integer grid limits)
pub fn quantize_tensor(comptime T: type, comptime U: type, input: *Tensor(T), output: *Tensor(U), scale: T, zero: U, minInt: U, maxInt: U) void {
    for (input.data, 0..) |val, i| {
        // quantize every val
        output.data[i] = clamp(T, U, val, scale, zero, minInt, maxInt);
    }
}

fn dequantize_tensor(comptime T: type, comptime U: type, input: *Tensor(U), output: *Tensor(T), scale: T, zero: U) void {
    for (input.data, 0..) |val, i| {
        // dequantize every val
        output.data[i] = scale * (val - zero);
    }
}

/// This function quantizes the input tensor using min/max method
pub fn minmax_quant(comptime T: type, comptime U: type, scheme: quantScheme, input: *Tensor(T), output: *Tensor(U)) void {
    var minFloat: T = input.data[0];
    var maxFloat: T = input.data[0];

    // compute the min and max value if the input tensor
    for (input.data[1..]) |val| {
        if (minFloat > val)
            minFloat = val;
        if (maxFloat < val)
            maxFloat = val;
    }

    // compute minInt and maxInt
    var minInt: U = undefined;
    var maxInt: U = undefined;

    if (@typeInfo(U).int.signedness == .signed) {
        minInt = @as(U, -(1 << (@bitSizeOf(U) - 1))); // minInt = - 2^(b-1)
        maxInt = @as(U, (1 << (@bitSizeOf(U) - 1)) - 1); // maxInt = 2^(b-1) - 1
    } else {
        minInt = 0; // minInt = 0
        maxInt = @as(U, (1 << @bitSizeOf(U)) - 1); // maxInt = 2^b - 1
    }

    const scale: T = get_scale_factor(T, U, minFloat, maxFloat);

    var zero: U = undefined;
    switch (scheme) {
        quantScheme.SYMM => zero = 0,
        quantScheme.ASYM => zero = get_zero_point(T, U, scale, minFloat),
    }

    quantize_tensor(T, U, input, output, scale, zero, minInt, maxInt);
}

/// this function computes the forbenius norm of the difference between two tensors
pub fn compute_MSE_norm(comptime T: type, comptime U: type, tensor1: *Tensor(T), tensor2: *Tensor(U)) T {
    var sum: T = 0;
    for (tensor1.data, 0..) |val, i| {
        sum = @abs((val - @as(T, @floatFromInt(tensor2.data[i]))) * (val - @as(T, @floatFromInt(tensor2.data[i]))));
    }
    return @sqrt(sum);
}

/// This function quantizes the input tensor, using the best scale factor and zero point
/// computed by minimizing the MSE between the input tensor and the quantized one.
pub fn MSE_grid_search_quant(comptime T: type, comptime U: type, scheme: quantScheme, input: *Tensor(T), output: *Tensor(U)) void {
    // compute max and min from input tensor
    var minFloat: T = input.data[0];
    var maxFloat: T = input.data[0];

    for (input.data[1..]) |val| {
        minFloat = if (val < minFloat) val else minFloat;
        maxFloat = if (val > maxFloat) val else maxFloat;
    }

    // compute maxInt and minInt
    var maxInt: U = undefined;
    var minInt: U = undefined;

    if (minFloat < 0) {
        minInt = -(1 << (@bitSizeOf(U) - 1)); // minInt = - 2^(b-1)
        maxInt = 1 << (@bitSizeOf(U) - 1) - 1; // maxInt = 2^(b-1) - 1
    } else {
        minInt = 0; // minInt = 0
        maxInt = 1 << @bitSizeOf(U) - 1; // maxInt = 2^b - 1
    }

    // grid search parameters setting
    const numCandidates: usize = 100; // arbitrary choice
    const meanFloat: T = (maxFloat + minFloat) / 2;
    const deltaFloat = 0.2 * std.math.abs(maxFloat - meanFloat);
    const maxStart: T = maxFloat - deltaFloat;
    const minStart: T = minFloat + deltaFloat;
    const step: T = (maxFloat - maxStart) / (numCandidates - 1);

    // current best result variables
    var bestMSE: T = std.math.inf(T);
    var bestScale: T = undefined;
    var bestZero: U = undefined;

    // grid search
    var candidateMin: T = minStart;
    var candidateMax: T = maxStart;
    for (0..numCandidates) |_| {
        const candidateScale: T = get_scale_factor(T, U, candidateMin, candidateMax);

        const candidateZero: U = if (scheme == 0) 0 else get_zero_point(T, U, candidateScale, candidateMin);

        // quantize
        quantize_tensor(T, U, input, output, candidateScale, candidateZero, minInt, maxInt);

        // compute mse between original input tensor and the quantized one
        const mseCandidate: T = compute_MSE_norm(T, U, input, output);

        // update parameters, if mse has improved
        if (mseCandidate < bestMSE) {
            bestMSE = mseCandidate;
            bestScale = candidateScale;
            bestZero = candidateZero;
        }

        // add/sub the step
        candidateMax += step;
        candidateMin -= step;
    }

    // quantization based on the computed best scale factor and best zero point
    quantize_tensor(T, U, input, output, bestScale, bestZero, minInt, maxInt);
}
