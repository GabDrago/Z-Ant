const std = @import("std");
const zant = @import("../../zant.zig");
const Tensor = zant.core.tensor.Tensor;

pub const quantScheme = enum {
    SYMM,
    ASYM,
};

// AUSILIARIE
// - normalize (maxFloat, minFloat, top, bottom)
// - get_symmetric_scale_factor
// - get_asymmetric_scale_factor
// - get_zero_point
// - minimize_mse

// EFFETTIVE
// - minmax_symmetric_quantization
// - minmax_asymmetric_quantization
// - mse_symmetric_quantization
// - mse_asymmetric_quantization
// - cross_entropy_quantization

// ========== auxiliary functions
pub fn clamp(comptime T: type, comptime U: type, value: T, scale: T, zero: U, minInt: U, maxInt: U) U {
    var roundedVal: U = @intFromFloat(@round(value / scale)); // U must be int type
    roundedVal += zero;

    if (roundedVal <= minInt)
        return minInt;
    if (roundedVal >= maxInt)
        return maxInt;

    return roundedVal;
}

pub inline fn get_scale_factor(comptime T: type, comptime U: type, minFloat: T, maxFloat: T, minInt: U, maxInt: U) T {
    return (maxFloat - minFloat) / (maxInt - minInt);
}

pub inline fn get_zero_point(comptime T: type, comptime U: type, scale: T, minFloat: T, minInt: U, maxInt: U) U {
    const zeroPointFloat: T = -minFloat / scale;
    var zeroPointInt: U = @intFromFloat(@round(zeroPointFloat));

    if (zeroPointInt < minInt) {
        zeroPointInt = minInt;
    } else if (zeroPointInt > maxInt) {
        zeroPointInt = maxInt;
    }

    return zeroPointInt;
}

// ========== quantization

/// This function quantizes the input tensor, using the given parameters:
/// scale factor, zero point, minInt/maxInt (aka the integer grid limits)
fn quantize_tensor(comptime T: type, comptime U: type, input: *Tensor(T), output: *Tensor(U), scale: T, zero: U, minInt: U, maxInt: U) void {
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

    if (minFloat < 0) {
        minInt = -(1 << (@bitSizeOf(U) - 1)); // minInt = - 2^(b-1)
        maxInt = 1 << (@bitSizeOf(U) - 1) - 1; // maxInt = 2^(b-1) - 1
    } else {
        minInt = 0; // minInt = 0
        maxInt = 1 << @bitSizeOf(U) - 1; // maxInt = 2^b - 1
    }

    const scale: T = get_scale_factor(T, U, minFloat, maxFloat, minInt, maxInt);

    var zero: U = undefined;
    switch (scheme) {
        0 => zero = 0,
        1 => zero = get_zero_point(T, U, scale, minFloat, minInt, maxInt),
    }

    quantize_tensor(T, U, input, output, scale, zero, minInt, maxInt);
}

/// this function computes the forbenius norm of the difference between two tensors
fn compute_MSE_norm(comptime T: type, tensor1: *Tensor, tensor2: *Tensor) T {
    var sum: T = 0;
    for (tensor1, 0..) |val, i| {
        sum = @abs((val - tensor2.data[i]) * (val - tensor2.data[i]));
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
        const candidateScale: T = get_scale_factor(T, U, candidateMin, candidateMax, minInt, maxInt);

        const candidateZero: U = if (scheme == 0) 0 else get_zero_point(T, U, candidateScale, candidateMin, minInt, maxInt);

        // quantize
        quantize_tensor(T, U, input, output, candidateScale, candidateZero, minInt, maxInt);

        // compute mse between original input tensor and the quantized one
        const mseCandidate: T = compute_MSE_norm(T, input, output);

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
