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
