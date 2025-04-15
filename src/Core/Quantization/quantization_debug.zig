const std = @import("std");
const zant = @import("../../zant.zig");
const quant = zant.core.quantization;
const Tensor = zant.core.tensor.Tensor;

pub const quantScheme = enum {
    SYMM,
    ASYM,
};

// ========== auxiliary functions

pub fn clamp_debug(comptime T: type, comptime U: type, value: T, scale: T, zero: T, minInt: U, maxInt: U) U {
    const roundedVal: T = @round(value / scale + zero);
    std.debug.print("\nRounded val: {}", .{roundedVal});

    if (roundedVal <= @as(T, @floatFromInt(minInt)))
        return minInt;
    if (roundedVal >= @as(T, @floatFromInt(maxInt)))
        return maxInt;

    const roundedValInt: U = @as(U, @intFromFloat(roundedVal));
    std.debug.print("\nRounded val (as int) + zero: {}", .{roundedValInt});

    return roundedValInt;
}

pub inline fn get_scale_factor(comptime T: type, comptime U: type, minFloat: T, maxFloat: T) T {
    const num: T = maxFloat - minFloat;
    std.debug.print("\nscale num {}", .{num});

    const num_elements = (1 << @bitSizeOf(U)) - 1; // 2^b - 1 values
    const denom: T = @as(T, @floatFromInt(num_elements));
    std.debug.print("\nscale denom = {}", .{denom});

    return num/denom;
}

pub inline fn get_zero_point(comptime T: type, scale: T, minFloat: T) T {
    const zeroPointFloat: T = -minFloat / scale;

    return zeroPointFloat;
}

// ==========

fn quantize_tensor_debug(comptime T: type, comptime U: type, input: *Tensor(T), output: *Tensor(U), scale: T, zero: T, minInt: U, maxInt: U) void {
    for (input.data, 0..) |val, i| {
        // quantize every val
        output.data[i] = clamp_debug(T, U, val, scale, zero, minInt, maxInt);
    }
}

pub fn debug_minmax_quant(comptime T: type, comptime U: type, scheme: quantScheme, input: *Tensor(T), output: *Tensor(U)) void {
    var minFloat: T = input.data[0];
    var maxFloat: T = input.data[0];

    // compute the min and max value if the input tensor
    for (input.data[1..]) |val| {
        if (minFloat > val)
            minFloat = val;
        if (maxFloat < val)
            maxFloat = val;
    }
    std.debug.print("\nminFloat = {d:.3}", .{minFloat});
    std.debug.print("\nmaxFloat = {d:.3}", .{maxFloat});

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

    std.debug.print("\nminInt = {}", .{minInt});
    std.debug.print("\nmaxInt = {}", .{maxInt});

    const scale: T = get_scale_factor(T, U, minFloat, maxFloat);
    std.debug.print("\nscale = {d:.3}", .{scale});

    var zero: T = undefined;
    switch (scheme) {
        quantScheme.SYMM => zero = 0,
        quantScheme.ASYM => zero = get_zero_point(T, scale, minFloat),
    }
    std.debug.print("\nzero point = {}", .{zero});

    quantize_tensor_debug(T, U, input, output, scale, zero, minInt, maxInt);
}
