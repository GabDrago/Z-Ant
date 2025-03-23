// const tensor = @import("../Tensor/TensorMath/");

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
pub fn clamp(comptime T: type, comptime U: type, value: T, scale: T, zero: U, minFloat: U, maxFloat: U) U {
    if (value <= minFloat)
        return minFloat;
    if (value >= maxFloat)
        return maxFloat;

    const roundedVal: U = @intFromFloat(@round(value / scale + zero));
    return roundedVal;
}

// TODO
// pub fn normalize(comptime T: type, comptime U: type, value: T, maxFloat: T, minFloat: T, top: U, bottom: U) U {
//     return top - bottom - value - maxFloat - minFloat;
// }

// pub fn minmax_scale_factor(comptime T: type, comptime U: type) T {
//     return
// }

// ========== quantization

// TODO: vedi x quant asimmetrica, cambia solo lo zero factor (si può "giocare" con la get_zero_factor, però server param per tipo di quant)
pub fn minmax_sym_quant(comptime T: type, comptime U: type, scheme: quantScheme, input: *Tensor(T), output: *Tensor(U)) void {
    var minFloat: T = input.data[0];
    var maxFloat: T = input.data[0];

    // inserire check su dimensioni/shape...

    for (input.data[1..]) |val| {
        if (minFloat > val)
            minFloat = val;
        if (maxFloat < val)
            maxFloat = val;
    }

    // calcolo minInt e maxInt a seconda dei valori presenti
    var minInt: U = undefined;
    var maxInt: U = undefined;

    if (minFloat < 0) {
        minInt = -(1 << (@bitSizeOf(U) - 1)); // minInt = - 2^(b-1)
        maxInt = 1 << (@bitSizeOf(U) - 1) - 1; // maxInt = 2^(b-1) - 1
    } else {
        minInt = 0; // minInt = 0
        maxInt = 1 << @bitSizeOf(U) - 1; // maxInt = 2^b - 1
    }

    const scale: T = (maxFloat - minFloat) / (maxInt - minInt); // sostituire con get_scale_factor(T, U, );

    var zero: U = undefined;
    switch (scheme) {
        0 => zero = 0,
        1 => zero = 1, // zero factor da calcolare caso asymm
    }

    for (input.data, 0..) |val, i| {
        // quantize ogni val
        output.data[i] = clamp(T, U, val, scale, zero, minFloat, maxFloat);
    }
}
