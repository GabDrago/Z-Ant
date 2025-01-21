const std = @import("std");
const pkgAllocator = @import("pkgAllocator");
const TensMath = @import("tensor_m");
const Tensor = @import("tensor").Tensor;
const TensorMathError = @import("errorHandler").TensorMathError;

test "Sum two tensors on CPU architecture" {
    std.debug.print("\n     test: Sum two tensors on CPU architecture", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2.deinit();

    var t3 = try TensMath.sum_tensors(f32, f64, &t1, &t2); // Output tensor with larger type
    defer t3.deinit();

    // Check if the values in t3 are as expected
    try std.testing.expect(2.0 == t3.data[0]);
    try std.testing.expect(4.0 == t3.data[1]);
}

test "test tensor element-wise multiplication" {
    std.debug.print("\n     test: tensor element-wise multiplication ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray1: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var inputArray2: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &shape);
    defer tensor1.deinit();

    var tensor2 = try Tensor(u8).fromArray(&allocator, &inputArray2, &shape);
    defer tensor2.deinit();

    var tensor3 = try TensMath.mul(u8, &tensor1, &tensor2);
    defer tensor3.deinit();

    for (0..tensor3.size) |i| {
        try std.testing.expect(tensor3.data[i] == tensor1.data[i] * tensor2.data[i]);
    }
}

test "Error when input tensors have different sizes" {
    std.debug.print("\n     test: Error when input tensors have different sizes", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };
    var inputArray2: [3][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
        [_]f32{ 14.0, 15.0 },
    };

    var shape1 = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2 = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape1);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorDifferentSize, TensMath.sum_tensors(f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "add bias" {
    std.debug.print("\n     test:add bias", .{});
    const allocator = pkgAllocator.allocator;

    var shape_tensor: [2]usize = [_]usize{ 2, 3 }; // 2x3 matrix
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    const flatArr: [6]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    var shape_bias: [1]usize = [_]usize{3};
    var bias_array: [3]f32 = [_]f32{ 1.0, 1.0, 1.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &shape_bias);

    try TensMath.add_bias(f32, &t1, &bias);

    for (t1.data, 0..) |*data, i| {
        try std.testing.expect(data.* == flatArr[i] + 1);
    }

    t1.deinit();
    bias.deinit();
}

test "Subtraction with same shape tensors" {
    std.debug.print("\n     test: Subtraction with same shape tensors", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 4.0);
    try std.testing.expectEqual(result.data[1], 5.0);
    try std.testing.expectEqual(result.data[2], 6.0);
    try std.testing.expectEqual(result.data[3], 7.0);
}

test "Subtraction with broadcasting - scalar and matrix" {
    std.debug.print("\n     test: Subtraction with broadcasting - scalar and matrix", .{});
    const allocator = pkgAllocator.allocator;

    // Matrix 2x2
    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };
    var shape1: [2]usize = [_]usize{ 2, 2 };

    // Scalar (1x1)
    var inputArray2 = [_]f32{2.0};
    var shape2: [1]usize = [_]usize{1};

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 3.0);
    try std.testing.expectEqual(result.data[1], 5.0);
    try std.testing.expectEqual(result.data[2], 7.0);
    try std.testing.expectEqual(result.data[3], 9.0);
}

test "Subtraction with broadcasting - row and matrix" {
    std.debug.print("\n     test: Subtraction with broadcasting - row and matrix", .{});
    const allocator = pkgAllocator.allocator;

    // Matrix 2x2
    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };
    var shape1: [2]usize = [_]usize{ 2, 2 };

    // Row vector as 2D array with broadcasting shape
    var inputArray2: [1][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
    };
    var shape2: [2]usize = [_]usize{ 1, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 4.0);
    try std.testing.expectEqual(result.data[1], 5.0);
    try std.testing.expectEqual(result.data[2], 8.0);
    try std.testing.expectEqual(result.data[3], 9.0);
}

test "Subtraction with incompatible shapes" {
    std.debug.print("\n     test: Subtraction with incompatible shapes", .{});
    const allocator = pkgAllocator.allocator;

    // Matrix 2x2
    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };
    var shape1: [2]usize = [_]usize{ 2, 2 };

    // Matrix 3x2 (incompatible)
    var inputArray2: [3][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
        [_]f32{ 5.0, 6.0 },
    };
    var shape2: [2]usize = [_]usize{ 3, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    try std.testing.expectError(TensorMathError.IncompatibleBroadcastShapes, TensMath.sub_tensors(f32, f32, &t1, &t2));
}

test "test tensor element-wise division" {
    std.debug.print("\n     test: tensor element-wise division ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray1: [2][3]u8 = [_][3]u8{
        [_]u8{ 2, 4, 6 },
        [_]u8{ 8, 10, 12 },
    };

    var inputArray2: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &shape);
    defer tensor1.deinit();

    var tensor2 = try Tensor(u8).fromArray(&allocator, &inputArray2, &shape);
    defer tensor2.deinit();

    var tensor3 = try TensMath.div(u8, &tensor1, &tensor2);
    defer tensor3.deinit();

    for (0..tensor3.size) |i| {
        try std.testing.expect(tensor3.data[i] == tensor1.data[i] / tensor2.data[i]);
    }
}
