const std = @import("std");
const zant = @import("zant");
const quant = zant.core.quantization;
const quant_debug = zant.core.quant_debug;
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;

const testing = std.testing;

test "MSE norm" {
    std.debug.print("\n    test: MSE tensor norm computation", .{});
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ -0.5, 0.8, -0.46 },
        [_]f32{ 0.234, 0.3435, -0.231 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator.allocator, &inputArray, &shape);

    defer inputTensor.deinit();

    var outputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 0, 255, 20 },
        [_]u8{ 195, 225, 55 },
    };
    var outputTensor = try Tensor(u8).fromArray(&pkgAllocator.allocator, &outputArray, &shape);
    defer outputTensor.deinit();

    std.debug.print("\n --> MSE : {}\n", .{quant.compute_MSE_norm(f32, u8, &inputTensor, &outputTensor)});

}

// asymmetric quantization

test "asymm signed range grid limits test" {
    std.debug.print("\n    test: asymm signed quantization grid limits\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ -0.5, 0.8, -0.46 },
        [_]f32{ 0.234, 0.3435, -0.231 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator.allocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try Tensor(u8).fromShape(&pkgAllocator.allocator, &shape);
    defer outputTensor.deinit();

    quant_debug.debug_minmax_quant(f32, u8, quant_debug.quantScheme.ASYM, &inputTensor, &outputTensor);
    outputTensor.printMultidim();

    std.debug.print("\n --> MSE : {}\n", .{quant.compute_MSE_norm(f32, u8, &inputTensor, &outputTensor)});
    try testing.expectEqual(0, outputTensor.get(0));
    try testing.expectEqual(255, outputTensor.get(1));
}

test "asymm unsigned range grid limits test" {
    std.debug.print("\n    test: asymm unsigned range grid limits\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 0.1, 5.0, 2.0 },
        [_]f32{ 0.4, 3.0, 0.6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator.allocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try Tensor(u8).fromShape(&pkgAllocator.allocator, &shape);
    defer outputTensor.deinit();

    quant_debug.debug_minmax_quant(f32, u8, quant_debug.quantScheme.ASYM, &inputTensor, &outputTensor);
    outputTensor.printMultidim();

    std.debug.print("\n --> MSE : {}\n", .{quant.compute_MSE_norm(f32, u8, &inputTensor, &outputTensor)});
    try testing.expectEqual(0, outputTensor.get(0));
    try testing.expectEqual(255, outputTensor.get(1));
}

// symmetric quantization

test "symm signed range grid limits test" {
    std.debug.print("\n    test: symm signed quantization grid limits\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ -0.5, 0.8, -0.46 },
        [_]f32{ 0.234, 0, -0.231 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator.allocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try Tensor(i8).fromShape(&pkgAllocator.allocator, &shape);
    defer outputTensor.deinit();

    quant_debug.debug_minmax_quant(f32, i8, quant_debug.quantScheme.SYMM, &inputTensor, &outputTensor);
    outputTensor.printMultidim();

    std.debug.print("\n --> MSE : {}\n", .{quant.compute_MSE_norm(f32, i8, &inputTensor, &outputTensor)});
    try testing.expectEqual(127, outputTensor.get(1));
    try testing.expectEqual(0, outputTensor.get(4));
}

test "symm unsigned range grid limits test (0 as min)" {
    std.debug.print("\n    test: symm unsigned range grid limits (with 0 as min value)\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 0.0, 5.0, 2.0 },
        [_]f32{ 0.4, 3.0, 0.6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator.allocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try Tensor(u8).fromShape(&pkgAllocator.allocator, &shape);
    defer outputTensor.deinit();

    quant_debug.debug_minmax_quant(f32, u8, quant_debug.quantScheme.SYMM, &inputTensor, &outputTensor);
    outputTensor.printMultidim();

    std.debug.print("\n --> MSE : {}\n", .{quant.compute_MSE_norm(f32, u8, &inputTensor, &outputTensor)});
    try testing.expectEqual(0, outputTensor.get(0));
    try testing.expectEqual(255, outputTensor.get(1));
}

test "symm unsigned range grid limits test (with negative val)" {
    std.debug.print("\n    test: symm unsigned range grid limits (with negative values)\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ -0.5, 0.8, -0.46 },
        [_]f32{ 0.0, 0.3435, -0.231 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator.allocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try Tensor(u8).fromShape(&pkgAllocator.allocator, &shape);
    defer outputTensor.deinit();

    quant_debug.debug_minmax_quant(f32, u8, quant_debug.quantScheme.SYMM, &inputTensor, &outputTensor);
    outputTensor.printMultidim();

    std.debug.print("\n --> MSE : {}\n", .{quant.compute_MSE_norm(f32, u8, &inputTensor, &outputTensor)});
    try testing.expectEqual(0, outputTensor.get(0));
    try testing.expectEqual(0, outputTensor.get(3));
}
