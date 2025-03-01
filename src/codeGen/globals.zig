const std = @import("std");
const zant = @import("zant");
const Tensor = zant.tensor.Tensor;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
//--- proto
const TensorProto = onnx.TensorProto;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;
const AttributeProto = onnx.AttributeProto;
const allocator = zant.utils.allocator.allocator;

//--- other
const utils = @import("codeGen_utils.zig");
const mathGen = @import("codeGen_math_handler.zig");
const codegen_options = @import("codegen_options");

pub var readyGraph: std.ArrayList(ReadyNode) = std.ArrayList(ReadyNode).init(allocator);
pub var tensorHashMap: std.StringHashMap(ReadyTensor) = std.StringHashMap(ReadyTensor).init(allocator); //key: TensorProto.name

pub var networkInput: []const u8 = undefined;
pub var networkOutput: []const u8 = undefined;

pub var inputType: type = f32;

pub const TensorTag = enum {
    INITIALIZER,
    CONSTANT,
    INPUT,
    OUTPUT,
    LINK, //with "LINK" I mean a tensor that is used to link two nodes, it means that is is the output of a node and the input of another
};

// Struct to represent a tensor that is ready for computation
pub const ReadyTensor = struct {
    name: []const u8,
    ready: bool,
    shape: []const i64,
    tensorProto: ?*TensorProto,
    tag: TensorTag,

    pub fn createInitializer(tensorProto: *TensorProto) !ReadyTensor {
        return ReadyTensor{
            .name = tensorProto.name.?,
            .ready = true,
            .shape = tensorProto.dims,
            .tensorProto = tensorProto,
            .tag = TensorTag.INITIALIZER,
        };
    }

    pub fn createInput(name: []const u8) !ReadyTensor {
        networkInput = name;
        return ReadyTensor{
            .name = name,
            .ready = true,
            .shape = try utils.parseNumbers(codegen_options.shape),
            .tensorProto = null,
            .tag = TensorTag.INPUT,
        };
    }

    pub fn createConstant(name: []const u8) !ReadyTensor {
        return ReadyTensor{
            .name = name,
            .ready = true,
            .shape = &[_]i64{ 1, 1, 1, 1 }, // it will be changed in the graph creation
            .tensorProto = null,
            .tag = TensorTag.CONSTANT,
        };
    }

    pub fn createLink(name: []const u8) !ReadyTensor {
        return ReadyTensor{ //default
            .name = name,
            .ready = false,
            .shape = &[_]i64{ 1, 1, 1, 1 }, // it will be changed in the graph creation
            .tensorProto = null,
            .tag = TensorTag.LINK,
        };
    }
};

// Struct representing a computational node in the ONNX model
pub const ReadyNode = struct {
    nodeProto: *NodeProto,
    inputs: std.ArrayList(*ReadyTensor),
    outputs: std.ArrayList(*ReadyTensor),
    ready: bool,

    // Creates a ReadyNode by preparing its input and output tensors
    pub fn create(nodeProto: *NodeProto) !ReadyNode {
        // std.debug.print("\n\nReadyNode.create() --> {s}", .{nodeProto.name.?});
        var newReadyNode = ReadyNode{
            .nodeProto = nodeProto,
            .inputs = std.ArrayList(*ReadyTensor).init(allocator),
            .outputs = std.ArrayList(*ReadyTensor).init(allocator),
            .ready = false,
        };

        for (nodeProto.input) |input_name| { //for each input tensor in NodeProto

            //adding the readyTensor to the model
            try newReadyNode.inputs.append(if (tensorHashMap.getPtr(input_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            // std.debug.print("\n   added input {s} to node {s} ", .{ input_name, nodeProto.name.? });

        }
        for (nodeProto.output) |output_name| { //for each output tensor

            //adding the readyTensor to the model
            try newReadyNode.outputs.append(if (tensorHashMap.getPtr(output_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            // std.debug.print("\n   added output {s} to node {s} ", .{ output_name, nodeProto.name.? });
        }

        // -- COMPUTING THE OUTPUT SHAPE --
        try mathGen.compute_output_shape(&newReadyNode);

        return newReadyNode;
    }
};

// ----------------------- HASH MAP -----------------------
// Populates tensorHashMap with the tensors used in the onnx graph, where the key is the name of the tensor
pub fn populateReadyTensorHashMap(model: ModelOnnx) !void {
    const protoGraph = try if (model.graph) |graph| graph else error.GraphNotAvailable;

    //adding initializers to the hash map
    for (protoGraph.initializers) |init_ptr| {
        //create the readyTensor
        const readyTensor: ReadyTensor = try ReadyTensor.createInitializer(init_ptr);
        //add the readyTensor to the HashMap
        try tensorHashMap.put(readyTensor.name, readyTensor);
    }

    //adding all the nodes inputs and outputs
    for (protoGraph.nodes) |node| { //for each NodeProto in the GraphProto
        for (node.input) |input_name| {
            try addToTensorHashMap(input_name);
        }
        for (node.output) |output_name| {
            try addToTensorHashMap(output_name);
        }
    }
}

pub fn addToTensorHashMap(name: []const u8) !void {
    if (tensorHashMap.get(name)) |_| {
        std.debug.print("\n ----- Tensor {s} already present!! ", .{name});
        return;
    } else {

        //if input
        if (std.mem.indexOf(u8, try utils.getSanitizedName(name), "input") != null) {
            //add the readyTensor to the HashMap
            try tensorHashMap.put(name, try ReadyTensor.createInput(name));
            std.debug.print("\n ----- Tensor {s} is INPUT!! ", .{name});
            return;
        }
        //if input
        if (std.mem.indexOf(u8, try utils.getSanitizedName(name), "constant") != null) {
            //add the readyTensor to the HashMap
            try tensorHashMap.put(name, try ReadyTensor.createConstant(name));
            std.debug.print("\n ----- Tensor {s} is CONSTANT!! ", .{name});
            return;
        }

        //else default

        //add the readyTensor to the HashMap
        try tensorHashMap.put(name, try ReadyTensor.createLink(name));
        std.debug.print("\n ----- Tensor {s} is LINK!! ", .{name});
    }
}

// ----------------------- READY GRAPH -----------------------
// Creates a graph representation with all nodes in a ready-to-compute state
pub fn populateReadyGraph(model: ModelOnnx) !void {
    const graph = try if (model.graph) |graph| graph else error.GraphNotAvailable;

    for (graph.nodes) |node_ptr| { //for each NodeProto in the GraphProto

        try readyGraph.append(try ReadyNode.create(node_ptr));
    }
}
