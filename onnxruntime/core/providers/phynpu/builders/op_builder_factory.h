 
 

#pragma once
#include <string>
#include <memory>
#include <map>
#include <utility>
#include "impl/conv_builder.h"
#include "impl/convtranspose_builder.h"
#include "impl/pool_builder.h"
#include "impl/cast_builder.h"
#include "impl/unary_activations.h"
#include "impl/elementwise_builder.h"
#include "impl/concat_builder.h"
#include "impl/pad_builder.h"
#include "impl/argmax_builder.h"
#include "impl/batchnorm_builder.h"
#include "impl/flatten_builder.h"
#include "impl/clip_builder.h"
#include "impl/squeeze_builder.h"
#include "impl/reshape_builder.h"
#include "impl/transpose_builder.h"
#include "impl/gemm_builder.h"
#include "impl/softmax_builder.h"
#include "impl/sum_builder.h"
#include "impl/prelu_builder.h"
#include "impl/gather_builder.h"
#include "impl/slice_builder.h"
#include "impl/depthToSpace_builder.h"
#include "impl/spaceToDepth_builder.h"
#include "impl/resize_builder.h"
#include "impl/lrn_builder.h"
#include "impl/split_builder.h"
#include "impl/hardsigmoid_builder.h"
#include "impl/hardswish_builder.h"
// #include "impl/quantizeLinear.h"
// #include "impl/dequantizeLinear.h"


namespace onnxruntime {
namespace phytium_npu {
using createIOpBuildItemFunc = std::function<std::unique_ptr<IOpBuilder>()>;
using OpBuildItemType = std::map<std::string, std::unique_ptr<IOpBuilder>>;

static const std::map<std::string, createIOpBuildItemFunc> ops_list = {
#define REGISTER_OP_BUILDER(ONNX_NODE_TYPE, BUILDER_TYPE) \
  { ONNX_NODE_TYPE, [] { return std::make_unique<BUILDER_TYPE>(); }}
    REGISTER_OP_BUILDER("Conv", ConvOpBuilder),
    REGISTER_OP_BUILDER("ConvTranspose", ConvTransposeOpBuilder),
    REGISTER_OP_BUILDER("MaxPool", MaxPoolOpBuilder),
    REGISTER_OP_BUILDER("AveragePool", AveragePoolOpBuilder),
    REGISTER_OP_BUILDER("GlobalAveragePool", GlobalAvgPoolOpBuilder),
    REGISTER_OP_BUILDER("GlobalMaxPool", GlobalMaxPoolOpBuilder),
    REGISTER_OP_BUILDER("Cast", CastOpBuilder),
    REGISTER_OP_BUILDER("LeakyRelu", LeakyReluOpBuilder),
    REGISTER_OP_BUILDER("Relu", ReluOpBuilder),
    REGISTER_OP_BUILDER("Relu1", Relu1OpBuilder),
    REGISTER_OP_BUILDER("Relu6", Relu6OpBuilder),
    REGISTER_OP_BUILDER("Sigmoid", SigmoidOpBuilder),
    REGISTER_OP_BUILDER("Tanh", TanhOpBuilder),
    REGISTER_OP_BUILDER("Sqrt", SqrtOpBuilder),    
    REGISTER_OP_BUILDER("Exp", ExpOpBuilder),
    REGISTER_OP_BUILDER("Add", AddOpBuilder),
    REGISTER_OP_BUILDER("Sub", SubOpBuilder),
    REGISTER_OP_BUILDER("Mul", MulOpBuilder),
    REGISTER_OP_BUILDER("Div", DivOpBuilder),     
    REGISTER_OP_BUILDER("And", AndOpBuilder),
    REGISTER_OP_BUILDER("Sum", SumOpBuilder),
    REGISTER_OP_BUILDER("Or", OrOpBuilder),
    REGISTER_OP_BUILDER("Xor", XorOpBuilder),
    REGISTER_OP_BUILDER("Max", MaxOpBuilder),
    REGISTER_OP_BUILDER("Min", MinOpBuilder),
    REGISTER_OP_BUILDER("MatMul",MatmulOpBuilder),
    REGISTER_OP_BUILDER("PRelu", PReluOpBuilder),
    REGISTER_OP_BUILDER("Concat", ConcatOpBuilder),
    REGISTER_OP_BUILDER("Pad", PadOpBuilder),
    REGISTER_OP_BUILDER("ArgMax", ArgMaxOpBuilder),
    REGISTER_OP_BUILDER("ReduceMean", ReduceMeanOpBuilder),
    REGISTER_OP_BUILDER("BatchNormalization", BatchNormOpBuilder),
    REGISTER_OP_BUILDER("Flatten", FlattenOpBuilder),
    REGISTER_OP_BUILDER("Clip", ClipOpBuilder),
    REGISTER_OP_BUILDER("Reshape", ReshapeOpBuilder),
    REGISTER_OP_BUILDER("Transpose", TransposeOpBuilder),
    REGISTER_OP_BUILDER("Squeeze", SqueezeOpBuilder),
    REGISTER_OP_BUILDER("Gemm", GemmOpBuilder),      
    REGISTER_OP_BUILDER("Gather", GatherOpBuilder),
    REGISTER_OP_BUILDER("DepthToSpace", DepthToSpaceOpBuilder),
    REGISTER_OP_BUILDER("SpaceToDepth", SpaceToDepthOpBuilder),
    REGISTER_OP_BUILDER("Slice", SliceOpBuilder),
    REGISTER_OP_BUILDER("Resize", ResizeOpBuilder),
    REGISTER_OP_BUILDER("LRN", LRNOpBuilder),
    REGISTER_OP_BUILDER("Split", SplitOpBuilder),
    REGISTER_OP_BUILDER("HardSigmoid", HardSigmoidOpBuilder),
    REGISTER_OP_BUILDER("HardSwish", HardSwishOpBuilder),
    // REGISTER_OP_BUILDER("QuantizeLinear", QuantizeLinearOpBuilder),
    // REGISTER_OP_BUILDER("DequantizeLinear", DequantizeLinearOpBuilder),


#undef REGISTER_OP_BUILDER
};

template <typename T>
struct OpBuildConstructor {
  T supported_ops;
  OpBuildConstructor(
      const std::map<typename T::key_type, createIOpBuildItemFunc> ops_list) {
    LOGS_DEFAULT(INFO) << "Initialize supported ops";
    for (const auto& kv : ops_list) {
      supported_ops.insert(std::make_pair(kv.first, kv.second()));
    }
  }
};

inline const OpBuildItemType& SupportedBuiltinOps() {
  static OpBuildConstructor<OpBuildItemType> op_build(ops_list);
  return op_build.supported_ops;
}

}  // namespace phytium_npu
}  // namespace onnxruntime
