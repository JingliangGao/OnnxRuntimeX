 
//Data: 2026 -01-23

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();

    NodeAttrHelper helper(*node);
    const bool allow_zero = helper.Get("allowzero", 0) == 1;
    auto& perm_tensor_proto = *graph_viewer.GetConstantInitializer(input_defs[1]->Name(), true);
    std::vector<int64_t> perm(perm_tensor_proto.dims()[0]);
    auto status = onnxruntime::utils::UnpackTensor(
        perm_tensor_proto,
        perm_tensor_proto.has_raw_data() ? perm_tensor_proto.raw_data().data() : nullptr,
        perm_tensor_proto.has_raw_data() ? perm_tensor_proto.raw_data().size() : 0,
        perm.data(), perm.size());

    // Check if perm has any 0's when allow zero is enabled.
    if (allow_zero && std::find(perm.begin(), perm.end(), 0L) != perm.end()) {
      LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 as dimension when allowzero is enabled";
      return false;
    }

    return true;
  }

  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc, shape_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO) <<"ReshapeOpBuilder input_desc.type"<<input_desc.type;  

    const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
    const auto& shape_tensor = *initializers.at(node_unit.GetNode().InputDefs()[1]->Name());
    Initializer unpacked_tensor(shape_tensor);
    auto raw_shape = unpacked_tensor.DataAsSpan<int64_t>();
    const auto size = SafeInt<size_t>(shape_tensor.dims()[0]);

    std::vector<int32_t> shape(size);
    for (uint32_t i = 0; i < size; i++) {
      int32_t dim = SafeInt<int32_t>(raw_shape[i]);
      // NNAPI reshape does not support 0 as dimension
      shape[i] = dim == 0 ? input_desc.size[i] : dim;
    }

    phydnnGetTensorDescriptor(input_tensor, &shape_desc);
    shape_desc.type = input_desc.type;
    shape_desc.dimensions = size;
    for (unsigned i = 0; i < size; i++) {
      shape_desc.size[i] = shape[i];
    }
    auto output_tensor =
        phydnnNetworkReshapeOp(graph_ep->GetGraph(), input_tensor, &shape_desc, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReshapeOp!";
      return output_tensors;
    } 

    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions;i++){
      LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }

    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }   
};  // ReshapeOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
