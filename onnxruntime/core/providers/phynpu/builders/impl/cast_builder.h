 
//Data: 2026 -01-16

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {
class CastOpBuilder : public BaseOpBuilder {
 public:
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {   
    NodeAttrHelper helper(*node);
    const auto to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    phydnn_type operand_type = GetPhyDnnType(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(to_type));     
    LOGS_DEFAULT(INFO) << "Cast Op get type:"<<operand_type;                                                     
    if(operand_type == PHYDNN_TYPE_U8  || operand_type == PHYDNN_TYPE_I8  || 
       operand_type >= PHYDNN_TYPE_Q_I8) {
      return true;
    }                                                                                       
    return false;                                                                                
  }   

  std::vector<phydnn_tensor> HandleBuildOp(
            phytium_npu::GraphPhynpuEP* graph_ep,
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
            const NodeUnit& node_unit,
            phydnn_quant_param in_quant,
            phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Cast Op.";
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc;
    bool bQuantFlag = false;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    if(input_desc.type >= PHYDNN_TYPE_Q_I8) {
      bQuantFlag = true;
    }
   
    // We already checked the "to" type in IsOpSupportedImpl.
    NodeAttrHelper helper(node_unit.GetNode());
    const auto to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    phydnn_type operand_type = GetPhyDnnType(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(to_type), bQuantFlag);
    LOGS_DEFAULT(INFO) <<"CreateConvertCast input_desc.type"<<input_desc.type<<" convert dtype:"<<operand_type; 

    phydnn_tensor output_tensor;
    if(bQuantFlag){
      output_tensor =
        phydnnNetworkCastOp(graph_ep->GetGraph(), input_tensor, operand_type, &input_desc.quant_param, &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkCastOp!";
        return output_tensors;
      }
    } else {
      output_tensor =
        phydnnNetworkCastOp(graph_ep->GetGraph(), input_tensor, operand_type, nullptr, &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkCastOp!";
        return output_tensors;
      }
    }
    
    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }
    
    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }
};

}  // namespace phytium_npu
}  // namespace onnxruntime
