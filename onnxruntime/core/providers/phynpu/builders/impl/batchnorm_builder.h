 
 

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class BatchNormOpBuilder : public BaseOpBuilder {
 public:  
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    for (auto input : node->InputDefs()) {                                                      
      if (*input->Type() == "tensor(int8)" || *input->Type() == "tensor(uint8)") {                                                  
        LOGS_DEFAULT(INFO) << "Int8/UInt8 quant type is not supported as BatchNorm ops."; 
        return false;                                                                           
      }                                                                                         
    }                                                                                           
    return true;                                                                                
  }  

  std::vector<phydnn_tensor> HandleBuildOp(
      phytium_npu::GraphPhynpuEP* graph_ep,
      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
      const NodeUnit& node_unit,
      phydnn_quant_param in_quant,
      phydnn_quant_param out_quant) override {
    std::vector<phydnn_tensor> output_tensors;    
    auto input_cout = inputs.size();
    if(input_cout < 5) {
      LOGS_DEFAULT(ERROR) << "BatchNormOpBuilder with input tensor size < 5 is not supported.";
      return output_tensors;
    }    

    phydnn_tensor input_tensor    = *inputs[0];
    phydnn_tensor gamma_tensor    = *inputs[1];
    phydnn_tensor beta_tensor     = *inputs[2];
    phydnn_tensor mean_tensor     = *inputs[3];
    phydnn_tensor variance_tensor = *inputs[4];

    NodeAttrHelper helper(node_unit.GetNode());
    auto epsilon = helper.Get("epsilon", 1e-5f);    

    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc, gamma_desc, out_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    phydnnGetTensorDescriptor(gamma_tensor, &gamma_desc);
    LOGS_DEFAULT(INFO) <<"BatchNormOpBuilder input_desc.type:"<<input_desc.type;

    // epsilon tensor
    phydnn_tensor epsilon_tensor = phydnnNetworkFixedScalar(graph_ep->GetGraph(), (double)epsilon, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedScalar, error_code:" << error_code;
      return output_tensors;
    } 

    for(int i=0; i<input_desc.dimensions; i++) {
      LOGS_DEFAULT(INFO)<<"CreateBatchNormLayer input_desc.size:"<<input_desc.size[i];
      //only one axis  
      if(gamma_desc.size[0] == input_desc.size[i]){
        continue;
      }

      if(gamma_tensor) {
        gamma_tensor = phydnnNetworkBroadcastOp(
          graph_ep->GetGraph(), gamma_tensor, i, input_desc.size[i], &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBroadcastOp, error_code:" << error_code;
          return output_tensors;
        } 
      }

      if(beta_tensor) {
        beta_tensor = phydnnNetworkBroadcastOp(
          graph_ep->GetGraph(), beta_tensor, i, input_desc.size[i], &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBroadcastOp, error_code:" << error_code;
          return output_tensors;
        } 
      }

      mean_tensor = phydnnNetworkBroadcastOp(
        graph_ep->GetGraph(), mean_tensor, i, input_desc.size[i], &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBroadcastOp, error_code:" << error_code;
        return output_tensors;
      } 
      
      variance_tensor = phydnnNetworkBroadcastOp(
        graph_ep->GetGraph(), variance_tensor, i, input_desc.size[i], &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBroadcastOp, error_code:" << error_code;
        return output_tensors;
      } 
    }

    phydnn_tensor_descriptor mean_desc;
    phydnnGetTensorDescriptor(mean_tensor, &mean_desc);
    LOGS_DEFAULT(INFO)<<"CreateBatchNormLayer mean_desc.dimensions:"<<mean_desc.dimensions;
    for(int i=0; i<mean_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"CreateBatchNormLayer mean_desc.size:"<<mean_desc.size[i];
    }

    //type
    if(input_desc.type != mean_desc.type) {
      LOGS_DEFAULT(INFO)<<"input_desc type:"<<input_desc.type<<" mean_desc type:"<<mean_desc.type;
      input_tensor = phydnnNetworkCastOp(
                  graph_ep->GetGraph(), input_tensor, mean_desc.type, nullptr, &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
        return output_tensors;
      }  
    }

    phydnn_tensor output_tensor = phydnnNetworkBinaryOp(
        graph_ep->GetGraph(), input_tensor, mean_tensor, PHYDNN_OPERATION_SUB, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code;
      return output_tensors;
    } 

    phydnn_tensor denominator = phydnnNetworkBinaryOp(
        graph_ep->GetGraph(), variance_tensor, epsilon_tensor, PHYDNN_OPERATION_ADD, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code;
      return output_tensors;
    }

    denominator = phydnnNetworkUnaryOp(
        graph_ep->GetGraph(), denominator, PHYDNN_OPERATION_SQRT, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code;
      return output_tensors;
    }

    // type
    phydnn_tensor_descriptor denominator_desc;
    phydnnGetTensorDescriptor(output_tensor, &out_desc);
    phydnnGetTensorDescriptor(denominator, &denominator_desc);
    if(out_desc.type != denominator_desc.type) {
      LOGS_DEFAULT(INFO)<<"out_desc type:"<<out_desc.type<<" denominator_desc type:"<<denominator_desc.type;
      output_tensor = phydnnNetworkCastOp(
                  graph_ep->GetGraph(), output_tensor, denominator_desc.type, nullptr, &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
        return output_tensors;
      }  
    }

    output_tensor = phydnnNetworkBinaryOp(
        graph_ep->GetGraph(), output_tensor, denominator, PHYDNN_OPERATION_DIV, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code;
      return output_tensors;
    }

    if(gamma_tensor) {
      // type
      phydnn_tensor_descriptor gamma_desc;
      phydnnGetTensorDescriptor(output_tensor, &out_desc);
      phydnnGetTensorDescriptor(gamma_tensor, &gamma_desc);
      if(out_desc.type != gamma_desc.type) {
        LOGS_DEFAULT(INFO)<<"out_desc type:"<<out_desc.type<<" gamma_desc type:"<<gamma_desc.type;
        output_tensor = phydnnNetworkCastOp(
                    graph_ep->GetGraph(), output_tensor, gamma_desc.type, &gamma_desc.quant_param, &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
          return output_tensors;
        }  
      }
      
      output_tensor = phydnnNetworkBinaryOp(
        graph_ep->GetGraph(), output_tensor, gamma_tensor, PHYDNN_OPERATION_MUL, &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code;
        return output_tensors;
      }
    }

    if(beta_tensor) {
       // type
      phydnn_tensor_descriptor beta_desc;
      phydnnGetTensorDescriptor(output_tensor, &out_desc);
      phydnnGetTensorDescriptor(beta_tensor, &beta_desc);
      if(out_desc.type != beta_desc.type) {
        LOGS_DEFAULT(INFO)<<"out_desc type:"<<out_desc.type<<" beta_desc type:"<<beta_desc.type;
        output_tensor = phydnnNetworkCastOp(
                    graph_ep->GetGraph(), output_tensor, beta_desc.type, &beta_desc.quant_param, &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
          return output_tensors;
        }  
      }

      output_tensor = phydnnNetworkBinaryOp(
        graph_ep->GetGraph(), output_tensor, beta_tensor, PHYDNN_OPERATION_ADD, &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code;
        return output_tensors;
      }
    }
    
    //
    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions;i++){
      LOGS_DEFAULT(INFO)<<"get result output_desc.size:"<<output_desc.size[i];
    }

    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }         
};

}  // namespace phytium_npu
}  // namespace onnxruntime
