 
//Data: 2026 -02-17

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {
class GemmOpBuilder : public BaseOpBuilder {
 public:
  // bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
  //                    const Node* node) const override {
  //   auto input_defs = node->InputDefs();         
  //   bool bConstant = graph_viewer.IsConstantInitializer(input_defs[1]->Name(), true);    
  //   LOGS_DEFAULT(INFO)<<"GemmOpBuilder the weights is Constant:"<<bConstant;             
  //   return bConstant;
  // }

  std::vector<phydnn_tensor> HandleBuildOp(
                phytium_npu::GraphPhynpuEP* graph_ep,
                std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
                const NodeUnit& node_unit,
                phydnn_quant_param in_quant,
                phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Gemm Op.";
    auto inputA_tensor = *inputs[0];
    auto inputB_tensor = *inputs[1];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_tensor inputC_tensor = nullptr;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor inputA_desc, inputB_desc;
    phydnnGetTensorDescriptor(inputA_tensor, &inputA_desc);
    phydnnGetTensorDescriptor(inputB_tensor, &inputB_desc);
    uint32_t inputA_size = 1, inputC_size = 1;
    for(unsigned int i=0; i<inputA_desc.dimensions; i++){
      inputA_size *= inputA_desc.size[i];
      LOGS_DEFAULT(INFO)<<"GemmOpBuilder get result: "<<i<<" inputA_desc.size:"<<inputA_desc.size[i];
    }
    for(unsigned int i=0; i<inputB_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"GemmOpBuilder get result: "<<i<<" inputB_desc.size:"<<inputB_desc.size[i];
    }

    /**  
    Compute  Y = alpha * A' * B' + beta * C 
    A' = transpose(A) if transA else A 。 
    B' = transpose(B) if transB else B 。 
    **/
    // get attributes
    NodeAttrHelper helper(node_unit.GetNode());
    auto trans_A = helper.Get("transA", 0);
    auto trans_B = helper.Get("transB", 0);
    const bool has_alpha = (helper.Get("alpha", 1.0f) != 1.0);
    const bool has_beta = (helper.Get("beta", 1.0f) != 1.0);
    const bool has_C = (inputs.size() == 3);
    LOGS_DEFAULT(INFO)<<"GemmOpBuilder has_alpha:"<<has_alpha<<", has_beta:"<<has_beta<<", has_C: "<<has_C;

    // alpha * A
    if (has_alpha) {
      auto alpha = helper.Get("alpha", 1.0f);
      std::vector<float> alpha_data;
      for(int i=0; i<inputA_size; i++) {
        alpha_data.push_back(alpha);
      }
      LOGS_DEFAULT(INFO)<<"GemmOpBuilder alpha:"<<alpha<<", size:"<<inputA_size;
      phydnn_tensor alpha_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &inputA_desc, alpha_data.data(), &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
      }
      inputA_tensor = MMBinaryBuildOp(graph_ep, inputA_tensor, alpha_tensor, PHYDNN_OPERATION_MUL, node_unit);  
    }
    
    // beta * C
    if(has_C) {
      inputC_tensor = *inputs[2];
      phydnn_tensor_descriptor inputC_desc;
      phydnnGetTensorDescriptor(inputC_tensor, &inputC_desc);
      for(unsigned int i=0; i<inputC_desc.dimensions; i++){
        inputC_size *= inputC_desc.size[i];
      }
      if (has_beta) {
        auto beta = helper.Get("beta", 1.0f);
        std::vector<float> beta_data;
        for(int i=0; i<inputC_size; i++) {
          beta_data.push_back(beta);
        }
        phydnn_tensor beta_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &inputC_desc, beta_data.data(), &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
        }
        LOGS_DEFAULT(INFO)<<"GemmOpBuilder beta:"<<beta<<", size:"<<inputC_size;
        inputC_tensor = MMBinaryBuildOp(graph_ep, inputC_tensor, beta_tensor, PHYDNN_OPERATION_MUL, node_unit);
      } 
    }

    // transpose A
    if (trans_A) {
      std::vector<int> order;
      int num_axes = inputA_desc.dimensions;
      for (int i = num_axes - 1; i > -1; --i) {
        order.push_back(i);
      }
      inputA_tensor = phydnnNetworkTransposeOp(graph_ep->GetGraph(),
                                              inputA_tensor,
                                              order.data(),
                                              &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkTransposeOp!";
        return output_tensors;
      } 
    }

    // transpose B
    if (trans_B) {
      std::vector<int> order;
      int num_axes = inputB_desc.dimensions;
      for (int i = num_axes - 1; i > -1; --i) {
        order.push_back(i);
      }
      inputB_tensor = phydnnNetworkTransposeOp(graph_ep->GetGraph(),
                                              inputB_tensor,
                                              order.data(),
                                              &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkTransposeOp!";
        return output_tensors;
      } 
    }

    // gemm = A*B + C
    // Flatten the input tensor from dimension 2 (Only supports flatten_dim=1)
    phydnn_tensor_descriptor input_desc;
    phydnnGetTensorDescriptor(inputA_tensor, &input_desc);
    for (unsigned i = 2; i < input_desc.dimensions; i++) {
      input_desc.size[1] *= input_desc.size[i];
    }
    input_desc.dimensions = 2;
    inputA_tensor =
        phydnnNetworkReshapeOp(graph_ep->GetGraph(), inputA_tensor, &input_desc, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReshapeOp!";
      return output_tensors;
    }    
    
    auto output_tensor = MMBinaryBuildOp(graph_ep, inputA_tensor, inputB_tensor, PHYDNN_OPERATION_MATMUL, node_unit);
    if(has_C) {
      //type
      phydnn_tensor_descriptor output_desc, bias_desc;
      phydnnGetTensorDescriptor(output_tensor, &output_desc);
      phydnnGetTensorDescriptor(inputC_tensor, &bias_desc);
      if(output_desc.type != bias_desc.type) {
        LOGS_DEFAULT(INFO)<<"output_desc type:"<<output_desc.type<<", bias_desc type:"<<bias_desc.type;
        output_tensor = phydnnNetworkCastOp(
                    graph_ep->GetGraph(), output_tensor, bias_desc.type, nullptr, &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
          return output_tensors;
        }  
      }

      output_tensor = phydnnNetworkBinaryOp(graph_ep->GetGraph(),
                                          output_tensor,
                                          inputC_tensor,
                                          PHYDNN_OPERATION_ADD,
                                          &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReLUOp!";
        return output_tensors;
      }
    }

    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions; i++) {
      LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }
    
    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }
};

}  // namespace phytium_npu
}  // namespace onnxruntime
