 
//Data: 2026 -02-18

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class HardSigmoidOpBuilder : public BaseOpBuilder {
 public:
  std::vector<phydnn_tensor> HandleBuildOp(
      phytium_npu::GraphPhynpuEP* graph_ep,
      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
      const NodeUnit& node_unit,
      phydnn_quant_param in_quant,
      phydnn_quant_param out_quant) override {
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_tensor output_tensor;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc, desc, out_desc;
    long input_size = 1;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    auto buffer_size = phydnnGetDescriptorSize(&input_desc, &error_code);
    desc.type = PHYDNN_TYPE_F32;
    desc.dimensions = input_desc.dimensions;
    for(unsigned int i=0; i<input_desc.dimensions; i++){
      input_size *= input_desc.size[i];
      desc.size[i] = input_desc.size[i];
      LOGS_DEFAULT(INFO)<<"HardSigmoid index:"<<i<<", desc.size:"<<desc.size[i]<<", input type:"<<input_desc.type;
    }

    // get alpha and beta params
    float alpha = 0.2f, beta = 0.5f;
    const auto& node = node_unit.GetNode();
    NodeAttrHelper helper(node);
    bool has_alpha = helper.HasAttr("alpha");
    bool has_beta  = helper.HasAttr("beta");
    if(has_alpha) {
      alpha = helper.Get("alpha", 0.2f);
    }
    if(has_beta) {
      beta  = helper.Get("beta", 0.5f);
    }
    LOGS_DEFAULT(INFO)<<"HardSigmoid alpha:"<<alpha<<", beta:"<<beta<<", size:"<<input_size<<", buffer_size:"<<buffer_size;

    if(input_desc.type >= PHYDNN_TYPE_Q_I8) {
      LOGS_DEFAULT(INFO)<<"HardSigmoidOpBuilder QuantizedHardSigmoid.";
      // 定义输入范围和 LUT 大小
      const size_t lut_size = 257;

      // 生成 LUT
      std::vector<int32_t> Lut;
      bool is_quant = true;
      if (input_desc.type == PHYDNN_TYPE_Q_U8 || input_desc.type == PHYDNN_TYPE_QPA_U8) {
        is_quant = false;
      }
      Lut = GenerateQuantizedHardSigmoidLUT(out_quant, in_quant, lut_size, is_quant, alpha, beta);

      // 创建 LUT 查找操作
      output_tensor = phydnnNetworkInterpolatedLookupOp(
                                      graph_ep->GetGraph(),
                                      input_tensor,
                                      Lut.data(),
                                      lut_size,
                                      &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp!";
      }
    } else {
      /*y = max(0, min(1, alpha * x + beta))*/
      std::vector<float> data;
      for(int i=0; i<input_size; i++) {
        data.push_back(alpha);
      }

      std::vector<float *> alpha_data;    // 考虑优化，释放内存。目前是不能在这里直接去释放的
      alpha_data.push_back((float*)calloc(1, buffer_size));
      memcpy(alpha_data.back(), data.data(), buffer_size);

      LOGS_DEFAULT(INFO)<<"HardSigmoidOpBuilder alpha:"<<alpha<<", size:"<<input_size<<", buffer_size:"<<buffer_size;
      auto alpha_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &desc, alpha_data.back(), &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
      }
      output_tensor = MMBinaryBuildOp(graph_ep, input_tensor, alpha_tensor, PHYDNN_OPERATION_MUL, node_unit);  

      if (has_beta) {
        data.clear();
        for(int i=0; i<input_size; i++) {
          data.push_back(beta);
        }

        std::vector<float *> beta_data;
        beta_data.push_back((float*)calloc(1, buffer_size));
        memcpy(beta_data.back(), data.data(), buffer_size);
        phydnn_tensor beta_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &desc, beta_data.back(), &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
        }

        LOGS_DEFAULT(INFO)<<"HardSigmoidOpBuilder beta:"<<beta<<", size:"<<input_size;
        output_tensor = phydnnNetworkBinaryOp(                                                 
            graph_ep->GetGraph(), output_tensor, beta_tensor, PHYDNN_OPERATION_ADD, &error_code);     
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp!";
        }
      } 

      // create zeros and ones
      std::vector<float> zeros;
      std::vector<float> ones;
      for(int i=0; i<input_size; i++) {
        zeros.push_back(0.0f);
        ones.push_back(1.0f);
      }
      std::vector<float *> zero_data;
      std::vector<float *> one_data;
      zero_data.push_back((float*)calloc(1, buffer_size));
      memcpy(zero_data.back(), zeros.data(), buffer_size);
      one_data.push_back((float*)calloc(1, buffer_size));
      memcpy(one_data.back(), ones.data(), buffer_size);

      phydnn_tensor zero_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &desc, zero_data.back(), &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
      }
      phydnn_tensor one_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &desc, one_data.back(), &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
      }

      output_tensor = phydnnNetworkBinaryOp(                                                 
          graph_ep->GetGraph(), output_tensor, one_tensor, PHYDNN_OPERATION_MIN, &error_code);      
      if(error_code != PHYDNN_SUCCESS) {                                                          
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp OPERATION_MIN, error_code:" << error_code; 
        return output_tensors;                                                                             
      }  

      output_tensor = phydnnNetworkBinaryOp(                                                 
          graph_ep->GetGraph(), output_tensor, zero_tensor, PHYDNN_OPERATION_MAX, &error_code);  
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp OPERATION_MAX, error_code:" << error_code; 
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

};  // HardSigmoidOpBuilder
}  // namespace phytium_npu
}  // namespace onnxruntime
