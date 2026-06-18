 
//Data: 2026 -02-16

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class DepthToSpaceOpBuilder : public BaseOpBuilder {
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
    phydnn_tensor_descriptor input_desc, shape_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    phydnnGetTensorDescriptor(input_tensor, &shape_desc);
    LOGS_DEFAULT(INFO) <<"DepthToSpaceOpBuilder input.type"<<input_desc.type<<", dims:"<<input_desc.dimensions; 

    NodeAttrHelper helper(node_unit);
    int blocksize = helper.Get("blocksize", 1);  

    if(node_unit.SinceVersion() < 11) {
      output_tensor = phydnnNetworkDepthToSpaceOp(graph_ep->GetGraph(), 
                                                    input_tensor, 
                                                    blocksize, 
                                                    &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkDepthToSpaceOp!";
        return output_tensors;
      } 
    } else {
      // input tensor mode=="DCR" layout: NCHW ; mode=="CRD" layout: NHWC
      const auto mode = helper.Get("mode", "DCR");
      LOGS_DEFAULT(INFO) <<"DepthToSpaceOpBuilder mode:"<<mode<<", blocksize:"<<blocksize;
      if (mode == "DCR") {
        shape_desc.dimensions = 6;
        shape_desc.size[0] = input_desc.size[0];
        shape_desc.size[1] = blocksize;
        shape_desc.size[2] = blocksize;
        shape_desc.size[3] = input_desc.size[1] / (blocksize * blocksize);
        shape_desc.size[4] = input_desc.size[2];
        shape_desc.size[5] = input_desc.size[3];
        for (unsigned i = 0; i < shape_desc.dimensions; i++) {
          LOGS_DEFAULT(INFO) <<"DepthToSpaceOpBuilder shape_desc index:"<<i<<", size:"<<shape_desc.size[i];
        }
        auto output_tensor_A =
        phydnnNetworkReshapeOp(graph_ep->GetGraph(), input_tensor, &shape_desc, &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReshapeOp!";
          return output_tensors;
        } 

        std::vector<int> order = {0, 3, 4, 1, 5, 2};
        auto output_tensor_B = phydnnNetworkTransposeOp(graph_ep->GetGraph(),
                                                output_tensor_A,
                                                order.data(),
                                                &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkTransposeOp!";
          return output_tensors;
        } 

        shape_desc.dimensions = 4;
        shape_desc.size[0] = input_desc.size[0];
        shape_desc.size[1] = input_desc.size[1] / (blocksize * blocksize);
        shape_desc.size[2] = input_desc.size[2] * blocksize;
        shape_desc.size[3] = input_desc.size[3] * blocksize;
        for (unsigned i = 0; i < shape_desc.dimensions; i++) {
          LOGS_DEFAULT(INFO) <<"DepthToSpaceOpBuilder shape_desc index:"<<i<<", size:"<<shape_desc.size[i];
        }

        output_tensor =
        phydnnNetworkReshapeOp(graph_ep->GetGraph(), output_tensor_B, &shape_desc, &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReshapeOp!";
          return output_tensors;
        } 
      } else {
        shape_desc.dimensions = 6;
        shape_desc.size[0] = input_desc.size[0];
        shape_desc.size[1] = input_desc.size[1] / (blocksize * blocksize);
        shape_desc.size[2] = blocksize;
        shape_desc.size[3] = blocksize;
        shape_desc.size[4] = input_desc.size[2];
        shape_desc.size[5] = input_desc.size[3];
        for (unsigned i = 0; i < shape_desc.dimensions; i++) {
          LOGS_DEFAULT(INFO) <<"DepthToSpaceOpBuilder shape_desc index:"<<i<<", size:"<<shape_desc.size[i];
        }
        auto output_tensor_A =
        phydnnNetworkReshapeOp(graph_ep->GetGraph(), input_tensor, &shape_desc, &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReshapeOp!";
          return output_tensors;
        } 

        std::vector<int> order = {0, 1, 4, 2, 5, 3};
        auto output_tensor_B = phydnnNetworkTransposeOp(graph_ep->GetGraph(),
                                                output_tensor_A,
                                                order.data(),
                                                &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkTransposeOp!";
          return output_tensors;
        } 

        shape_desc.dimensions = 4;
        shape_desc.size[0] = input_desc.size[0];
        shape_desc.size[1] = input_desc.size[1] / (blocksize * blocksize);
        shape_desc.size[2] = input_desc.size[2] * blocksize;
        shape_desc.size[3] = input_desc.size[3] * blocksize;
        for (unsigned i = 0; i < shape_desc.dimensions; i++) {
          LOGS_DEFAULT(INFO) <<"DepthToSpaceOpBuilder shape_desc index:"<<i<<", size:"<<shape_desc.size[i];
        }

        output_tensor =
        phydnnNetworkReshapeOp(graph_ep->GetGraph(), output_tensor_B, &shape_desc, &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReshapeOp!";
          return output_tensors;
        } 
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
};  // DepthToSpaceOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
