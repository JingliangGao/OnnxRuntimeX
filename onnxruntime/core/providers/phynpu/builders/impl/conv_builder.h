 
 

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class ConvOpBuilder : public BaseOpBuilder {

  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    (void)graph_viewer; 
    auto input_defs = node->InputDefs();
    auto shape = GetTensorShape(*input_defs[0]);
    if (shape.NumDimensions() >= 5) {
      LOGS_DEFAULT(ERROR) << "nosupport conv2d yet.";
      return false;
    }

    if (shape.NumDimensions() == 4) {
      auto length = shape[shape.NumDimensions() -2];
      auto width  = shape[shape.NumDimensions() -1];
      if(length > 8192) {
        LOGS_DEFAULT(ERROR) << "conv2d length is:"<<length<<" over the max_length 8192.";
        LOGS_DEFAULT(ERROR) << "nosupport conv2d yet.";
        return false;
      }
      if(width > 2048) {
        LOGS_DEFAULT(ERROR) << "conv2d width is:"<<width<<" over the max_width 2048.";
        LOGS_DEFAULT(ERROR) << "nosupport conv2d yet.";
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
    auto input_tensor   = *inputs[0];
    auto weights_tensor = *inputs[1];
    std::vector<phydnn_tensor> output_tensors;
    // Add bias if present.
    bool bais_flag = inputs.size() > 2 ? true : false;
    
    phydnn_err_code error_code;
    phydnn_tensor output_tensor;
    phydnn_tensor_descriptor input_desc, weights_desc;
    phydnn_tensor_descriptor output_desc, bias_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    phydnnGetTensorDescriptor(weights_tensor, &weights_desc);

    // get attrs
    NodeAttrHelper helper(node_unit.GetNode());
    auto auto_pad  = helper.Get("auto_pad", std::string(""));
    auto group     = helper.Get("group", static_cast<uint32_t>(1));
    auto strides   = helper.Get("strides", std::vector<int>{1, 1});
    auto pads      = helper.Get("pads", std::vector<int>{0, 0, 0, 0});
    auto dilations = helper.Get("dilations", std::vector<int>{1, 1});
    
    auto pad_height_top    = pads[0];
    auto pad_height_bottom = pads[1];
    auto pad_width_left    = pads[2];
    auto pad_width_right   = pads[3];
    auto stride_height     = strides[0];                                      
    auto stride_width      = strides[1]; 
    auto dilation_height   = dilations[0];                               
    auto dilation_width    = dilations[1]; 
    // layout NCHW
    auto filter_height     = weights_desc.size[2];
    auto filter_width      = weights_desc.size[3];
    
    // is depthwise_mode
    auto OChannel_idx = input_desc.dimensions - 1;
    bool is_depthwise_mode = group != 1 && group == input_desc.size[1]; // NCHW layout -> channels
    LOGS_DEFAULT(INFO)<<"CreateConvolutionLayer group:"<<group<<", input_desc size:"<<input_desc.size[1]<<", idx:"<<OChannel_idx;

    // set auto_pad shape
    if (auto_pad != "NOTSET") { 
      UpdateConv2DPadAndDilation(
        input_desc.size[2],
        filter_height,
        auto_pad,
        &pad_height_top,
        &pad_height_bottom,
        stride_height,
        &dilation_height);

      UpdateConv2DPadAndDilation(
        input_desc.size[3],
        filter_width,
        auto_pad,
        &pad_width_left,
        &pad_width_right,
        stride_width,
        &dilation_width);   
    }

    unsigned int ksizes[2]    = {static_cast<unsigned int>(filter_height), static_cast<unsigned int>(filter_width)};
    unsigned int stride_t[2]  = {static_cast<unsigned int>(stride_height), static_cast<unsigned int>(stride_width)};
    unsigned int pad_begin[2] = {static_cast<unsigned int>(pad_height_top), static_cast<unsigned int>(pad_width_left)};
    unsigned int pad_end[2]   = {static_cast<unsigned int>(pad_height_bottom), static_cast<unsigned int>(pad_width_right)};
    unsigned int dilation[2]  = {static_cast<unsigned int>(dilation_height), static_cast<unsigned int>(dilation_width)};

    // create convolution layer
    if (is_depthwise_mode) {
      // Transpose weight tensor
      int order[4] = {1, 0, 2, 3};
      weights_tensor =
          phydnnNetworkTransposeOp(graph_ep->GetGraph(), weights_tensor, order, &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkTransposeOp, error_code:" << error_code;
        return output_tensors;
      } 
      LOGS_DEFAULT(INFO)<<"CreateConvolutionLayer depth input_desc type:"<<input_desc.type<<", weights_desc type:"<<weights_desc.type;
      for(unsigned int i=0; i<input_desc.dimensions; i++){
        LOGS_DEFAULT(INFO)<<"CreateConvolutionLayer depth input_desc.size:"<<input_desc.size[i];
      }

      if(input_desc.type != weights_desc.type) {
        LOGS_DEFAULT(INFO)<<"set weights_tensor type:"<<input_desc.type;
        weights_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                    weights_tensor,
                                    input_desc.type,
                                    &input_desc.quant_param,
                                    &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call CreateConvolutionLayer, error_code:" << error_code;
          return output_tensors;
        } 
      }

      output_tensor = phydnnNetworkDepthConvolution2dOp_v2(graph_ep->GetGraph(),
                                                          input_tensor,
                                                          weights_tensor,
                                                          stride_t,
                                                          pad_begin,
                                                          pad_end,
                                                          dilation,
                                                          &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkDepthConvolution2dOp_v2, error_code:" << error_code;
        return output_tensors;
      } 

    } else if(group == 1) {
      //set type
      LOGS_DEFAULT(INFO)<<"CreateConvolutionLayer convert2d input_desc type:"<<input_desc.type<<", weights_desc type:"<<weights_desc.type;
      for(unsigned int i=0; i<input_desc.dimensions; i++){
        LOGS_DEFAULT(INFO)<<"CreateConvolutionLayer convert2d input_desc.size:"<<input_desc.size[i];
      }
      for(unsigned int i=0; i<weights_desc.dimensions; i++){
        LOGS_DEFAULT(INFO)<<"CreateConvolutionLayer convert2d weights_desc.size:"<<weights_desc.size[i];
      }


      if(input_desc.type != weights_desc.type) {
        LOGS_DEFAULT(INFO)<<"set weights_tensor type:"<<input_desc.type;
        weights_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                    weights_tensor,
                                    input_desc.type,
                                    &input_desc.quant_param,
                                    &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call CreateConvolutionLayer, error_code:" << error_code;
          return output_tensors;
        } 
      }

      output_tensor = phydnnNetworkConvolution2dOp_v2(graph_ep->GetGraph(),
                                                      input_tensor,
                                                      weights_tensor,
                                                      stride_t,
                                                      pad_begin,
                                                      pad_end,
                                                      dilation,
                                                      &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkConvolution2dOp_v2, error_code:" << error_code;
        return output_tensors;
      }     
    } else {
      if(input_desc.type != weights_desc.type) {
        LOGS_DEFAULT(INFO)<<"set weights_tensor type:"<<input_desc.type;
        weights_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                    weights_tensor,
                                    input_desc.type,
                                    &input_desc.quant_param,
                                    &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call CreateConvolutionLayer, error_code:" << error_code;
          return output_tensors;
        } 
      }
      
      output_tensor = phydnnNetworkGroupedConvolution2dOp(graph_ep->GetGraph(),
                                                      input_tensor,
                                                      weights_tensor,
                                                      stride_t,
                                                      pad_begin,
                                                      pad_end,
                                                      dilation,
                                                      group,
                                                      &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkConvolution2dOp_v2, error_code:" << error_code;
        return output_tensors;
      } 
    }

    if (bais_flag) {
      auto bias_tensor = *inputs[2]; // Get bias tensor

      // Broadcast bias tensor on height and width dimension
      phydnnGetTensorDescriptor(output_tensor, &output_desc);
      bias_tensor = phydnnNetworkBroadcastOp(
          graph_ep->GetGraph(), bias_tensor, 1, output_desc.size[2], &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBroadcastOp, error_code:" << error_code;
        return output_tensors;
      } 

      bias_tensor = phydnnNetworkBroadcastOp(
          graph_ep->GetGraph(), bias_tensor, 2, output_desc.size[3], &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBroadcastOp, error_code:" << error_code;
        return output_tensors;
      } 

      //type
      phydnnGetTensorDescriptor(bias_tensor, &bias_desc);
      if(output_desc.type != bias_desc.type) {
        LOGS_DEFAULT(INFO)<<"output_desc type:"<<output_desc.type<<" bias_desc type:"<<bias_desc.type;
        output_tensor = phydnnNetworkCastOp(
                    graph_ep->GetGraph(), output_tensor, bias_desc.type, nullptr, &error_code);
        if(error_code != PHYDNN_SUCCESS) {
          LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
          return output_tensors;
        }  
      }
      
      // Add bias tensor
      output_tensor = phydnnNetworkBinaryOp(graph_ep->GetGraph(),
                                            output_tensor,
                                            bias_tensor,
                                            PHYDNN_OPERATION_ADD,
                                            &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code;
        return output_tensors;
      }
    }
    
    phydnn_tensor_descriptor out_desc;
    phydnnGetTensorDescriptor(output_tensor, &out_desc);
    for(unsigned int i=0; i<out_desc.dimensions; i++) {
       LOGS_DEFAULT(INFO)<<"Conv2dOpBuilder get result: "<<i<<" out_desc.size:"<<out_desc.size[i]<<", type:"<<out_desc.type;
    }

    output_tensors.push_back(output_tensor);
    return output_tensors;   
  } 
};
    
}  // namespace phytium_npu
}  // namespace onnxruntime
