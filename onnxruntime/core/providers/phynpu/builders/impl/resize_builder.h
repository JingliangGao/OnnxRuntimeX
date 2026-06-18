 
//Data: 2026 -02-16

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class ResizeOpBuilder : public BaseOpBuilder {
 public:  
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    if (node->SinceVersion() > 10) {
      NodeAttrHelper helper(*node);
      const auto mode = helper.Get("mode", "nearest");
      auto coordinate_transformation = helper.Get("coordinate_transformation_mode", "half_pixel");
      bool align_corners = false;
      if(coordinate_transformation == "align_corners" || coordinate_transformation == "asymmetric"){
      //if(coordinate_transformation == "align_corners"){  
        align_corners = true;
      }

      if((mode == "nearest" || mode == "linear" || mode == "bilinear") && align_corners) {
        LOGS_DEFAULT(INFO) << "Resize ops can only support align_corners by Version > opset v10. The mode is:"<<mode;
        return true;
      }
      LOGS_DEFAULT(INFO) << "Resize ops get mode:"<<mode<<", coordinate is:" << coordinate_transformation;
    } else {
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
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO) <<"ResizeOpBuilder input.type"<<input_desc.type<<", dims:"<<input_desc.dimensions; 
    size_t input_rank = input_desc.dimensions;

    // get params.
    NodeAttrHelper helper(node_unit.GetNode());
    const auto mode = helper.Get("mode", "nearest");
    auto axes = helper.Get("axes", std::vector<int64_t>{});
    bool is_linear = mode == "linear";
    auto coordinate_transformation = helper.Get("coordinate_transformation_mode", "half_pixel");
    bool align_corners = coordinate_transformation == "align_corners";
    bool half_pixel_center = coordinate_transformation == "half_pixel";
    //bool align_corners = true;
    const auto& input_defs = node_unit.GetNode().InputDefs();

    // get scales or sizes. only one of them can be set.
    std::vector<float> scales;
    std::vector<int64_t> sizes;
    int nheight = 0;
    int nwidth  = 0;
    bool is_scales = false;
    const int32_t index = node_unit.SinceVersion() > 10 ? 2 : 1;

    LOGS_DEFAULT(INFO) <<"ResizeOpBuilder scales index:"<<index;
    if(*inputs[index] != nullptr) {  
      auto scales_name = (input_defs.size() > index) ? std::string(input_defs[index]->Name()) : "";
      if(!scales_name.empty()) {
        is_scales = true;  
      }
    }

    LOGS_DEFAULT(INFO) <<"ResizeOpBuilder is_scales:"<<is_scales;
    if(is_scales) {
      const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
      const auto& scales_tensor = *initializers.at(node_unit.GetNode().InputDefs()[index]->Name());
      Initializer unpacked_tensor(scales_tensor);
      auto raw_scales = unpacked_tensor.DataAsSpan<float>();
      const auto len = SafeInt<size_t>(scales_tensor.dims()[0]);
      scales.reserve(len);
      scales.insert(scales.end(), raw_scales.begin(), raw_scales.end());

      // set out sizes.
      int num_scales = scales.size();
      {
        // only the last two dims have their size changed
        nheight = static_cast<int64_t>(input_desc.size[input_rank - 2] * scales[num_scales - 2]);
        nwidth  = static_cast<int64_t>(input_desc.size[input_rank - 1] * scales[num_scales - 1]);
      }
    } else {
      const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
      const auto& sizes_tensor = *initializers.at(node_unit.GetNode().InputDefs()[3]->Name());
      Initializer unpacked_tensor(sizes_tensor);
      auto raw_size = unpacked_tensor.DataAsSpan<int64_t>();
      const auto len = SafeInt<size_t>(sizes_tensor.dims()[0]);
      sizes.reserve(len);
      sizes.insert(sizes.end(), raw_size.begin(), raw_size.end());
    }

    for(int i=0; i<sizes.size(); i++){
      LOGS_DEFAULT(INFO) <<"ResizeOpBuilder get size:"<<sizes[i];
    }

    if(sizes.size() >= 2) {
      nheight = sizes[sizes.size() -2];
      nwidth  = sizes[sizes.size() -1];
    }

    LOGS_DEFAULT(INFO) <<"ResizeOpBuilder nheight:"<<nheight<<", nwidth:"<<nwidth<<", align_corners:"<<align_corners;
    // create output tensor
    phydnn_tensor output_tensor = nullptr;
    if (mode == "linear" || mode == "bilinear") {
      LOGS_DEFAULT(INFO) <<"ResizeOpBuilder linear.";
      output_tensor = phydnnNetworkResizeBilinearOp(graph_ep->GetGraph(),
                                                    input_tensor,
                                                    nheight, nwidth,
                                                    align_corners,
                                                    &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkResizeBilinearOp!";
        return output_tensors;
      }
    } else if(mode == "nearest"){  
      LOGS_DEFAULT(INFO) <<"ResizeOpBuilder Nearest.";
      output_tensor = phydnnNetworkResizeNearestNeighbourOp(graph_ep->GetGraph(),
                                                            input_tensor,
                                                            nheight, nwidth,
                                                            align_corners,
                                                            &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkResizeNearestNeighbourOp!";
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
};  // ResizeOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
