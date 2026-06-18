 
 

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class MaxPoolOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer, 
                     const Node* node) const override {
    // auto shape = GetTensorShape(*node->InputDefs()[0]);
    // if (shape.NumDimensions() >= 5) {
    //   LOGS_DEFAULT(ERROR) << "3DPool is not supported yet.";
    //   return false;
    // }

    return true;
  }

  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
      return PoolBaseOp(graph_ep, inputs, node_unit, PHYDNN_POOLING_MAX);
  }   
};

class AveragePoolOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer, 
                     const Node* node) const override {
    // auto shape = GetTensorShape(*node->InputDefs()[0]);
    // if (shape.NumDimensions() >= 5) {
    //   LOGS_DEFAULT(ERROR) << "3DPool is not supported yet.";
    //   return false;
    // }
    return true;
  }

  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
      return PoolBaseOp(graph_ep, inputs, node_unit, PHYDNN_POOLING_AVERAGE);
  }   
};


class GlobalAvgPoolOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer, 
                     const Node* node) const override {
    // auto shape = GetTensorShape(*node->InputDefs()[0]);
    // if (shape.NumDimensions() >= 5) {
    //   LOGS_DEFAULT(ERROR) << "3DPool is not supported yet.";
    //   return false;
    // }
    return true;
  }

  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
      return GolabelPoolBaseOp(graph_ep, inputs, node_unit, PHYDNN_POOLING_AVERAGE);
  }   
};


class GlobalMaxPoolOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer, 
                     const Node* node) const override {
    // auto shape = GetTensorShape(*node->InputDefs()[0]);
    // if (shape.NumDimensions() >= 5) {
    //   LOGS_DEFAULT(ERROR) << "3DPool is not supported yet.";
    //   return false;
    // }
    return true;
  }

  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
      return GolabelPoolBaseOp(graph_ep, inputs, node_unit, PHYDNN_POOLING_MAX);
  }   
};

}  // namespace phytium_npu
}  // namespace onnxruntime
