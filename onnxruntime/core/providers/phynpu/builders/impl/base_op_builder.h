 
 

#pragma once
#include <memory>
#include <vector>
#include "../../phynpu_ep_graph.h"
#include "../op_builder.h"
#include "../../phynpu_util.h"

namespace onnxruntime {
namespace phytium_npu {

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;

  bool IsSupported(const onnxruntime::GraphViewer& graph_viewer,
                   const NodeUnit& node_unit) const override;
  bool BuildOp(phytium_npu::GraphPhynpuEP* graph_ep,
               const onnxruntime::GraphViewer& graph_viewer, 
               const NodeUnit& node_unit) override;
  virtual bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                             const Node* node) const {
    (void)graph_viewer;
    (void)node;
    return true;
  }

  virtual bool IsQuantizedOp(const NodeUnit& /* node_unit */) const { return false; }

  virtual int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const { return 1; }
  virtual int GetMaxSupportedOpSet(const NodeUnit& /* node_unit */) const { return 25; }

  virtual bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit) const;

  // TODO(cfy): Check if this node_unit's type is supported
  virtual bool IsNodeUnitTypeSupported(const NodeUnit& node_unit) const { (void)node_unit; return true; } 

  virtual std::vector<phydnn_tensor> HandleBuildOp(
      phytium_npu::GraphPhynpuEP* graph_ep,
      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
      const NodeUnit& node_unit,
      phydnn_quant_param in_quant,
      phydnn_quant_param out_quant) {
    (void)graph_ep; 
    (void)inputs; 
    (void)node_unit; 
    (void)in_quant; 
    (void)out_quant;
    return std::vector<phydnn_tensor>();
  }

 private:
  bool HasSupportedOpSet(const NodeUnit& node_unit) const;
  bool HasSupportedInputOutputs(const InitializedTensorSet& initializers,
                                const NodeUnit& node_unit) const;
};

}  // namespace phytium_npu
}  // namespace onnxruntime
