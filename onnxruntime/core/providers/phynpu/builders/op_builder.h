 
 

#pragma once
#include "core/graph/graph_viewer.h"
#include "core/framework/node_unit.h"

namespace onnxruntime {
namespace phytium_npu {
class GraphPhynpuEP; //需要添加一个类，用于图的构建

class IOpBuilder {
 public:
  IOpBuilder() {}
  virtual ~IOpBuilder() {}
  virtual bool IsSupported(const onnxruntime::GraphViewer& graph_viewer,
                             const NodeUnit& node_unit) const {
    (void)graph_viewer;
    (void)node_unit;
    return true;
  }

  virtual bool BuildOp(GraphPhynpuEP* graph_ep,
                       const onnxruntime::GraphViewer& graph_viewer,
                       const NodeUnit& node_unit) = 0;
};

}  // namespace phytium_npu
}  // namespace onnxruntime
