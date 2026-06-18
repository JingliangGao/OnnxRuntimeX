 
 

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {                                                                                       \

class ArgMaxOpBuilder : public BaseOpBuilder {                                            
  public:                                                                                         
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                                
                      const Node* node) const override {                                          
    NodeAttrHelper helper(*node);                                                                 
    const auto select_last_index = helper.Get("select_last_index", 0);                            
    if (select_last_index != 0) {                                                                                         
      LOGS_DEFAULT(ERROR) << "select_last_index for ArgMax is not supported";                     
      return false;                                                                               
    }                                                                                                     
    return true;                                                                                     
  }                                                                                               

  std::vector<phydnn_tensor> HandleBuildOp(                                                                    
      phytium_npu::GraphPhynpuEP* graph_ep,                                                       
      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                                  
      const NodeUnit& node_unit,
      phydnn_quant_param in_quant,
      phydnn_quant_param out_quant) override {      
    return ReduceBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_REDUCE_ARGMAX);  
  }
};

class ArgMinOpBuilder : public BaseOpBuilder {                                            
 public:                                                                                         
                                                                                         
  std::vector<phydnn_tensor> HandleBuildOp(                                                                    
      phytium_npu::GraphPhynpuEP* graph_ep,                                                       
      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                                  
      const NodeUnit& node_unit,
      phydnn_quant_param in_quant,
      phydnn_quant_param out_quant) override {        
    return ReduceBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_REDUCE_MIN);    
  }
};

class ReduceMinOpBuilder : public BaseOpBuilder {                                            
 public:                                                                                         
                                                                                         
  std::vector<phydnn_tensor> HandleBuildOp(                                                                    
      phytium_npu::GraphPhynpuEP* graph_ep,                                                       
      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                                  
      const NodeUnit& node_unit,
      phydnn_quant_param in_quant,
      phydnn_quant_param out_quant) override {        
    return ReduceBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_REDUCE_MIN);    
  }
};

class ReduceMaxOpBuilder : public BaseOpBuilder {                                            
 public:                                                                                         
                                                                                         
  std::vector<phydnn_tensor> HandleBuildOp(                                                                    
      phytium_npu::GraphPhynpuEP* graph_ep,                                                       
      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                                  
      const NodeUnit& node_unit,
      phydnn_quant_param in_quant,
      phydnn_quant_param out_quant) override {        
    return ReduceBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_REDUCE_MAX);    
  }
};

class ReduceSumOpBuilder : public BaseOpBuilder {                                            
 public:                                                                                         
                                                                                         
  std::vector<phydnn_tensor> HandleBuildOp(                                                                    
      phytium_npu::GraphPhynpuEP* graph_ep,                                                       
      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                                  
      const NodeUnit& node_unit,
      phydnn_quant_param in_quant,
      phydnn_quant_param out_quant) override {        
    return ReduceBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_REDUCE_SUM);    
  }
};

class ReduceMeanOpBuilder : public BaseOpBuilder {                                            
 public:                                                                                         
                                                                                         
  std::vector<phydnn_tensor> HandleBuildOp(                                                                    
      phytium_npu::GraphPhynpuEP* graph_ep,                                                       
      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                                  
      const NodeUnit& node_unit,
      phydnn_quant_param in_quant,
      phydnn_quant_param out_quant) override {        
    return ReduceBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_REDUCE_MEAN);    
  }
};

}  // namespace phytium_npu
}  // namespace onnxruntime
