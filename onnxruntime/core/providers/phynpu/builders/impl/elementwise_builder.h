 
//Data: 2026 -01-22

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class AddOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_ADD);                                                                       
  }                                                                                             
};                                                                                                     

class SubOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_SUB);                                                                       
  }                                                                                             
};                                                                                                     

class MulOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_MUL);                                                                       
  }                                                                                             
};                                                                                                     

class DivOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    for (auto input : node->InputDefs()) {                                                      
      if (*input->Type() == "tensor(int8)" || *input->Type() == "tensor(uint8)") {                                                  
        LOGS_DEFAULT(INFO) << "Int8/UInt8 quant type is not supported as PHYDNN_OPERATION_DIV ops."; 
        return false;                                                                           
      }                                                                                         
    }                                                                                           
    return true;                                                                                
  }  

  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_DIV);                                                                       
  }                                                                                             
};     

class AndOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_AND);                                                                       
  }                                                                                             
};     

class OrOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_OR);                                                                       
  }                                                                                             
};     

class XorOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_XOR);                                                                       
  }                                                                                             
};     

class MaxOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_MAX);                                                                       
  }                                                                                             
};     

class MinOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_MIN);                                                                       
  }                                                                                             
};     

class MatmulOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override { 
    auto input0_tensor = *inputs[0];                                                            
    auto input1_tensor = *inputs[1];       
    std::vector<phydnn_tensor> output_tensors;                                                                                                                                        
    auto output_tensor = MMBinaryBuildOp(graph_ep, input0_tensor, input1_tensor, PHYDNN_OPERATION_MATMUL, node_unit);    
    output_tensors.push_back(output_tensor);
    return output_tensors;                                                                    
  }                                                                                             
}; 

class Add_satOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_ADD_SAT);                                                                       
  }                                                                                             
}; 

class Sub_satOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_SUB_SAT);                                                                       
  }                                                                                             
}; 

class Mul_satOpBuilder : public BaseOpBuilder {                                          
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    // for (auto input : node->InputDefs()) {                                                      
    //   if (*input->Type() == "tensor(int64)") {                                                  
    //     LOGS_DEFAULT(INFO) << "Int64 type is not supported as elementwise operation input."; 
    //     return false;                                                                           
    //   }                                                                                         
    // }                                                                                           
    return true;                                                                                
  }    
  std::vector<phydnn_tensor> HandleBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_quant_param in_quant,
                      phydnn_quant_param out_quant) override {                                                                                                                           
    return BinaryBaseBuildOp(graph_ep, inputs, node_unit, PHYDNN_OPERATION_MUL_SAT);                                                                       
  }                                                                                             
}; 

}  // namespace phytium_npu
}  // namespace onnxruntime
