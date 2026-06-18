 
 

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/node_unit.h"
#include "core/optimizer/initializer.h"
#include "core/common/safeint.h"
#include "phynpu_ep_graph.h"  

namespace onnxruntime {
namespace phytium_npu {
phydnn_type GetPhyDnnType(ONNX_NAMESPACE::TensorProto_DataType to_type, bool is_qtensor=false);

std::string PrintNode(const onnxruntime::NodeArg& node_arg);

std::string PrintNode(const std::vector<int64_t> shape);

TensorShape GetTensorShape(const onnxruntime::NodeArg& node_arg);

std::shared_ptr<uint8_t> UnpackTensor(
    const NodeArg* node_arg, const ONNX_NAMESPACE::TensorProto& initializer);
    
void UpdateConv2DPadAndDilation(
    int32_t input_size,
    int32_t filter_height_or_width,
    std::string auto_pad,
    int32_t* pad_top_or_left,
    int32_t* pad_bottom_or_right,
    int32_t stride_height_or_width,
    int32_t* dilation_height_or_width);

std::vector<phydnn_tensor> CreateElementwiseOps(
    phytium_npu::GraphPhynpuEP* graph_ep,
    phydnn_tensor input0_tensor,
    phydnn_tensor input1_tensor,
    phydnn_operation_binary phydnn_operation,
    int32_t fuse_code) ;

std::vector<phydnn_tensor> CreateConvertUnaryActivations(
    phytium_npu::GraphPhynpuEP* graph_ep,
    phydnn_tensor input_tensor,
    phydnn_operation_unary op) ;

std::vector<phydnn_tensor> CreateReLULayer(
    phytium_npu::GraphPhynpuEP* graph_ep,
    phydnn_tensor input_tensor,
    bool has_min_clamp,
    float min_clamp,
    bool has_max_clamp,
    float max_clamp,
    float negative_slope);

size_t GetTensorElementSize(const ONNXTensorElementDataType type);

size_t GetTensorBytes(const Ort::TensorTypeAndShapeInfo& info);

std::vector<phydnn_tensor> PoolBaseOp(
            phytium_npu::GraphPhynpuEP* graph_ep,
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
            const NodeUnit& node_unit,
            phydnn_pooling_type pool_type);

std::vector<phydnn_tensor> GolabelPoolBaseOp(
            phytium_npu::GraphPhynpuEP* graph_ep,
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
            const NodeUnit& node_unit,
            phydnn_pooling_type pool_type);

std::vector<phydnn_tensor> BinaryBaseBuildOp(
            phytium_npu::GraphPhynpuEP* graph_ep,                             
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
            const NodeUnit& node_unit,
            phydnn_operation_binary op_type);

phydnn_tensor MMBinaryBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
            phydnn_tensor x_tensor,
            phydnn_tensor y_tensor,
            phydnn_operation_binary op_type,                    
            const NodeUnit& node_unit);

std::vector<phydnn_tensor> ReduceBaseBuildOp(                                                                    
          phytium_npu::GraphPhynpuEP* graph_ep,                                                       
          std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                                  
          const NodeUnit& node_unit,
          phydnn_reduce_type reduce_type);
          
bool GetphynpuClipMinMax(const GraphViewer& graph_viewer, const Node& node, float& min, float& max);

std::vector<int32_t> GetDataFromInputTensors(phytium_npu::GraphPhynpuEP* graph_ep, 
                            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
                            const NodeUnit& node_unit,
                            int index);
                            
void GetQuantizationScaleAndZeroPoint(
    const GraphViewer& graph_viewer, const NodeUnitIODef& io_def, 
    const std::filesystem::path& model_path,
    float& scale, int32_t& zero_point, 
    std::optional<std::vector<float>>& pcq_scales,
    std::optional<std::vector<int32_t>>& pcq_zps);

phydnn_tensor ConvertQuantTensorType(phytium_npu::GraphPhynpuEP* graph_ep,
                                    phydnn_tensor tensor, 
                                    phydnn_type out_type,
                                    phydnn_quant_param *quant_param);

phydnn_tensor SetQuantOutputTensor(const onnxruntime::GraphViewer& graph_viewer,
                                  phytium_npu::GraphPhynpuEP* graph_ep,
                                  const NodeUnitIODef nudef,
                                  const NodeUnit& node_unit,
                                  phydnn_tensor tensor,
                                  bool is_lut);     

bool GetLUTOpsname(const NodeUnit& node_unit);

std::vector<int32_t> GenerateQuantizedHardSigmoidLUT(
                                         phydnn_quant_param& out_attrs, 
                                         phydnn_quant_param& in_attrs, 
                                         size_t lut_size,
                                         bool is_signed,
                                         float alpha,
                                         float beta);
std::vector<int32_t> GenerateQuantizedHardSwishLUT(phydnn_quant_param& out_attrs, 
                                                  phydnn_quant_param& in_attrs, 
                                                  size_t lut_size,
                                                  bool is_signed=true);
std::vector<int32_t> GenerateQuantizedLeakyreluLUT(phydnn_quant_param& out_attrs, 
                                                  phydnn_quant_param& in_attrs, 
                                                  size_t lut_size,
                                                  float alpha_value,
                                                  bool is_signed=true) ;
std::vector<int32_t> GenerateQuantizedSigmoidLUT(phydnn_quant_param& out_attrs, 
                                                  phydnn_quant_param& in_attrs, 
                                                  size_t lut_size,
                                                  bool is_signed=true) ;
std::vector<int32_t> GenerateQuantizedTanhLUT(phydnn_quant_param& out_attrs, 
                                              phydnn_quant_param& in_attrs, 
                                              size_t lut_size,
                                              bool is_signed=true);
std::vector<int32_t> GenerateQuantizedExpLUT(phydnn_quant_param& out_attrs, 
                                              phydnn_quant_param& in_attrs, 
                                              size_t lut_size,
                                              bool is_signed=true);

}  // namespace phytium_npu
}  // namespace onnxruntime
