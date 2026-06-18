/*************************************************************************/ /*!
@Copyright      Copyright (c) Phytium Technology Co., Ltd. All Rights Reserved
@License        MIT

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

@file
@brief          PHY DNN - the Phytium Neural Network programming interface
*/ /**************************************************************************/
#ifndef _PHY_DNN_COMPILE_H
#define _PHY_DNN_COMPILE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "phydnn_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _phydnn_device_capabilities_t* phydnn_device_capabilities;

typedef enum _phydnn_operation_unary_t
{
  PHYDNN_OPERATION_SIGN,
  PHYDNN_OPERATION_SIGMOID,
  PHYDNN_OPERATION_TANH,
  PHYDNN_OPERATION_RELU, /* max(input_value, 0) */
  PHYDNN_OPERATION_NEGATE,
  PHYDNN_OPERATION_ABS,
  PHYDNN_OPERATION_NOT,
  PHYDNN_OPERATION_LOG,
  PHYDNN_OPERATION_SQRT,
  PHYDNN_OPERATION_EXP,
  PHYDNN_OPERATION_FLOOR,
  PHYDNN_OPERATION_CEIL,
#if PHYDNN_MINOR_VERSION >= 6
  PHYDNN_OPERATION_SIN,
  PHYDNN_OPERATION_RSQRT
#endif
} phydnn_operation_unary;

typedef enum _phydnn_operation_binary_t
{
  PHYDNN_OPERATION_ADD,
  PHYDNN_OPERATION_SUB,
  PHYDNN_OPERATION_MUL,
  PHYDNN_OPERATION_DIV,
  PHYDNN_OPERATION_AND,
  PHYDNN_OPERATION_OR,
  PHYDNN_OPERATION_XOR,
  PHYDNN_OPERATION_MAX,
  PHYDNN_OPERATION_MIN,
  PHYDNN_OPERATION_MATMUL,
  PHYDNN_OPERATION_ADD_SAT,
  PHYDNN_OPERATION_SUB_SAT,
  PHYDNN_OPERATION_MUL_SAT,
#if PHYDNN_MINOR_VERSION >= 4
  PHYDNN_OPERATION_PRELU,
#endif
} phydnn_operation_binary;

typedef enum _phydnn_pooling_type_t
{
  PHYDNN_POOLING_MAX,
  PHYDNN_POOLING_AVERAGE,
  PHYDNN_POOLING_MIN,
  PHYDNN_POOLING_L2
} phydnn_pooling_type;

typedef enum _phydnn_lrn_type_t
{
  PHYDNN_LRN_ACROSS,
  PHYDNN_LRN_WITHIN
} phydnn_lrn_type;

typedef enum _phydnn_image_transform_type_t
{
  PHYDNN_IMAGE_TRANSFORM_NEAREST,
  PHYDNN_IMAGE_TRANSFORM_BILINEAR
} phydnn_image_transform_type;

typedef enum _phydnn_reduce_type_t
{
  PHYDNN_REDUCE_SUM,
  PHYDNN_REDUCE_MEAN,
  PHYDNN_REDUCE_MAX,
#if PHYDNN_MINOR_VERSION >= 3
  PHYDNN_REDUCE_ARGMAX,
#endif
#if PHYDNN_MINOR_VERSION >=6
  PHYDNN_REDUCE_MIN
#endif
} phydnn_reduce_type;

/**
 * Obtain the descriptor of a tensor.
 *
 * If the tensor is of type PHYDNN_TYPE_QPA_* the caller is resposible of destroying the
 * phydnn_per_axis_quant_param structure (desc->quant_param->per_axis) by calling
 * phydnnDestroyPerAxisQuantParam
 *
 * @Input  tensor  A tensor previously created.
 * @Output desc    The descriptor of the given tensor. May not be NULL.
 * @Return PHYDNN_SUCCESS or PHYDNN_INVALID_VALUE for invalid tensor.
 */
phydnn_err_code phydnnGetTensorDescriptor(phydnn_tensor tensor,
                                          phydnn_tensor_descriptor *desc);

/**
 * Create a Network
 *
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE.
 * @Return The network created.
 */
phydnn_network phydnnCreateNetwork(phydnn_err_code *errcode_ret);

/**
 * Create a Network from IR
 *
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE.
 * @Return The network created.
 */
phydnn_network phydnnCreateNetworkFromIR(const void *arch_data,
                                         size_t arch_data_size,
                                         const void *params_data,
                                         size_t params_data_size,
                                         phydnn_err_code *errcode_ret);

/**
 * Destroy a previously created Network
 *
 * @Input  network      A previously created network.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_INVALID_VALUE for invalid network.
 */
phydnn_err_code phydnnNetworkDestroy(phydnn_network network);

/**
 * Add an input to the network.
 *
 * @Input  network      Handle to the network.
 * @Input  descriptor   The description of the input to add.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid descriptor,
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY. May be NULL.
 * @Return The created input tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkInput(phydnn_network network,
                                 const phydnn_tensor_descriptor *const descriptor,
                                 phydnn_err_code *errcode_ret);

/**
 * Add a tensor with fixed data to the network.
 *
 * @Input  network      Handle to the network.
 * @Input  descriptor   The description of the fixed tensor to add.
 * @Input  fixed_data   The data to use for the tensor. This data is fixed and will
 *                      likely be optimised by the implementation. May not be NULL.
 *                      Must stay valid until the network object is created.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid descriptor,
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 * @Return The created fixed tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkFixedInput(phydnn_network network,
                                      const phydnn_tensor_descriptor *const descriptor,
                                      const void *const fixed_data,
                                      phydnn_err_code *errcode_ret);

phydnn_tensor phydnnNetworkFixedScalar(phydnn_network network,
                                      double scalar,
                                      phydnn_err_code *errcode_ret);

/**
 * Sets the name of a tensor.
 *
 * @Input  tensor  The tensor to set name of. May not be NULL.
 * @Input  name  The given name, may not be NULL.
 * @Return errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid tensor,
 *                      or PHYDNN_OUT_OF_MEMORY.
 */
phydnn_err_code phydnnTensorSetName(phydnn_tensor tensor, const char *name);

/**
 * Get the name of a tensor.
 *
 * The returned pointer will be internal and stay valid until at least the next
 * API call on the same thread.
 *
 * @Input  tensor  The tensor to get the name of. May not be NULL.
 * @Output errcode_ret  PHYDNN_SUCCESS or PHYDNN_INVALID_VALUE for invalid tensor.
 * @Return The given name, NULL on failure.
 */
const char *phydnnTensorGetName(phydnn_tensor tensor, phydnn_err_code *errcode_ret);

/**
 * Find tensor by name.
 *
 * If multiple tensors in this network have the same name, then only one of them
 * will be returned.
 *
 * @Input  network      Handle to the network.
 * @Input  name         The name to look for, may not be NULL.
 * @Output errcode_ret  PHYDNN_SUCCESS or PHYDNN_INVALID_VALUE for NULL name.
 * @Return The tensor with the given name, or NULL if not found.
 */
phydnn_tensor phydnnNetworkFindTensor(phydnn_network network,
                                      const char *name,
                                      phydnn_err_code *errcode_ret);


#if PHYDNN_MINOR_VERSION >= 6

/**
 * Set the name of the network.
 *
 * @Input network  The network to set name of.
 * @Input network_name The name for the network
 * @Return errcode_ret PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid network/network_name,
 *                      or PHYDNN_OUT_OF_MEMORY
 */
phydnn_err_code phydnnNetworkSetName(phydnn_network network, const char *network_name);

/**
 * Get the name of the network.
 *
 * @Input network The network to get the name of.
 * @Output errcode_ret PHYDNN_SUCCESS, or PHYDNN_INVALID_VALUE for invalid network.
 * @Return The given name, NULL on failure
 */
const char *phydnnNetworkGetName(const phydnn_network network, phydnn_err_code *errcode_ret);
#endif

#if PHYDNN_MINOR_VERSION >= 2
/**
 * Find non-fixed input tensors of the network.
 *
 * These tensors can then be used as input tensors for the network object.
 *
 * @Input  network      Handle to the network.
 * @Output num_inputs   Number of input tensors found.
 * @Output errcode_ret  PHYDNN_SUCCESS or PHYDNN_FAILURE.
 * @Return An array containing the non-fixed input tensors of the network.
 */
phydnn_tensor* phydnnNetworkFindInputs(phydnn_network network,
                                       unsigned *num_inputs,
                                       phydnn_err_code *errcode_ret);

/**
 * Find default output tensors of the network.
 *
 * Default output tensors are either:
 * - Tensors with "output" attribute defined, if there is such a tensor in the network
 *   this attribute is an index defining their position in the outputs vector.
 * - The tensors which are not inputs of any other tensor, otherwise.
 *
 * These tensors can then be used as output tensors for the network object.
 *
 * @Input  network      Handle to the network.
 * @Output num_outputs  Number of default output tensors found.
 * @Output errcode_ret  PHYDNN_SUCCESS or PHYDNN_FAILURE.
 * @Return A vector containing the default output tensors of the network.
 */
phydnn_tensor* phydnnNetworkFindDefaultOutputs(phydnn_network network,
                                               unsigned *num_outputs,
                                               phydnn_err_code *errcode_ret);
#endif

/**
 * Add Reshape operation of a tensor to a different layout.
 *
 * @Input  network      Handle to the network.
 * @Input  tensor       The input tensor to reshape. May not be NULL.
 * @Input  descriptor   The tensor descriptor of the new tensor. Must be the
 *                      same tensor type as the input tensor. May not be NULL.
 *                      Can contain 0: Copy one dimension from input tensor descriptor
 *                      Can contain -1 only once: Infer single dimension from
 *                      the rest of this descriptor and the input tensor descriptor
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid descriptor.
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 * @Return The output reshaped tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkReshapeOp(phydnn_network network,
                                     phydnn_tensor tensor,
                                     const phydnn_tensor_descriptor *const descriptor,
                                     phydnn_err_code *errcode_ret);

/**
 * Add Transpose dimensions operation of a tensor.
 *
 * e.g.
 * tensor=[[1,2,3],[4,5,6]] order=[0,1] => [[1,2,3],[4,5,6]]
 * tensor=[[1,2,3],[4,5,6]] order=[1,0] => [[1,4],[2,5],[3,6]]
 * tensor=[[1],[1],[1],[1],[1]] order=[1,0] => [[1,1,1,1,1]]
 * tensor=[[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]] order=[1,2,0] => [[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]
 *
 * @Input  network      Handle to the network.
 * @Input  tensor       The input tensor to transpose. May not be NULL.
 * @Input  order        The new ordering of tensor dimensions. Must contain every
 *                      dimension of the input tensor exactly once. Length is the
 *                      same size as the input tensor dimensions.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid tensor or order,
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 * @Return The output transposed tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkTransposeOp(phydnn_network network,
                                       phydnn_tensor tensor,
                                       const int order[],
                                       phydnn_err_code *errcode_ret);

/**
 * Add data-type cast operation of a tensor.
 *
 * @Input  network         Handle to the network.
 * @Input  tensor          The input data to cast. May not be NULL.
 * @Input  dst_type        phydnn_type to cast to
 * @Input  dst_quant_param quant params if the dst_type is quantized else NULL
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid tensor,
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 * @Return The output type casted tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkCastOp(phydnn_network network,
                                  phydnn_tensor tensor,
                                  phydnn_type dst_type,
                                  const phydnn_quant_param *const dst_quant_param,
                                  phydnn_err_code *errcode_ret);

/**
 * Add Broadcast operation of a tensor along a new dimension.
 *
 * e.g.
 * tensor=[1], dim=0, size=2 => [[1], [1]]
 * tensor=[1], dim=1, size=3 => [[1, 1, 1]]
 * tensor=[[1,2],[3,4],[5,6]], dim=0, size=2 => [[[1,2],[3,4],[5,6]], [[1,2],[3,4],[5,6]]]
 * tensor=[[1,2],[3,4],[5,6]], dim=2, size=2 => [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
 * tensor=1, dim=0, size=5 => [1,1,1,1,1]
 *
 * @Input  network      Handle to the network.
 * @Input  input        The input tensor to broadcast. May not be NULL.
 * @Input  dimension    The dimension of the new broadcasted dimension in the
 *                      resulting tensor. Must be <= the rank of the input tensor.
 * @Input  size         The size of the newly broadcast dimension.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid tensor,
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 * @Return The output broadcasted tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkBroadcastOp(phydnn_network network,
                                       phydnn_tensor tensor,
                                       unsigned int dimension,
                                       size_t size,
                                       phydnn_err_code *errcode_ret);

/**
 * Obtain subset of a tensor in the network.
 *
 * start, end and stride must have the same length as number of dimensions in
 * input tensor. Start and end are inclusive.
 * @Input  network      Handle to the network.
 * @Input  tensor       The input tensor. May not be NULL.
 * @Input  start        The start of the sub-tensor in each dimension.
 * @Input  end          The end of the sub-tensor in each dimension.
 * @Input  stride       The stride of the sub-tensor in each dimension.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid tensor,
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 * @Return The output sub-tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkSubTensor(phydnn_network network,
                                     phydnn_tensor tensor,
                                     const size_t start[],
                                     const size_t end[],
                                     const size_t stride[],
                                     phydnn_err_code *errcode_ret);

/**
 * Add Interleave Operation to interleave two tensors along a given dimension.
 *
 * Slice tensor2 into tensor1 along one dimension. Place initial element of
 * tensor2 at start[] index of tensor1, shifting tensor1 values higher by one.
 * Skip stride[] elements and place the next element of tensor2, etc. Extra
 * elements are appended.
 * Dimensions other than interleave dimension must have the same sizes.
 * e.g.
 * tensor1=[0,1,2,3,4,5,6,7,8], tensor2=[10,11,12,13], dim=0, start=4, stride=2 => [0,1,2,3,10,4,5,11,6,7,12,8,13]
 * tensor1=[[0,1,2],[3,4,5]], tensor2=[[6,7],[8,9]], dim=1, start=2, stride=0 => [[0,1,6,7,2],[3,4,8,9,5]]
 *
 * @Input  network      Handle to the network.
 * @Input  tensor1      First tensor. May not be NULL
 * @Input  tensor2      Second tensor. Must have same number of dimensions as
 *                      tensor1. May not be NULL
 * @Input  dimension    The dimension to join along. Must be < the rank of the input tensors.
 * @Input  start        The starting element to place tensor2
 * @Input  stride       The stride of splicing in tensor2 (tensor1 elements between
 *                      tensor2 elements).
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid tensor,
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 * @Return The output interleaved tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkInterleaveOp(phydnn_network network,
                                        phydnn_tensor tensor1,
                                        phydnn_tensor tensor2,
                                        unsigned int dimension,
                                        size_t start,
                                        size_t stride,
                                        phydnn_err_code *errcode_ret);

/**
 * Add Split Operation to evenly split a tensor along an axis in the network.
 *
 * @Input  network      Handle to the network.
 * @Input  tensor       The input tensor. May not be NULL.
 * @Input  dimension    The dimension to slice along. Must be < the rank of
 *                      the input tensor.
 * @Input  num_slices   The number of slices to be sliced. Must evenly divide
 *                      dimension size. Must be > 0.
 * @Output out_tensors  The multiple tensors along the requested dimension. Must
 *                      have space for num_slices tensors.
 * @Return PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid tensor,
 *         PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 */
phydnn_err_code phydnnNetworkSplitOp(phydnn_network network,
                                     phydnn_tensor tensor,
                                     unsigned int dimension,
                                     unsigned int num_slices,
                                     phydnn_tensor out_tensors[]);

/**
 * Add Concat Operation to concat multiple tensors along an axis in the network.
 *
 * @Input  network      Handle to the network.
 * @Input  tensor       The input tensors. Array size of num_concats. May not be NULL.
 *                      All tensors must have the same rank.
 *                      The size of all the dimensions in the tensor except the dimension
 *                      being concatenated must match.
 * @Input  dimension    The dimension to concat along.
 * @Input  num_concats  The number of concats along the requested dimension. Must be >= 2.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid tensor,
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 * @Return The concated tensor along the requested dimension and size or NULL on failure.
 */
phydnn_tensor phydnnNetworkConcatOp(phydnn_network network,
                                    const phydnn_tensor tensors[],
                                    unsigned int dimension,
                                    unsigned int num_concats,
                                    phydnn_err_code *errcode_ret);

/**
 * Add unary operation on a tensor
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input tensor to operate on
 * @Input  operation    The operation to perform on the input tensor.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkUnaryOp(phydnn_network network,
                                   phydnn_tensor in_tensor,
                                   phydnn_operation_unary operation,
                                   phydnn_err_code *errcode_ret);

/**
 * Add ReLU operation on a tensor
 *
 * output = min(max(input, min_clamp), max_clamp)  if input > 0
 * 		  = min(max(input, min_clamp), max_clamp) * negative_slope  if input < 0
 *
 * @Input  network          Handle to the network.
 * @Input  in_tensor        The input tensor to operate on
 * @Input  has_min_clamp    Whether there is a minimal clamping value.
 * @Input  min_clamp        The minimal clamping value.
 * @Input  has_max_clamp    Whether there is a maximal clamping value.
 * @Input  max_clamp        The maximal clamping value.
 * @Input  negative_slope   Multiplier to apply to negative input values
 * @Output errcode_ret      PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                          PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                          PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkReLUOp(phydnn_network network,
                                  phydnn_tensor in_tensor,
                                  bool has_min_clamp,
                                  float min_clamp,
                                  bool has_max_clamp,
                                  float max_clamp,
                                  float negative_slope,
                                  phydnn_err_code *errcode_ret);

#if PHYDNN_MINOR_VERSION >= 10
/**
 * Add Interpolated Lookup LUT operation on a tensor
 * This node requires and integer input and integer output datatype.
 *
 * @Input  network          Handle to the network.
 * @Input  in_tensor        The input tensor to operate on
 * @Input  lut              LUT table pointer.
 * @Input  lut_size         Size of the LUT array
 * @Output errcode_ret      PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                          PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                          PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkInterpolatedLookupOp(phydnn_network network,
                                                phydnn_tensor in_tensor,
                                                const int32_t lut[],
                                                size_t lut_size,
                                                phydnn_err_code *errcode_ret);
#endif

/**
 * Add a binary operation on a set of tensors
 * Tensor type must be the same for both arguments.
 * The two tensors will be automatically broadcasted for element-wise operations.
 * The Tensor dimensions will be matched from highest to lowest.
 * A dimension size of 1 (or none) will be broadcasted to the size of the other tensor's dimension.
 * e.g. tensor1 size of [1,7,3,1], tensor2 size of [2,1,1,3,4].
 * tensor1+tensor2 size = [2,1,7,3,4].
 * @Input  network      Handle to the network.
 * @Input  in_tensor1   The LHS input tensor to operate on
 * @Input  in_tensor2   The RHS input tensor to operate on
 * @Input  operation    The operation to perform on the input tensors.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkBinaryOp(phydnn_network network,
                                    phydnn_tensor in_tensor1,
                                    phydnn_tensor in_tensor2,
                                    phydnn_operation_binary operation,
                                    phydnn_err_code *errcode_ret);

/**
 * Add a 2D convolution operation on a set of tensors.
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to convolve. 4D: [N, C, H, W]
 * @Input  filter       The filter to use. 4D: [Co, Ci, Hf, Wf]
 * @Input  stride       The stride in each dimension. 2D: [Hs, Ws]
 * @Input  pad          The padding in each dimension. 2D: [Hp, Wp]
 * @Input  dilation     The input dilation in each dimension. 2D: [Hd, Wd]
 * @Input  with_partial Perform the last calculation in each dimension if it is
 *                      not complete, ignoring out of bounds values.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkConvolution2dOp(phydnn_network network,
                                           phydnn_tensor in_tensor,
                                           phydnn_tensor filter,
                                           const unsigned int stride[2],
                                           const unsigned int pad[2],
                                           const unsigned int dilation[2],
                                           bool with_partial,
                                           phydnn_err_code *errcode_ret);

/**
 * Add a 2D convolution operation on a set of tensors.
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to convolve. 4D: [N, C, H, W]
 * @Input  filter       The filter to use. 4D: [Co, Ci, Hf, Wf]
 * @Input  stride       The stride in each dimension. 2D: [Hs, Ws]
 * @Input  pad_to_begin The padding added before data in each dimension. 2D: [H, W]
 * @Input  pad_to_end   The padding added after data in each dimension. 2D: [H, W]
 * @Input  dilation     The input dilation in each dimension. 2D: [Hd, Wd]
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkConvolution2dOp_v2(phydnn_network network,
                                           phydnn_tensor in_tensor,
                                           phydnn_tensor filter,
                                           const unsigned int stride[2],
                                           const unsigned int pad_to_begin[2],
                                           const unsigned int pad_to_end[2],
                                           const unsigned int dilation[2],
                                           phydnn_err_code *errcode_ret);

#if PHYDNN_MINOR_VERSION >= 2
/**
 * Add a grouped 2D convolution operation on a set of tensors.
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to convolve. 4D: [N, C, H, W]
 * @Input  filter       The filter to use. 4D: [Co, Ci/groups, Hf, Wf]
 * @Input  stride       The stride in each dimension. 2D: [Hs, Ws]
 * @Input  pad_to_begin The padding added before data in each dimension. 2D: [H, W]
 * @Input  pad_to_end   The padding added after data in each dimension. 2D: [H, W]
 * @Input  dilation     The input dilation in each dimension. 2D: [Hd, Wd]
 * @Input  groups     The grouping parameter of the convolution. 1 for a normal convolution.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkGroupedConvolution2dOp(phydnn_network network,
                                           phydnn_tensor in_tensor,
                                           phydnn_tensor filter,
                                           const unsigned int stride[2],
                                           const unsigned int pad_to_begin[2],
                                           const unsigned int pad_to_end[2],
                                           const unsigned int dilation[2],
                                           unsigned groups,
                                           phydnn_err_code *errcode_ret);
#endif

/**
 * Add a 2D depthwise convolution operation on a set of tensors.
 * out[n, c, h, w] = sum_{hf \in Hf, wf \in Wf} {
 * in_tensor[n, c/(Co/Ci), h*Hs+hf*Hd-Hp, w*Ws+wf*Wd-Wp] *
 * filter[c%(Co/Ci), c/(Co/Ci), hf, wf]
 * }
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to convolve. 4D: [N, C, H, W]
 * @Input  filter       The filter to use. 4D: [Co/Ci, Ci, Hf, Wf]
 *                      Must be the same type as in_tensor.
 *                      Number of channels must be the same as in in_tensor.
 *                      Size must be > 0.
 * @Input  stride       The stride in each dimension. 2D: [Hs, Ws]
 *                      Must be > 0 in both dimensions.
 * @Input  pad          The padding in each dimension. 2D: [Hp, Wp]
 * @Input  dilation     The input dilation in each dimension. 2D: [Hd, Wd]
 *                      Must be > 0 in both dimensions.
 * @Input  with_partial Perform the last calculation in each dimension if it is
 *                      not complete, ignoring out of bounds values.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkDepthConvolution2dOp(phydnn_network network,
                                                phydnn_tensor in_tensor,
                                                phydnn_tensor filter,
                                                const unsigned int stride[2],
                                                const unsigned int pad[2],
                                                const unsigned int dilation[2],
                                                bool with_partial,
                                                phydnn_err_code *errcode_ret);

/**
 * Add a 2D depthwise convolution operation on a set of tensors.
 * out[n, c, h, w] = sum_{hf \in Hf, wf \in Wf} {
 * in_tensor[n, c/(Co/Ci), h*Hs+hf*Hd-Hp, w*Ws+wf*Wd-Wp] *
 * filter[c%(Co/Ci), c/(Co/Ci), hf, wf]
 * }
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to convolve. 4D: [N, C, H, W]
 * @Input  filter       The filter to use. 4D: [Co/Ci, Ci, Hf, Wf]
 *                      Must be the same type as in_tensor.
 *                      Number of channels must be the same as in in_tensor.
 *                      Size must be > 0.
 * @Input  stride       The stride in each dimension. 2D: [Hs, Ws]
 *                      Must be > 0 in both dimensions.
 * @Input  pad_to_begin The padding added before data in each dimension. 2D: [H, W]
 * @Input  pad_to_end   The padding added after data in each dimension. 2D: [H, W]
 * @Input  dilation     The input dilation in each dimension. 2D: [Hd, Wd]
 *                      Must be > 0 in both dimensions.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkDepthConvolution2dOp_v2(phydnn_network network,
                                                phydnn_tensor in_tensor,
                                                phydnn_tensor filter,
                                                const unsigned int stride[2],
                                                const unsigned int pad_to_begin[2],
                                                const unsigned int pad_to_end[2],
                                                const unsigned int dilation[2],
                                                phydnn_err_code *errcode_ret);

#if PHYDNN_MINOR_VERSION >= 2

/**
 * Add a 2D deconvolution operation.
 * out_h = stride * (input_h - 1) + filter_h - pad_to_begin[0] - pad_to_end[0];
 * out_w = stride * (input_w - 1) + filter_w - pad_to_begin[1] - pad_to_end[1];
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to deconvolve. 4D: [N, C, H, W]
 * @Input  filter       The filter to use. 4D: [Co, Ci, Hf, Wf]
 *                      Number of input channels must match number of channels in in_tensor.
 *                      Filter width and height must be > 0.
 * @Input  stride       The stride in each dimension. 2D: [Hs, Ws]
 *                      Must be > 0 in both dimensions.
 * @Input  pad_to_begin The padding added before data in each dimension. 2D: [H, W]
 * @Input  pad_to_end   The padding added after data in each dimension. 2D: [H, W]
 * @Input  dilation     The input dilation in each dimension. 2D: [Hd, Wd]
 *                      Must be > 0 in both dimensions.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkDeconvolution2dOp_v2(phydnn_network network,
        phydnn_tensor in_tensor,
        phydnn_tensor filter,
        const unsigned int stride[2],
        const unsigned int pad_to_begin[2],
        const unsigned int pad_to_end[2],
        const unsigned int dilation[2],
        phydnn_err_code *errcode_ret);

#endif

/**
 * Add a 2D deconvolution operation.
 * output_size = ((in_size - 1) * stride) - 2*pad + filter_size
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to deconvolve. 4D: [N, C, H, W]
 * @Input  filter       The filter to use. 4D: [Co, Ci, Hf, Wf]
 *                      Number of input channels must match number of channels in in_tensor.
 *                      Filter width and height must be > 0.
 * @Input  stride       The stride in each dimension. 2D: [Hs, Ws]
 *                      Must be > 0 in both dimensions.
 * @Input  pad          The padding in each dimension. 2D: [Hp, Wp]
 * @Input  dilation     The input dilation in each dimension. 2D: [Hd, Wd]
 *                      Must be > 0 in both dimensions.
 * @Input  partial_size The extra padding to add to the right and bottom of the
 *                      transposed convolution.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkDeconvolution2dOp(phydnn_network network,
                                             phydnn_tensor in_tensor,
                                             phydnn_tensor filter,
                                             const unsigned int stride[2],
                                             const unsigned int pad[2],
                                             const unsigned int dilation[2],
                                             const unsigned int partial_size[2],
                                             phydnn_err_code *errcode_ret);

#if PHYDNN_MINOR_VERSION >= 2
/**
 * Add a grouped 2D deconvolution operation on a set of tensors.
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to deconvolve. 4D: [N, C, H, W]
 * @Input  filter       The filter to use. 4D: [Co/groups, Ci, Hf, Wf]
 *                      Number of input channels must match number of channels in in_tensor.
 *                      Filter width and height must be > 0.
 * @Input  stride       The stride in each dimension. 2D: [Hs, Ws]
 *                      Filter width and height must be > 0.
 * @Input  pad_to_begin The padding added before data in each dimension. 2D: [H, W]
 * @Input  pad_to_end   The padding added after data in each dimension. 2D: [H, W]
 * @Input  dilation     The input dilation in each dimension. 2D: [Hd, Wd]
 *                      Filter width and height must be > 0.
 * @Input  groups       The grouping parameter of the convolution. 1 for a normal convolution.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkGroupedDeconvolution2dOp(phydnn_network network,
                                                    phydnn_tensor in_tensor,
                                                    phydnn_tensor filter,
                                                    const unsigned int stride[2],
                                                    const unsigned int pad_to_begin[2],
                                                    const unsigned int pad_to_end[2],
                                                    const unsigned int dilation[2],
                                                    unsigned groups,
                                                    phydnn_err_code *errcode_ret);
#endif

/**
 * Add 2D pooling operation on a tensor.
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to pool. 4D: [N, C, H, W]
 * @Input  size         The size of pooling window for the 2 end dimensions. Must be > 0.
 * @Input  stride       The pooling stride in each pooled dimension. Must be > 0.
 * @Input  pad          The padding in each pooled dimension.
 * @Input  type         The type of pooling to perform.
 * @Input  with_partial To do pooling with or without partial data.
 *                      If true, all values are used and
 *                      output_size = ceil((input_size + 2 * pad_size - pool_size) / stride) + 1.
 *                      If false, not all values used and
 *                      output_size = floor((input_size + 2 * pad_size - pool_size) / stride) + 1.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkPooling2dOp(phydnn_network network,
                                       phydnn_tensor in_tensor,
                                       const unsigned int size[2],
                                       const unsigned int stride[2],
                                       const unsigned int pad[2],
                                       phydnn_pooling_type type,
                                       bool with_partial,
                                       phydnn_err_code *errcode_ret);

/**
 * Add 2D pooling operation on a tensor.
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to pool. 4D: [N, C, H, W]
 * @Input  size         The size of pooling window for the 2 end dimensions. Must be > 0.
 * @Input  stride       The pooling stride in each pooled dimension. Must be > 0.
 * @Input  pad_to_begin The padding added before data in each dimension. 2D: [H, W]
 * @Input  pad_to_end   The padding added after data in each dimension. 2D: [H, W]
 * @Input  type         The type of pooling to perform.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkPooling2dOp_v2(phydnn_network network,
                                       phydnn_tensor in_tensor,
                                       const unsigned int size[2],
                                       const unsigned int stride[2],
                                       const unsigned int pad_to_begin[2],
                                       const unsigned int pad_to_end[2],
                                       phydnn_pooling_type type,
                                       phydnn_err_code *errcode_ret);

#if PHYDNN_MINOR_VERSION >= 5
/**
 * Add 2D pooling operation on a tensor.
 *
 * @Input  network            Handle to the network.
 * @Input  in_tensor          The input data to pool. 4D: [N, C, H, W]
 * @Input  size               The size of pooling window for the 2 end dimensions. Must be > 0.
 * @Input  stride             The pooling stride in each pooled dimension. Must be > 0.
 * @Input  pad_to_begin       The padding added before data in each dimension. 2D: [H, W]
 * @Input  pad_to_end         The padding added after data in each dimension. 2D: [H, W]
 * @Input  type               The type of pooling to perform.
 * @Input  count_include_pad  If type == PHYDNN_POOLING_AVERAGE
 *                            Whether pooling calculation should take padding value into account
 *                            Ignored for other types.
 * @Output errcode_ret        PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                            PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                            PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkPooling2dOp_v3(phydnn_network network,
                                       phydnn_tensor in_tensor,
                                       const unsigned int size[2],
                                       const unsigned int stride[2],
                                       const unsigned int pad_to_begin[2],
                                       const unsigned int pad_to_end[2],
                                       phydnn_pooling_type type,
                                       bool count_include_pad,
                                       phydnn_err_code *errcode_ret);
#endif

/**
 * Add local response normalisation operation on a tensor.
 *
 * pre_pad = floor((window_size - 1) / 2)
 * post_pad = window_size - pre_pad - 1
 *
 * - Python array slicing is non-inclusive on the right.
 * - Values outside bounds of array are 0.
 *
 * Across:
 * a = alpha / window_size
 * sum_squares[n, c, h, w] = sum(input[n, c - pre_pad : c + post_pad + 1, h, w] ^ 2)
 * output[n, c, h, w] = input[n, c, h, w] / ((k + a * sum_squares[n, c, h, w]) ^ beta)
 *
 * Within:
 * a = alpha / (window_size ^ 2)
 * sum_squares[n, c, h, w] = sum(input[n, c, h - pre_pad : h + post_pad + 1, w - pre_pad : w + post_pad + 1] ^ 2)
 * output[n, c, h, w] = input[n, c, h, w] / ((k + a * sum_squares[n, c, h, w]) ^ beta)
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to normalise. 4D: [N, C, H, W]
 * @Input  type         Across/within channel
 * @Input  window_size  Number of channels to sum over (for cross channel) or
 *                      the side length of the square region to sum over (for
 *                      within channel). Must be > 0.
 * @Input  k            k value
 * @Input  alpha        Scaling Parameter
 * @Input  beta         Exponent
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkLrnOp(phydnn_network network,
                                 phydnn_tensor in_tensor,
                                 phydnn_lrn_type type,
                                 size_t window_size,
                                 float k,
                                 float alpha,
                                 float beta,
                                 phydnn_err_code *errcode_ret);

/**
 * Add projective image transform.
 *
 * output[n, c, h, w] = filter(input[n, c, (b0*x + b1*y + b2) / k, (a0*x + a1*y + a2) / k]),
 * where k = (c0*x + c1*y + 1)
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input data to transform. 4D: [N, C, H, W]
 * @Input  transform    The parameters of the transform. 2-dimensions: (N, 8)
 *                      transform[n] = (a0, a1, a2, b0, b1, b2, c0, c1)
 *                      Must be the same type as in_tensor.
 * @Input  type         The filtering type.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkImageTransformOp(phydnn_network network,
                                            phydnn_tensor in_tensor,
                                            phydnn_tensor transform,
                                            phydnn_image_transform_type type,
                                            phydnn_err_code *errcode_ret);

/**
 * Dimension reduction
 *
 * @Input  network  Handle to the network.
 * @Input  in_tensor  The input data to reduce.
 * @Input  type  The type of reduction to perform.
 * @Input  axis  List of axis to reduce. The dimension of the input tensor
 *               is reduced by 1 for each entry in this list. Each specified axis
 *               must be in the range [0, dimensions - 1].
 * @Input  num_axis  number of items in the axis parameter.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkReduceOp(phydnn_network network,
                                    phydnn_tensor in_tensor,
                                    phydnn_reduce_type type,
                                    const int axis[],
                                    size_t num_axis,
                                    phydnn_err_code *errcode_ret);

/**
 * Compute Softmax Activation
 *
 * softmax[i] = exp((input[i] - max(input, axis)) * beta)/sum_{j}{exp((input[j] - max(input, axis)) * beta), axis}
 * @Input network		Handle to the network
 * @Input in_tensor 	input data
 * @Input beta			beta in calculation
 * @Input axis  		the axis to sum over when computing softmax
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_VALUE for invalid argument (e.g axis
 *                      outside of input tensor dimensions)
 *                      PHYDNN_OUT_OF_MEMORY
 * @Return The output tensor with same shape as input, or NULL on failure
 */
phydnn_tensor phydnnNetworkSoftmaxOp(phydnn_network network,
                                    phydnn_tensor in_tensor,
                                    float beta,
									unsigned int axis,
									phydnn_err_code *errcode_ret);

typedef struct _phydnn_lstm_weight_tensors_t
{
	/* Weights for the input data
	 * Size [input_size, num_units]
	 */
	phydnn_tensor in_to_remember; // Set to NULL for coupled remember/forget
	phydnn_tensor in_to_forget; // Must be non-NULL
	phydnn_tensor in_to_state; // Must be non-NULL
	phydnn_tensor in_to_output; // Must be non-NULL

	/* Weights for previous
	 * Size [output_size, num_units]
	 */
	phydnn_tensor prev_to_remember; // Set to NULL for coupled remember/forget
	phydnn_tensor prev_to_forget; // Must be non-NULL
	phydnn_tensor prev_to_state; // Must be non-NULL
	phydnn_tensor prev_to_output; // Must be non-NULL

	/* Weights for peephole connections, set all to NULL for no-peephole.
	 * Size [num_units]
	 */
	phydnn_tensor state_to_remember; // Set to NULL for coupled remember/forget or no-peephole
	phydnn_tensor state_to_forget; // Set to NULL for no-peephole
	phydnn_tensor state_to_output; // Set to NULL for no-peephole

	/* Bias for each stage
	 * Size [num_units]
	 */
	phydnn_tensor remember_bias; // Set to NULL for coupled remember/forget, otherwise optional
	phydnn_tensor forget_bias; // Optional
	phydnn_tensor state_bias; // Optional
	phydnn_tensor output_bias; // Optional

	/* Output projection weights
	 * Size [num_units, output_size]
	 */
	phydnn_tensor projection; // Set to NULL for no-projection
	/* Output projection bias
	 * Size [output_size]
	 */
	phydnn_tensor projection_bias; // Set to NULL for no-projection, otherwise optional
} phydnn_lstm_weight_tensors;

typedef struct _phydnn_lstm_output_tensors_t
{
	phydnn_tensor output;
	phydnn_tensor state;
} phydnn_lstm_output_tensors;

/**
 * LSTM Cell
 *
 * @Input  network  Handle to the network.
 * @Input  in_tensor        The input data. Size [batch_size, input_size].
 * @Input  prev_tensor      The cell output from previous iteration.
 *                          Size [batch_size, output_size].
 *                          Must be the same type as in_tensor.
 * @Input  state_tensor     The cell state. Size [batch_size, num_units].
 *                          Must be the same type as in_tensor.
 * @Input  weights          Structure of all weight tensors.
 * @Input  state_clip       Value to clip the resultant state.
 * @Input  projection_clip  Value to clip the output projection.
 * @Output errcode_ret      PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                          PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                          PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return Structure of output and state tensors. Tensors will be NULL on failure.
 */
phydnn_lstm_output_tensors phydnnNetworkLSTMOp(phydnn_network network,
		phydnn_tensor in_tensor,
		phydnn_tensor prev_tensor,
		phydnn_tensor state_tensor,
		phydnn_lstm_weight_tensors *weights,
		float state_clip,
		float projection_clip,
		phydnn_err_code *errcode_ret);

/**
 * Rearrange the data from depth into blocks of spatial data
 *
 * @Input network Handle to the network.
 * @Input in_tensor    The input data for depth to space operation. 4D: [N, C, H, W]
 *                     Number of channels should be divisible by block size squared.
 * @Input block_size   It indicates the input block size and how data is moved.
 *                     Must be >= 2.
 * @Output errcode_ret PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                     PHYDNN_OUT_OF_MEMORY or PHYDNN_INVALID_VALUE
 *                     for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkDepthToSpaceOp(phydnn_network network,
		phydnn_tensor in_tensor,
		size_t block_size,
		phydnn_err_code *errcode_ret);

#if PHYDNN_MINOR_VERSION >= 4
/**
 * Add Batch to Space operation on a tensor
 * @Input network      Handle to the network.
 * @Input in_tensor    The input data for batch to space operation. 4D: [N, C, H, W]
 * @Input block_size   The input block_size for each spatial dimension.
 *                     Right now we support two spatial dimensions Height(H) and Width(W).
 *                     1D tensor, block_size[2] = [block_size_H, block_size_W].
 * @Output errcode_ret PHYDNN_SUCCESS, PHYDNN_FAILURE or PHYDNN_INVALID_VALUE
 *                     for invalid tensor or argument
 * @Return  The output tensor or NULL on failure
 */
phydnn_tensor phydnnNetworkBatchToSpaceNDOp(phydnn_network network,
		phydnn_tensor in_tensor,
		phydnn_tensor block_size,
		phydnn_err_code *errcode_ret);
#endif

/**
 * Gather slices of the input tensor along axis based on indices.
 *
 * @Input network       Handle to the network
 * @Input in_tensor     The tensor from which to gather values.
 *                      Rank must be at least 1.
 * @Input indices       The tensor of indices used.
 *                      Rank must be at least 1 and of signed or unsigned integer type.
 * @Input axis          The axis in in_tensor to gather indices from.
 *                      It must be of signed or unsigned integer type.
 *                      It must be less than rank of in_tensor.
 * @Output errcode_ret  PHYDNN_FAILURE, PHYDNN_SUCCESS or PHYDNN_INVALID_VALUE
 *                      for invalid tensor or argument.
 * @Return              The ouput tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkGatherOp(phydnn_network network,
                                  phydnn_tensor in_tensor,
                                  phydnn_tensor indices,
                                  unsigned int axis,
                                  phydnn_err_code *errcode_ret);

/**
 * Add Space to Depth operation on a tensor
 *
 * @Input  network      Handle to the network.
 * @Input  in_tensor    The input tensor to operate on. 4D: [N, C, H, W]
 *                      Width and height must be divisible by block_size.
 * @Input  block_size   The size of the spatial block.
 *                      Must be >= 2.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_OUT_OF_MEMORY or PHYDNN_INVALID_VALUE for
 *                      invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkSpaceToDepthOp(phydnn_network network,
                                          phydnn_tensor in_tensor,
                                          size_t block_size,
                                          phydnn_err_code *errcode_ret);
#if PHYDNN_MINOR_VERSION >= 4
/**
 * Add Space to Batch operation on a tensor
 *
 * @Input network      Handle to the network.
 * @Input in_tensor    The input tensor to operate on 4D: [N, C, H, W]
 * @Input in_padding   The input padding 2D tensor for each spatial dimension.
 *                     Right now we support only two spatial dimension[H, W].
 *                     in_padding[2][2] = [padding_start_H, padding_end_H,
 *                                         padding_start_W, padding_end_W].
 * @Input block_size    The input block_size 1D tensor.
 *                      block_size[2] = [block_size_H, block_size_W]
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_INVALID_VALUE for
 *                      invalid tensor or argument.
 * @Return               The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkSpaceToBatchNDOp(phydnn_network network,
                                            phydnn_tensor in_tensor,
                                            phydnn_tensor in_padding,
                                            phydnn_tensor block_size,
                                            phydnn_err_code *errcode_ret);
#endif

/**
 * Add Resize Image using Bilinear interpolation.
 *
 * @Input  network        Handle to the network.
 * @Input  in_tensor      The 4D input tensor to resize, in format [N, C, H, W].
 * @Input  height         New height of the output tensor.  Must be > 0.
 * @Input  width          New width of the output tensor.  Must be > 0.
 * @Input  align_corners  If True, the centers of the 4 corner pixels of the
 *                        input and output tensors are aligned, preserving
 *                        the values at the corner pixels.
 * @Output errcode_ret    PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                        PHYDNN_OUT_OF_MEMORY or PHYDNN_INVALID_VALUE
 *                        for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkResizeBilinearOp(phydnn_network network,
					    phydnn_tensor in_tensor,
					    unsigned int height,
					    unsigned int width,
					    bool align_corners,
					    phydnn_err_code *errcode_ret);

/**
 * Pad tensor
 * @Input network		Handle to network
 * @Input in_tensor		tensor to be padded
 * @Input pad_before	value at idx N correspond to the number of values to add before
 * 						tensor contents in dimensions N. The length of this array must be
 * 						equal to the input tensor's number of dimensions.
 * @Input pad_after		value at idx N correspond to the number of values to add after
 * 						tensor contents in dimensions N. The length of this array must be
 * 						equal to the input tensor's number of dimensions.
 * @Input pad_value		constant pad value.  This float value is converted to the type of
 * 						in_tensor when used for padding
 * @Output errcode_ret	PHY_SUCCESS,
 *                      CL_INVALID_VALUE for invalid tensor or other arguments,
 *                      PHYDNN_OUT_OF_MEMORY. May be NULL.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkPadOp(phydnn_network network,
						phydnn_tensor in_tensor,
						const unsigned int pad_before[],
						const unsigned int pad_after[],
						float pad_value,
						phydnn_err_code *errcode_ret);

#if PHYDNN_MINOR_VERSION >= 2

typedef enum _phydnn_pad_mode_t
{
  PHYDNN_PAD_CONSTANT,
  PHYDNN_PAD_SYMMETRIC,
  PHYDNN_PAD_REFLECT
} phydnn_pad_mode;


/**
 * Pad tensor
 * @Input network     Handle to network
 * @Input in_tensor   tensor to be padded
 * @Input pad_before  value at idx N correspond to the number of values to add before
 *            tensor contents in dimensions N. The length of this array must be
 *            equal to the input tensor's number of dimensions.
 * @Input pad_after   value at idx N correspond to the number of values to add after
 *            tensor contents in dimensions N. The length of this array must be
 *            equal to the input tensor's number of dimensions.
 * @Input pad_value   constant pad value.  This float value is converted to the type of
 *            in_tensor when used for padding
 * @Input pad_mode    Padding mode (constant, symmetric, or reflect)
 * @Output errcode_ret  PHY_SUCCESS,
 *                      CL_INVALID_VALUE for invalid tensor or other arguments,
 *                      PHYDNN_OUT_OF_MEMORY. May be NULL.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkPadOp_v2(phydnn_network network,
                                    phydnn_tensor in_tensor,
                                    const unsigned int pad_before[],
                                    const unsigned int pad_after[],
                                    float pad_value,
                                    phydnn_pad_mode pad_mode,
                                    phydnn_err_code *errcode_ret);
#endif

#if PHYDNN_MINOR_VERSION >= 3

/**
 * Add Resize Image using Nearest Neighbour interpolation.
 *
 * @Input  network        Handle to the network.
 * @Input  in_tensor      The 4D input tensor to resize, in format [N, C, H, W].
 * @Input  height         New height of the output tensor.  Must be > 0.
 * @Input  width          New width of the output tensor.  Must be > 0.
 * @Input  align_corners  If True, the centers of the 4 corner pixels of the
 *                        input and output tensors are aligned, preserving
 *                        the values at the corner pixels.
 * @Output errcode_ret    PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                        PHYDNN_OUT_OF_MEMORY or PHYDNN_INVALID_VALUE
 *                        for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkResizeNearestNeighbourOp(phydnn_network network,
					    phydnn_tensor in_tensor,
					    unsigned int height,
					    unsigned int width,
					    bool align_corners,
					    phydnn_err_code *errcode_ret);

#endif

#if PHYDNN_MINOR_VERSION >= 4

/**
 * Add ROI Pooling.
 *
 * @Input  network		Handle to the network.
 * @input  in_tensor		A 4D input feature map, in format [N, C, H, W].
 * @input  rois_tensor		A 2D input tensor, in format [num_rois, 4],
 *				specifying the locations of Region of Interests in the form {x1, y1, x2, y2}.
 *				An ROI is represented by its upper-left coordinate (x1, y1) and
 *				lower-right coordinate (x2, y2) in the original image.
 *				A spatial scaling factor is applied to map into feature map coordinate.
 *				A valid region of interest should satisfy x1 <= x2 and y1 <= y2.
 *
 * @input  batch_idx		A 1D input tensor specifying batch indices of each box (ROI).
 * @input  out_height		A scalar, the height of the output tensor.
 * @input  out_width		A scalar, the width of the output tensor.
 * @input  scaled_height	A scalar, specifying the ratio from the height of original image
 *				to the height of feature map.
 * @input  scaled_width		A scalar, specifying the ratio from the width of original image
 *				to the width of feature map.
 *
 * @Output errcode_ret    	PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY
 *                        	or PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor in format [num_rois, C, out_height, out_width] or NULL on failure.
 */
phydnn_tensor phydnnNetworkROIPoolingOp(phydnn_network network,
					phydnn_tensor in_tensor,
					phydnn_tensor roi_tensor,
					phydnn_tensor batch_idx_tensor,
					unsigned int out_height,
					unsigned int out_width,
					float scaled_height,
					float scaled_width,
					phydnn_err_code *errcode_ret);

/**
 * Add ROI Align.
 *
 * @Input  network		Handle to the network.
 * @input  in_tensor		A 4D input feature map, in format [N, C, H, W].
 * @input  rois_tensor		A 2D input tensor, in format [num_rois, 4],
 *				specifying the locations of Region of Interests in the form {x1, y1, x2, y2}.
 *				An ROI is represented by its upper-left coordinate (x1, y1) and
 *				lower-right coordinate (x2, y2) in the original image.
 *				A spatial scaling factor is applied to map into feature map coordinate.
 *				A valid region of interest should satisfy x1 <= x2 and y1 <= y2.
 *
 * @input  batch_idx		A 1D input tensor specifying batch indices of each box (ROI).
 * @input  out_height		A scalar, the height of the output tensor.
 * @input  out_width		A scalar, the width of the output tensor.
 * @input  scaled_height	A scalar, specifying the ratio from the height of original image
 *				to the height of feature map.
 * @input  scaled_widtht	A scalar, specifying the ratio from the width of original image
 *				to the width of feature map.
 * @input  num_samples_height	A scalar, the number of sampling points in height dimension used to compute the output.
 *				Set to 0 for adaptive value of ceil(roi_height/out_height).
 * @input  num_samples_width	A scalar, the number of sampling points in width dimension used to compute the output.
 *				Set to 0 for adaptive value of ceil(roi_width/out_width).
 *
 * @Output errcode_ret    	PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY
 *				or PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor in format [num_rois, C, out_height, out_width] or NULL on failure.
 */
phydnn_tensor phydnnNetworkROIAlignOp(phydnn_network network,
				      phydnn_tensor in_tensor,
				      phydnn_tensor roi_tensor,
				      phydnn_tensor batch_idx_tensor,
				      unsigned int out_height,
				      unsigned int out_width,
				      float scaled_height,
				      float scaled_width,
				      unsigned int num_samples_height,
				      unsigned int num_samples_width,
				      phydnn_err_code *errcode_ret);

#endif

#if PHYDNN_MINOR_VERSION >= 4
/**
 * Enum to list commonly defined data layouts.
 */
typedef enum _phydnn_predefined_data_layout_t {
  RGBA,
  PLANAR_RGB,
  PLANAR_BGR
} phydnn_predefined_data_layout;

/**
 * Fill the data layout parameters structure based on the specified pre-defined layout.
 *
 * @Input  data_layout   Pointer to the data layout structure to be filled.
 * @Input  predef_layout Pre-defined data layout to be set for this tensor.
 * @Input  tensor_desc   Tensor descriptor to check that the predef layout is compatible with the tensor.
 * @Return PHYDNN_SUCCESS or PHYDNN_INVALID_VALUE for invalid tensor or invalid pre-defined data layout.
 */
phydnn_err_code phydnnFillDataLayoutParameters(phydnn_data_layout_param *data_layout,
                                               phydnn_predefined_data_layout predef_layout,
                                               const phydnn_tensor_descriptor* const desc);
#endif

/**
 * Stores serialised compiled binary object
 */
typedef struct _phydnn_network_binary_t {
    size_t size;
    void *data;
} phydnn_network_binary;

/**
 * Produces a network binary used for storing to a file or loading for
 * execution.
 *
 * The call to phydnnNetworkBinaryDestroy() is needed to deallocate the memory
 *
 * @Input  device       The PHYDNN device for which to create this network.
 * @Input  context      The PHYDNN context in which to create this network output.
 * @Input  network      Network object for which object is being created.
 * @Input  num_inputs   The number of elements in the input tensor array.
 * @Input  inputs       Array of tensors to use as network inputs.
 * @Input  num_outputs  The number of elements in the output tensor array.
 * @Input  outputs      Array of tensors to create network outputs from.
 * @Input  flags        Flags specifying the options to create the network object.
 * @Input  options      String of options to pass to the graph compiler,
 *                      device dependent.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument or
 *                      PHYDNN_INVALID_DEVICE for invalid device.
 * @Return The output network binary. Returns {0, NULL} on failure.
 */
phydnn_network_binary phydnnCreateNetworkBinary(const phydnn_device device,
                                                const phydnn_context context,
                                                const phydnn_network network,
                                                unsigned int num_inputs,
                                                const phydnn_tensor inputs[],
                                                unsigned int num_outputs,
                                                const phydnn_tensor outputs[],
                                                const phydnn_network_object_flags flags,
                                                const char *options,
                                                phydnn_err_code *errcode_ret);

#if PHYDNN_MINOR_VERSION >= 4
/**
 * Produces a network binary used for storing to a file or loading for
 * execution.
 *
 * The call to phydnnNetworkBinaryDestroy() is needed to deallocate the memory
 *
 * @Input  device         The PHYDNN device for which to create this network.
 * @Input  context        The PHYDNN context in which to create this network output.
 * @Input  network        Network object for which object is being created.
 * @Input  num_inputs     The number of elements in the input tensor array.
 * @Input  inputs         Array of tensors to use as network inputs.
 * @Input  inputs_layout  Array of data layout parameters for the specified input tensors.
 * @Input  num_outputs    The number of elements in the output tensor array.
 * @Input  outputs        Array of tensors to create network outputs from.
 * @Input  outputs_layout Array of data layout parameters for the specified output tensors.
 * @Input  flags          Flags specifying the options to create the network object.
 * @Input  options        String of options to pass to the graph compiler,
 *                        device dependent.
 * @Output errcode_ret    PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY or
 *                        PHYDNN_INVALID_VALUE for invalid tensor or argument or
 *                        PHYDNN_INVALID_DEVICE for invalid device.
 * @Return The output network binary. Returns {0, NULL} on failure.
 */
phydnn_network_binary phydnnCreateNetworkBinary_v2(const phydnn_device device,
                                                   const phydnn_context context,
                                                   const phydnn_network network,
                                                   unsigned int num_inputs,
                                                   const phydnn_tensor inputs[],
                                                   const phydnn_data_layout_param inputs_layout[],
                                                   unsigned int num_outputs,
                                                   const phydnn_tensor outputs[],
                                                   const phydnn_data_layout_param outputs_layout[],
                                                   const phydnn_network_object_flags flags,
                                                   const char *options,
                                                   phydnn_err_code *errcode_ret);
#endif

/**
 * Destroy compiled network binary.
 *
 * @Input  binary  The network binary object to destroy.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE or PHYDNN_INVALID_VALUE
 */
phydnn_err_code phydnnNetworkBinaryDestroy(phydnn_network_binary *binary);

#if PHYDNN_MINOR_VERSION >= 5
/**
 * Allocates a new phydnn_per_axis_quant_param structure and its internal arrays.
 * After being used the memory needs to be freed with phydnnDestroyPerAxisQuantParam function.
 *
 * @Input  axis        axis along which the quantization parameters are applied
 * @Input  count       number of scales/zero_points
 * @Input  scales      array of integers representing the scales of the quantization parameters
 *                     along the specified axis. If NULL the internal array is allocated and filled
 *                     with 1.0f
 * @Input  zero_points array of integers representing the zero_points of the quantization parameters
 *                     along the specified axis. If NULL the internal array is allocated and filled
 *                     with 0
 *
 * @Return pointer to the new and filled phydnn_per_axis_quant_param structure or NULL in case of
 *         failure
 */
phydnn_per_axis_quant_param *phydnnCreatePerAxisQuantParam(unsigned axis,
                                                           unsigned count,
                                                           const float *scales,
                                                           const int *zero_points);

/**
 * Frees the memory used by the phydnn_per_axis_quant_param structure and its internal arrays.
 *
 * @Input  pa_param parameters structure to destroy
 */
void phydnnDestroyPerAxisQuantParam(phydnn_per_axis_quant_param *pa_param);
#endif


/***** Get capabilities API ***********************************************************************/

/* Level of support for operations */
typedef enum _phydnn_support_level_
{
  PHYDNN_UNSUPPORTED_OPERATION,
  PHYDNN_SUPPORTED_BY_HW,
  PHYDNN_SUPPORTED_BY_SW
} phydnn_support_level;

/**
 * Structure to contain support levels for different operations and devices.
 * We do not replicate the list of devices, as the user will have created it prior to calling
 * one of the functions.
 * support_levels[d][j] indicates how operation j (in operation_names) is supported on device d
 */
struct _phydnn_device_capabilities_t
{
  unsigned num_operations;
  unsigned num_devices;
  char** operation_names;
  phydnn_support_level** support_levels;
};

#if PHYDNN_MINOR_VERSION >= 7
/**
 * Return support levels of the network's operations for the given devices.
 * If the network has been created from IR, the operation names will be the ones of the IR.
 * If it has not, the operation names will be created internally by PHYDNN.
 *
 * @Input network        Handle to the network
 * @Input num_devices    Number of devices to check capabilities for
 * @Input device_types   Array containing device types to check
 * @Input device_options Array containing device options (nullptr, HW config filepath, ...)
 * @Output errcode_ret   PHYDNN_SUCCESS or error code
 * @Return               Capabilities structure
 */
phydnn_device_capabilities phydnnGetDeviceCapabilities(phydnn_network network,
                                                       unsigned num_devices,
                                                       const phydnn_device_type* device_types,
                                                       const void** device_options,
                                                       phydnn_err_code* err);

/* Destroy the given capabilities structure. */
phydnn_err_code phydnnDeviceCapabilitiesDestroy(phydnn_device_capabilities capabilities);

#endif /* PHYDNN_MINOR_VERSION >= 7 */

/** @} */
#ifdef __cplusplus
}
#endif
#endif /* _PHY_DNN_COMPILE_H */
