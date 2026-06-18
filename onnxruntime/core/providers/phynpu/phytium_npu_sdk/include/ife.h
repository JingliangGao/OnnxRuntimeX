/*************************************************************************/ /*!
@Title          ife.h
@Copyright      Copyright (c) Phytium Technology Co., Ltd. All Rights Reserved
@Description    Top level header file to create DNN Internal Graph Using NNVM
@CodingStyle    Google
@License        Strictly Confidential.
*/ /**************************************************************************/
#ifndef PHY_IFE_H_
#define PHY_IFE_H_

#include <iomanip>
#include <iterator>
#include <mutex>
#include <sstream>
#include <vector>


// Internal only
phydnn_tensor phydnnNetworkFixedScalar(phydnn_network network,
		double scalar,
		phydnn_err_code *errcode_ret);

/**
 * Creates a network output tensor, not meant to be reused as input
 * by another op in this network.
 *
 * @Input  network  Handle to the network.
 * @Input  in_tensor  The original network output tensor.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor, meant to be input to no other operation, or NULL on failure.
 */
phydnn_tensor phydnnNetworkOutput(phydnn_network network,
        phydnn_tensor in_tensor,
        phydnn_err_code *errcode_ret);

/**
 * Add data-type cast operation of a tensor.
 *
 * @Input  network              Handle to the network.
 * @Input  tensor               The input data to cast. May not be NULL.
 * @Input  dst_type             phydnn_type to cast to
 * @Input  dst_quant_param      quant params if the dst_type is quantized else NULL
 * @Input  explicit_conversion  Whether this cast performs an explicit mandatory conversion or not
 *
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid tensor,
 *                      PHYDNN_FAILURE or PHYDNN_OUT_OF_MEMORY.
 * @Return The output type casted tensor or NULL on failure.
 */
extern "C"
phydnn_tensor phydnnNetworkCastOp_v2(phydnn_network network,
                                  phydnn_tensor tensor,
                                  phydnn_type dst_type,
                                  const phydnn_quant_param *const dst_quant_param,
                                  bool explicit_conversion,
                                  phydnn_err_code *errcode_ret);

/**
 * Add a 2D deconvolution operation.
 * out_h = stride * (input_h - 1) + filter_h - pad_to_begin[0] - pad_to_end[0] + out_pad[0];
 * out_w = stride * (input_w - 1) + filter_w - pad_to_begin[1] - pad_to_end[1] + out_pad[1];
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
 * @Input  out_pad      Adjustment to the output shape. For convolution, several input shapes can
 *                      lead to the same output shape. So, for transposed convolution, several output
 *                      shapes should be reachable from a single input shape. 2D: [H, W]
 *
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkDeconvolution2dOp_v3(phydnn_network network,
        phydnn_tensor in_tensor,
        phydnn_tensor filter,
        const unsigned int stride[2],
        const unsigned int pad_to_begin[2],
        const unsigned int pad_to_end[2],
        const unsigned int dilation[2],
        const unsigned int out_pad[2],
        phydnn_err_code *errcode_ret);

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
 * @Input  out_pad      Adjustment to the output shape. For convolution, several input shapes can
 *                      lead to the same output shape. So, for transposed convolution, several output
 *                      shapes should be reachable from a single input shape. 2D: [H, W]
 * @Input  groups       The grouping parameter of the convolution. 1 for a normal convolution.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_OPERATION, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument.
 * @Return The output tensor or NULL on failure.
 */
phydnn_tensor phydnnNetworkGroupedDeconvolution2dOp_v2(phydnn_network network,
                                                    phydnn_tensor in_tensor,
                                                    phydnn_tensor filter,
                                                    const unsigned int stride[2],
                                                    const unsigned int pad_to_begin[2],
                                                    const unsigned int pad_to_end[2],
                                                    const unsigned int dilation[2],
                                                    const unsigned int out_pad[2],
                                                    unsigned groups,
                                                    phydnn_err_code *errcode_ret);

#endif /* PHY_IFE_H_ */
