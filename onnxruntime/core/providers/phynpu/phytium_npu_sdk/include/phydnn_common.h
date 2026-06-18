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
#ifndef _PHY_DNN_COMMON_H
#define _PHY_DNN_COMMON_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(PHYDNN_MAJOR_VERSION)
#define PHYDNN_MAJOR_VERSION 1
#endif

#if !defined(PHYDNN_MINOR_VERSION)
#define PHYDNN_MINOR_VERSION 10
#endif

#define PHYDNN_DESCRIPTOR_MAX_DIM (6u)
#define PHYDNN_CTX_FLAGS_NONE (0)
#define PHYDNN_NETWORK_OBJ_FLAG_NONE (0U)
#define PHYDNN_NETWORK_OBJ_FLAG_BLOCKING_OPTIMISATION (1U << 0)

typedef unsigned phydnn_context_flags;
typedef unsigned phydnn_network_object_flags;

typedef struct _phydnn_device_t *phydnn_device;
typedef struct _phydnn_context_t *phydnn_context;
typedef struct _phydnn_memory_t *phydnn_memory;
typedef struct _phydnn_tensor_t *phydnn_tensor;
typedef struct _phydnn_network_t *phydnn_network;
typedef struct _phydnn_input_t *phydnn_input;
typedef struct _phydnn_output_t *phydnn_output;
typedef struct _phydnn_network_object_t *phydnn_network_object;

typedef enum _phydnn_device_type_t
{
  PHYDNN_DEVICE_TYPE_CPU,
  PHYDNN_DEVICE_TYPE_GPU,
  PHYDNN_DEVICE_TYPE_ACCELERATOR,
  PHYDNN_DEVICE_TYPE_ALL
} phydnn_device_type;

typedef enum _phydnn_device_info_t
{
  PHYDNN_DEVICE_TYPE,
  PHYDNN_DEVICE_ID,
  PHYDNN_DEVICE_BVNC,

} phydnn_device_info;

typedef enum _phydnn_err_code_
{
  PHYDNN_SUCCESS = 0,
  PHYDNN_FAILURE,
  PHYDNN_INVALID_DEVICE,
  PHYDNN_INVALID_CONTEXT,
  PHYDNN_INVALID_VALUE,
  PHYDNN_INVALID_OPERATION,
  PHYDNN_OUT_OF_MEMORY,
  PHYDNN_UNSUPPORTED
} phydnn_err_code;

typedef enum _phydnn_report_flags_
{
  PHYDNN_REPORT_VERBOSE,
  PHYDNN_REPORT_INFO,
  PHYDNN_REPORT_WARNING,
  PHYDNN_REPORT_ERROR
} phydnn_report_flags;

typedef enum _phydnn_type_t
{
  PHYDNN_TYPE_I8,
  PHYDNN_TYPE_U8,
  PHYDNN_TYPE_I16,
  PHYDNN_TYPE_U16,
  PHYDNN_TYPE_I32,
  PHYDNN_TYPE_U32,
  PHYDNN_TYPE_F16,
  PHYDNN_TYPE_F32,
  PHYDNN_TYPE_Q_I8,
  PHYDNN_TYPE_Q_U8,
#if PHYDNN_MINOR_VERSION >= 5
  PHYDNN_TYPE_QPA_I8,
  PHYDNN_TYPE_QPA_U8,
#endif
  PHYDNN_TYPE_MAX
} phydnn_type;

#if PHYDNN_MINOR_VERSION >= 4
typedef enum _phydnn_dimensions_order_t
{
  PHYDNN_UNKNOWN,
  PHYDNN_NCHW,
  PHYDNN_NHWC
} phydnn_dimensions_order;

/**
 * A structure to hold data layout parameters.
 * 'data_stride' array of per dimension strides, in bytes. Should have the same number of dimensions as the tensor. Can be 0 (contiguous data) or -1 (device can decide the best option)
 * 'interleave' amount of data (elements) from 2nd dimension interleaved at innermost dimension (only makes sense for 4D tensors).
 * 'order' array of chars specifying the data order. This is optional and currently only 4D orders "NCHW" and "NHWC" are supported. Set to "\0" when not used
 */
typedef struct _phydnn_data_layout_param_t {
  long long int data_stride[PHYDNN_DESCRIPTOR_MAX_DIM];
  uint32_t interleave;
  phydnn_dimensions_order order;
} phydnn_data_layout_param;
#endif

#if PHYDNN_MINOR_VERSION >= 5
/**
 * A structure to hold per axis quantization parameters 'scale' and 'zero_point'.
 *       real_value = scale * (quantized_value - zero_point)
 * 'scales'      is an array of per axis scales. This is only used for tensors of
 *               PHYDNN_TYPE_QPA_*
 * 'zero_points' is an array of per axis scales. This is only used for tensors of
 *               PHYDNN_TYPE_QPA_*
 * 'axis'        the axis to which the scales and zero_points apply.
 * 'count'       number of scales and zero_points. Should match the dimension of the tensor in the
 *               specified axis
 */
typedef struct _phydnn_per_axis_quant_param_t
{
  float *scales;
  int *zero_points;
  unsigned axis;
  unsigned count;
} phydnn_per_axis_quant_param;
#endif

/**
 * A structure to hold quantization parameters 'scale' and 'zero_point'.
 *       real_value = scale * (quantized_value - zero_point)
 * 'zero_point'     is the quantized value that corresponds to the real value 0
 * 'scale'          is the difference of real values corresponding to consecutive quantized values,
 *                  i.e. the quantization step
 * 'per_axis'       pointer to structure containing per axis quantization parameters. Only used
 *                  by PHYDNN_TYPE_QA* type tensors. Should be created and destroyed using the
 *                  appropriate functions.
 *                  \sa phydnn_per_axis_quant_param,phydnnCreatePerAxisQuantParam,phydnnDestroyPerAxisQuantParam
 */
typedef struct _phydnn_quant_param_t
{
  union {
    struct {
      float scale;
      int zero_point;
    };
#if PHYDNN_MINOR_VERSION >= 5
    phydnn_per_axis_quant_param *per_axis;
#endif
  };
} phydnn_quant_param;

/* Tensor **********************************************************/

/* A tensor is an opaque memory region. It may correspond to actual memory, or
 * it may be eliminated in whole network optimisation and hence not actually exist
 * when executing. In either case its use is to connect nodes together
 * logically.
 *
 * A tensor layout is described by a tensor descriptor which fully defines the
 * layout. This allows the user to access data in a memory object without necessarily
 * needing to pass through an opaque tensor. This is useful for custom kernels
 * in a network.
 *
 * The layout of all tensors can be considered to be linearly ordered in row
 * major order and alignment of 16 bytes.
 */
typedef struct _phydnn_tensor_descriptor_t
{
  unsigned int dimensions;
  phydnn_type type;
  size_t size[PHYDNN_DESCRIPTOR_MAX_DIM];
  phydnn_quant_param quant_param;
} phydnn_tensor_descriptor;

#if PHYDNN_MINOR_VERSION >= 9
/* Available network priorities. */
typedef enum _phydnn_network_priority_t
{
    PHYDNN_NETWORK_PRIORITY_LOWEST = 0,
    PHYDNN_NETWORK_PRIORITY_MEDIUM,
    PHYDNN_NETWORK_PRIORITY_HIGHEST
} phydnn_network_priority;
#endif

/* Network Object **********************************************************/

/* Network Object creation functions. These functions compile and then load objects for the phy_npu
so they're handled as common functions. They are only available on full phydnn library so we guard
them against compilation on exec-only lib*/
#ifndef PHYDNN_EXECUTE_ONLY
/**
 * Produces a network object used for execution of network.
 *
 * The difference between using multiple outputs here, or multiple calls with
 * single outputs is the network optimisation. A tensor that is not an output
 * may be elided completely.
 *
 * @Input  device         The PHYDNN device for which to create this network.
 * @Input  context        The PHYDNN context in which to create this network output.
 * @Input  network        Network object for which object is being created.
 * @Input  num_inputs     The number of elements in the input tensor array.
 * @Input  inputs         Array of tensors to use as network inputs.
 * @Input  num_outputs    The number of elements in the output tensor array.
 * @Input  outputs        Array of tensors to create network outputs from.
 * @Input  flags          Flags specifying the options to create the network object.
 * @Input  options        String of options to pass to the graph compiler,
 *                        device dependent.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument or
 *                      PHYDNN_INVALID_DEVICE for invalid device.
 * @Return The output network object node for execution or NULL on failure.
 */
phydnn_network_object phydnnCreateNetworkObject(const phydnn_device device,
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
 * Produces a network object used for execution of network.
 *
 * The difference between using multiple outputs here, or multiple calls with
 * single outputs is the network optimisation. A tensor that is not an output
 * may be elided completely.
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
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument or
 *                      PHYDNN_INVALID_DEVICE for invalid device.
 * @Return The output network object node for execution or NULL on failure.
 */
phydnn_network_object phydnnCreateNetworkObject_v2(const phydnn_device device,
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

#if PHYDNN_MINOR_VERSION >= 9
/**
 * Produces a prioritised network object used for execution of network.
 *
 * This function is a version of the phydnnCreateNetworkObject_v2() extended
 * with the network priority level.
 *
 * The priority level for a network created with phydnnCreateNetworkObject*()
 * functions is always set to PHYDNN_NETWORK_PRIORITY_LOWEST. This function
 * allows for creating networks with raised priority level.
 *
 * The priority level set for a network should be only treated as a "hint"
 * for execution scheduling algorithms.
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
 * @Input  priority       Network priority level.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY or
 *                      PHYDNN_INVALID_VALUE for invalid tensor or argument or
 *                      PHYDNN_INVALID_DEVICE for invalid device.
 * @Return The output network object node for execution or NULL on failure.
 */
phydnn_network_object phydnnCreatePrioritisedNetworkObject(const phydnn_device device,
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
                                                   phydnn_network_priority priority,
                                                   phydnn_err_code *errcode_ret);
#endif
#endif /* FULL_PHYDNN_BUILD */

/**
 * Destroy a Network Object previously created.
 *
 * @Input  network_object  A network object created previously.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *         PHYDNN_INVALID_VALUE for invalid network object
 */
phydnn_err_code phydnnNetworkObjectDestroy(phydnn_network_object network_object);

/**
 * Get the list of available devices.
 *
 * @Input  device_type  The type of requested devices.
 * @Input  max_devices  The maximum number of devices that can be added to devices list if not NULL.
 * @Output devices      The list of returned PHYDNN devices if not NULL
 * @Output num_devices  The number of devices of a particular type that are present. May not be NULL.
 * @Return PHYDNN_SUCCESS on success. PHYDNN_INVALID_VALUE or PHYDNN_FAILURE for failure.
 */
phydnn_err_code phydnnGetDevices(phydnn_device_type device_type,
                                 unsigned int max_devices,
                                 phydnn_device devices[],
                                 unsigned int *num_devices);

/**
 * Get the  specific information of a particular device.
 *
 * @Input  device                 The device for which information is requested.
 * @Input  device_info            The identifier for the device information being requested.
 * @Input  device_info_data_size  The size in bytes of memory pointed by device_info_data.
 * @Output device_info_data       Pointer to memory location where device info data is stored. May not be NULL.
 * @Return PHYDNN_SUCCESS on success. PHYDNN_INVALID_VALUE or PHYDNN_FAILURE for
 * failure
 */
phydnn_err_code phydnnGetDeviceInfo(phydnn_device device,
                                    phydnn_device_info device_info,
                                    size_t device_info_data_size,
                                    void *device_info_data);

/**
 * Create a PHYDNN context from a number of PHYDNN devices.
 *
 * @Input  num_devices    Number of devices in the device array. Must be > 0
 * @Input  devices[]      An array of PHYDNN devices for which DNN context is created.
 * @Input  context_flags  Flags to modify how the device context behaves.
 * @Output errcode_ret    PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_INVALID_DEVICE
 * @Return The returned PHYDNN context created or NULL on failure.
 */
phydnn_context phydnnCreateContext(unsigned int num_devices,
                                   const phydnn_device devices[],
                                   const phydnn_context_flags context_flags,
                                   phydnn_err_code *errcode_ret);

/**
 * Destroy a previously created PHYDNN context.
 *
 * @Input  context  A previously created PHYDNN context
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_INVALID_CONTEXT
 */
phydnn_err_code phydnnContextDestroy(phydnn_context context);

/**
 * Utility function to get size of a descriptor
 *
 * @Input  descriptor   The descriptor to find total size of
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid descriptor.
 *                      May be NULL.
 * @Return The total size of the data referenced by this descriptor.
 */
size_t phydnnGetDescriptorSize(const phydnn_tensor_descriptor *const descriptor,
                               phydnn_err_code *errcode_ret);

/**
 * Utility function to obtain the descriptor of an input.
 *
 * If the tensor is of type PHYDNN_TYPE_QPA_* the caller is resposible of destroying the
 * phydnn_per_axis_quant_param structure (ret.quant_param->per_axis) by calling
 * phydnnDestroyPerAxisQuantParam
 *
 * @Input  input        An input previously created.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid input.
 * @Return The descriptor of the given input.
 */
phydnn_tensor_descriptor phydnnGetInputDescriptor(phydnn_input input,
                                                  phydnn_err_code *errcode_ret);

/**
 * Utility function to obtain the descriptor of an output.
 *
 * If the tensor is of type PHYDNN_TYPE_QPA_* the caller is resposible of destroying the
 * phydnn_per_axis_quant_param structure (ret.quant_param->per_axis) by calling
 * phydnnDestroyPerAxisQuantParam
 *
 * @Input  output       An output previously created.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE for invalid output.
 * @Return The descriptor of the given output.
 */
phydnn_tensor_descriptor phydnnGetOutputDescriptor(phydnn_output output,
                                                   phydnn_err_code *errcode_ret);

/**
 * Callback function typedef for PHYDNN error reporting functions. The callback function
 * should be made thread-safe by the user.
 *
 * @Input   flags               E.g. Information/warning/error/verbose/etc.
 * @Input   tensor_names        Array of tensor names where the error occurred.
 * @Input   num_tensor_names    Number of tensor names in tensor_names array
 * @Input   error_code          Error code for the error which caused the callback to be called.
 * @Input   error_message       Error message for the error which caused the callback to be called.
 */
typedef void (*PFN_phydnn_debug_report_callback)(phydnn_report_flags        flags,
                                                 const char**                     tensor_names,
                                                 int                       num_tensor_names,
                                                 phydnn_err_code            error_code,
                                                 const char*                error_message);

/**
 * Set the error handling function. This fuction is called by the driver when
 * error occurs in the driver and will provide some information about the
 * natute of the error. See definition of PFN_phydnn_debug_report_callack for
 * more detail.
 *
 * @Input   err_callback      The callback function to use.
 * @Return  PHYDNN_SUCCESS if the error handling function has been set successfully, else returns PHYDNN_FAILURE
 */
phydnn_err_code phydnnSetErrorHandler(PFN_phydnn_debug_report_callback err_callback);

/**
 * Return the phydnn API version.
 *
 * @Input  api_version    Pointer to the cstring that will hold the API version.
 * @Input  param_size     When api_version is not NULL this parameter is the size of the input string and it is used for
 *                        size checking. If api_version is NULL param_size is used to return the size of the version.
 * @Return PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE
 */
phydnn_err_code  phydnnGetApiVersion(char *api_version, size_t *param_size);

/**
 * Return the phydnn version.
 *
 * @Input  driver_version  Pointer to the cstring with the phydnn version.
 * @Input  param_size      When driver_version is not NULL this parameter is the size of the input string and it is used for
 *                         size checking. If driver_version is NULL param_size is used to return the size of the version.
 * @Return PHYDNN_SUCCESS, PHYDNN_INVALID_VALUE,
 *         PHYDNN_OUT_OF_MEMORY.
 */
phydnn_err_code  phydnnGetDriverVersion(char *driver_version, size_t *param_size);

/** @} */
#ifdef __cplusplus
}
#endif
#endif /* _PHY_DNN_COMMON_H */
