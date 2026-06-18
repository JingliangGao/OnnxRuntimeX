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
#ifndef _PHY_DNN_EXECUTE_H
#define _PHY_DNN_EXECUTE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "phydnn_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup PHYDNN PHYDNN Execution API
 *
 * @{
 * PHYDNN API - The PHYDNN Execution API.
 *
 * The Phytium Neural Network interface is used for controlling the Phytium
 * Neural Network GPU and hardware acceleration solutions.
 */

typedef struct _phydnn_event_t *phydnn_event;
typedef struct _phydnn_binding_t *phydnn_binding;

/* Memory **********************************************************/
typedef enum _phydnn_lock_access_t
{
  PHYDNN_LOCK_ACCESS_READ_ONLY,
  PHYDNN_LOCK_ACCESS_WRITE_ONLY,
  PHYDNN_LOCK_ACCESS_READ_WRITE
} phydnn_lock_access;

typedef enum _phydnn_import_mem_type_t
{
  PHYDNN_IMPORT_MEM_TYPE_CPU,
  /* Only valid to use with a context created with the PHYDNN CL extension.
   * Imports a cl_mem to use as an phydnn_memory object. */
  PHYDNN_IMPORT_MEM_TYPE_OPENCL,
  /* Used for importing buffer file descriptors to be used as phydnn_memory objects */
  PHYDNN_IMPORT_MEM_TYPE_FD
} phydnn_import_mem_type;

#if PHYDNN_MINOR_VERSION >= 4
/*
 * Enum to list the different types of I/O data layout parameters.
 */
typedef enum _phydnn_data_layout_param_type_t {
  PHYDNN_DATA_LAYOUT_INTERLEAVE, // int representing the data interleaving
  PHYDNN_DATA_LAYOUT_STRIDES, // long long int[PHYDNN_DESCRIPTOR_MAX_DIM] representing strides in each dimension in bytes
  PHYDNN_DATA_LAYOUT_ORDER, // phydnn_dimensions_order. Currently only valid for 4D tensors
  PHYDNN_DATA_LAYOUT_BYTE_SIZE // uint32 representing the size of the expected buffer in bytes
} phydnn_data_layout_param_type;
#endif

#if PHYDNN_MINOR_VERSION >= 8
/**
 * Enum to list supported sync types.
 */
typedef enum _phydnn_external_sync_type_t
{
  PHYDNN_SYNC_FD  // external sync based on file descriptor, data type expected is: int.
} phydnn_external_sync_type;
#endif

#if PHYDNN_MINOR_VERSION >= 9
typedef enum _phydnn_lstm_segment_flag_
{
 PHYDNN_LSTM_SEGMENT_NONE,      //Executes nothing (no segment). Functionalities will be added later.
 PHYDNN_LSTM_SEGMENT_STANDARD,  //Executes the "main" part/cell of the LSTM network.
 PHYDNN_LSTM_SEGMENT_GET_STATE, //Executes the parts of the network which get the LSTM state data. Can be used to store any intermediate LSTM states.
 PHYDNN_LSTM_SEGMENT_SET_STATE  //Executes the parts of the network which set the LSTM state data. If not called then then initial LSTM state data will be uninitialised
} phydnn_lstm_segment_flag;  //Used only for LSTM networks (for now)

/**
 * A union to hold detailed error codes.
 *
 * 'cpu_err_code'               detailed error code reported by CPU
 * 'gpu_err_code'               detailed error code reported by GPU
 * 'accelerator_err_code'       detailed error code reported by the accelerator
 */
typedef union _phydnn_detailed_err_code_t
{
  uint64_t cpu_err_code;
  uint64_t gpu_err_code;
  uint64_t accelerator_err_code;  
} phydnn_detailed_err_code;
#endif

/**
 * Allocate Device Memory.
 *
 * @Input  context      A previously obtained PHYDNN context
 * @Input  size         Size of allocation in bytes
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY
 * @Return phydnn_memory object on success and NULL on failure
 */
phydnn_memory phydnnAllocateMemory(phydnn_context context,
                                   size_t size,
                                   phydnn_err_code *errcode_ret);

/**
 * Import Memory from externally allocated memory.
 *
 * @Input  context           A previously obtained PHYDNN context
 * @Input  memory            External memory to be imported to img dnn memory
 *                           in case of PHYDNN_IMPORT_MEM_TYPE_FD import, shall point to single 'int'
 *                           containing buffer file descriptor
 * @Input  size              Size of allocation in bytes
 * @Input  import_mem_type   Type of external memory to be imported.
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY
 * @Return phydnn_memory object on success and NULL on failure
 */
phydnn_memory phydnnImportMemory(phydnn_context context,
                                 void* memory,
                                 size_t size,
                                 phydnn_import_mem_type import_mem_type,
                                 phydnn_err_code *errcode_ret);
/**
 * Release Previously allocated Device Memory.
 *
 * @Input  memory  Previously allocated device memory
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE
 */
phydnn_err_code phydnnMemoryDestroy(phydnn_memory memory);

/**
 * Lock the memory for host access. Obtain host accessible pointer to memory.
 *
 * @Input  memory             Memory object for which host access is sought.
 * @Input  lock_access        Lock access type.
 * @Output errcode_ret        PHYDNN_SUCCESS, PHYDNN_FAILURE
 * @Return Host accessible pointer to memory on success and NULL on failure
 */
void* phydnnMemoryLock(phydnn_memory memory,
                       phydnn_lock_access lock_access,
                       phydnn_err_code *errcode_ret);

/**
 * Unlock the memory that was previously locked for host access.
 *
 * @Input  memory    Memory object to unlock.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE
 */
phydnn_err_code phydnnMemoryUnlock(phydnn_memory memory);

#if PHYDNN_MINOR_VERSION >= 5
/**
* Divide PHY DNN memory into smaller buffers
* Note:
*       - creating overlapping buffers is not supported
*       - only one level allowed (original buffer might be divided into smaller ones,
*         but the smaller buffers cannot be subdivided further)
*       - original buffer may only be freed when all the sub-buffers are freed
*
* @Input  memory            Original memory (obtained via Allocate or Import)
* @Input  offset            Offset of the smaller buffer in memory
* @Input  size              Size of the smaller buffer
* @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE
* @Return phydnn_memory object on success and NULL on failure
*/
phydnn_memory phydnnSubdivideMemory(phydnn_memory memory,
                                 uint32_t offset,
                                 size_t size,
                                 phydnn_err_code *errcode_ret);
#endif

/* Data Layout **********************************************************/
#if PHYDNN_MINOR_VERSION >= 4
/**
 * Get the requested parameter from the specified input
 *
 * @Input  input     The input from which to retrieve the parameter.
 * @Input  parameter The requested parameter.
 * @Output out       Pointer to the memory to fill with parameter value.
 */
phydnn_err_code phydnnGetInputTensorParameter(phydnn_input input, phydnn_data_layout_param_type type, void *out);

/**
 * Get the requested parameter from the specified output
 *
 * @Input  output     The output from which to retrieve the parameter.
 * @Input  parameter The requested parameter.
 * @Output out       Pointer to the memory to fill with parameter value.
 */
phydnn_err_code phydnnGetOutputTensorParameter(phydnn_output output, phydnn_data_layout_param_type type, void *out);
#endif

/* Network Object **********************************************************/
/**
 * Produces a network object used for execution using binary object.
 *
 * @Input  device        The PHYDNN device for which to create this network.
 * @Input  context       The PHYDNN context in which to create this network output.
 * @Output size          Size of the object data in bytes.
 * @Output object_data   The object data from binary file.
 * @Output errcode_ret   PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY or
 *                       PHYDNN_INVALID_VALUE for invalid tensor or argument or
 *                       PHYDNN_INVALID_DEVICE for invalid device.
 * @Return The output network object for execution or NULL on failure.
 */
phydnn_network_object phydnnLoadNetworkObject(phydnn_device device,
                                              phydnn_context context,
                                              size_t size,
                                              const void* object_data,
                                              phydnn_err_code *errcode_ret);

#if PHYDNN_MINOR_VERSION >= 9
/**
 * Produces a prioritised network object used for execution using binary object.
 *
 * This function is a version of the phydnnLoadNetworkObject() extended
 * with the network priority level.
 *
 * The priority level for a network created with phydnnLoadNetworkObject()
 * function is always set to PHYDNN_NETWORK_PRIORITY_LOWEST. This function
 * allows for creating networks with raised priority level.
 *
 * The priority level set for a network should be only treated as a "hint"
 * for execution scheduling algorithms.
 *
 * @Input  device        The PHYDNN device for which to create this network.
 * @Input  context       The PHYDNN context in which to create this network output.
 * @Output size          Size of the object data in bytes.
 * @Output object_data   The object data from binary file.
 * @Input  priority      Network priority level.
 * @Output errcode_ret   PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_OUT_OF_MEMORY or
 *                       PHYDNN_INVALID_VALUE for invalid tensor or argument or
 *                       PHYDNN_INVALID_DEVICE for invalid device.
 * @Return The output network object for execution or NULL on failure.
 */
phydnn_network_object phydnnLoadPrioritisedNetworkObject(phydnn_device device,
                                                 phydnn_context context,
                                                 size_t size,
                                                 const void* object_data,
                                                 phydnn_network_priority priority,
                                                 phydnn_err_code *errcode_ret);
#endif

/**
 * Get inputs to a network.
 *
 * @Input  network_object  The network from which inputs requested.
 * @Input  max_inputs      The maximum number of elements in the inputs array.
 *                         Must be same as the total number of network inputs.
 * @Output inputs[]        Array of inputs to network created if not NULL.
 * @Output num_inputs      The number of inputs present in the network.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE or
 *         PHYDNN_INVALID_VALUE for invalid network or arguments
 */
phydnn_err_code phydnnNetworkObjectGetInputs(const phydnn_network_object network_object,
                                             unsigned int max_inputs,
                                             phydnn_input inputs[],
                                             unsigned int *num_inputs);

/**
 * Get outputs from a network.
 *
 * @Input  network_object  The network from which outputs requested.
 * @Input  max_outputs     The maximum number of elements in the outputs array.
 * @Output outputs[]       Array of outputs from network created if not NULL.
 * @Output num_outputs     The number of outputs present in the network.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE or
 *         PHYDNN_INVALID_VALUE for invalid network or arguments
 */
phydnn_err_code phydnnNetworkObjectGetOutputs(const phydnn_network_object network_object,
                                              unsigned int max_outputs,
                                              phydnn_output outputs[],
                                              unsigned int *num_outputs);

/**
 * Execute the network object with the given bindings.
 *
 * When the network object is executed, the event object follows the object's execution
 * and when complete the given output here contains the output tensor.
 *
 * A binding descriptor is allowed to have bindings for inputs and ouputs that do
 * not exist in the network object.
 *
 * After this, the user still will need to wait for the event to finish.
 *
 * @Input  network_object          The network object to execute.
 * @Input  bindings                The bindings to make to the input and output objects in the network.
 * @Input  blocking_execute        To indicate whether the execute call is blocking or non-blocking.
 * @Input  num_events_in_wait_list The number of events in wait list that need to be completed before
 *                                 executing this network object.
 * @Input  event_wait_list         List of events to wait before executing this network object.
 *                                 These events must have been created from within the same phydnn_context.
 *                                 (i.e. Executing a network object that was compiled for the same context).
 * @Output event                   Returns an event object that identifies this execute command.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_INVALID_CONTEXT for an invalid context,
 *         PHYDNN_OUT_OF_MEMORY or PHYDNN_INVALID_VALUE for invalid argument
 */
phydnn_err_code phydnnNetworkObjectExecute(const phydnn_network_object network_object,
                                           const phydnn_binding bindings,
                                           bool blocking_execute,
                                           unsigned int num_events_in_wait_list,
                                           const phydnn_event event_wait_list[],
                                           phydnn_event *event);

#if PHYDNN_MINOR_VERSION >= 9
/**
 * Execute the LSTM network object with the given bindings.
 *
 * When the LSTM network object is executed, the event object follows the object's execution
 * and when complete the given output here contains the output tensor.
 *
 * A binding descriptor is allowed to have bindings for inputs and ouputs that do
 * not exist in the network object.
 *
 * After this, the user still will need to wait for the event to finish.
 *
 * @Input  network_object          The network object to execute.
 * @Input  bindings                Multiple sets of bindings to make to the input and output objects in the network.
 * @Input  blocking_execute        To indicate whether the execute call is blocking or non-blocking.
 * @Input  num_events_in_wait_list The number of events in wait list that need to be completed before
 *                                 executing this network object.
 * @Input  event_wait_list         List of events to wait before executing this network object.
 *                                 These events must have been created from within the same phydnn_context.
 *                                 (i.e. Executing a network object that was compiled for the same context).
 * @Output event                   Returns an event object that identifies this execute command.
 * @Input  segment_flag            Flags of segments to be executed (SET_STATE,STANDARD,GET_STATE).
 * @Input  start_offset            2D ([num_inputs,tensor_dimension]) array of offsets for all inputs.
 *                                 It is used to exclude batches of inputs during execute, when multiple bindings are allocated.
 *                                 Basic examples assuming allocated bindings of size [10,1,1,1] and input initial size [1,1,1,1]:
 *                                   If offset is [0,0,0,0], all 10 batches will be executed.
 *                                   If [4,0,0,0], the first 4 batches will be skipped and only the remaining 6 will be.
 * @Input  max_dim                 It is the maximum number of dimensions of all the inputs.
 * @Input  timesteps               Number of timesteps (>0) of the LSTM network to execute (used only on PHYDNN_LSTM_SEGMENT_STANDARD segment_flag).
 *                                 The standard segment will be executed 'timesteps' times. 
 *                                 For any other value of segment_flag this argument is unused. 
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE, PHYDNN_INVALID_CONTEXT for an invalid context,
 *         PHYDNN_OUT_OF_MEMORY or PHYDNN_INVALID_VALUE for invalid argument
 */
phydnn_err_code phydnnLSTMNetworkObjectExecute(const phydnn_network_object network_object,
                                           phydnn_binding* bindings,
                                           bool blocking_execute,
                                           unsigned int num_events_in_wait_list,
                                           const phydnn_event event_wait_list[],
                                           phydnn_event *event,
                                           phydnn_lstm_segment_flag segment_flag,
                                           unsigned** start_offset,
                                           unsigned max_dim,
                                           unsigned timesteps);
#endif

#if PHYDNN_MINOR_VERSION >= 6
/**
 *  Get the name of the network_object.
 *
 * @Input network_object The network object to get name of.
 * @Output errcode_ret PHYDNN_SUCCESS, or PHYDNN_INVALID_VALUE for invalid network object.
 * @Return The given name on success, NULL on failure
 */
const char *phydnnNetworkGetObjectName(const phydnn_network_object network_object,
                                        phydnn_err_code *errcode_ret);
#endif

/* Binding **********************************************************/
/**
 * Create a binding descriptor.
 * Effectively a map of memory to input or output objects.
 *
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE or
 *                      PHYDNN_OUT_OF_MEMORY.
 * @Return A new binding descriptor or NULL on failure.
 */
phydnn_binding phydnnCreateBinding(phydnn_err_code *errcode_ret);

/**
 * Destroy a binding descriptor.
 *
 * @Input  binding      The binding descriptor to destroy
 * @Output errcode_ret  PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *                      PHYDNN_INVALID_VALUE for invalid descriptor.
 */
phydnn_err_code phydnnBindingDestroy(phydnn_binding binding);

/**
 * Add an input binding to a descriptor.
 * If a binding already exists for this input in the descriptor, then it is
 * updated.
 *
 * @Input  descriptor  The binding descriptor to add to.
 * @Input  input       The input object to bind memory to.
 * @Input  memory      The memory to bind to a tensor.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *         PHYDNN_INVALID_VALUE for invalid input or memory.
 */
phydnn_err_code phydnnBindingAddInput(phydnn_binding binding,
                                      phydnn_input input,
                                      phydnn_memory memory);

/**
 * Add an input size binding to a descriptor.
 * If a binding already exists for this input size in the descriptor, then it is
 * updated.
 *
 * @Input  descriptor  The binding descriptor to add to.
 * @Input  input       The input object to bind size of.
 * @Input  dimension   The dimension number of the size to set. Must have been
 *                     set to 0 in the original descriptor.
 * @Input  size        The dimension size to set.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *         PHYDNN_INVALID_VALUE for invalid input or dimension.
 */
phydnn_err_code phydnnBindingAddInputSize(phydnn_binding binding,
                                          phydnn_input input,
                                          unsigned int dimension,
                                          size_t size);

/**
 * Add a output binding to a descriptor.
 * If a binding already exists for this output in the descriptor, then it is
 * updated.
 *
 * @Input  descriptor  The binding descriptor to add to.
 * @Input  output      The output object to bind memory to.
 * @Input  memory      The memory to bind to a tensor.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE,
 *         PHYDNN_INVALID_VALUE for invalid output or memory.
 */
phydnn_err_code phydnnBindingAddOutput(phydnn_binding descriptor,
                                       phydnn_output output,
                                       phydnn_memory memory);

/* Event handling **********************************************************/
/**
 * Wait for a img dnn event.
 *
 * @Input  event  The event object on which to wait for.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE
 */
phydnn_err_code phydnnWaitForEvent(phydnn_event event);

/**
 * Destroy a img dnn event object.
 *
 * @Input  event  The event object to release.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE
 */
phydnn_err_code phydnnEventDestroy(phydnn_event event);

#if PHYDNN_MINOR_VERSION >= 8
/**
 * Get external sync object of selected type from PHYDNN event.
 *
 * Caller takes ownership of the returned sync. For example, in case of PHYDNN_SYNC_FD - for closing
 * the file descriptor, when sync is not needed anymore.
 *
 * The source PHYDNN event retains original sync, caller gets independent (lifetime wise) copy
 * of the sync object (in case of PHYDNN_SYNC_FD via dup() call).
 *
 * @Input  event          The event object to get sync object from.
 * @Input  type           The type of expected sync object.
 * @Output external_sync  Requested external sync.
 *
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE
 */
phydnn_err_code phydnnEventGetExternalSync(const phydnn_event event,
                                           phydnn_external_sync_type type,
                                           void* external_sync);

/**
 * Create PHYDNN event based on external sync of specific type.
 *
 * PHYDNN does not take ownership of provided external sync (however it should remain valid for
 * the duration of this call).
 *
 * Created PHYDNN event holds a copy of provided sync object (in case of PHYDNN_SYNC_FD
 * via dup() call), in order to have independent lifetime of the sync. Duplicated sync is closed
 * during mandatory call to phydnnEventDestroy().
 *
 * @Input  type           The type of provided external sync object.
 * @Input  external_sync  Provided external sync.
 * @Output event          The PHYDNN event object based on provided sync.
 *
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE
 */
phydnn_err_code phydnnCreateEventFromExternalSync(phydnn_external_sync_type type,
                                                  void* external_sync,
                                                  phydnn_event* event);
#endif

#if PHYDNN_MINOR_VERSION >= 9
/**
 * Get detailed error code for a img dnn event.
 *
 * @Input event                   The event object to get the detailed error from.
 * @Output detailed error code    The detailed error code.
 * @Return PHYDNN_SUCCESS, PHYDNN_FAILURE
 */
phydnn_err_code phydnnGetDetailedErrorCode(phydnn_event event,
                                           phydnn_detailed_err_code *detailed_err_code);

/* Get utilities **********************************************************/
/**
 * Utility function to obtain the name of an input.
 *
 * @Input  input  An input previously created.
 * @Output err    PHYDNN_SUCCESS or PHYDNN_INVALID_VALUE for invalid tensor.
 * @Return        The given name on success, NULL on failure.
 */
const char *phydnnGetInputName(phydnn_input input, phydnn_err_code* errcode_ret);

/**
 * Utility function to obtain the name of an output.
 *
 * @Input  output An output previously created.
 * @Output err    PHYDNN_SUCCESS or PHYDNN_INVALID_VALUE for invalid tensor.
 * @Return        The given name on success, NULL on failure.
 */
const char *phydnnGetOutputName(phydnn_output output, phydnn_err_code* errcode_ret);

#endif /* PHYDNN_MINOR_VERSION >= 9 */

/** @} */
#ifdef __cplusplus
}
#endif
#endif /* _PHY_DNN_EXECUTE_H */
