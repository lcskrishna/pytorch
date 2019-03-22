#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/SpatialDilatedMaxPooling.cu"
#else

#include <THCUNN/common.h>
#include <THCUNN/generic/pooling_shape.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>

#include <tuple>

#include <THH/THH.h>

#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>

#include <ATen/TensorUtils.h>

#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

#include <iostream>

static inline void THNN_(SpatialDilatedMaxPooling_shapeCheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *gradOutput, THCIndexTensor *indices,
                         int kH, int kW, int dH, int dW, int padH, int padW,
                         int dilationH, int dilationW, bool ceil_mode) {

  THArgCheck(kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(dilationH > 0 && dilationW > 0, 12,
             "dilation should be greater than zero, but got dilationH: %d dilationW: %d",
             dilationH, dilationW);

  int ndim = input->dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;
  int batchSize = 1;

  if (ndim == 4) {
    batchSize = input->size(0);
    dimf++;
    dimh++;
    dimw++;
  }

  THCUNN_argCheck(state, !input->is_empty() && (ndim == 3 || ndim == 4), 2, input,
                  "non-empty 3D or 4D input tensor expected but got: %s");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2,
             "pad should be smaller than half of kernel size, but got "
             "padW = %d, padH = %d, kW = %d, kH = %d",
             padW, padH, kW, kH);

  int64_t nInputPlane = input->size(dimh-1);
  int64_t nInputRows = input->size(dimh);
  int64_t nInputCols = input->size(dimw);
  int64_t nOutputPlane = nInputPlane;

  int64_t nOutputRows = pooling_output_shape<int64_t>(nInputRows, kH, padH, dH, dilationH, ceil_mode);
  int64_t nOutputCols = pooling_output_shape<int64_t>(nInputCols, kW, padW, dW, dilationW, ceil_mode);

  if (nOutputCols < 1 || nOutputRows < 1)
    THError("Given input size: (%dx%dx%d). "
            "Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimh, nOutputRows);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimw, nOutputCols);
  }
  if (indices != NULL) {
    THCUNN_check_dim_size_indices(state, indices, 4, 0, batchSize);
    THCUNN_check_dim_size_indices(state, indices, 4, 1, nOutputPlane);
    THCUNN_check_dim_size_indices(state, indices, 4, 2, nOutputRows);
    THCUNN_check_dim_size_indices(state, indices, 4, 3, nOutputCols);
  }
}

void THNN_(SpatialDilatedMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           bool ceil_mode)
{

  THCUNN_assertSameGPU(state, 3, input, output, indices);
  THNN_(SpatialDilatedMaxPooling_shapeCheck)
       (state, input, NULL, NULL, kH, kW, dH, dW,
        padH, padW, dilationH, dilationW, ceil_mode);

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int64_t nOutputCols, nOutputRows;

  if (input->dim() == 3) {
    nInputCols = input->size(2);
    nInputRows = input->size(1);
    nInputPlane = input->size(0);
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size(3);
    nInputRows = input->size(2);
    nInputPlane = input->size(1);
    batchSize = input->size(0);
  }

  nOutputCols = pooling_output_shape<int64_t>(nInputCols, kW, padW, dW, dilationW, ceil_mode);
  nOutputRows = pooling_output_shape<int64_t>(nInputRows, kH, padH, dH, dilationH, ceil_mode);

  input = THCTensor_(newContiguous)(state, input);
  scalar_t* input_data = THCTensor_(data)(state, input);

  THCTensor_(resize4d)(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THCUNN_resizeAs_indices(state, indices, output);

  THCIndex_t* indices_data = THCIndexTensor_(data)(state, indices);
  scalar_t* output_data = THCTensor_(data)(state, output);

  int count = THCTensor_(nElement)(state, output);
#if defined (__HIP_PLATFORM_HCC__)
  int kernel_size[2] = {kH, kW};
  int stride[2] = {dH, dW};
  int padding[2] = {padH, padW};
  int dilation[2] = {dilationH, dilationW};

  //Write an miopen implementation.
  miopenPoolingMode_t mode = miopenPoolingMax;
  auto handle = at::native::getMiopenHandle();
  miopenDataType_t datatype = miopenFloat;

  //Input and output tensor descriptors.
  miopenTensorDescriptor_t idesc;
  miopenTensorDescriptor_t odesc;
  miopenCreateTensorDescriptor(&idesc);
  miopenCreateTensorDescriptor(&odesc);

  miopenSet4dTensorDescriptor(idesc, datatype, batchSize, nInputPlane, nInputCols, nInputRows);
  miopenSet4dTensorDescriptor(odesc, datatype, batchSize, nInputPlane, nOutputCols, nOutputRows);

  //Pooling Descriptor.
  miopenPoolingDescriptor_t pdesc;
  miopenCreatePoolingDescriptor(&pdesc);
  miopenSet2dPoolingDescriptor(pdesc, mode, kH, kW, padH, padW, dH, dW);

  //Get workspace size.
  size_t ws_size;
  miopenPoolingGetWorkSpaceSize(odesc, &ws_size);

  at::native::Constant one(datatype, 1);
  at::native::Constant zero(datatype, 0);

  miopenPoolingForward(handle, pdesc, &one, idesc, (void *) input_data, &zero, odesc, (void *) output_data, true, indices_data, ws_size);
  indices_data = (THCIndex_t *) indices_data;


  //Destroy descriptors.
  miopenDestroyPoolingDescriptor(pdesc);
  miopenDestroyTensorDescriptor(odesc);
  miopenDestroyTensorDescriptor(idesc);

#else
  MaxPoolForward<scalar_t, accreal> <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, input_data,
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
  THCudaCheck(cudaGetLastError());
#endif
  if(input->dim() == 3)
    THCTensor_(resize3d)(state, output, nInputPlane, nOutputRows, nOutputCols);



  THCTensor_(free)(state, input);
}

void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCIndexTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           bool ceil_mode)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, indices, gradInput);
  THNN_(SpatialDilatedMaxPooling_shapeCheck)
       (state, input, gradOutput, indices, kH, kW, dH, dW,
       padH, padW, dilationH, dilationW, ceil_mode);

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int64_t nOutputCols, nOutputRows;

  if (THTensor_nDimensionLegacyAll(input) == 3) {
    nInputCols = input->size(2);
    nInputRows = input->size(1);
    nInputPlane = input->size(0);
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size(3);
    nInputRows = input->size(2);
    nInputPlane = input->size(1);
    batchSize = input->size(0);
  }

  nOutputCols = pooling_output_shape<int64_t>(nInputCols, kW, padW, dW, dilationW, ceil_mode);
  nOutputRows = pooling_output_shape<int64_t>(nInputRows, kH, padH, dH, dilationH, ceil_mode);

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);

  int count = THCTensor_(nElement)(state, input);
  dim3 grid;
  int imgcount = nInputCols * nInputRows;
  const int blocks = (imgcount + BACKWARD_THREADS - 1) / BACKWARD_THREADS;
  grid.x = blocks;
  grid.y = batchSize;
  grid.z = nInputPlane;
  uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  uint64_t maxGridZ = at::cuda::getCurrentDeviceProperties()->maxGridSize[2];
  if (maxGridY < grid.y) grid.y = maxGridY;
  if (maxGridZ < grid.z) grid.z = maxGridZ;

#if defined (__HIP_PLATFORM_HCC__)
  
  miopenPoolingMode_t mode = miopenPoolingMax;
  auto handle = at::native::getMiopenHandle();
  miopenDataType_t datatype = miopenFloat;

  scalar_t * input_data = THCTensor_(data)(state, input);
  scalar_t * gradOutput_data = THCTensor_(data)(state, gradOutput);
  scalar_t * indices_data = THCTensor_(data)(state, indices);
  scalar_t * grad_input_data = THCTensor_(data)(state, gradInput);
  scalar_t * output_data = THCTensor_(data)(state, input);

  //Create tensor descriptors.
  miopenTensorDescriptor_t yDesc, dyDesc, xDesc, dxDesc; 
  miopenCreateTensorDescriptor(&yDesc);
  miopenCreateTensorDescriptor(&dyDesc);
  miopenCreateTensorDescriptor(&xDesc);
  miopenCreateTensorDescriptor(&dxDesc);

  miopenSet4dTensorDescriptor(yDesc, datatype, batchSize, nInputPlane, nInputCols, nInputRows);
  miopenSet4dTensorDescriptor(dyDesc, datatype, batchSize, nInputPlane, nInputCols, nInputRows);
  miopenSet4dTensorDescriptor(xDesc, datatype, batchSize, nInputPlane, nOutputCols, nOutputRows);
  miopenSet4dTensorDescriptor(dxDesc, datatype, batchSize, nInputPlane, nOutputCols, nOutputRows);

  //Pooling descriptor.
  miopenPoolingDescriptor_t pdesc;
  miopenCreatePoolingDescriptor(&pdesc);
  miopenSet2dPoolingDescriptor(pdesc, mode, kH, kW, padH, padW, dH, dW);

  //Constants.
  at::native::Constant one(datatype, 1);
  at::native::Constant zero(datatype, 0);

  miopenPoolingBackward(handle, pdesc, &one, yDesc, (void *)input_data, dyDesc, (void *) gradOutput_data, xDesc, (void *) output_data, &zero, dxDesc, 
                      (void *) grad_input_data, (void *) indices_data );

#else  
  MaxPoolBackward<scalar_t, accreal> <<< grid, BACKWARD_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count,
      THCTensor_(data)(state, gradOutput),
      THCIndexTensor_(data)(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW,
      THCTensor_(data)(state, gradInput));
  THCudaCheck(cudaGetLastError());
#endif

  THCTensor_(free)(state, gradOutput);

  // clean
  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
