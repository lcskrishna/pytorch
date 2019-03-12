#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at { namespace native {

    // See Note [ATen preprocessor philosophy]

    std::tuple<at::Tensor, at::Tensor> miopen_max_pool2d(
        const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, 
        IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
      AT_ERROR("miopen_pooling: ATen not compiled with MIOpen support");
    }                                  

}}

#else //AT_ROCM_ENABLED

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

namespace at { namespace native {


    std::tuple<at::Tensor, at::Tensor> miopen_max_pool2d(
        const Tensor& input_t, IntArrayRef kernel_size, IntArrayRef stride,
        IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
    {
        TensorArg input { input_t, "input", 1};
        setMIOpenStreamToCurrent();
        CheckedFrom c = "miopen_pooling";

        checkAllDefined(c, {input});

        //Pooling mode.
        miopenPoolingMode_t mode = miopenPoolingMax;
        auto handle = getMiopenHandle();
        auto datatype = getMiopenDataType(*input);

        //Input and output descriptors.
        TensorDescriptor idesc{ *input, 4}; //input descriptor
        //TODO: calculate output shape of the pooling and create an output descriptor.

        //Pooling Descriptor.    
        miopenPoolingDescriptor_t pdesc;
        miopenCreatePoolingDescriptor(&pdesc);
        miopenSet2dPoolingDescriptor(pdesc, mode, kernel_size[0], kernel_size[1], padding[0], padding[1], stride[0], stride[1]);
 
        /*TODO:
        Get pooling workspace size and assign memory for workspace (indices).
        Need to cast the long tensor into int8 tensor for maxpooling (although it's slow until miopen1.8 release.) */

        //Run miopen pooling forward and return the indices and the output tensor.
      
        return output_t;
    }   


}} //namespace at::native

#endif
