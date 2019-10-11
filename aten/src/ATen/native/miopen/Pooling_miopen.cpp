#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at { namespace native {

    // See Note [ATen preprocessor philosophy]
    std::tuple<Tensor, Tensor> miopen_max_pooling(
        const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, 
        IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
      AT_ERROR("miopen_pooling: ATen not compiled with MIOpen support");
    }

    Tensor miopen_max_pooling_backward(
        const Tensor& grad_output, const at::Tensor& input_t, IntArrayRef kernel_size,
        IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& indices_t) {
      AT_ERROR("miopen_pooling_backward: ATen not compiled with MIOpen support");
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

    //Calculate Pooling output shape.
    static std::vector<int64_t> pooling_output_size(
        IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef padding,
        IntArrayRef stride, IntArrayRef dilation, bool ceil_mode)
    {
        auto dim = input_size.size();
        std::vector<int64_t> output_size(dim);
        output_size[0] = input_size[0]; // output batch_size = input batch_size
        output_size[1] = input_size[1]; // output channel_dim = input channel_dim
        for (size_t d = 2; d < dim ; ++d) {
            output_size[d] = ((input_size[d] + 2 * padding[d - 2] - dilation[d - 2] * (kernel_size[d - 2] - 1) - 1 + (ceil_mode ? stride[d - 2] : 0))/ stride[d - 2] + 1);
        }

        return output_size;
    }

    //Perform miopen max pool operation.
    std::tuple<at::Tensor, at::Tensor> miopen_max_pooling(
        const Tensor& input_t, IntArrayRef kernel_size, IntArrayRef stride,
        IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
    {
        //std::cout << "Pooling_miopen : MIOpen pooling forward called. " << std::endl;
        TensorArg input { input_t, "input", 0};
        setMIOpenStreamToCurrent();
        CheckedFrom c = "miopen_pooling";

        checkAllDefined(c, {input});

        //create output and indices tensor.
        auto output_t = at::empty(pooling_output_size(input->sizes(), kernel_size, padding, stride, dilation, ceil_mode), input->options());
        TensorArg output { output_t, "result", 1 };

        //auto indices_t = at::empty(output->sizes(), output->options());
        auto indices_t = at::empty(output->sizes(), output->options().dtype(kLong));
        TensorArg indices {indices_t, "indices", 2};

        //Pooling mode.
        miopenPoolingMode_t mode = miopenPoolingMax;
        auto handle = getMiopenHandle();
        auto datatype = getMiopenDataType(*input);

        //Input and output descriptors.
        TensorDescriptor idesc{ *input, 4}; //input descriptor
        TensorDescriptor odesc{ *output, 4}; //output descriptor

        //Pooling Descriptor.    
        miopenPoolingDescriptor_t pdesc;
        miopenCreatePoolingDescriptor(&pdesc);
        MIOPEN_CHECK(miopenSet2dPoolingDescriptor(pdesc, mode, kernel_size[0], kernel_size[1], padding[0], padding[1], stride[0], stride[1]));

        //Get pooling workspace size. 
        size_t ws_size;
        miopenIndexType_t index_type = miopenIndexUint64;
        miopenSetPoolingIndexType(pdesc, index_type);
        MIOPEN_CHECK(miopenPoolingGetWorkSpaceSizeV2(pdesc, odesc.desc(), &ws_size));

        Constant one(datatype, 1);
        Constant zero(datatype, 0);
       
        //Run miopen pooling forward and return the indices and the output tensor.
        MIOPEN_CHECK(miopenPoolingForward(handle, pdesc, &one, 
                            idesc.desc(), input->data_ptr(),
                            &zero, odesc.desc(), output->data_ptr(),
                            true, indices->data_ptr(), ws_size));

        return std::tuple<at::Tensor, at::Tensor>{output_t, indices_t};
    } 

    //Perform miopen maxpooling backwards.
    Tensor miopen_max_pooling_backward(
        const Tensor& grad_output_t, const Tensor& input_t, IntArrayRef kernel_size, IntArrayRef stride,
        IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& indices_t)
    {
        TensorArg grad_output {grad_output_t, "grad_output", 1},
                  input {input_t, "input", 2},
                  indices {indices_t, "indices", 3};

        //std::cout << "Pooling_miopen.cpp : Entered backward pass for MIOpen max pooling." << std::endl;

        CheckedFrom c = "miopen_max_pooling_backwards";
        setMIOpenStreamToCurrent();
        
        checkAllDefined(c, {grad_output, input, indices});
        checkAllSameGPU(c, {grad_output, input, indices});

        auto grad_input_t = at::empty(input->sizes(), input->options());
        auto output_t = at::empty(pooling_output_size(input->sizes(), kernel_size, padding, stride, dilation, ceil_mode), grad_output->options());

        TensorArg grad_input {grad_input_t, "grad_input", 0};
        TensorArg output {output_t, "output", 4};
           
        auto handle = getMiopenHandle();
        auto datatype = getMiopenDataType(*input);

        TensorDescriptor godesc{ *grad_output, 4};
        TensorDescriptor odesc {*output, 4};
        TensorDescriptor idesc{ *input, 4};

        //Pooling mode.
        miopenPoolingMode_t mode = miopenPoolingMax;
        
        //Pooling descriptor.
        miopenPoolingDescriptor_t pdesc;
        miopenCreatePoolingDescriptor(&pdesc);
        MIOPEN_CHECK(miopenSet2dPoolingDescriptor(pdesc, mode, kernel_size[0], kernel_size[1], padding[0], padding[1], stride[0], stride[1]));

        //Get pooling workspace size. 
        size_t ws_size;
        miopenIndexType_t index_type = miopenIndexUint64;
        miopenSetPoolingIndexType(pdesc, index_type);
        MIOPEN_CHECK(miopenPoolingGetWorkSpaceSizeV2(pdesc, godesc.desc(), &ws_size));

        Constant one(datatype, 1);
        Constant zero(datatype, 0);

        //Run MIOpen backward Pooling.
        MIOPEN_CHECK(miopenPoolingBackward(handle, pdesc, &one,
                        odesc.desc(), output->data_ptr(),
                        godesc.desc(), grad_output->data_ptr(),
                        idesc.desc(), input->data_ptr(), &zero, 
                        idesc.desc(), grad_input->data_ptr(), indices->data_ptr()                 
                        ));
                
        return grad_input_t;
    }

}} //namespace at::native

#endif

