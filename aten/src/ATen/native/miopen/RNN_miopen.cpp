#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>

#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/Exception.h>

#if !AT_ROCM_ENABLED()

namespace at { namespace native {

    Tensor miopen_rnn_flatten_weight(
            TensorList weight_arr, int64_t weight_stride0, int64_t input_size,
            int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
            bool batch_first, bool fn_bidirectional
            ) {
        AT_ERROR("miopen_flatten_weight: ATen not compiled with MIOpen support.");
    }

    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> miopen_rnn(
            const Tensor& input_r, TensorList weight, int64_t weight_stride0,
            const Tensor& weight_buf_r, const Tensor& hx, const Tensor& cx,
            int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
            bool batch_first, double fn_dropout, bool fn_train, bool fn_bidirectional,
            IntArrayRef fn_batch_sizes, const Tensor& fn_dropout_state
            ) {
        AT_ERROR("miopen_rnn : ATen not compiled with MIOpen support.");
    }

    std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> miopen_rnn_backward(
            const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
            const Tensor& output, const Tensor& grad_output_r, const Tensor& grad_hy_r,
            const Tensor& grad_cy_r, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first,
            double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor& dropout_state, 
            const Tensor& reserve, std::array<bool, 4> output_mask
            ) {
        AT_ERROR("miopen_rnn_backward: ATen not compiled with MIOpen support.");
    }

}} //namespace at::native

#else // AT_CUDNN_ENABLED()

//RNNDescriptor.
struct RNNDescriptorParams {
    int64_t hidden_size;
    int64_t num_layers;
    miopenRNNDirectionMode_t direction;
    miopenRNNMode_t rnn_mode;
    miopenDataType_t datatype;
    miopenRNNAlgo_t algo = miopenRNNdefault;
    miopenRNNInputMode_t input_mode = miopenRNNlinear;
    miopenBiasMode_t bias_mode = miopenRNNNoBias;

    void set_bidirectional(bool fn_bidirectional) {
        direction = fn_bidirectional ? miopenRNNbidirection : miopenRNNunidirection;
    }

    void set_algo(miopenRNNAlgo_t algo) {
        this->algo = algo;
    }

    /*fn_mode is set in torch.backends.cudnn (get_cudnn_mode() method) 
      Need to modify the interface to the frontend to make this function useful.
     */
    void set_mode(int64_t fn_mode) {
        switch (fn_mode) {
            case 0:
                rnn_mode = miopenRNNRELU;
                break;
            case 1:
                rnn_mode = miopenRNNTANH;
                break;
            case 2:
                rnn_mode = miopenLSTM;
                break;
            case 3:
                rnn_mode = miopenGRU;
                break;
            default:
                {
                    std::ostringstream oss;
                    oss << "unrecognized miopen RNN mode " << fn_mode;
                    AT_ERROR(oss.str());
                }
        }	
    }

    void set(int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional, miopenDataType_t datatype, miopenBiasMode_t bias_mode) {
        this->set_mode(mode);
        this->hidden_size = hidden_size;
        this->num_layers = num_layers;
        this->direction = this->set_bidirectional(bidirectional);
        this->datatype = datatype;
        this->bias_mode = bias_mode;
    }

    RNNDescriptor descriptor() const {
        RNNDescriptor rnn_desc;
        rnn_desc.set(hidden_size, num_layers, input_mode, direction, rnn_mode, bias_mode, algo, datatype);
        return rnn_desc;
    }
};

void _viewOrCopyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to, bool copy) {
    AT_ASSERTM(params_from.size(0) == params_to.size(0), "number of layers mismatch");
    for (size_t i = 0; i < params_from.size(0); i++) {
        auto layer_params_from = params_from[i];
        auto layer_params_to = params_to[i];
        // NOTE: these lists have all weights before all biases, so if the layer
        // doesn't use biases, iteration will terminate once layer_params_from ends
        // and ignore them.
        for (auto a = layer_params_from.begin(), b = layer_params_to.begin();
                a != layer_params_from.end() && b != layer_params_to.end();
                ++a, ++b) {
            auto param_from = *a, param_to = *b;
            AT_ASSERTM(param_from.type() == param_to.type(), "parameter types mismatch");
            if (copy) {
                param_to.copy_(param_from.view_as(param_to));
            } else {
                param_from.resize_as_(param_to);
            }
        }
    }
}

void _copyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
    _viewOrCopyParams(params_from, params_to, true);
}

void _viewParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
    _viewOrCopyParams(params_from, params_to, false);
}

int64_t get_num_weights(miopenHandle_t handle, const RNNDescriptor& rnn_desc,
        const TensorDescriptor& x_desc, miopenDataType_t datatype)
{
    size_t weight_size;
    MIOPEN_CHECK(miopenGetRNNParamsSize(handle, rnn_desc.desc(), x_desc.desc(), &weight_size, datatype));
    auto element_size = dataSize(datatype);
    AT_ASSERTM(weight_size % element_size == 0, "miopenGetRNNParamsSize returned nonsensical weight_size.");
    return weight_size / element_size;
}

Tensor miopen_rnn_flatten_weight(
        TensorList weight_arr, int64_t weight_stride0, int64_t input_size,
        miopenRNNMode_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
        bool batch_first, bool fn_bidirectional
        ) {
    AT_ERROR("miopen_flatten_weight: not implemented yet.");

    AT_CHECK(weight_arr.size() > 0, "miopen_rnn_flatten_weight : cannot flatten empty weight list.");

    auto any_param = weight_arr[0];
    auto datatype = getMiopenDataType(any_param);

    RNNDescriptorParam rnn;
    rnn.set(fn_mode, hidden_size, num_layers, bidirectional, datatype);

    RNNDescriptor rnn_desc = rnn.descriptor();

    TensorGeometry x_geom({1, input_size});
    TensorDescriptor x_desc;
    x_desc.set(getMiopenDataType(any_param), x_geom.sizes(), x_geom.strides(), 5);

    auto num_weights = get_num_weights(handle, rnn_desc, x_desc, datatype);
    auto weight_buf = at::zeros(num_weights, any_param.options());

    FilterDescriptor w_desc;
    w_desc.set(weight_buf, 3);

    //Slice off views into weight_buf.
    std::vector<Tensor> params_arr;
    size_t params_stride0;
    std::tie(params_arr, params_stride0) = get_parameters(handle, rnn, rnn_desc, x_desc, w_desc, weight_buf);

    MatrixRef<Tensor> weight {weight_arr, static_cast<size_t>(weight_stride0)},
        params {params_arr, params_stride0};

    //Copy weights.
    _copyParams(weight, params);

    // Update the storage
    for (size_t i = 0; i < weight.size(0); i++) {
        for (auto orig_param_it = weight[i].begin(), new_param_it = params[i].begin();
                orig_param_it != weight[i].end() && new_param_it != params[i].end();
                orig_param_it++, new_param_it++) {
            auto orig_param = *orig_param_it, new_param = *new_param_it;
            orig_param.set_(new_param.view_as(orig_param));
        }
    }

    return weight_buf;

}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> miopen_rnn(
        const Tensor& input_r, TensorList weight, int64_t weight_stride0,
        const Tensor& weight_buf_r, const Tensor& hx, const Tensor& cx,
        int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
        bool batch_first, double fn_dropout, bool fn_train, bool fn_bidirectional,
        IntArrayRef fn_batch_sizes, const Tensor& fn_dropout_state
        ) {
    AT_ERROR("miopen_rnn : not implemented yet.");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> miopen_rnn_backward(
        const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
        const Tensor& output, const Tensor& grad_output_r, const Tensor& grad_hy_r,
        const Tensor& grad_cy_r, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first,
        double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor& dropout_state, 
        const Tensor& reserve, std::array<bool, 4> output_mask
        ) {
    AT_ERROR("miopen_rnn_backward: not implemented yet.");
}


#endif 
