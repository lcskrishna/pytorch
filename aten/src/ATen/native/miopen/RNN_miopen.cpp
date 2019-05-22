#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>

#include <ATen/cuda/CUDAConfig.h>
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

#else // AT_ROCM_ENABLED()

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

namespace at { namespace native {

//RNNDescriptor.
struct RNNDescriptorParams {
    int64_t hidden_size;
    int64_t num_layers;
    miopenRNNDirectionMode_t direction;
    miopenRNNMode_t rnn_mode;
    miopenDataType_t datatype;
    miopenRNNAlgo_t algo = miopenRNNdefault;
    miopenRNNInputMode_t input_mode = miopenRNNlinear;
    miopenRNNBiasMode_t bias_mode = miopenRNNNoBias;

    int64_t num_directions() const {
    	return (direction == miopenRNNbidirection) ? 2 : 1;
    }

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

    void set(int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional, miopenDataType_t datatype, miopenRNNBiasMode_t bias_mode) {
        this->set_mode(mode);
        this->hidden_size = hidden_size;
        this->num_layers = num_layers;
        this->set_bidirectional(bidirectional);
        this->datatype = datatype;
        this->bias_mode = bias_mode;
    }

    RNNDescriptor descriptor() const {
        RNNDescriptor rnn_desc;
        rnn_desc.set(hidden_size, num_layers, input_mode, direction, rnn_mode, bias_mode, algo, datatype);
        return rnn_desc;
    }
};

//TensorDescriptor list.
std::vector<TensorDescriptor> rnn_descriptor_sequence(const Tensor& tensor, IntArrayRef batch_sizes) {
	std::vector<TensorDescriptor> descriptors(batch_sizes.size());
	size_t i =0;

	auto batch_tensor_size = tensor.sizes().vec();
	for (auto batch_size : batch_sizes) {
		batch_tensor_size[0] = batch_size;

		descriptors[i].set(getMiopenDataType(tensor), batch_tensor_size, tensor.strides(), 3);
		i++;
	}

	return descriptors;
}

std::vector<TensorDescriptor> rnn_descriptor(const Tensor& tensor, int64_t N) {
	std::vector<TensorDescriptor> descriptors(N);
	for (int64_t i = 0; i < N ; i++) {
		descriptors[i].set(tensor, 5);
	}

	return descriptors;
}

struct TensorDescriptorListParams {
	IntArrayRef batch_sizes;
	int64_t seq_length;
	int64_t mini_batch;

	int64_t input_size;
	int64_t batch_sizes_sum;

	bool is_input_packed() const {
		return batch_sizes.size() != 0;
	}

	void set(IntArrayRef input_sizes, IntArrayRef batch_sizes_, bool batch_first) {
		batch_sizes = batch_sizes_;
		if (is_input_packed()) {
			seq_length = batch_sizes.size();
			mini_batch = batch_sizes[0];
			batch_sizes_sum = input_sizes[0];
			input_size = input_sizes[1];
		} else {
			if (batch_first) {
				seq_length = input_sizes[1];
				mini_batch = input_sizes[0];
			} else {
				seq_length = input_sizes[0];
				mini_batch = input_sizes[1];
			}
			input_size = input_sizes[2];
			batch_sizes_sum = -1;
		}
	}

	std::vector<TensorDescriptor> descriptors(Tensor x) const {
		auto is_input_packed = batch_sizes.size() != 0;
		if (is_input_packed) {
			return rnn_descriptor_sequence(x, batch_sizes);
		} else {
			return rnn_descriptor(x[0], seq_length);
		}
	}
};

struct RNNParams {
	RNNDescriptorParams rnn;
	TensorDescriptorListParams tensors;
};

struct RNNDescriptors {
	RNNDescriptor rnn_desc;
	std::vector<TensorDescriptor> x_descs;
	std::vector<TensorDescriptor> y_descs;
	TensorDescriptor hx_desc;
	TensorDescriptor hy_desc;
	TensorDescriptor cx_desc;
	TensorDescriptor cy_desc;

	RNNDescriptors(const RNNParams& fn, miopenHandle_t handle, Tensor x, Tensor y, Tensor hx, Tensor cx) {
		rnn_desc = fn.rnn.descriptor();
		x_descs = fn.tensors.descriptors(x);
		y_descs = fn.tensors.descriptors(y);
		hx_desc.set(hx, 5);
		hy_desc.set(hx, 5);
		if (cx.defined()) {
			cx_desc.set(cx, 5);
			cy_desc.set(cx, 5);
		}
	}

	std::vector<miopenTensorDescriptor_t> get_descs(const std::vector<TensorDescriptor>& descs) {
		std::vector<miopenTensorDescriptor_t> r;
		r.reserve(descs.size());
		for (auto& desc : descs) {
			r.emplace_back(desc.desc());
		}
		return r;
	}

	std::vector<miopenTensorDescriptor_t> get_x_descs() {
		return get_descs(x_descs);
	}

	std::vector<miopenTensorDescriptor_t> get_y_descs() {
		return get_descs(y_descs);
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

int64_t _num_linear_layers(miopenRNNMode_t mode) {
	switch(mode) {
		case miopenLSTM:
			return 8;
		case miopenGRU:
			return 6;
		case miopenRNNRELU:
			return 2;
		case miopenRNNTANH:
			return 2;
		default:
			AT_ERROR("Unknown miopen RNN mode : ", mode);
	}
}

// This is a lightweight version of the method above used to quickly get the expected
// parameter offsets.
std::vector<void*> get_expected_data_ptrs(
        const Tensor& weight_buf, miopenHandle_t handle, const RNNDescriptorParams& rnn,
        const RNNDescriptor& rnn_desc, const TensorDescriptor& x_desc, miopenDataType_t datatype) {
    int64_t num_linear_layers = _num_linear_layers(rnn.rnn_mode);
    int64_t num_dir_layers = rnn.num_directions() * rnn.num_layers;
    const auto miopen_methods = { miopenGetRNNLayerParamOffset, miopenGetRNNLayerBiasOffset };
    std::vector<void*> data_ptrs;
    data_ptrs.reserve(num_dir_layers * 2 * 2);
    auto element_size = dataSize(datatype);
    for (int64_t layer = 0; layer < num_dir_layers; layer++) {
        for (auto miopen_method : miopen_methods) {
            const std::array<int64_t, 2> linear_offsets = { 0, num_linear_layers / 2 };
            for (int64_t linear_id : linear_offsets) {
                FilterDescriptor lin_layer_mat_desc;
                void* matrix_pointer;
                size_t param_offset;
                MIOPEN_CHECK(miopen_method(
                    rnn_desc.desc(),
                    layer,
                    x_desc.desc(),
                    linear_id,
                    lin_layer_mat_desc.mut_desc(),
                    &param_offset
                    ));
                matrix_pointer = (char*)weight_buf.data_ptr() + (param_offset * element_size);
                data_ptrs.push_back(matrix_pointer);
           }
        }
    }
    return data_ptrs;
}

std::pair<std::vector<Tensor>, size_t> get_parameters(miopenHandle_t handle, const RNNDescriptorParams& rnn,
					const RNNDescriptor& rnn_desc, const TensorDescriptor& x_desc, const FilterDescriptor& w_desc,
					const Tensor& weight_buf)
{
	std::vector<Tensor> params;
	int64_t num_linear_layers = _num_linear_layers(rnn.rnn_mode);
	int64_t num_layers = rnn.num_directions() * rnn.num_layers;
	size_t cur_offset = 0;
	size_t global_layer_params_count = 0;
	auto elem_size = dataSize(getMiopenDataType(weight_buf));

	for (int64_t layer = 0; layer < num_layers; layer++) {
		size_t layer_params_count = 0;

		// Get layer params
		for (int64_t linear_id = 0; linear_id < num_linear_layers; linear_id++) {
			FilterDescriptor lin_layer_mat_desc;
			size_t offset;
			MIOPEN_CHECK(miopenGetRNNLayerParamOffset(
				rnn_desc.desc(),
				layer,
				x_desc.desc(),
				linear_id,
				lin_layer_mat_desc.mut_desc(),
				&offset));

			size_t param_size;
			MIOPEN_CHECK(miopenGetRNNLayerParamSize(
				handle,
				rnn_desc.desc(),
				layer,
				x_desc.desc(),
				linear_id,
				&param_size));
			param_size /= elem_size;

			if(linear_id == 0 || linear_id == num_linear_layers / 2) {
				std::initializer_list<int64_t> size = { param_size * num_linear_layers / 2, 1};
				Tensor param = at::empty({0}, weight_buf.options()).set_(weight_buf.storage(), offset, size);
				params.emplace_back(std::move(param));
				layer_params_count++;
			} else {
				AT_ASSERTM(cur_offset == offset, "cur_offset = ", cur_offset, " ; offset = ", offset);
			}
			cur_offset = offset + param_size;
		}

		// Get bias params
		for (int64_t linear_id = 0; linear_id < num_linear_layers; linear_id++) {
			FilterDescriptor lin_layer_mat_desc;
			size_t offset;
			MIOPEN_CHECK(miopenGetRNNLayerBiasOffset(
				rnn_desc.desc(),
				layer,
				x_desc.desc(),
				linear_id,
				lin_layer_mat_desc.mut_desc(),
				&offset));

			size_t bias_size;
			MIOPEN_CHECK(miopenGetRNNLayerBiasSize(
				handle,
				rnn_desc.desc(),
				layer,
				linear_id,
				&bias_size));
			bias_size /= elem_size;

			if(linear_id == 0 || linear_id == num_linear_layers / 2) {
				std::initializer_list<int64_t> size = { bias_size * num_linear_layers / 2, 1};
				Tensor param = at::empty({0}, weight_buf.options()).set_(weight_buf.storage(), offset, size);
				params.emplace_back(std::move(param));
				layer_params_count++;
			} else {
				AT_ASSERTM(cur_offset == offset, "cur_offset = ", cur_offset, " ; offset = ", offset);
			}
			cur_offset = offset + bias_size;
		}

		if (layer == 0) {
			global_layer_params_count = layer_params_count;
		} else {
			AT_ASSERTM(global_layer_params_count == layer_params_count,
				"global_layer_params_count = ", global_layer_params_count,
				"; layer_params_count = ", layer_params_count);
		}
	} // layer
	return std::make_pair(params, global_layer_params_count);
}

std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
	if (tensors.is_input_packed()) {
		return {tensors.batch_sizes_sum, tensors.input_size};
	} else {
		return {tensors.seq_length, tensors.mini_batch, tensors.input_size};
	}
}

std::vector<int64_t> _hidden_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
	return {rnn.num_layers * rnn.num_directions(), tensors.mini_batch, rnn.hidden_size};
}

std::vector<int64_t> _output_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
	if (tensors.is_input_packed()) {
		return {tensors.batch_sizes_sum, rnn.hidden_size * rnn.num_directions()};
	} else {
		return {tensors.seq_length, tensors.mini_batch, rnn.hidden_size * rnn.num_directions()};
	}
}

Tensor miopen_rnn_flatten_weight(
        TensorList weight_arr, int64_t weight_stride0, int64_t input_size,
        int64_t fn_mode, int64_t fn_hidden_size, int64_t fn_num_layers,
        bool batch_first, bool fn_bidirectional
        ) {
    //AT_ERROR("miopen_flatten_weight: not implemented yet.");

    AT_CHECK(weight_arr.size() > 0, "miopen_rnn_flatten_weight : cannot flatten empty weight list.");

    auto any_param = weight_arr[0];
    auto handle = getMiopenHandle();
    auto datatype = getMiopenDataType(any_param);

    RNNDescriptorParams rnn;
    miopenRNNBiasMode_t bias_mode = (weight_stride0 == 4) ? miopenRNNwithBias : miopenRNNNoBias;
    rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, datatype, bias_mode);

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
    //AT_ERROR("miopen_rnn : not implemented yet.");

    check_device(input_r, weight, {hx, cx});
    auto input = input_r;
    auto weight_buf = weight_buf_r;
    
    RNNParams fn;
    auto datatype = getMiopenDataType(input);
    miopenRNNBiasMode_t bias_mode = (weight_stride0 == 4) ? miopenRNNwithBias : miopenRNNNoBias;
    fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, datatype, bias_mode);
    fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

    if (fn.rnn.rnn_mode != miopenLSTM) {
    	AT_CHECK(!cx.defined(), "miopen_rnn: illegal defined cx for non-LSTM RNN.");
    }

    auto is_input_packed = fn.tensors.batch_sizes.size() != 0;
    if (batch_first && !is_input_packed) {
    	input = input.transpose(0, 1);
    }

    auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
    auto output_size = _output_size(fn.rnn, fn.tensors);

    AT_CHECK(hx.is_contiguous(), "miopen_rnn : hx is not contiguous.");
    AT_CHECK(!cx.defined() || cx.is_contiguous(), "miopen_rnn : cx is not contiguous.");

    auto x = input.contiguous();
    auto output = at::empty(output_size, input.options());
    auto hy = at::empty(hidden_size, hx.options());
    Tensor cy;
    if (cx.defined()) {
    	cy = at::empty(hidden_size, cx.options());
    } else {
    	cy = at::empty({0}, hx.options());
    }

    auto y = output;
    auto handle = getMiopenHandle();
    miopenRNNAlgo_t algo = miopenRNNdefault;
    fn.rnn.set_algo(algo);

    RNNDescriptors descs(fn, handle, x, y, hx, cx);

    //TODO: Need to implement get_parameters that gets params and params_stride0. [Done.]
    FilterDescriptor w_desc;
    if (!weight_buf.defined()) {
    	auto num_weights = get_num_weights(handle, descs.rnn_desc, descs.x_descs[0], datatype);
    	weight_buf = at::empty(num_weights, x.options());
    	w_desc.set(weight_buf, 3);
    	weight_buf.zero_();
    	std::vector<Tensor> params;
    	size_t params_stride0;
    	std::tie(params, params_stride0) = get_parameters(handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, weight_buf);
    	_copyParams(MatrixRef<Tensor>{weight, static_cast<size_t>(weight_stride0)},
    				MatrixRef<Tensor>{params, params_stride0});
    } else {
    	w_desc.set(weight_buf, 3);
    }

    AT_CHECK(!cx.defined() || cx.sizes().equals(hidden_size), "Expected cell size ", IntArrayRef{hidden_size}, ", got", cx.sizes());

    size_t workspace_size;
    auto x_descs_arr = descs.get_x_descs();
    auto y_descs_arr = descs.get_y_descs();

    //Allocate workspace size.
    MIOPEN_CHECK(miopenGetRNNWorkspaceSize(handle, descs.rnn_desc.desc(), fn.tensors.seq_length, x_descs_arr.data(), &workspace_size));
    auto workspace = at::empty(workspace_size, input.options().dtype(kByte));

    //Train or inference.
    Tensor reserve;
    if (fn_train) { //Train.
    	size_t reserver_size;
    	MIOPEN_CHECK(miopenGetRNNTrainingReserveSize(handle, descs.rnn_desc.desc(), fn.tensors.seq_length, x_descs_arr.data(), &reserver_size));
    	reserve = at::empty(reserver_size, input.options().dtype(kByte));

    	MIOPEN_CHECK(miopenRNNForwardTraining(handle, descs.rnn_desc.desc(), fn.tensors.seq_length,
    			x_descs_arr.data(), x.data_ptr(),
    			descs.hx_desc.desc(), hx.data_ptr(),
    			descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
    			w_desc.desc(), weight_buf.data_ptr(),
    			y_descs_arr.data(), y.data_ptr(),
    			descs.hy_desc.desc(), hy.data_ptr(),
    			descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr, 
    			workspace.data_ptr(), workspace_size, reserve.data_ptr(), reserver_size ));
    } else { //Inference.
    	reserve = at::empty({0}, input.options().dtype(kByte));
    	MIOPEN_CHECK(miopenRNNForwardInference(handle, descs.rnn_desc.desc(), fn.tensors.seq_length,
    			x_descs_arr.data(), x.data_ptr(),
    			descs.hx_desc.desc(), hx.data_ptr(),
    			descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
    			w_desc.desc(), weight_buf.data_ptr(),
    			y_descs_arr.data(), y.data_ptr(),
    			descs.hy_desc.desc(), hy.data_ptr(),
    			descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr,
    			workspace.data_ptr(), workspace_size));
    }

    if (batch_first && !is_input_packed) {
    	output.transpose_(0, 1);
    }

    return std::make_tuple(output, hy, cy, reserve, weight_buf);

}

std::tuple<Tensor, Tensor, Tensor> miopen_rnn_backward_input(
	    const Tensor& input_r, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
	    const Tensor& output_r, const Tensor& grad_output_r, const Tensor& grad_hy,
	    const Tensor& grad_cy,
	    int64_t fn_mode, int64_t fn_hidden_size,
	    int64_t fn_num_layers, bool batch_first, double fn_dropout,
	    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
	    const Tensor& fn_dropout_state, const Tensor& fn_reserve,
	    std::array<bool, 3> output_mask
	    ) {
	auto input = input_r;
	auto grad_output = grad_output_r;
	auto output = output_r;

	RNNParams fn;
	auto datatype = getMiopenDataType(input);
	fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, datatype, miopenRNNwithBias);	// AF TODO: Check the bias bool
	fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

	auto handle = getMiopenHandle();

	if(fn.rnn.rnn_mode != miopenLSTM) {
		AT_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
	}

	auto is_input_packed = fn_batch_sizes.size() != 0;
	if (batch_first && !is_input_packed) {
		input = input.transpose(0, 1);
		grad_output = grad_output.transpose(0, 1);
		output = output.transpose(0, 1);
	}

	auto input_size = _input_size(fn.tensors);
	auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
	auto output_size = _output_size(fn.rnn, fn.tensors);

	AT_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
	AT_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

	auto x = input.contiguous();
    auto dy = grad_output.contiguous();
    auto y = output;
    auto w = weight_buf;
    auto dx = at::empty(input.sizes(), input.options());
    auto dhy = grad_hy.contiguous().view(hidden_size);
    auto dcy = grad_cy.defined() ? grad_cy.contiguous().view(hidden_size) : Tensor();
    auto dhx = at::empty(hidden_size, hx.options());
    AT_ASSERTM(cx.defined() || !output_mask[2], "illegally required grad of cx for non-LSTM RNN");
    auto dcx = cx.defined() ? at::empty(hidden_size, cx.options()) : Tensor();

    AT_CHECK(fn_train, "miopen RNN backward can only be called in training mode");

    AT_CHECK(input.sizes().equals(input_size),
		"Expected input size ", IntArrayRef{input_size}, ", got ", input.sizes());
	AT_CHECK(output.sizes().equals(output_size),
        "Expected output size ", IntArrayRef{output_size}, ", got ", output.sizes());

	AT_CHECK(!hx.defined() || hx.sizes().equals(hidden_size),
        "Expected hidden size ", IntArrayRef{hidden_size}, ", got ", hx.sizes());
	AT_CHECK(!cx.defined() || cx.sizes().equals(hidden_size),
        "Expected cell size ", IntArrayRef{hidden_size}, ", got ", cx.sizes());
	AT_CHECK(!dhy.defined() || dhy.sizes().equals(hidden_size),
        "Expected d_hidden size ", IntArrayRef{hidden_size}, ", got ", dhy.sizes());
	AT_CHECK(!dcy.defined() || dcy.sizes().equals(hidden_size),
        "Expected d_cell size ", IntArrayRef{hidden_size}, ", got ", dcy.sizes());

	AT_CHECK(dhy.is_cuda() && dy.is_cuda() && (!dcy.defined() || dcy.is_cuda()),
        "Gradients aren't HIP tensors");

	miopenRNNAlgo_t algo = miopenRNNdefault;
	fn.rnn.set_algo(algo);
	RNNDescriptors descs(fn, handle, x, y, hx, cx);

	FilterDescriptor w_desc;
	w_desc.set(weight_buf, 3);

	size_t workspace_size;
	auto x_descs_arr = descs.get_x_descs();
	auto y_descs_arr = descs.get_y_descs();

	MIOPEN_CHECK(miopenGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
	auto workspace = at::empty(workspace_size, input.options().dtype(kByte));

	MIOPEN_CHECK(miopenRNNBackwardData(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        y_descs_arr.data(), y.data_ptr(),
        y_descs_arr.data(), dy.data_ptr(),
        descs.hy_desc.desc(), dhy.data_ptr(),
        descs.cy_desc.desc(), cx.defined() ? dcy.data_ptr() : nullptr,
        w_desc.desc(), w.data_ptr(),
        descs.hx_desc.desc(), hx.data_ptr(),
        descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
        x_descs_arr.data(), dx.data_ptr(),
        descs.hx_desc.desc(), dhx.data_ptr(),
        descs.cx_desc.desc(), cx.defined() ? dcx.data_ptr() : nullptr,
        workspace.data_ptr(), workspace.size(0),
        fn_reserve.data_ptr(), fn_reserve.size(0)
        ));

	if(batch_first && !is_input_packed) {
		dx = dx.transpose_(0, 1);
	}

	return std::make_tuple(dx, dhx, dcx);
}

std::vector<Tensor> miopen_rnn_backward_weight(
	    const Tensor& input_r, TensorList weight_arr, int64_t weight_stride0,
	    const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
	    const Tensor& output_r,
	    int64_t fn_mode, int64_t fn_hidden_size,
	    int64_t fn_num_layers, bool batch_first, double fn_dropout,
	    bool fn_train, bool fn_bidirectional, IntArrayRef fn_batch_sizes,
	    const Tensor& fn_dropout_state, const Tensor& fn_reserve
	    ) {
	MatrixRef<Tensor> weight{ weight_arr, static_cast<size_t>(weight_stride0) };

	auto input = input_r;
	auto output = output_r;

	RNNParams fn;
	auto datatype = getMiopenDataType(input);
	miopenRNNBiasMode_t bias_mode = (weight_stride0 == 4) ? miopenRNNwithBias : miopenRNNNoBias;
	fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional, datatype, bias_mode);
	fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

	auto handle = getMiopenHandle();

	if (fn.rnn.rnn_mode != miopenLSTM) {
		AT_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
	}

	auto is_input_packed = fn_batch_sizes.size() != 0;
	if (batch_first && !is_input_packed) {
		input = input.transpose(0, 1);
		output = output.transpose(0, 1);
	}

	auto input_size = _input_size(fn.tensors);
	auto hidden_size = _hidden_size(fn.rnn, fn.tensors);

	AT_CHECK(fn_train, "miopen RNN backward can only be called in training mode");

	AT_CHECK(input.sizes().equals(input_size),
		"Expected input size ", IntArrayRef{input_size}, ", got ", input.sizes());
	AT_CHECK(!hx.defined() || hx.sizes().equals(hidden_size),
		"Expected hidden size ", IntArrayRef{hidden_size}, ", got ", hx.sizes());

	AT_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
	AT_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

	auto x = input.contiguous();
	const auto& y = output;
	auto dw = at::zeros(weight_buf.sizes(), weight_buf.options());

	miopenRNNAlgo_t algo = miopenRNNdefault;
	fn.rnn.set_algo(algo);
	RNNDescriptors descs(fn, handle, x, y, hx, cx);

	FilterDescriptor w_desc;
	w_desc.set(weight_buf, 3);

	size_t workspace_size;
	auto x_descs_arr = descs.get_x_descs();
	auto y_descs_arr = descs.get_y_descs();

	MIOPEN_CHECK(miopenGetRNNWorkspaceSize(
		handle,
		descs.rnn_desc.desc(),
		fn.tensors.seq_length,
		x_descs_arr.data(),
		&workspace_size
		));
	auto workspace = at::empty(workspace_size, input.options().dtype(kByte));

	MIOPEN_CHECK(miopenRNNBackwardWeights(
		handle,
		descs.rnn_desc.desc(),
		fn.tensors.seq_length,
		x_descs_arr.data(), x.data_ptr(),
		descs.hx_desc.desc(), hx.data_ptr(),
		y_descs_arr.data(), y.data_ptr(),
		w_desc.desc(), dw.data_ptr(),
		workspace.data_ptr(), workspace.size(0),
		fn_reserve.data_ptr(), fn_reserve.size(0)
		));

	std::vector<Tensor> grad_params_arr;
	size_t grad_params_stride0;
	std::tie(grad_params_arr, grad_params_stride0) = get_parameters(handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, dw);
	if (grad_params_stride0 == static_cast<size_t>(weight_stride0)) {
		_viewParams(MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
			MatrixRef<Tensor>{weight_arr, static_cast<size_t>(weight_stride0)});
		return grad_params_arr;
	} else {
		std::vector<Tensor> grad_weight_arr;
		grad_weight_arr.reserve( weight.numel() );
		for (const auto& w : weight_arr) {
			grad_weight_arr.emplace_back(at::empty(w.sizes(), w.options()));
		}
		_copyParams(MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
			MatrixRef<Tensor>{grad_weight_arr, static_cast<size_t>(weight_stride0)});
		return grad_weight_arr;
	}
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> miopen_rnn_backward(
        const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
        const Tensor& output, const Tensor& grad_output_r, const Tensor& grad_hy_r,
        const Tensor& grad_cy_r, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first,
        double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor& dropout_state, 
        const Tensor& reserve, std::array<bool, 4> output_mask
        ) {
    auto grad_output = grad_output_r.defined() ? grad_output_r : at::zeros_like(output);
    auto grad_hy = grad_hy_r.defined() ? grad_hy_r : at::zeros_like(hx);
    auto grad_cy = cx.defined() ? (grad_cy_r.defined() ? grad_cy_r : at::zeros_like(cx)) : grad_cy_r;

    Tensor dx, dhx, dcx;
    std::tie(dx, dhx, dcx) = at::native::miopen_rnn_backward_input(input, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, {output_mask[0], output_mask[1], output_mask[2]});
    std::vector<Tensor> dw;
    if (output_mask[3]) {
		dw = at::native::miopen_rnn_backward_weight(input, weight, weight_stride0, weight_buf, hx, cx, output, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve);
    }
    return std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>{dx, dhx, dcx, dw};
}

namespace {

std::tuple<Tensor, Tensor> unpack_hidden(const Tensor& hidden) {
	return std::make_tuple(hidden, at::Tensor{});
}

std::tuple<Tensor, Tensor> unpack_hidden(const std::tuple<Tensor, Tensor>& hidden) {
	return hidden;
}

template<typename hidden_type>
hidden_type pack_hidden(const Tensor& hx, const Tensor& cx) {
  static_assert(std::is_same<hidden_type, void>::value, "pack_hidden not implemented for this type");
  AT_ERROR("NOT IMPLEMENTED");
}

template<>
Tensor pack_hidden<Tensor>(const Tensor& hx, const Tensor& cx) {
  AT_ASSERT(cx.numel() == 0);
  return hx;
}

template<>
std::tuple<Tensor, Tensor> pack_hidden<std::tuple<Tensor, Tensor>>(const Tensor& hx, const Tensor& cx) {
  return std::make_tuple(hx, cx);
}

Tensor try_get_weight_buf(
      const Tensor& input, TensorList parameters, bool has_biases,
      miopenRNNMode_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional) {
  // Prepare all relevant descriptors
  auto handle = getMiopenHandle();
  auto datatype = getMiopenDataType(input);

  RNNDescriptorParams rnn;
  miopenRNNBiasMode_t bias_mode = (has_biases) ? miopenRNNwithBias : miopenRNNNoBias;
  rnn.set(mode, hidden_size, num_layers, bidirectional, datatype, bias_mode);
  RNNDescriptor rnn_desc = rnn.descriptor();

  TensorGeometry x_geom ({1, input.size(-1)});
  TensorDescriptor x_desc;
  x_desc.set(datatype, x_geom.sizes(), x_geom.strides(), 5);

  auto num_params = get_num_weights(handle, rnn_desc, x_desc, datatype);

  // Try to get parameter storage
  auto & any_param = parameters.at(0);
  auto param_storage = any_param.storage();
  auto weight_buf = at::empty({0}, any_param.options()).set_(param_storage);
  if (weight_buf.size(0) < num_params) {
    return {};
  } else if (weight_buf.size(0) > num_params) {
    weight_buf = weight_buf.narrow(0, 0, num_params);
  }

  // Get and check data pointers
  //TODO: implement get_expected_data_ptrs.
  auto expected_data_ptrs = get_expected_data_ptrs(
      weight_buf, handle, rnn, rnn_desc, x_desc, datatype);

  int64_t num_parameters = parameters.size();
  int64_t num_ptrs = expected_data_ptrs.size();
  AT_ASSERT(num_ptrs == (num_parameters * (has_biases ? 1 : 2)));
  AT_ASSERT(num_ptrs % (has_biases ? 4 : 2) == 0);
  for (int64_t param_i = 0, ptr_i = 0;
       ptr_i < num_ptrs;
       ptr_i += (has_biases ? 2 : 4), param_i += 2) {
    if (expected_data_ptrs[ptr_i] != parameters[param_i].data_ptr()) return {};
    if (expected_data_ptrs[ptr_i + 1] != parameters[param_i + 1].data_ptr()) return {};
  }
  if (!parameters[num_parameters - 1].is_contiguous()) return {};
  return weight_buf;
}

const char * WEIGHT_FORMAT_WARN = "RNN module weights are not part of single contiguous "
                                  "chunk of memory. This means they need to be compacted "
                                  "at every call, possibly greatly increasing memory usage. "
                                  "To compact weights again call flatten_parameters().";

template<typename hidden_type>
std::pair<Tensor, hidden_type> _miopen_impl(
      const Tensor& input, const Tensor& _batch_sizes, const hidden_type& hidden,
      TensorList params, bool has_biases, miopenRNNMode_t mode,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  Tensor hx, cx;
  std::tie(hx, cx) = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  auto weight_buf = try_get_weight_buf(
      input, params, has_biases, mode, hidden_size, num_layers, bidirectional);
  if (!weight_buf.defined()) {
    AT_WARN(WEIGHT_FORMAT_WARN);
  }

  AT_CHECK(_batch_sizes.dim() == 1, "batch_sizes tensor should be 1D");
  IntArrayRef batch_sizes { _batch_sizes.data<int64_t>(), static_cast<size_t>(_batch_sizes.size(0)) };

  Tensor dropout_state = at::empty({0}, input.options());

  auto miopen_output = at::miopen_rnn(
      input, params, has_biases ? 4 : 2, weight_buf,
      hx, cx, static_cast<int>(mode), hidden_size, num_layers, /*batch_first=*/false,
      dropout_p, train, bidirectional, batch_sizes, dropout_state);

  return {std::get<0>(miopen_output),
          pack_hidden<hidden_type>(std::get<1>(miopen_output), std::get<2>(miopen_output))};
}

template<typename hidden_type>
std::pair<Tensor, hidden_type> _miopen_impl(
      const Tensor& input, const hidden_type& hidden,
      TensorList params, bool has_biases, miopenRNNMode_t mode,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  Tensor hx, cx;
  std::tie(hx, cx) = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  auto weight_buf = try_get_weight_buf(
      input, params, has_biases, mode, hidden_size, num_layers, bidirectional);
  if (!weight_buf.defined()) {
    AT_WARN(WEIGHT_FORMAT_WARN);
  }

  Tensor dropout_state = at::empty({0}, input.options());

  auto miopen_output = at::miopen_rnn(
      input, params, has_biases ? 4 : 2, weight_buf,
      hx, cx, static_cast<int>(mode), hidden_size, num_layers, batch_first, dropout_p,
      train, bidirectional, /*batch_sizes=*/{}, dropout_state);

  return {std::get<0>(miopen_output),
          pack_hidden<hidden_type>(std::get<1>(miopen_output), std::get<2>(miopen_output))};
}

#define ONE_HIDDEN_RNN(NAME, MODE)                                             \
void NAME##_miopen(Tensor& output, Tensor& hy,                                 \
      const Tensor& input, const Tensor& hx,                                   \
      TensorList params, bool has_biases,                                      \
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) { \
  std::tie(output, hy) = _miopen_impl(input, hx, params, has_biases,           \
      MODE, num_layers, dropout_p, train, bidirectional, batch_first);         \
}                                                                              \
                                                                               \
void NAME##_packed_miopen(Tensor& output, Tensor& hy,                          \
      const Tensor& data, const Tensor& batch_sizes, const Tensor& hx,         \
      TensorList params, bool has_biases,                                      \
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {  \
  std::tie(output, hy) = _miopen_impl(data, batch_sizes, hx, params,           \
      has_biases, MODE, num_layers, dropout_p, train, bidirectional);          \
}                                                                              \
                                                                               \
REGISTER_CUDA_DISPATCH(NAME##_miopen_stub, &NAME##_miopen);                    \
REGISTER_CUDA_DISPATCH(NAME##_packed_miopen_stub, &NAME##_packed_miopen);

ONE_HIDDEN_RNN(gru, miopenGRU)
ONE_HIDDEN_RNN(rnn_tanh, miopenRNNTANH)
ONE_HIDDEN_RNN(rnn_relu, miopenRNNRELU)

void lstm_miopen(Tensor& output, Tensor& hy, Tensor& cy,
      const Tensor& input, TensorList hx,
      TensorList params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  auto result = _miopen_impl(input, std::make_tuple(hx[0], hx[1]), params, has_biases,
      miopenLSTM, num_layers, dropout_p, train, bidirectional, batch_first);
  output = result.first;
  hy = std::get<0>(result.second);
  cy = std::get<1>(result.second);
}

void lstm_packed_miopen(Tensor& output, Tensor& hy, Tensor& cy,
      const Tensor& data, const Tensor& batch_sizes, TensorList hx,
      TensorList params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  auto result = _miopen_impl(data, batch_sizes, std::make_tuple(hx[0], hx[1]),
      params, has_biases, miopenLSTM, num_layers, dropout_p, train, bidirectional);
  output = result.first;
  hy = std::get<0>(result.second);
  cy = std::get<1>(result.second);
}

REGISTER_CUDA_DISPATCH(lstm_miopen_stub, &lstm_miopen);
REGISTER_CUDA_DISPATCH(lstm_packed_miopen_stub, &lstm_packed_miopen);

} // anonymous namepsace
}} //namespace native.

#endif 
