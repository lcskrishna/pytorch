import torch
import sys

MIOPEN_TENSOR_TYPES = {
    'torch.cuda.HalfTensor',
    'torch.cuda.FloatTensor',
}

MIOPEN_RNN_RELU = 0
MIOPEN_RNN_TANH = 1
MIOPEN_LSTM = 2
MIOPEN_GRU = 3

def is_available():
    r"""Returns whether PyTorch is built with MIOpen support."""
    return torch._C.has_miopen

def get_miopen_rnn_mode(mode):
    if mode == 'RNN_RELU':
        return MIOPEN_RNN_RELU
    elif mode == 'RNN_TANH':
        return MIOPEN_RNN_TANH
    elif mode == 'LSTM':
        return MIOPEN_LSTM
    elif mode == 'GRU':
        return MIOPEN_GRU
    else:
        raise Exception("Unknown RNN mode: {}".format(mode))

def is_acceptable(tensor):
	if tensor.type() not in MIOPEN_TENSOR_TYPES:
		return False
	if not is_available():
		return False
	return True

def is_rnn_acceptable(tensor):
	# MIOpen RNNs only work for fp32 datatype
	if tensor.type() != 'torch.cuda.FloatTensor':
		return False
	if not is_available():
		return False
	return True

def permute_rnn_weights(mode, wei_tensor):
	if mode == 'LSTM':
		split_param = torch.chunk(wei_tensor, 4, 0)
		permuted_param = torch.cat((split_param[0], split_param[1], split_param[3], split_param[2]), 0)
		return permuted_param
	elif mode == 'GRU':
		split_param = torch.chunk(wei_tensor, 3, 0)
		permuted_param = torch.cat((split_param[1], split_param[0], split_param[2]), 0)
		return permuted_param
	else:
		return wei_tensor