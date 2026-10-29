import copy
import pytest

from vlmbench.additional_adapters import verify_quantized_experts


def worker():
    return [{'parameter_devices': {'cuda:0': 3}, 'experts': [{
        'quant_method': 'CompressedTensorsWNA16MoEMethod',
        'tensors': [
            {'name': name, 'dtype': 'torch.int32', 'device': 'cuda:0'}
            for name in ('w13_weight', 'w2_weight')
        ] + [{'name': 'w13_weight_scale', 'dtype': 'torch.bfloat16', 'device': 'cuda:0'}]
    }]}]


def test_packed_weights_with_bf16_scales_are_valid():
    verify_quantized_experts(worker(), 1)


@pytest.mark.parametrize('failure', ['dequantized', 'unquantized_method', 'offload', 'missing_layer'])
def test_false_int4_success_is_rejected(failure):
    evidence = copy.deepcopy(worker())
    if failure == 'dequantized':
        for tensor in evidence[0]['experts'][0]['tensors']:
            tensor['dtype'] = 'torch.bfloat16'
    elif failure == 'unquantized_method':
        evidence[0]['experts'][0]['quant_method'] = 'UnquantizedFusedMoEMethod'
    elif failure == 'offload':
        evidence[0]['parameter_devices']['cpu'] = 1
    else:
        evidence[0]['experts'] = []
    with pytest.raises(AssertionError):
        verify_quantized_experts(evidence, 1)
