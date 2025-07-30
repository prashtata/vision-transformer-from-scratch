# tests/test_mha.py

import torch
import pytest
from attention import *

torch.manual_seed(42)

IMPLEMENTATIONS = {
    "ParamLooped": MultiHeadedAttention_ParameterLooped,
    "ParamVectorized": MultiHeadedAttention_ParameterVec,
    "Param+Bias": MultiHeadedAttention_ParamBias,
    "Linear": MultiHeadedAttention_Linear,
}

@pytest.mark.parametrize("impl_name", IMPLEMENTATIONS.keys())
def test_output_shape(impl_name):
    model = IMPLEMENTATIONS[impl_name](embed_dim=64, num_heads=4)
    x = torch.randn(8, 10, 64)  # batch=8, seq=10, embed=64
    out = model(x)
    assert out.shape == x.shape, f"{impl_name}: Output shape mismatch."


@pytest.mark.parametrize("impl_name", IMPLEMENTATIONS.keys())
def test_gradient_flow(impl_name):
    model = IMPLEMENTATIONS[impl_name](embed_dim=64, num_heads=4)
    x = torch.randn(4, 10, 64, requires_grad=True)
    out = model(x)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None, f"{impl_name}: Gradients did not flow."


@pytest.mark.parametrize("impl_name", IMPLEMENTATIONS.keys())
def test_deterministic_output(impl_name):
    model = IMPLEMENTATIONS[impl_name](embed_dim=64, num_heads=4)
    x = torch.randn(2, 5, 64)
    torch.manual_seed(0)
    out1 = model(x)
    torch.manual_seed(0)
    out2 = model(x)
    assert torch.allclose(out1, out2, atol=1e-5), f"{impl_name}: Outputs are not deterministic."


@pytest.mark.parametrize("impl_name", IMPLEMENTATIONS.keys())
def test_invalid_head_config_raises(impl_name):
    # embed_dim not divisible by num_heads should raise
    with pytest.raises(AssertionError):
        _ = IMPLEMENTATIONS[impl_name](embed_dim=60, num_heads=8)


#@pytest.mark.parametrize("impl_name", IMPLEMENTATIONS.keys())
#def test_consistency_with_torch(impl_name):
#    if impl_name == "Torch":
#        return  # don't compare Torch to itself
#
#    custom_model = IMPLEMENTATIONS[impl_name](embed_dim=64, num_heads=4)
#    torch_model = TorchMultiHeadedAttention(embed_dim=64, num_heads=4)
#
#    x = torch.randn(2, 5, 64)
#    out_custom = custom_model(x)
#    out_torch = torch_model(x)
#
#    # Allow minor numerical differences
#    cos_sim = torch.nn.functional.cosine_similarity(
#        out_custom.flatten(), out_torch.flatten(), dim=0
#    )
#    assert cos_sim > 0.98, f"{impl_name}: Cosine similarity too low vs TorchMHA: {cos_sim:.4f}"

