import pytest
import torch

from ultralytics.nn.modules import ProtoHypergraphFusion


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_soft_relation_cuda_amp_backward():
    """Soft top-k relations must scatter with the AMP similarity dtype."""
    fusion = ProtoHypergraphFusion(128, relation="soft").cuda().train()
    rgb = torch.randn(2, 128, 20, 20, device="cuda", requires_grad=True)
    ir = torch.randn(2, 128, 20, 20, device="cuda", requires_grad=True)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = fusion([rgb, ir])
        loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert all(parameter.grad is not None for parameter in fusion.parameters() if parameter.requires_grad)
