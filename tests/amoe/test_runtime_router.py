import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / "ultralytics/nn/modules/drmamba_router.py"
SPEC = importlib.util.spec_from_file_location("drmamba_router_under_test", MODULE_PATH)
router_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router_module)


def test_exact_top2_contract_for_deterministic_logits():
    router = router_module.GAPTopKRouter(dim=8, experts=4, top_k=2)
    output = router.route_logits(torch.tensor([[0.0, 3.0, 2.0, 1.0]]))

    assert set(output.selected_indices[0].tolist()) == {1, 2}
    assert output.weights[0, 0] == 0
    assert output.weights[0, 3] == 0
    torch.testing.assert_close(output.weights.sum(dim=1), torch.ones(1))
    expected = torch.softmax(torch.tensor([3.0, 2.0]), dim=0)
    torch.testing.assert_close(output.weights[0, [1, 2]], expected)


def test_axis_descriptor_distinguishes_horizontal_and_vertical_stripes():
    router = router_module.AxisAwareTopKRouter(dim=1, detach_descriptor=False)
    horizontal = torch.tensor([[[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]]])
    vertical = horizontal.transpose(2, 3).contiguous()

    horizontal_descriptor = router.describe(horizontal)
    vertical_descriptor = router.describe(vertical)

    assert horizontal_descriptor.shape == vertical_descriptor.shape == (1, 4)
    assert not torch.equal(horizontal_descriptor, vertical_descriptor)


def test_detached_axis_descriptor_blocks_input_gradient_but_not_router_gradient():
    router = router_module.AxisAwareTopKRouter(dim=4, detach_descriptor=True)
    x = torch.randn(2, 4, 3, 5, requires_grad=True)

    output = router(x)
    output.logits.square().mean().backward()

    assert x.grad is None
    gradients = [parameter.grad for parameter in router.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_train_eval_top2_selection_is_identical_without_router_noise():
    torch.manual_seed(17)
    x = torch.randn(3, 8, 4, 5)
    for router in (
        router_module.GAPTopKRouter(dim=8),
        router_module.AxisAwareTopKRouter(dim=8),
    ):
        router.train()
        training_output = router(x)
        router.eval()
        evaluation_output = router(x)

        torch.testing.assert_close(training_output.weights, evaluation_output.weights)
        torch.testing.assert_close(training_output.selected_indices, evaluation_output.selected_indices)
        assert (training_output.weights != 0).sum(dim=1).eq(2).all()
