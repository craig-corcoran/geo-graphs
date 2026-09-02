import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from geo_graphs.model import (
    UNet,
    logit,
    predict_mask,
    segmentation_loss,
    soft_dice_loss,
)


def targets(batch: int = 2, size: int = 32) -> torch.Tensor:
    t = torch.zeros(batch, 1, size, size)
    t[:, :, size // 3 : 2 * size // 3, :] = 1.0
    return t


@pytest.mark.parametrize("widths", [(8, 16), (8, 16, 32), (4, 8, 16, 32)])
def test_output_matches_input_resolution(widths):
    model = UNet(in_channels=3, widths=widths)
    size = 8 * 2 ** len(widths)
    out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 1, size, size)


def test_depth_is_the_number_of_downsamples():
    assert UNet(widths=(8, 16, 32)).depth == 2


def test_rejects_a_degenerate_width_list():
    with pytest.raises(ValueError, match="at least an encoder"):
        UNet(widths=(8,))


def test_accepts_non_rgb_input():
    out = UNet(in_channels=8, widths=(8, 16))(torch.randn(1, 8, 32, 32))
    assert out.shape == (1, 1, 32, 32)


def test_loss_at_initialization_is_about_ln_two():
    """The standard check: uninformative logits should cost ln(2) under BCE.

    A different value here means the targets or the logits are wrong, and it is
    far easier to catch now than after a training run has gone sideways.
    """
    t = targets()
    bce = F.binary_cross_entropy_with_logits(torch.zeros_like(t), t)
    assert float(bce) == pytest.approx(0.6931, abs=1e-3)


def test_dice_rewards_a_perfect_prediction():
    t = targets()
    assert float(soft_dice_loss(torch.where(t > 0, 20.0, -20.0), t)) == pytest.approx(
        0.0, abs=1e-4
    )


def test_dice_punishes_predicting_nothing():
    """The failure cross-entropy alone would tolerate on a sparse target."""
    t = targets()
    assert float(soft_dice_loss(torch.full_like(t, -20.0), t)) > 0.9


def test_dice_is_zero_when_both_are_empty():
    empty = torch.zeros(1, 1, 16, 16)
    assert float(soft_dice_loss(torch.full_like(empty, -20.0), empty)) == pytest.approx(
        0.0, abs=1e-3
    )


def test_segmentation_loss_blends_its_two_terms():
    t = targets()
    logits = torch.randn_like(t)

    bce_only = float(segmentation_loss(logits, t, dice_weight=0.0))
    dice_only = float(segmentation_loss(logits, t, dice_weight=1.0))
    blended = float(segmentation_loss(logits, t, dice_weight=0.5))

    assert bce_only != dice_only
    assert blended == pytest.approx(0.5 * (bce_only + dice_only), abs=1e-5)


def test_gradients_reach_every_parameter():
    """A None grad means something was detached or never used in the loss."""
    model = UNet(in_channels=3, widths=(8, 16))
    segmentation_loss(model(torch.randn(2, 3, 32, 32)), targets()).backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []


def test_predict_mask_thresholds_on_probability():
    # sigmoid(-2, 0, 2) = 0.119, 0.5, 0.881
    logits = torch.tensor([[-2.0, 0.0, 2.0]])
    assert predict_mask(logits, threshold=0.5).tolist() == [[False, False, True]]
    assert predict_mask(logits, threshold=0.2).tolist() == [[False, True, True]]
    assert predict_mask(logits, threshold=0.05).tolist() == [[True, True, True]]


def test_model_is_deterministic_in_eval_mode():
    model = UNet(in_channels=3, widths=(8, 16)).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_logit_inverts_sigmoid():
    for p in (0.1, 0.5, 0.9):
        assert float(torch.sigmoid(torch.tensor(logit(p)))) == pytest.approx(p)


def test_logit_is_infinite_at_the_endpoints():
    assert logit(0.0) == -math.inf
    assert logit(1.0) == math.inf


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_logit_rejects_a_probability_outside_the_unit_interval(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        logit(bad)


def test_predict_mask_preserves_array_type():
    """Inference returns numpy, training stays in torch; one definition serves both."""
    values = [-2.0, 0.0, 2.0]
    assert isinstance(predict_mask(torch.tensor(values)), torch.Tensor)
    assert isinstance(predict_mask(np.array(values)), np.ndarray)


def test_predict_mask_agrees_across_array_types():
    values = np.linspace(-6.0, 6.0, 101, dtype=np.float32)
    for threshold in (0.1, 0.5, 0.9):
        torch_mask = predict_mask(torch.from_numpy(values), threshold).numpy()
        assert np.array_equal(predict_mask(values, threshold), torch_mask)


def test_predict_mask_matches_the_probability_space_comparison():
    """Logit-space thresholding is an optimization, not a different decision."""
    values = torch.linspace(-8.0, 8.0, 257)
    for threshold in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert torch.equal(
            predict_mask(values, threshold), torch.sigmoid(values) > threshold
        )


def test_predict_mask_at_extreme_thresholds():
    values = torch.tensor([-50.0, 0.0, 50.0])
    assert predict_mask(values, threshold=0.0).all()
    assert not predict_mask(values, threshold=1.0).any()
