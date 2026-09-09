import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from geo_graphs import model as model_module
from geo_graphs.model import (
    FUSION_REGISTRY,
    Fusion,
    UNet,
    logit,
    predict_mask,
    segmentation_loss,
    soft_cldice_loss,
    soft_dice_loss,
    soft_skeleton,
)


def targets(batch: int = 2, size: int = 32) -> torch.Tensor:
    t = torch.zeros(batch, 1, size, size)
    t[:, :, size // 3 : 2 * size // 3, :] = 1.0
    return t


def road_bar(size: int = 32, width: int = 9, margin: int = 2) -> torch.Tensor:
    """A horizontal road that stops short of the frame, so its ends can erode.

    The margin matters: max-pooling treats out-of-frame as background for
    dilation and as foreground for erosion, so a bar running to the image edge
    never loses length to skeletonization and the centreline effect vanishes.
    """
    t = torch.zeros(1, 1, size, size)
    row = (size - width) // 2
    t[:, :, row : row + width, margin : size - margin] = 1.0
    return t


def sever(mask: torch.Tensor, gap: int = 3) -> torch.Tensor:
    """Cut the bar clean through with a narrow vertical gap."""
    cut = mask.clone()
    middle = mask.shape[-1] // 2
    cut[:, :, :, middle : middle + gap] = 0.0
    return cut


def saturated(mask: torch.Tensor) -> torch.Tensor:
    """Logits that decode back to ``mask`` under a 0.5 threshold."""
    return torch.where(mask > 0, 20.0, -20.0)


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


def fused_model(aux_channels: int = 3, widths=(8, 16)) -> UNet:
    """A tiny fused network, in eval mode so batch norm is deterministic."""
    return UNet(
        in_channels=3,
        widths=widths,
        aux_channels=aux_channels,
        fusion=FUSION_REGISTRY["early"](),
    ).eval()


def aux_batch(
    observed: torch.Tensor, fill: float, size: int = 32, channels: int = 2
) -> torch.Tensor:
    """An aux stack whose unobserved data channels are set to ``fill``."""
    values = torch.randn(1, channels, size, size)
    values = torch.where(observed > 0, values, torch.full_like(values, fill))
    return torch.cat([values, observed], dim=1)


def test_the_fill_value_under_an_invalid_pixel_cannot_reach_the_logits():
    """The single guarantee the availability channel exists to make.

    Two crops that differ only in what was written where nothing was measured
    must be indistinguishable to the model. If they are not, the network is
    reading the coverage mask off the data channels, and every number from the
    coverage sweep is measuring the fill rather than the lidar.
    """
    torch.manual_seed(0)
    model = fused_model()

    observed = torch.zeros(1, 1, 32, 32)
    observed[:, :, :16, :] = 1.0
    # The observed half is identical between the two; only the fill differs.
    torch.manual_seed(1)
    quiet = aux_batch(observed, fill=0.0)
    torch.manual_seed(1)
    loud = aux_batch(observed, fill=-999.0)

    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(image, quiet), model(image, loud))
    # The probe is only meaningful if the two stacks actually differ.
    assert not torch.equal(quiet, loud)


def test_observed_lidar_does_reach_the_logits():
    """The other half of the leakage test: masking must not mask everything."""
    torch.manual_seed(0)
    model = fused_model()
    observed = torch.ones(1, 1, 32, 32)
    image = torch.randn(1, 3, 32, 32)

    flat = torch.cat([torch.zeros(1, 2, 32, 32), observed], dim=1)
    raised = torch.cat([torch.ones(1, 2, 32, 32), observed], dim=1)
    with torch.no_grad():
        assert not torch.equal(model(image, flat), model(image, raised))


def test_mask_unobserved_zeroes_data_and_keeps_the_indicator():
    aux = torch.ones(1, 3, 4, 4) * 5.0
    aux[:, -1] = 1.0
    aux[:, -1, :2, :] = 0.0

    masked = model_module.mask_unobserved(aux)

    assert (masked[:, :2, :2, :] == 0.0).all()
    assert (masked[:, :2, 2:, :] == 5.0).all()
    assert torch.equal(masked[:, -1], aux[:, -1])


def test_mask_unobserved_rejects_a_stack_with_no_room_for_an_indicator():
    with pytest.raises(ValueError, match="K >= 1"):
        model_module.mask_unobserved(torch.zeros(1, 1, 4, 4))


def test_early_fusion_widens_the_stem_by_the_whole_aux_stack():
    fusion = FUSION_REGISTRY["early"]()
    assert fusion.stem_channels(3, 3) == 6
    assert model_module._first_conv(fused_model().encoders[0]).in_channels == 6


def test_the_new_stem_channels_start_at_the_mean_of_the_rgb_ones():
    """So a pretrained RGB encoder survives being widened."""
    torch.manual_seed(0)
    stem = model_module._first_conv(fused_model(aux_channels=3).encoders[0])
    expected = stem.weight[:, :3].mean(dim=1)
    assert torch.allclose(stem.weight[:, 3], expected)
    assert torch.allclose(stem.weight[:, 4], expected)
    assert torch.allclose(stem.weight[:, 5], expected)


def test_a_fused_model_still_maps_input_resolution_to_output():
    out = fused_model()(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 1, 32, 32)


def test_a_fused_model_refuses_to_run_without_its_aux_stack():
    with pytest.raises(ValueError, match="got no aux stack"):
        fused_model()(torch.randn(1, 3, 32, 32))


def test_an_imagery_only_model_refuses_an_aux_stack():
    """Silently ignoring it would hide a run that thought it was using lidar."""
    with pytest.raises(ValueError, match="built without fusion"):
        UNet(in_channels=3, widths=(8, 16))(
            torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
        )


def test_aux_channels_and_fusion_must_agree():
    with pytest.raises(ValueError, match="disagree"):
        UNet(widths=(8, 16), aux_channels=3)
    with pytest.raises(ValueError, match="disagree"):
        UNet(widths=(8, 16), fusion=FUSION_REGISTRY["early"]())


def test_fusion_registry_yields_a_fresh_instance_each_lookup():
    factory = FUSION_REGISTRY["early"]
    assert factory() is not factory()


def test_fusion_registry_entries_satisfy_the_protocol():
    assert all(isinstance(factory(), Fusion) for factory in FUSION_REGISTRY.values())


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


def test_soft_skeleton_is_thinner_than_the_shape():
    bar = road_bar()
    skeleton = soft_skeleton(bar)
    assert 0.0 < float(skeleton.sum()) < 0.25 * float(bar.sum())
    assert torch.all(skeleton <= bar + 1e-6)


def test_soft_skeleton_stays_inside_the_unit_interval():
    probs = torch.rand(1, 1, 24, 24)
    skeleton = soft_skeleton(probs)
    assert float(skeleton.min()) >= 0.0
    assert float(skeleton.max()) <= 1.0 + 1e-6


def test_too_few_iterations_leave_the_skeleton_hollow():
    """Why the iteration default is set above the road half-width, not below it.

    A 9px bar needs four peels before anything is thin enough to survive an
    opening. Stop short and the centreline never appears, which reads as a loss
    that does nothing rather than as a mis-set knob.
    """
    bar = road_bar(width=9)
    assert float(soft_skeleton(bar, iterations=3).sum()) == 0.0
    assert float(soft_skeleton(bar, iterations=10).sum()) > 0.0


def test_cldice_rewards_a_perfect_prediction():
    bar = road_bar()
    assert float(soft_cldice_loss(saturated(bar), bar)) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("empty", ["prediction", "target", "both"])
def test_cldice_is_finite_on_degenerate_masks(empty):
    bar = road_bar()
    blank = torch.zeros_like(bar)
    predicted = blank if empty in ("prediction", "both") else bar
    target = blank if empty in ("target", "both") else bar

    loss = soft_cldice_loss(saturated(predicted), target)
    assert math.isfinite(float(loss))
    assert 0.0 <= float(loss) <= 1.0


def test_cldice_is_zero_when_both_are_empty():
    blank = torch.zeros(1, 1, 16, 16)
    loss = soft_cldice_loss(torch.full_like(blank, -20.0), blank)
    assert float(loss) == pytest.approx(0.0, abs=1e-3)


def test_cldice_gradients_reach_the_logits():
    """Non-None, finite and non-zero: min/max pooling routes gradient, not blocks it."""
    bar = road_bar()
    logits = torch.where(bar > 0, 1.5, -1.5).requires_grad_(True)

    soft_cldice_loss(logits, bar).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_a_severing_gap_costs_more_on_cldice_than_on_dice():
    """The entire justification for the loss, as a number.

    A narrow cut through a wide road removes a small share of the area but a
    large share of the centreline, because skeleton ends retract by roughly the
    half-width. Dice sees the area; clDice sees the severed route.
    """
    bar = road_bar()
    cut = sever(bar, gap=3)
    perfect, gapped = saturated(bar), saturated(cut)

    dice_cost = float(soft_dice_loss(gapped, bar)) - float(soft_dice_loss(perfect, bar))
    cldice_cost = float(soft_cldice_loss(gapped, bar)) - float(
        soft_cldice_loss(perfect, bar)
    )

    assert dice_cost > 0.0
    assert cldice_cost > 1.2 * dice_cost


def test_cldice_weight_zero_is_an_exact_no_op():
    """Bit-identical, so turning the term off cannot perturb a baseline run."""
    t = targets()
    logits = torch.randn_like(t)
    for dice_weight in (0.0, 0.5, 1.0):
        before = segmentation_loss(logits, t, dice_weight=dice_weight)
        after = segmentation_loss(logits, t, dice_weight=dice_weight, cldice_weight=0.0)
        assert torch.equal(before, after)


def test_cldice_weight_zero_skips_the_skeleton(monkeypatch):
    """The iterated pooling is the expensive part; weight 0 must not pay for it."""

    def explode(*args, **kwargs):
        raise AssertionError("skeletonized despite cldice_weight=0")

    monkeypatch.setattr(model_module, "soft_skeleton", explode)
    t = targets()
    segmentation_loss(torch.randn_like(t), t, cldice_weight=0.0)


def test_segmentation_loss_blends_the_centreline_term():
    """Nested convex blend: clDice mixes with the pixel loss, it does not replace it."""
    bar = road_bar()
    logits = torch.randn_like(bar)

    pixel = float(segmentation_loss(logits, bar, dice_weight=0.5))
    centreline = float(soft_cldice_loss(logits, bar))
    blended = float(segmentation_loss(logits, bar, dice_weight=0.5, cldice_weight=0.25))

    assert blended == pytest.approx(0.75 * pixel + 0.25 * centreline, abs=1e-5)


def test_fewer_skeleton_iterations_stay_a_call_site_choice():
    """The cost knob is a parameter, so a smoke run turns it down without an edit."""
    bar = road_bar()
    logits = saturated(sever(bar))
    cheap = float(segmentation_loss(logits, bar, cldice_weight=0.5, cldice_iterations=2))
    full = float(segmentation_loss(logits, bar, cldice_weight=0.5))
    assert cheap != full


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
