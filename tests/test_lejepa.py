import torch

from bias import augment, pretrain_lejepa


def test_sigreg_is_small_for_gaussian_and_large_for_collapse() -> None:
    torch.manual_seed(0)
    sigreg = pretrain_lejepa.SIGReg()
    gaussian = torch.randn(4, 512, 16)
    collapsed = torch.zeros(4, 512, 16) + 0.01 * torch.randn(4, 512, 16)
    assert sigreg(gaussian) < 3.0
    assert sigreg(collapsed) > 50 * sigreg(gaussian)


def test_prediction_term_is_zero_when_views_agree() -> None:
    sigreg = pretrain_lejepa.SIGReg()
    same = torch.randn(1, 64, 8).expand(4, 64, 8)
    _, prediction, _ = pretrain_lejepa.lejepa_loss(same, sigreg, lam=0.02)
    assert prediction.item() == 0.0


def test_view_transform_keeps_shape_and_dtype() -> None:
    image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    out = augment.cifar_ssl_view()(image)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.uint8
