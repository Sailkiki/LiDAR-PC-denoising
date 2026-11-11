import numpy as np
import torch

from r2net.models import DenoiseNet
from r2net.ops import knn_group, knn_points, normalize_unit_sphere


def test_ops_shapes():
    x = torch.randn(2, 20, 3)
    y = torch.randn(2, 30, 3)
    d2, idx, nn = knn_points(x, y, K=5, return_nn=True)
    assert d2.shape == (2, 20, 5)
    assert idx.shape == (2, 20, 5)
    assert nn.shape == (2, 20, 5, 3)
    _, idx_x, nn_x = knn_points(x, x, K=5, return_nn=True)
    g = knn_group(x, idx_x)
    assert g.shape == (2, 20, 5, 3)
    assert torch.allclose(g, nn_x)


def test_normalize_unit_sphere():
    p = torch.randn(100, 3) * 3 + 5
    n, c, r = normalize_unit_sphere(p)
    assert torch.isfinite(n).all()
    assert n.norm(dim=-1).max() <= 1.0 + 1e-5
    assert torch.allclose(n * r + c, p, atol=1e-5)


def test_model_loss_and_inference():
    model = DenoiseNet(
        frame_knn=8,
        num_train_points=8,
        num_clean_nbs=3,
        score_net_hidden_dim=32,
        score_net_num_blocks=2,
        conv_channels=16,
        num_convs=2,
        num_fc_layers=3,
        growth_rate=8,
        conv_knn=6,
        use_oc=True,
        use_ace=True,
    )
    pn = torch.randn(1, 128, 3)
    pc = torch.randn(1, 160, 3)
    loss = model.get_supervised_loss(pn, pc, loss_type="weighted_huber", rbf_sigma=0.02, loss_scale=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    out = model.infer_cloud(torch.randn(200, 3), patch_size=64, patch_batch_size=2, num_steps=2, denoise_knn=3)
    assert out.shape == (200, 3)
    assert torch.isfinite(out).all()
