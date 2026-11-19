#!/usr/bin/env python3
# Benchmark arms: checkpoint and architecture per model, and one prediction function
import copy
import glob

import numpy as np
import torch
from model import TimePFN, TimePFN_PPD_unified_fold, TimePFN_PPD_unified_joint

D_FLOOR = 1e-4  # = experiments/exp_unified_joint.D_FLOOR


MODELS = {
    'timepfn_original': dict(
        cls='TimePFN', ckpt='checkpoints/TimePFN/checkpoint.pth',
        patch_size=16, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        # point forecaster
        use_rope=False, conditions=['F'], probabilistic=False),

    # marginal-only ablations
    'Marginal_JTPFN_nll': dict(
        cls='TimePFN_PPD_unified_fold',
        ckpt='checkpoints/Marginal_JTPFN_nll/*/checkpoint_ppd_matern.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0),
    'Marginal_JTPFN_beta': dict(
        cls='TimePFN_PPD_unified_fold',
        ckpt='checkpoints/Marginal_JTPFN_beta/*/checkpoint_ppd_matern.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0),

    # lambda ablations (R=32); l0.5_r32 is the reported model
    'JointTimePFN_l0.0_r32': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l0.0_r32/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=32),
    'JointTimePFN_l0.25_r32': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l0.25_r32/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=32),
    'JointTimePFN_l0.5_r32': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l0.5_r32/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=32),
    'JointTimePFN_l0.75_r32': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l0.75_r32/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=32),
    'JointTimePFN_l1.0_r32': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l1.0_r32/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=32),

    #trained without RoPE
    'JointTimePFN_worope_l0.5_r32': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_worope_l0.5_r32/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=False, rope_base=10.0, joint_rank=32),

    # rank ablations (lambda=0.5)
    'JointTimePFN_l0.5_r4': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l0.5_r4/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=4),
    'JointTimePFN_l0.5_r8': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l0.5_r8/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=8),
    'JointTimePFN_l0.5_r16': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l0.5_r16/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=16),
    'JointTimePFN_l0.5_r64': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l0.5_r64/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=64),
    'JointTimePFN_l0.5_r128': dict(
        cls='TimePFN_PPD_unified_joint',
        ckpt='checkpoints/JointTimePFN_l0.5_r128/*/checkpoint_joint_nll.pth',
        patch_size=32, embed_dim=256, d_model=1024, d_ff=512, e_layers=8, n_heads=8,
        use_rope=True, rope_base=10.0, joint_rank=128),
}


def build(spec, base_args, device):
    # Build a model and load its checkpoint
    cfg = copy.deepcopy(base_args)
    for k in ('patch_size', 'embed_dim', 'd_model', 'd_ff', 'e_layers', 'n_heads', 'use_rope'):
        setattr(cfg, k, spec[k])
    cfg.joint_rank = spec.get('joint_rank', 32)
    cfg.rope_base = spec.get('rope_base', 10.0)
    paths = sorted(glob.glob(spec['ckpt']))

    if not paths:
        return None, None, f"checkpoint not found: {spec['ckpt']}"

    cls = {'TimePFN': TimePFN, 'TimePFN_PPD_unified_fold': TimePFN_PPD_unified_fold, 'TimePFN_PPD_unified_joint': TimePFN_PPD_unified_joint}[spec['cls']]

    net = cls.Model(cfg).to(device)
    missing, unexpected = net.load_state_dict(torch.load(paths[0], map_location='cpu'), strict=False)

    if missing:
        return None, None, (f"state_dict MISSING {len(missing)} keys "
                            f"(first: {missing[:3]}) -- architecture mismatch")
    if unexpected:
        print(f"[warn] {spec['cls']}: {len(unexpected)} unexpected key(s) in checkpoint, ignored: {unexpected[:4]}")
    net.eval()
    return net, sum(p.numel() for p in net.parameters()), None


def predict(net, spec, Wgrid, observed, marks, device, t_star, want_factor=False):
    """(mean [T,N], var [T,N] or None), plus factor [T,N,R] when want_factor."""
    T, N = Wgrid.shape
    if spec['cls'] == 'TimePFN':
        ctx = Wgrid[:t_star].copy()
        # dense context: forward-fill, channel mean before the first value
        if observed is not None:
            obs_c = observed[:t_star]
            if not obs_c.all():
                for c in range(N):
                    oc = obs_c[:, c]
                    if not oc.any():
                        continue
                    idx = np.where(oc, np.arange(t_star), 0)
                    np.maximum.accumulate(idx, out=idx)
                    filled = ctx[idx, c]
                    lead = np.arange(t_star) < np.argmax(oc)
                    filled[lead] = ctx[oc, c].mean()
                    ctx[:, c] = filled
        x = torch.tensor(ctx[None].astype('float32'), device=device)
        # four all-zero time-mark channels
        x_mark = torch.zeros(1, t_star, 4, device=device)
        with torch.no_grad():
            out = net(x, x_mark, None, None)[0].cpu().numpy()
        g = np.zeros((T, N), dtype='float32'); g[t_star:] = out
        return (g, None, None) if want_factor else (g, None)
    fm = torch.tensor((~observed)[None].astype('float32'), device=device)
    x = torch.tensor(Wgrid[None].astype('float32'), device=device)

    with torch.no_grad():
        out = net(x * (1 - fm), fm, marks)
    # joint checkpoint: Sigma_ii = max(d, D_FLOOR) + ||w_i||^2, as in training

    if len(out) == 3:
        mean, logvar, fac = out
        w = fac[0].cpu().numpy()
        d = np.maximum(np.exp(logvar[0].cpu().numpy().clip(-10, 10)), D_FLOOR)
        var = d + (w ** 2).sum(-1)
        m = mean[0].cpu().numpy()
        return (m, var, w) if want_factor else (m, var)

    # marginal head: its loss clamps logvar to [-10, 10] and applies no floor
    mean, logvar = out[0], out[1]
    m, v = mean[0].cpu().numpy(), np.exp(logvar[0].cpu().numpy().clip(-10, 10))
    return (m, v, None) if want_factor else (m, v)
