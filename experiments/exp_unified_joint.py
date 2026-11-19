
import math
import os
import time
import warnings

import numpy as np
import torch
from torch.optim import lr_scheduler

from experiments.exp_unified_masked import (Exp_UnifiedMasked, batch_sample_mask, masked_mse)
from data_provider.mask_sampling import PATTERNS
from layers.joint_head import lrd_joint_nll, lrd_marginal_var
from utils.tools import adjust_learning_rate, fast_loader


# Pretraining of the joint model: joint NLL over the masked cells plus lambda * marginal NLL.

# Usage (scripts/pretrain.sh, ARM=joint):
#    --exp_name unified_joint --model TimePFN_PPD_unified_joint --joint_rank 32 --joint_lambda 0.5


warnings.filterwarnings('ignore')

LOGVAR_CLAMP = 10.0
D_FLOOR = 1e-4  # d_min


def gather_masked(mean, logvar, factor, target, mask, max_cells=0, rng=None):
    # Gather the masked cells per sample, padded to K_max (d=1, W=0, y=mu).

    # Returns y, mu, d [B, K_max], W [B, K_max, R], k_eff [B], valid [B, K_max] (bool).
    # max_cells > 0 keeps a random subset, which is the exact marginal of the joint.

    B, T, N = mean.shape
    R = factor.shape[-1]
    flat_m = mask.reshape(B, T * N)
    if max_cells:
        n_masked = flat_m.sum(1)  # [B]
        over = n_masked > max_cells
        if bool(over.any()):
            keep_p = torch.where(over, max_cells / n_masked.clamp(min=1.0),
                                 torch.ones_like(n_masked))  # [B]
            flat_m = flat_m * (torch.rand_like(flat_m) < keep_p.unsqueeze(1)).to(flat_m.dtype)

    k_eff = flat_m.sum(1)  # [B]
    K_max = int(k_eff.max().item())
    if K_max == 0:
        return None

    # masked slots (1.0) sort before observed (0.0); stable keeps a deterministic order
    order = torch.argsort(flat_m, dim=1, descending=True, stable=True)[:, :K_max]  # [B, K_max]
    valid = torch.gather(flat_m, 1, order) > 0  # [B, K_max]

    y = torch.gather(target.reshape(B, T * N), 1, order)
    mu = torch.gather(mean.reshape(B, T * N), 1, order)
    lv = torch.gather(logvar.reshape(B, T * N), 1, order).clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
    d = lv.exp().clamp(min=D_FLOOR)
    W = torch.gather(factor.reshape(B, T * N, R), 1, order.unsqueeze(-1).expand(-1, -1, R))

    # neutralise the padding
    y = torch.where(valid, y, mu)
    d = torch.where(valid, d, torch.ones_like(d))
    W = W * valid.unsqueeze(-1).to(W.dtype)
    return y, mu, d, W, k_eff, valid


def joint_nll_percell(mean, logvar, factor, target, mask, max_cells=0, rng=None, jitter=1e-6,
                      norm='sample'):
    # Joint NLL in nats per cell; norm='sample' (reported) averages per window.
    g = gather_masked(mean, logvar, factor, target, mask, max_cells, rng)
    if g is None:
        return mean.sum() * 0.0
    y, mu, d, W, k_eff, _ = g
    nll = lrd_joint_nll(y, mu, d, W, k_eff=k_eff, jitter=jitter)  # [B]
    if norm == 'sample':
        return (nll / k_eff.clamp(min=1.0)).mean()
    return nll.sum() / k_eff.sum().clamp(min=1.0)


def joint_composite_loss(mean, logvar, factor, target, mask, max_cells=0, rng=None, jitter=1e-6,
                         norm='sample'):
    # Returns (joint_nll, marginal_nll) in nats per cell, from one gathered subset.
    g = gather_masked(mean, logvar, factor, target, mask, max_cells, rng)
    if g is None:
        z = mean.sum() * 0.0
        return z, z
    y, mu, d, W, k_eff, valid = g
    jn = lrd_joint_nll(y, mu, d, W, k_eff=k_eff, jitter=jitter)  # [B]
    # Sigma_ii = d_i + ||w_i||^2
    var = (d + (W ** 2).sum(-1)).clamp(min=D_FLOOR)
    per = 0.5 * (torch.log(var) + (y - mu) ** 2 / var + math.log(2 * math.pi))
    # padding costs 0.5*log(2pi) per slot here, so mask it out
    mn = (per * valid.to(per.dtype)).sum(1)  # [B]
    if norm == 'sample':
        return (jn / k_eff.clamp(min=1.0)).mean(), (mn / k_eff.clamp(min=1.0)).mean()
    ke = k_eff.sum().clamp(min=1.0)
    return jn.sum() / ke, mn.sum() / ke


def implied_marginal_metrics(mean, logvar, factor, target, mask):

    # Marginal NLL and Cov95 from the implied Sigma_ii = d + ||w||^2

    lv = logvar.clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
    var = lrd_marginal_var(lv.exp().clamp(min=D_FLOOR), factor)
    per = 0.5 * (torch.log(var) + (target - mean) ** 2 / var + math.log(2 * math.pi))
    nll = (per * mask).sum() / mask.sum().clamp(min=1.0)

    std = var.sqrt()

    cov = (((target >= mean - 1.96 * std) & (target <= mean + 1.96 * std)).float()
           * mask).sum() / mask.sum().clamp(min=1.0)
    return nll.item(), cov.item()


class Exp_UnifiedJoint(Exp_UnifiedMasked):
    def __init__(self, args):
        super().__init__(args)
        self.max_cells = int(getattr(args, 'joint_max_cells', 0))

    def _select_optimizer(self):
        ps = [p for p in self.model.parameters() if p.requires_grad]
        if self.args.optimizer == 'adamw':
            return torch.optim.AdamW(ps, lr=self.args.learning_rate)
        return torch.optim.Adam(ps, lr=self.args.learning_rate)

    # validation
    def _vali_loop(self, vali_loader, forced_pattern):

        jn, mn, ms, cv, wn, cells = [], [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for input_vals, mask, full, full_mark in vali_loader:
                full = full.float().to(self.device)
                full_mark = self._maybe_marks(full_mark)
                B, T, N = full.shape
                new_mask = torch.from_numpy(
                    batch_sample_mask(B, T, N, self._eval_rng, forced_pattern)).to(self.device)
                mean, logvar, factor = self.model(full * (1.0 - new_mask), new_mask, full_mark)

                jn.append(joint_nll_percell(mean, logvar, factor, full, new_mask,
                                            self.max_cells, self._eval_rng,
                                            norm=getattr(self.args, 'loss_norm', 'sample')).item())

                m_nll, m_cov = implied_marginal_metrics(mean, logvar, factor, full, new_mask)

                mn.append(m_nll); cv.append(m_cov)
                ms.append(masked_mse(mean, full, new_mask).item())
                wn.append(((factor ** 2).sum(-1) * new_mask).sum().item()
                          / max(new_mask.sum().item(), 1.0))
                cells.append(new_mask.sum().item())


        self.model.train()
        c = np.array(cells); w = c / c.sum().clip(min=1)
        return (float(np.average(jn, weights=w)), float(np.average(ms, weights=w)),
                float(np.average(cv, weights=w)), float(np.average(mn, weights=w)),
                float(np.average(wn, weights=w)))

    def vali(self, vali_loader):
        results = {p: self._vali_loop(vali_loader, forced_pattern=p) for p in PATTERNS}
        results['mixed_overall'] = self._vali_loop(vali_loader, forced_pattern=None)
        return results

    def vali_mixed(self, vali_loader):
        return {'mixed_overall': self._vali_loop(vali_loader, forced_pattern=None)}

    def _print_vali(self, tag, results):
        print(f"[{tag}]")
        for pat, (jnll, mse, cov, mnll, w2) in results.items():
            print(f'  {pat:<20}  jointNLL {jnll:.4f}  MSE {mse:.4f}  Cov95 {cov:.4f}  margNLL {mnll:.4f}  |w|^2 {w2:.4f}')

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        best_ckpt = os.path.join(path, 'checkpoint_joint_nll.pth')  # selected by joint + lambda*marg NLL

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim, steps_per_epoch=train_steps, pct_start=self.args.pct_start,
            epochs=self.args.train_epochs, max_lr=self.args.learning_rate)

        val_every = int(getattr(self.args, 'val_every', 2500))
        lam = float(getattr(self.args, 'joint_lambda', 0.0))
        print(f'[joint] composite loss: joint_NLL + {lam:g} * marginal_NLL' + ('  (lam=0: pure joint, control arm)' if lam == 0 else ''))
        best_nll, global_step = float('inf'), 0
        time_now = time.time()
        train_rng = np.random.default_rng(0)

        for epoch in range(self.args.train_epochs):
            iter_count, train_loss = 0, []
            self.model.train()
            epoch_time = time.time()

            for i, (input_vals, mask, full, full_mark) in enumerate(train_loader):
                global_step += 1; iter_count += 1
                model_optim.zero_grad()
                input_vals = input_vals.float().to(self.device)
                mask = mask.float().to(self.device)
                full = full.float().to(self.device)
                full_mark = self._maybe_marks(full_mark)

                mean, logvar, factor = self.model(input_vals, mask, full_mark)
                j_nll, m_nll = joint_composite_loss(mean, logvar, factor, full, mask,
                                                    self.max_cells, train_rng,
                                                    norm=getattr(self.args, 'loss_norm', 'sample'))
                loss = j_nll + lam * m_nll
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    w2 = ((factor ** 2).sum(-1) * mask).sum().item() / max(mask.sum().item(), 1.0)
                    print(f'\titers: {i + 1}, epoch: {epoch + 1} | loss {loss.item():.4f} (jointNLL {j_nll.item():.4f} + {lam:g}*margNLL {m_nll.item():.4f}) | |w|^2 {w2:.4f} | speed {speed:.3f}s/it; left {left / 3600:.2f}h')
                    iter_count = 0; time_now = time.time()

                if (i + 1) % val_every == 0:
                    r = self.vali_mixed(vali_loader)
                    self._print_vali(f"step {global_step} VAL (mixed)", r)
                    # index 3 of _vali_loop is the marginal NLL
                    cur_nll = r['mixed_overall'][0] + lam * r['mixed_overall'][3]
                    if cur_nll < best_nll:
                        best_nll = cur_nll; torch.save(self.model.state_dict(), best_ckpt)
                        print(f"  improved (joint + lambda*marg NLL), saved {best_ckpt}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                model_optim.step()

                if self.args.lradj == 'synthetic':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
                elif (i + 1) % 10000 == 0:
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f'Epoch {epoch + 1} done in {time.time() - epoch_time:.1f}s | train loss {np.mean(train_loss):.4f}')
            r = self.vali(vali_loader)
            self._print_vali(f"epoch {epoch+1} VAL (per-pattern)", r)
            cur_nll = r['mixed_overall'][0] + lam * r['mixed_overall'][3]
            if cur_nll < best_nll:
                best_nll = cur_nll; torch.save(self.model.state_dict(), best_ckpt)
                print(f"  improved (joint + lambda*marg NLL), saved {best_ckpt}")

        if not os.path.exists(best_ckpt):
            print("Warning: no best checkpoint found, saving current model as fallback.")
            torch.save(self.model.state_dict(), best_ckpt)
        self.model.load_state_dict(torch.load(best_ckpt, map_location='cpu'))
        return self.model

    def test(self, setting, test=0):
        _, raw = self._get_data(flag='test')
        test_loader = fast_loader(raw, num_workers=self.args.num_workers)
        if test:
            ckpt = os.path.join('./checkpoints/' + setting, 'checkpoint_joint_nll.pth')
            self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        results = self.vali(test_loader)
        self._print_vali("TEST (per-pattern)", results)
        with open("result_unified_joint.txt", 'a') as f:
            f.write(setting + "\n")
            for pat, (jnll, mse, cov, mnll, w2) in results.items():
                f.write(f"  {pat:<20}  jointNLL {jnll:.4f}  MSE {mse:.4f}  Cov95 {cov:.4f}  "
                        f"margNLL {mnll:.4f}  |w|^2 {w2:.4f}\n")
            f.write("\n")
