# Masked pretraining of the marginal model (TimePFN_PPD_unified_fold).

# Usage (scripts/pretrain.sh, ARM=nll or ARM=beta):
#    --exp_name unified_masked --model TimePFN_PPD_unified_fold --beta_nll {0.0, 0.5}


from data_provider.data_factory import data_provider
from data_provider.mask_sampling import sample_mask, PATTERNS
from experiments.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, fast_loader
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')


def masked_beta_nll(mean, logvar, target, mask, beta=0.5, norm='cell'):
    # beta-NLL over masked cells (beta = 0 is the log score). We report norm='sample'.
    logvar = logvar.clamp(-10, 10)
    per_cell = 0.5 * (logvar + (target - mean) ** 2 / logvar.exp())
    weight = logvar.exp().detach() ** beta
    weighted = per_cell * weight * mask
    if norm == 'sample':
        per_sample = weighted.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)
        return per_sample.mean()
    return weighted.sum() / mask.sum().clamp(min=1.0)


def masked_mse(mean, target, mask):
    err = ((mean - target) ** 2) * mask
    return err.sum() / mask.sum().clamp(min=1.0)


def masked_cov95(mean, logvar, target, mask):
    # 95% interval coverage over the masked cells
    std = torch.exp(0.5 * logvar.clamp(-10, 10))
    lo, hi = mean - 1.96 * std, mean + 1.96 * std
    covered = ((target >= lo) & (target <= hi)).float() * mask
    return covered.sum() / mask.sum().clamp(min=1.0)


def batch_sample_mask(B, T, N, rng, forced_pattern=None):
    M = np.zeros((B, T, N), dtype=np.float32)
    for b in range(B):
        M[b], _ = sample_mask(T, N, rng, forced_pattern=forced_pattern)
    return M



class Exp_UnifiedMasked(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        # eval RNG, separate from the dataset RNGs
        self._eval_rng = np.random.default_rng(20260511)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        if self.args.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        else:
            raise ValueError('Invalid optimizer')

    def _maybe_marks(self, full_mark):
        return full_mark.float().to(self.device)

    def _vali_loop(self, vali_loader, forced_pattern):
        # one pass through vali_loader with the mask overridden to a fixed pattern
        total_nll, total_mse, total_cov, total_cells = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for input_vals, mask, full, full_mark in vali_loader:
                full = full.float().to(self.device)
                full_mark = self._maybe_marks(full_mark)
                B, T, N = full.shape

                new_mask = torch.from_numpy(
                    batch_sample_mask(B, T, N, self._eval_rng, forced_pattern)
                ).to(self.device)
                new_input = full * (1.0 - new_mask)

                mean, logvar = self.model(new_input, new_mask, full_mark)

                total_nll.append(self._nll(mean, logvar, full, new_mask).item())
                total_mse.append(masked_mse(mean, full, new_mask).item())
                total_cov.append(masked_cov95(mean, logvar, full, new_mask).item())
                total_cells.append(new_mask.sum().item())

        self.model.train()
        cells = np.array(total_cells)
        w = cells / cells.sum().clip(min=1)
        return (
            float(np.average(total_nll, weights=w)),
            float(np.average(total_mse, weights=w)),
            float(np.average(total_cov, weights=w)),
        )

    def vali(self, vali_loader):
        # each pattern of mask_sampling.py, plus one mixture-sampled pass
        results = {}
        for pat in PATTERNS:
            nll, mse, cov = self._vali_loop(vali_loader, forced_pattern=pat)
            results[pat] = (nll, mse, cov)
        nll, mse, cov = self._vali_loop(vali_loader, forced_pattern=None)
        results['mixed_overall'] = (nll, mse, cov)
        return results

    def vali_mixed(self, vali_loader):
        nll, mse, cov = self._vali_loop(vali_loader, forced_pattern=None)
        return {'mixed_overall': (nll, mse, cov)}

    def _print_vali(self, tag, results):
        print(f"[{tag}]")
        for pat, (nll, mse, cov) in results.items():
            print(f"  {pat:<20}  NLL {nll:.4f}  MSE {mse:.4f}  Cov95 {cov:.4f}")

    def _nll(self, mean, logvar, target, mask):
        norm = getattr(self.args, 'loss_norm', 'cell')
        beta = float(getattr(self.args, 'beta_nll', 0.5))
        return masked_beta_nll(mean, logvar, target, mask, beta=beta, norm=norm)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        best_ckpt = os.path.join(path, 'checkpoint_ppd_matern.pth')

        train_steps = len(train_loader)
        # no early stopping; the best validation checkpoint is kept
        model_optim = self._select_optimizer()
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        best_val_nll = float('inf')
        global_step = 0
        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (input_vals, mask, full, full_mark) in enumerate(train_loader):
                global_step += 1
                iter_count += 1
                model_optim.zero_grad()

                input_vals = input_vals.float().to(self.device)
                mask = mask.float().to(self.device)
                full = full.float().to(self.device)
                full_mark = self._maybe_marks(full_mark)

                mean, logvar = self.model(input_vals, mask, full_mark)
                loss = self._nll(mean, logvar, full, mask)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.4f} | speed {speed:.3f}s/it; left {left / 3600:.2f}h')
                    iter_count = 0
                    time_now = time.time()

                if (i + 1) % 2500 == 0:
                    vali_results = self.vali_mixed(vali_loader)
                    self._print_vali(f"step {global_step} VAL (mixed)", vali_results)
                    cur_nll = vali_results['mixed_overall'][0]
                    if cur_nll < best_val_nll:
                        best_val_nll = cur_nll
                        torch.save(self.model.state_dict(), best_ckpt)
                        print(f"  improved, saved {best_ckpt}")

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
            # end of epoch: every pattern plus the mixture
            vali_results = self.vali(vali_loader)
            self._print_vali(f"epoch {epoch+1} VAL (per-pattern)", vali_results)
            cur_nll = vali_results['mixed_overall'][0]
            if cur_nll < best_val_nll:
                best_val_nll = cur_nll
                torch.save(self.model.state_dict(), best_ckpt)
                print(f"  improved, saved {best_ckpt}")

        if not os.path.exists(best_ckpt):
            print("Warning: no best checkpoint found, saving current model as fallback.")
            torch.save(self.model.state_dict(), best_ckpt)
        self.model.load_state_dict(torch.load(best_ckpt, map_location='cpu'))
        return self.model

    def test(self, setting, test=0):
        _, test_loader_raw = self._get_data(flag='test')
        test_loader = fast_loader(test_loader_raw, num_workers=self.args.num_workers)
        if test:
            ckpt = os.path.join('./checkpoints/' + setting, 'checkpoint_ppd_matern.pth')
            self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))

        results = self.vali(test_loader)
        self._print_vali("TEST (per-pattern)", results)

        with open("result_unified_masked.txt", 'a') as f:
            f.write(setting + "\n")
            for pat, (nll, mse, cov) in results.items():
                f.write(f"  {pat:<20}  NLL {nll:.4f}  MSE {mse:.4f}  Cov95 {cov:.4f}\n")
            f.write("\n")
