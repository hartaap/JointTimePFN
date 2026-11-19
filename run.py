import argparse
import os
import torch
import random
import numpy as np
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time
from experiments.exp_unified_masked import Exp_UnifiedMasked
from experiments.exp_unified_joint import Exp_UnifiedJoint


def clear_gpu_memory():
    torch.cuda.empty_cache()
    variables = gc.collect()
    del variables

def wait_until_enough_gpu_memory(min_memory_available):
    max_retries, sleep_time = 10, 5
    if not torch.cuda.is_available(): return  # no GPU: nothing to wait for
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

if __name__ == '__main__':
    # seed from PFN_SEED (per Slurm task), else 2021
    fix_seed = int(os.environ.get('PFN_SEED', '2021'))
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    np.random.seed(fix_seed)
    print(f"[run.py] PFN_SEED={fix_seed}", flush=True)

    parser = argparse.ArgumentParser(description='JointTimePFN')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='TimePFN_PPD_unified_joint or TimePFN_PPD_unified_fold')

    # data loader
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='no effect: only the last pred_len steps of seq_y are used')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

    # optimization
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    # model and experiment
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='unified_joint or unified_masked')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--patch_size', type=int, default=16, help='patch_size')
    parser.add_argument('--embed_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--use_rope', action='store_true', help='RoPE on the patch axis')
    parser.add_argument('--loss_norm', type=str, default='cell', choices=['cell', 'sample'], help='per-window mean (sample) or mean over all queried cells of the batch (cell)')
    parser.add_argument('--beta_nll', type=float, default=0.5, help='beta-NLL exponent; 0.0 = log score')
    
    # LRD joint head (exp_name unified_joint / model TimePFN_PPD_unified_joint)
    parser.add_argument('--joint_rank', type=int, default=32, help='rank R of Sigma = D + W W^T')
    parser.add_argument('--val_every', type=int, default=2500,
                        help='validation interval in steps (joint model)')
    parser.add_argument('--joint_max_cells', type=int, default=0, help='cap on masked cells per sample in the joint loss (0 = none)')
    parser.add_argument('--joint_lambda', type=float, default=0.0,
                        help='marginal NLL weight lambda')
    parser.add_argument('--rope_base', type=float, default=10.0, help='RoPE base (not stored in the checkpoint)')
    parser.add_argument('--synth_split', type=str, default='801010', choices=['801010'],
                        help='80/10/10 split over series')
    parser.add_argument('--val_series_cap', type=int, default=0,
                        help='validation series cap (0 = all)')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--synthetic_data_path', type=str, default='kernelsynth-coregionalization.arrow', help='path to synthetic data')
    parser.add_argument('--synthetic_root_path', type=str, default='./dataset/synthetics/', help='root path of the synthetic data file')
    parser.add_argument('--synthetic_length', type=int, default=1024, help='synthetic sequence length')
    parser.add_argument('--stride', type=int, default=-1, help='window stride over the synthetic series')
    parser.add_argument('--load', type=bool, default=False, help='load model')
    parser.add_argument('--data_amount', type=int, default=-1, help='amount of data to use, -1 to use all')
    parser.add_argument('--max_channel', type=int, default=-1, help='max channel to use, -1 to use all')
    parser.add_argument('--min_channel', type=int, default=-1, help='if >0, variable-N training from min_channel channels')
    parser.add_argument('--combine_Gaussian_datasets', type=bool, default=False, help='make true if you would like to combine multiple Gaussian datasets' )
    parser.add_argument('--ordered_data', type=bool, default=False, help='make true if you would like to order the data in time horizon' )
    parser.add_argument('--total_synthetic_series', type=int, default=15000, help='total number of series in synthetic dataset for train/val/test split')


    args = parser.parse_args()
    args.use_gpu = torch.cuda.is_available()

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'unified_masked':
        Exp = Exp_UnifiedMasked
    elif args.exp_name == 'unified_joint':
        Exp = Exp_UnifiedJoint
    else:
        raise ValueError(
            f"--exp_name {args.exp_name!r} is not shipped with this bundle; "
            "use 'unified_joint' (joint model) or 'unified_masked' (marginal model)."
        )

    if torch.cuda.is_available():
        clear_gpu_memory()
        wait_until_enough_gpu_memory(2 * 1024 ** 3)
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_embDim{}_dataAmount{}_maxChannel{}_subset{}'.format(
                args.model_id,
                args.model,
                'ETTh1',
                'M',
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                1,
                args.d_ff,
                args.factor,
                'timeF',
                True,
                args.des,
                args.class_strategy,
                ii,
                args.embed_dim,
                args.data_amount,
                args.max_channel,
                -1)

            exp = Exp(args)  # set experiments
            print(f'train: {setting}')
            exp.train(setting)

            print(f'test: {setting}')
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_embDim{}_dataAmount{}_maxChannel{}_subset{}'.format(
            args.model_id,
            args.model,
            'ETTh1',
            'M',
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            1,
            args.d_ff,
            args.factor,
            'timeF',
            True,
            args.des,
            args.class_strategy,
            ii,
            args.embed_dim,
            args.data_amount,
            args.max_channel,
            -1)

        exp = Exp(args)  # set experiments
        print(f'test: {setting}')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
