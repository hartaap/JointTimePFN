from torch.utils.data import DataLoader

from data_provider.data_loader import (Dataset_GaussianPCoregionalization,
                                       make_variable_channel_collate)


def data_provider(args, flag):
    # Dataset and loader of windows from the synthetic LMC arrow
    if flag == 'test':
        shuffle, drop_last, batch_size = False, True, 1
    elif flag == 'val':
        shuffle, drop_last = False, True
        batch_size = getattr(args, 'val_batch_size', args.batch_size)
    else:
        shuffle, drop_last, batch_size = not args.ordered_data, True, args.batch_size

    # 80/10/10 over series; val_series_cap trims only the in-loop validation
    total = args.total_synthetic_series
    train_end, val_end = int(total * 0.80), int(total * 0.90)
    if flag == 'train':
        series_range = (0, train_end)
    elif flag == 'val':
        cap = getattr(args, 'val_series_cap', 0)
        series_range = (train_end, val_end if cap <= 0 else min(val_end, train_end + cap))
    else:
        series_range = (val_end, total)

    data_set = Dataset_GaussianPCoregionalization(
        root_path=args.synthetic_root_path, data_path=args.synthetic_data_path,
        n_variables=args.enc_in, seq_len=args.seq_len, pred_len=args.pred_len,
        size=[args.seq_len, args.label_len, args.pred_len],
        synthetic_length=args.synthetic_length, stride=args.stride,
        series_range=series_range, return_mask=True,
        max_channel=args.max_channel, min_channel=getattr(args, 'min_channel', -1))
    print(f'{flag}: series {series_range[0]}-{series_range[1]}, {len(data_set)} windows')

    # min_channel > 0 draws the channel count per batch
    min_ch = getattr(args, 'min_channel', -1)
    collate = make_variable_channel_collate(min_ch, args.max_channel) if min_ch > 0 else None
    return data_set, DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                                num_workers=args.num_workers, drop_last=drop_last,
                                collate_fn=collate)
