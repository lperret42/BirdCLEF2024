import time
import os
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import timm
import onnxruntime
from scipy.special import expit

from src.utils import set_seed, print_duration
from src.dataset import BirdDataset, collate_fn
from src.wave2spec import MelSpectrogram
from src.dl_wrapper import DataLoaderWrapper


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)

    parser.add_argument("--train_path", type=str, help="just to retrieve ordered class names", required=True)

    parser.add_argument("--checkpoint", type=str, nargs='*', default=None)
    parser.add_argument("--checkpoints", type=str, default=None)

    parser.add_argument("--folds", type=str, nargs='*', default=None)

    parser.add_argument("--mode", type=str, default='inference')

    parser.add_argument("--batch_size", type=int, default=1, help="Taille du lot pour le traitement.")
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--inf_type", type=str, default=None)

    parser.add_argument("--out_path", type=str, default='submission.csv')

    parser.add_argument("--n_file", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


@torch.no_grad
def build_sub(df, args):
    original_col_order = list(df.columns)   # to keep the original order of columns

    if args.checkpoint is None and args.checkpoints is None:
        raise ValueError('Exactly one of checkpoint and checkpoints args must be set')
    elif args.checkpoint is not None and args.checkpoints is not None:
        raise ValueError('Exactly one of checkpoint and checkpoints args must be set, not both')
    elif args.checkpoint is not None:
        checkpoints = args.checkpoint
        checkpoint = checkpoints[0]  # not support multiple checkpoint at now
    else:  # args.checkpoints is not None
        checkpoints = [os.path.join(args.checkpoints, d) for d in os.listdir(args.checkpoints) if os.path.isdir(os.path.join(args.checkpoints, d))]
        checkpoint = checkpoints[0]  # not support multiple checkpoint at now

    print(f'{checkpoint=}')

    config_path = os.path.join(checkpoint, "config.json")
    with open(config_path, 'r') as file:
        CFG = json.load(file)

    # just to retrieve label names
    if 'pretraining' in CFG['trainpath']:  # for local debug for model only pretrained
        train_path = CFG['trainpath']
    else:
        train_path = args.train_path

    train_df = pd.read_csv(train_path)
    labels = list(sorted(train_df['primary_label'].unique()))
    label_to_num = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)
    print(f'{num_classes=}')

    if args.mode == 'inference':
        df['filename'] = df['row_id'].apply(lambda x: '_'.join(x.split('_')[:-1]) + '.ogg')
        df['end'] = df['row_id'].apply(lambda x: int(x.split('_')[-1]))
        df['start'] = df['end'] - 5
    else:
        df['row_id'] = df['filename']
        df['target'] = df['primary_label'].map(label_to_num)

    if args.n_file is not None:
        unique_filenames = df['filename'].unique()
        if len(unique_filenames) >= args.n_file:
            selected_filenames = pd.Series(unique_filenames).sample(n=args.n_file)
            df = df[df['filename'].isin(selected_filenames)].reset_index(drop=True)
            print(f'{df.shape=} after keep only {args.n_file=}')
        else:
            print(f"Warning: Insufficient unique filenames. Keeping all {len(unique_filenames)} unique filenames.")

    if args.mode == 'inference':
        nb_filenames = len(sorted(set(df['filename'].values)))
        print(f'{nb_filenames=}')

    device = torch.device("cpu") if args.device == 'cpu' else torch.device("cuda:0")

    weights_filenames = [file for file in os.listdir(checkpoint) if file.endswith('.ckpt')]
    print(f'{len(weights_filenames)=}')
    if args.folds is not None:  # keep only asked folds
        print(f'use only models of {args.folds=}')
        weights_filenames = [file for file in weights_filenames if file in [f'model_{fold}.ckpt' for fold in args.folds]]
        print(f'{len(weights_filenames)=}')
    weights_path = [os.path.join(checkpoint, file) for file in weights_filenames]
    print(f'{weights_path=}')
    print(f'{len(weights_path)=}')
    if len(weights_path) != CFG['nfolds']:
        print(f'WARNING: nb_models = {len(weights_path)} != {CFG["nfolds"]=}')

    wave2spec_obj = MelSpectrogram(
        device=device,
        **CFG['spec_args'],
    )

    df.sort_values('row_id', inplace=True)

    ds = BirdDataset(
        df,
        args.audio_dir,
        num_classes=num_classes,
        sr=CFG['sr'],
        duration=5,
        split=None if args.mode == 'inference' else 'first',
        cache_audio_dir=None,
        mode=args.mode,
    )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    dl = DataLoaderWrapper(dl, wave2spec_obj)

    models = []
    for weight_path in weights_path:
        model = timm.create_model(CFG['model_name'], in_chans=1, pretrained=False, num_classes=num_classes)
        ckpt = torch.load(weight_path, map_location=device)
        state_dict = ckpt['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        models.append(model)

    if args.inf_type == 'onnx':
        ort_sessions = []
        for model in models:
            dummy_input = torch.randn(args.batch_size, 1, 256, 256, device=device)
            onnx_path = f"model_{len(ort_sessions)}.onnx"
            torch.onnx.export(model, dummy_input, onnx_path, opset_version=14)
            ort_session = onnxruntime.InferenceSession(onnx_path)
            ort_sessions.append(ort_session)

            print('start dummy call')
            fake_input_onnx = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_session.run(None, fake_input_onnx)

    elif args.inf_type is not None and 'openvino' in args.inf_type:
        import openvino.runtime as ov

        ov_sessions = []
        for model in models:
            dummy_input = torch.randn(args.batch_size, 1, 256, 256, device=device)
            onnx_path = f"model_{len(ov_sessions)}.onnx"
            torch.onnx.export(model, dummy_input, onnx_path, opset_version=14)

            core = ov.Core()
            if args.inf_type == 'openvino':
                compiled_model = core.compile_model(onnx_path, device_name="CPU")
            else:
                compiled_model = core.compile_model(onnx_path, device_name="CPU", config={"ENFORCE_FP16": "YES"})
            ov_sessions.append(compiled_model)

            print('start dummy call')
            fake_input_ov = {compiled_model.input(0).any_name: dummy_input.cpu().numpy()}
            if args.inf_type == 'openvino-16':
                fake_input_ov = fake_input_ov.astype(np.float16)
            compiled_model(fake_input_ov)

    elif args.inf_type == 'compile':
        compiled_models = []
        for model in models:
            compiled_model = torch.compile(model)
            compiled_models.append(compiled_model)

            print('start dummy call')
            dummy_input = torch.randn(args.batch_size, 1, 256, 256, device=device)
            with torch.no_grad():
                logits = compiled_model(dummy_input)
                probs = torch.nn.functional.sigmoid(logits).cpu().detach()

    original_row_ids = df['row_id'].values  # to keep the original order of row_id

    print(df)
    print(f'{df.shape=}')
    print(f'{len(dl)=}')

    row_ids = []
    probs_list = []

    start = time.time()
    for row_id, x, y, w in tqdm(dl, total=len(dl)):
        if args.inf_type == 'onnx':
            probs_sum = np.zeros((len(x), num_classes))
            for ort_session in ort_sessions:
                input_name = ort_session.get_inputs()[0].name
                logits = ort_session.run(None, {input_name: x.numpy()})[0]
                probs = expit(logits)
                probs_sum += probs
            probs_avg = probs_sum / len(ort_sessions)

        elif args.inf_type == 'openvino':
            probs_sum = np.zeros((len(x), num_classes))
            for compiled_model in ov_sessions:
                input_name = compiled_model.input(0).any_name
                if args.inf_type == 'openvino':
                    logits = compiled_model({input_name: x.numpy()})[compiled_model.output(0)]
                else:
                    logits = compiled_model({input_name: x.numpy().astype(np.float16)})[compiled_model.output(0)]
                probs = expit(logits)
                probs_sum += probs
            probs_avg = probs_sum / len(ov_sessions)

        elif args.inf_type == 'compile':
            probs_sum = torch.zeros(len(x), num_classes)
            for compiled_model in compiled_models:
                logits = compiled_model(x)
                probs = torch.nn.functional.sigmoid(logits).cpu().detach()
                probs_sum += probs_avg
            probs_avg = probs_sum / len(compiled_models)
        else:
            probs_sum = torch.zeros(len(x), num_classes)
            for model in models:
                logits = model(x)
                probs = torch.nn.functional.sigmoid(logits).cpu().detach()
                probs_sum += probs
            probs_avg = probs_sum / len(models)

        row_ids.extend(row_id)
        probs_list.append(probs_avg)

    if args.inf_type in ['onnx', 'openvino', 'openvino-16']:
        probs_concat = np.concatenate(probs_list, axis=0)
    else:
        probs_concat = torch.cat(probs_list, dim=0).numpy()

    df_sub = pd.DataFrame(probs_concat, columns=labels)
    df_sub.insert(0, 'row_id', row_ids)

    df_sub = df_sub.set_index('row_id')
    df_sub = df_sub.loc[original_row_ids]
    df_sub.reset_index(inplace=True)

    df_sub = df_sub[original_col_order]

    end = time.time()
    duration = end - start
    print_duration(duration)

    if args.mode == 'inference':
        LB_duration_estimation = duration * (1100 / nb_filenames)
        print('LB duration estimation: ', end='')
        print_duration(LB_duration_estimation)

    return df_sub


def main():
    args = parse_arguments()

    set_seed(args.seed)

    df = pd.read_parquet(args.dataset_path.replace('.csv', '.parquet')) if os.path.exists(args.dataset_path.replace('.csv', '.parquet')) else pd.read_csv(args.dataset_path)

    print(f'{df.shape=}')

    if len(df) == 3:
        df_sub = df
    else:
        df_sub = build_sub(df, args)

    df_sub.to_csv(args.out_path, index=False)
    print(f'sub saved in {args.out_path}')
    print(df_sub.head())


if __name__ == "__main__":
    main()
