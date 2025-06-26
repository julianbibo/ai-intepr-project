import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
from audiocraft.models import MusicGen
import argparse
import random
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def register_hooks(model: MusicGen, layers_to_capture: list[int], layer_outputs: dict) -> list:
    """
    Register forward hooks to capture outputs of specified layers in the model.
    """

    hooks = []

    def get_hook(layer_idx):
        layer_outputs[layer_idx] = []

        def hook(module, input, output):
            # print(f"Hook output shape: {output.shape}")

            half_batch_size = output.shape[0] // 2
            # select first half
            layer_outputs[layer_idx].append(output[:half_batch_size].detach().cpu())
            # om en om selectie, only select the even batch samples
            # layer_outputs[layer_idx].append(output[0::2].detach().cpu())  # train only on conditional representations
        return hook

    for idx, layer in enumerate(model.lm.transformer.layers):
        if idx in layers_to_capture:
            hooks.append(layer.register_forward_hook(get_hook(idx)))

    return hooks


def generate_audio_from_prompts(args, music_model: MusicGen, prompt_batches: list[list[str]], instrument, seed, split: str) -> None:
    """
    Generate audio from prompts and save the layer outputs for training.
    This function captures the outputs of the transformer layers during generation.
    """

    print(f"Generating audio for split: {split}")

    output_dir = Path(args.output_dir) / instrument / f"seed_{seed}" / split
    output_dir.mkdir(exist_ok=True, parents=True)

    # total amount of transformer layers in the model
    n_layers = len(music_model.lm.transformer.layers)
    # Exclude the last layer for training data
    n_train_layers = n_layers - 1

    X_list = {layer: [] for layer in range(n_train_layers)}
    Y_list = []

    with torch.no_grad():
        for idx, prompt_batch in enumerate(tqdm(prompt_batches, desc="  Capturing layer outputs", mininterval=5)):
            # layer_num -> generation step outputs
            layer_outputs: dict[int, list[torch.Tensor]] = {}
            hooks = register_hooks(music_model, range(n_layers), layer_outputs)

            # saves outputs in layer_outputs
            music_model.generate(prompt_batch, progress=False)

            for h in hooks:
                h.remove()

            # get the outputs of the first n-2 layers for our train data
            for layer in range(n_train_layers):
                x_outputs = layer_outputs[layer]  # list[Tensor(prompts, 1, hidden)]
                # concatenate on dim 1
                x_cat = torch.cat(x_outputs, dim=1)  # (prompts, n_steps, hidden)

                X_list[layer].append(x_cat[:, -1, :])  # last step output
                # X_all_list[layer].append(x)  # all steps output

            # get the output of the n-1th layer as our ground truth
            y_outputs = layer_outputs[n_layers - 1]  # list[Tensor(prompts, 1, hidden)]
            y_cat = torch.cat(y_outputs, dim=1)  # (prompts, n_steps, hidden)
            Y_list.append(y_cat[:, -1, :])  # (prompts, hidden)
            # Y_all_list.append(y)

            torch.cuda.empty_cache()
            del x_cat, y_cat, y_outputs, x_outputs, layer_outputs

    print("Saving data")
    Y = torch.cat(Y_list, dim=0)
    print(f"{Y.shape=}, {len(Y_list)=}, {len(X_list[0])=}")

    prompts = []
    for batch in prompt_batches:
        prompts.extend(batch)

    for layer in range(n_train_layers):
        X = torch.cat(X_list[layer], dim=0)
        # X_all = torch.cat(X_all_list[layer], dim=0)
        print(f"{layer=}, {X.shape=}, {Y.shape=}")

        torch.save({
            "X": X,
            "Y": Y,
            "prompts": prompts,
            "layer": layer,
            "instrument": instrument
        }, output_dir / f"layer_{layer:02d}.pt")


def run_prompt_batches(args, music_model: MusicGen, prompts_file: Path, seed) -> None:
    """
    Run the prompt batches for a specific instrument, generating audio and saving the layer outputs.
    This function handles the splitting of prompts into train, validation, and test sets.
    """

    instrument = prompts_file.stem.replace("prompts_", "")
    print(f"\nProcessing instrument: {instrument}")

    with open(prompts_file) as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    if args.max_prompts is not None:
        all_prompts = all_prompts[:args.max_prompts]
        print(f"Using only the first {args.max_prompts} prompts for {instrument}.")

    splits_str = args.splits.split('/')
    assert len(splits_str) == 3, "Splits must be in the format 'train/val/test', e.g., '80/10/10'."
    assert sum(float(s.strip()) for s in splits_str) == 100, "Splits must sum to 100."
    splits = [float(s) / 100 for s in splits_str]
    print(f"Creating splits: {splits}")

    n_prompts = len(all_prompts)
    prompt_indices = list(range(n_prompts))  # len = 1300
    random.shuffle(prompt_indices)
    n_val = int(n_prompts * splits[1])
    n_test = int(n_prompts * splits[2])
    n_train = n_prompts - n_val - n_test
    print(f"Total prompts: {n_prompts}, Train: {n_train}, Val: {n_val}, Test: {n_test}")

    train_indices = prompt_indices[:n_train]
    val_indices = prompt_indices[n_train:n_train + n_val]
    test_indices = prompt_indices[n_train + n_val:]

    train_prompts = [all_prompts[i] for i in train_indices]
    val_prompts = [all_prompts[i] for i in val_indices]
    test_prompts = [all_prompts[i] for i in test_indices]

    # # split {split_prompts into batches of args.prompt_bs
    train_batches = [train_prompts[i:i + args.prompt_bs]
                     for i in range(0, len(train_prompts), args.prompt_bs)]
    val_batches = [val_prompts[i:i + args.prompt_bs]
                   for i in range(0, len(val_prompts), args.prompt_bs)]
    test_batches = [test_prompts[i:i + args.prompt_bs]
                    for i in range(0, len(test_prompts), args.prompt_bs)]

    # generate train, val, test
    generate_audio_from_prompts(args, music_model, train_batches, instrument, seed, "train")
    generate_audio_from_prompts(args, music_model, val_batches, instrument, seed, "val")
    generate_audio_from_prompts(args, music_model, test_batches, instrument, seed, "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data for MusicGen residuals.")
    parser.add_argument("--duration", type=int, default=4,
                        help="Duration of generated audio in seconds.")
    parser.add_argument("--seeds", type=str, default="1,2,3",
                        help="Random seeds for reproducibility.")
    parser.add_argument("--prompts_dir", type=str, default="prompts",
                        help="Directory containing instrument prompts files.")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to save the generated training data.")
    parser.add_argument("--model_name", type=str, default="facebook/musicgen-small",
                        help="Name of the MusicGen model to use.")
    parser.add_argument("--prompt_bs", type=int, default=128,
                        help="Prompt batch size for audio generation.")
    parser.add_argument("--only_piano", action="store_true",
                        help="If set, only process prompts for the 'piano' instrument.")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Maximum number of prompts to process per instrument. If None, all prompts are used.")
    parser.add_argument("--splits", type=str, default="80/10/10",
                        help="Train/validation/test split ratios, e.g., '80/10/10'.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    music_model = MusicGen.get_pretrained(args.model_name, device=device)
    # music_model.set_generation_params(duration=args.duration, use_sampling=False)
    music_model.set_generation_params(duration=args.duration)
    music_model.compression_model.eval()
    music_model.lm.eval()
    print(f"Transformer layers: {len(music_model.lm.transformer.layers)}")

    prompts_dir = Path(args.prompts_dir)
    if args.only_piano:
        prompt_files = list(prompts_dir.glob("prompts_piano.txt"))
        print("Only processing 'piano' prompts")
    else:
        prompt_files = list(prompts_dir.glob("prompts_*.txt"))

    for prompts_file in prompt_files:
        for seed in args.seeds.split(','):
            seed = int(seed.strip())
            print(f"Running for seed {seed}")
            set_seed(seed)
            run_prompt_batches(args, music_model, prompts_file, seed)


if __name__ == "__main__":
    main()
