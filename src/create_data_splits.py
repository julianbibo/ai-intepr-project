import torch
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def combine_and_split_data(data_dir: Path, data_seed, output_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """
    Combine data from multiple seeds and split into train/val/test sets.
    
    Args:
        data_dir: Directory containing the original data structure
        data_seed: Specific seed to use (if None, uses all seeds)
        output_dir: Directory to save the combined and split data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation (test gets the remainder)
        seed: Random seed for reproducible splits
    """
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Find all instruments
    instruments = [d.name for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found instruments: {instruments}")
    
    for instrument in instruments:
        print(f"\nProcessing instrument: {instrument}")
        instrument_dir = data_dir / instrument
        
        if data_seed != None:
            # Use data from only the specified seed directory
            specific_seed_dir = instrument_dir / f"seed_{data_seed}"
            if specific_seed_dir.exists() and specific_seed_dir.is_dir():
                seed_dirs = [specific_seed_dir]
                print(f"Using specific seed: {specific_seed_dir.name}")
            else:
                print(f"Specified seed directory {specific_seed_dir} does not exist for {instrument}, skipping...")
                raise ValueError

        else:
            # Find all seed directories
            seed_dirs = [d for d in instrument_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
            print(f"Found seeds: {[d.name for d in seed_dirs]}")
        
        if not seed_dirs:
            print(f"No seed directories found for {instrument}, skipping...")
            raise ValueError
        
        # Find all layer files from the first seed to get layer numbers
        first_seed_dir = seed_dirs[0]
        layer_files = list(first_seed_dir.glob("layer_*_last.pt"))
        layer_numbers = sorted([int(f.stem.split('_')[1]) for f in layer_files])
        print(f"Found layers: {layer_numbers}")
        
        for layer_num in layer_numbers:
            print(f"  Processing layer {layer_num:02d}")
            
            # Collect data from all seeds for this instrument-layer combination
            all_X = []
            all_Y = []
            layer_info = None
            instrument_info = None
            
            for seed_dir in seed_dirs:
                layer_file = seed_dir / f"layer_{layer_num:02d}_last.pt"
                
                if not layer_file.exists():
                    print(f"    Warning: {layer_file} does not exist, skipping...")
                    raise ValueError
                
                # Load data
                data = torch.load(layer_file, map_location='cpu')
                all_X.append(data['X'])
                all_Y.append(data['Y'])
                
                # Store layer and instrument info (should be same across seeds)
                if layer_info is None:
                    layer_info = data['layer']
                    instrument_info = data['instrument']
            
            if not all_X:
                print(f"    No valid data found for layer {layer_num:02d}, skipping...")
                raise ValueError
            
            # Combine data from all seeds
            combined_X = torch.cat(all_X, dim=0)
            combined_Y = torch.cat(all_Y, dim=0)
            
            print(f"    Combined data shape: X={combined_X.shape}, Y={combined_Y.shape}")
            
            # Create random indices for splitting
            n_samples = combined_X.shape[0]
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            # Calculate split sizes
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            n_test = n_samples - n_train - n_val
            
            # Split indices
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
            
            print(f"    Split sizes: train={n_train}, val={n_val}, test={n_test}")
            
            # Create output directory
            output_instrument_dir = output_dir / instrument
            output_instrument_dir.mkdir(exist_ok=True, parents=True)
            
            # Save splits
            splits = {
                'train': train_indices,
                'val': val_indices,
                'test': test_indices
            }
            
            for split_name, split_indices in splits.items():
                if len(split_indices) == 0:
                    print(f"    Warning: {split_name} split is empty, skipping...")
                    raise ValueError
                
                split_X = combined_X[split_indices]
                split_Y = combined_Y[split_indices]
                
                save_data = {
                    'X': split_X,
                    'Y': split_Y,
                    'layer': layer_info,
                    'instrument': instrument_info,
                    'split': split_name,
                    'original_indices': split_indices
                }
                
                output_file = output_instrument_dir / f"layer_{layer_num:02d}_{split_name}.pt"
                torch.save(save_data, output_file)
                print(f"    Saved {split_name}: {output_file} (shape: {split_X.shape})")


def main():
    parser = argparse.ArgumentParser(description="Combine data from multiple seeds and split into train/val/test")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the original data structure")
    parser.add_argument("--data_seed", type=int, default=None,
                        help="Singular data generation seed to use for output split set")
    parser.add_argument("--output_dir", type=str, default="data_split",
                        help="Directory to save the combined and split data")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Proportion of data for training (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Proportion of data for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        raise ValueError("Train and validation ratios sum to more than 1.0")
    
    print(f"Split ratios: train={args.train_ratio:.1%}, val={args.val_ratio:.1%}, test={test_ratio:.1%}")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    combine_and_split_data(data_dir, args.data_seed, output_dir, args.train_ratio, args.val_ratio, args.seed)
    
    print(f"\nData combination and splitting complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()