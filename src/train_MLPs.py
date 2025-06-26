import wandb
import torch
import torch.nn as nn
from pathlib import Path
from activations_dataset import ActivationsDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os

from mlp import MLP

from dotenv import load_dotenv

load_dotenv()
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
SLURM_JOB_ID = os.getenv('SLURM_JOB_ID')

wandb.login(key=WANDB_API_KEY)


def calculate_identity_baseline_loss(dataloader, device='cuda', criterion=None):
    """
    Calculate validation loss using identity operation (input = output) as baseline.

    Args:
        dataloader: DataLoader containing (input, target) pairs
        device: Device to run computation on
        criterion: Loss function to use (defaults to MSELoss)

    Returns:
        float: Average identity baseline loss
    """
    if criterion is None:
        criterion = nn.MSELoss()

    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_prompts in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Identity operation: use input as prediction
            pred_y = batch_x
            avg_batch_loss = criterion(pred_y, batch_y)

            total_loss += avg_batch_loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

    avg_identity_loss = total_loss / total_samples
    return avg_identity_loss


class MLPTrainer:
    def __init__(self, args, layer: int, device='cuda'):
        self.layer = layer
        self.instruments = args.instruments
        self.device = device
        self.use_wandb = args.use_wandb
        self.hidden_dim = args.hidden_dim

        self.model = MLP.from_args(args).to(device)

        print("Model:")
        print(self.model)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=args.lrs_factor, patience=args.lrs_patience)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_samples = 0

        for batch_x, batch_y, batch_prompts in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            pred_y = self.model(batch_x)
            loss = self.criterion(pred_y, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_prompts in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                pred_y = self.model(batch_x)
                loss = self.criterion(pred_y, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    def save_model(self, path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'instruments': self.instruments,
            'layer': self.layer,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


def train_layer_mlp(args, layer: int, device):
    data_dir = Path(args.data_dir)

    print(f"Training MLP for layer {layer} with instruments: {args.instruments}")

    # Initialize wandb for this layer if requested
    if args.use_wandb:
        instruments_str = "-".join(args.instruments)
        run_name = f"layer_{layer:02d}_{instruments_str}_{SLURM_JOB_ID}"

        # Initialize trainer
        trainer = MLPTrainer(
            args=args,
            layer=layer,
            device=device,
        )

        hidden_dim = trainer.hidden_dim

        wandb.init(
            project=args.wandb_project or "mlp-layer-training-2",
            group=args.wandb_group,
            name=run_name,
            config={
                "job_id": SLURM_JOB_ID,
                "layer": layer,
                "hidden_dim": hidden_dim,
                "device": device,
                "args": vars(args),
            },
            tags=[f"layer_{layer}", *args.instruments]
        )

    try:
        # Load datasets
        train_set = ActivationsDataset(
            data_dir=data_dir,
            instruments=args.instruments,
            seeds=args.seeds.split(","),
            split="train",
            layer=layer,
        )
        val_set = ActivationsDataset(
            data_dir=data_dir,
            instruments=args.instruments,
            seeds=args.seeds.split(","),
            split="val",
            layer=layer,
        )
        test_set = ActivationsDataset(
            data_dir=data_dir,
            instruments=args.instruments,
            seeds=args.seeds.split(","),
            split="test",
            layer=layer,
        )

        print(
            f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # Create dataloaders
        train_dataloader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True
        )

        val_dataloader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False
        )

        test_dataloader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(args.epochs):
            # Training
            train_loss = trainer.train_epoch(train_dataloader)

            # Validation
            val_loss = trainer.validate(val_dataloader)
            identity_loss = calculate_identity_baseline_loss(
                val_dataloader, device, trainer.criterion)
            improvement = identity_loss - val_loss
            improvement_pct = (improvement / identity_loss) * 100 if identity_loss > 0 else 0

            current_residual_weight = trainer.model.residual_weight.item()
            current_adapted_weight = trainer.model.adapted_weight.item()

            # Learning rate scheduling
            trainer.scheduler.step(val_loss)

            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "identity_baseline_loss": identity_loss,
                    "improvement_over_identity": improvement,
                    "improvement_percentage": improvement_pct,
                    "current_residual_weight": current_residual_weight,
                    "current_adapted_weight": current_adapted_weight,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "learning_rate": trainer.optimizer.param_groups[0]['lr'],
                })

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = trainer.model.state_dict().copy()
            else:
                patience_counter += 1

            # Print progress
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                # print(f"Layer {layer}, Epoch {epoch:3d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                print(f"Layer {layer}, Epoch {epoch:3d}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Identity Baseline: {identity_loss:.6f}")
                print(f"  Improvement: {improvement:.6f} ({improvement_pct:.1f}%)")

            # Early stopping
            if patience_counter >= args.stop_patience:
                print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
                if args.use_wandb:
                    wandb.log({
                        "early_stopped": True,
                        "early_stop_epoch": epoch,
                    })
                break

        # Load best model
        if best_model_state is not None:
            trainer.model.load_state_dict(best_model_state)

        # After training, also compare test set
        test_loss = trainer.validate(test_dataloader)
        test_identity_loss = calculate_identity_baseline_loss(
            test_dataloader, device, trainer.criterion)
        test_improvement = test_identity_loss - test_loss
        test_improvement_pct = (test_improvement / test_identity_loss) * \
            100 if test_identity_loss > 0 else 0

        print(f"Layer {layer} - Final Results:")
        print(f"  Test Loss: {test_loss:.6f}")
        print(f"  Test Identity Baseline: {test_identity_loss:.6f}")
        print(f"  Test Improvement: {test_improvement:.6f} ({test_improvement_pct:.1f}%)")

        if args.use_wandb:
            wandb.log({
                "final_test_loss": test_loss,
                "final_test_identity_loss": test_identity_loss,
                "final_test_improvement": test_improvement,
                "final_test_improvement_pct": test_improvement_pct,
                "total_epochs": epoch + 1
            })

        # Save trained model
        instruments_string = "-".join(args.instruments)
        save_dir_expanded = os.path.expanduser(args.save_dir)
        save_path = os.path.join(
            save_dir_expanded, f"layer_{layer:02d}_{instruments_string}_seeds-{args.seeds}_mlp.pt")

        trainer.save_model(save_path)
        print(f"Saved MLP for layer {layer} to {save_path}")

        # # Log model artifact to wandb
        # if use_wandb:
        #     artifact = wandb.Artifact(f"model_layer_{layer:02d}", type="model")
        #     artifact.add_file(save_path)
        #     wandb.log_artifact(artifact)

        return trainer.model

    finally:
        # Always finish the wandb run
        if args.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train MLP adapters for transformer layers")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the activation data")
    parser.add_argument("--instruments", nargs="+", required=True,
                        help="List of instruments to train on")
    parser.add_argument("--seeds", type=str, required=True,
                        help="String identifier for the random seeds used")

    # Model arguments
    parser.add_argument("--input_dim", type=int, default=1024,
                        help="Input dimension of the MLP")
    parser.add_argument("--hidden_dim", type=int, default=2048,
                        help="Hidden dimension of the MLP")
    parser.add_argument("--output_dim", type=int, default=1024,
                        help="Output dimension of the MLP")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--stop_patience", type=int, default=50,
                        help="Patience for early stopping")
    parser.add_argument("--lrs_patience", type=int, default=15,
                        help="Patience for learning rate scheduler")
    parser.add_argument("--lrs_factor", type=float, default=0.1,
                        help="Factor by which to reduce learning rate")

    # Layer arguments
    parser.add_argument("--start_layer", type=int, default=0,
                        help="Starting layer to train (inclusive)")
    parser.add_argument("--end_layer", type=int, default=22,
                        help="Ending layer to train (inclusive)")
    parser.add_argument("--single_layer", type=int, default=None,
                        help="Train only a single specific layer")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="List of specific layers to train (e.g., --layers 0 5 10 15)")

    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cpu/cuda)")
    parser.add_argument("--save_dir", type=str, default="~/ai-intepr-project/weights/",
                        help="Directory to save trained models")

    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="mlp-layer-training-2",
                        help="Wandb project name")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Wandb group name (auto-generated if not provided)")

    args = parser.parse_args()
    # print args
    print("ARGS:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("Using CPU")

    # Determine layers to train
    if args.layers is not None:
        layers_to_train = args.layers
    elif args.single_layer is not None:
        layers_to_train = [args.single_layer]
    else:
        layers_to_train = list(range(args.start_layer, args.end_layer + 1))

    print(f"Training MLPs for layers: {layers_to_train}")

    # Create wandb group name if not provided
    if args.use_wandb and args.wandb_group is None:
        instruments_str = "-".join(args.instruments)
        args.wandb_group = f"mlp_training_{instruments_str}_seeds-{args.seeds}"

    # Train MLPs for specified layers
    layer_mlps = {}
    for layer in layers_to_train:
        mlp = train_layer_mlp(
            args=args,
            layer=layer,
            device=device,
        )
        layer_mlps[layer] = mlp
        print(f"Successfully trained MLP for layer {layer}")

    print(f"Training completed. Successfully trained {len(layer_mlps)} MLPs.")


if __name__ == "__main__":
    main()


# Usage in your DecoderLens
# class DecoderLens:
#     def __init__(self, musicgen_model, layer_mlps_dict):
#         self.musicgen = musicgen_model
#         self.layer_mlps = layer_mlps_dict  # {layer_idx: trained_mlp}

#     def generate_from_layer(self, layer_idx, layer_repr, **generation_params):
#         """
#         Args:
#             layer_idx: Which layer to start from
#             layer_repr: [2, seq_len, 1024] - CFG representations from hooks
#         """
#         # Transform middle layer to final layer format
#         final_repr = self.layer_mlps[layer_idx](layer_repr)  # [2, seq_len, 1024]

#         # Feed into MusicGen's existing pipeline
#         normed = self.musicgen.lm.out_norm(final_repr)

#         # This is where MusicGen would continue with:
#         # 1. Linear heads to get logits
#         # 2. Sampling with CFG combination
#         # 3. Token decoding to audio

#         # For now, return the normalized representation
#         return normed

# # Use in DecoderLens
# decoder_lens = DecoderLens(musicgen_model, layer_mlps)
