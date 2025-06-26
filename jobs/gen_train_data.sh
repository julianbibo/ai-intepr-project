#!/bin/bash
#SBATCH --output=/home/scur1188/ai-intepr-project/logs/%x/%j.log
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00

cd ~/ai-intepr-project
module purge
module load 2023
module load Anaconda3/2023.07-2
source activate myenv

############################################################

# parser.add_argument("--duration", type=int, default=4,
#                         help="Duration of generated audio in seconds.")
#     parser.add_argument("--seeds", type=str, default="1,2,3",
#                         help="Random seeds for reproducibility.")
#     parser.add_argument("--prompts_dir", type=str, default="prompts",
#                         help="Directory containing instrument prompts files.")
#     parser.add_argument("--output_dir", type=str, default="data",
#                         help="Directory to save the generated training data.")
#     parser.add_argument("--model_name", type=str, default="facebook/musicgen-small",
#                         help="Name of the MusicGen model to use.")
#     parser.add_argument("--prompt_bs", type=int, default=128,
#                         help="Prompt batch size for audio generation.")
#     parser.add_argument("--only_piano", action="store_true",
#     parser.add_argument("--max_prompts", type=int, default=None,
#                         help="Maximum number of prompts to process per instrument. If None, all prompts are used.")
#     parser.add_argument("--splits", type=str, default="80/10/10",
#                         help="Train/validation/test split ratios, e.g., '80/10/10'.")

python src/gen_train_data.py \
    --output_dir "/home/scur1188/ai-intepr-project/data" \
    --seeds 1,2,3 \
    --only_piano
