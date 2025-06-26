#!/bin/bash
#SBATCH --output=/home/scur1188/ai-intepr-project/logs/%x/%j.log
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

cd ~/ai-intepr-project
module purge
module load 2023
module load Anaconda3/2023.07-2
source activate myenv

############################################################


python src/create_data_splits.py --data_dir "/home/scur1188/ai-intepr-project/data/" --output_dir "/home/scur1188/ai-intepr-project/combined_data_seeds-1-2-3/"

python src/create_data_splits.py --data_dir "/home/scur1188/ai-intepr-project/data/" --data_seed 1 --output_dir "/home/scur1188/ai-intepr-project/combined_data_seed-1/"