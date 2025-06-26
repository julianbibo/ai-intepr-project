#!/bin/bash
#SBATCH --output=/home/scur1188/ai-intepr-project/logs/%x/%j.log
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

cd ~/ai-intepr-project
# module purge
# module load 2023
# module load Anaconda3/2023.07-2
# source activate myenv

############################################################

# python src/train_MLPs.py --data_dir "$1" --instruments "$2" --seeds "$3" --hidden_dim "$4" --single_layer "$5" --use_wandb 


sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 0
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 1
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 2
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 3
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 4
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 5
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 6
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 7
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 8
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 9
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 10
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 11
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 12
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 13
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 14
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 15
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 16
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 17
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 18
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 19
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 20
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 21
sbatch jobs/train.job "/home/scur1188/ai-intepr-project/data" "piano" "1,2,3" 2048 22


