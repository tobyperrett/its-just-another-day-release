#!/bin/bash

#SBATCH --job-name=e4d_encode
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=6-23:59:59
#SBATCH --mem=32G
#SBATCH --account=cosc028885
#SBATCH --array=0-50

source /user/home/tp8961/.bashrc
conda activate /user/work/tp8961/.conda/lavila

python /user/work/tp8961/LaViLa/scripts/convert_ego4d.py
