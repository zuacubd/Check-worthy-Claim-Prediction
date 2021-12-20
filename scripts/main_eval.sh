#!/bin/bash

#SBATCH --job-name=Checkthat_eval
#SBATCH --output=logs/checkthat_eval.log
#SBATCH --error=logs/checkthat_eval.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=8000M


#source scripts/aven.sh

data_dir="/projets/sig/mullah/nlp/checkthat18"
gold_data_dir="${data_dir}/data/raw/task1_gold"
output_dir="${data_dir}/output/test"

echo "launching ..."

python3 app/eval/scorer/task1.py --gold_file_path="${gold_data_dir}/English/task1-en-file1.txt,${gold_data_dir}/English/task1-en-file2.txt,${gold_data_dir}/English/task1-en-file3.txt,${gold_data_dir}/English/task1-en-file4.txt,${gold_data_dir}/English/task1-en-file5.txt,${gold_data_dir}/English/task1-en-file6.txt,${gold_data_dir}/English/task1-en-file7.txt" --pred_file_path="${output_dir}/NW/task1-en-file1_sgd_log_natural.txt,${output_dir}/NW/task1-en-file2_sgd_log_natural.txt,${output_dir}/NW/task1-en-file3_sgd_log_natural.txt,${output_dir}/NW/task1-en-file4_sgd_log_natural.txt,${output_dir}/NW/task1-en-file5_sgd_log_natural.txt,${output_dir}/NW/task1-en-file6_sgd_log_natural.txt,${output_dir}/NW/task1-en-file7_sgd_log_natural.txt"
python3 app/eval/scorer/task1.py --gold_file_path="${gold_data_dir}/English/task1-en-file1.txt,${gold_data_dir}/English/task1-en-file2.txt,${gold_data_dir}/English/task1-en-file3.txt,${gold_data_dir}/English/task1-en-file4.txt,${gold_data_dir}/English/task1-en-file5.txt,${gold_data_dir}/English/task1-en-file6.txt,${gold_data_dir}/English/task1-en-file7.txt" --pred_file_path="${output_dir}/NW/task1-en-file1_nn_lbfgs_natural.txt,${output_dir}/NW/task1-en-file2_nn_lbfgs_natural.txt,${output_dir}/NW/task1-en-file3_nn_lbfgs_natural.txt,${output_dir}/NW/task1-en-file4_nn_lbfgs_natural.txt,${output_dir}/NW/task1-en-file5_nn_lbfgs_natural.txt,${output_dir}/NW/task1-en-file6_nn_lbfgs_natural.txt,${output_dir}/NW/task1-en-file7_nn_lbfgs_natural.txt"
python3 app/eval/scorer/task1.py --gold_file_path="${gold_data_dir}/English/task1-en-file1.txt,${gold_data_dir}/English/task1-en-file2.txt,${gold_data_dir}/English/task1-en-file3.txt,${gold_data_dir}/English/task1-en-file4.txt,${gold_data_dir}/English/task1-en-file5.txt,${gold_data_dir}/English/task1-en-file6.txt,${gold_data_dir}/English/task1-en-file7.txt" --pred_file_path="${output_dir}/NW/task1-en-file1_svc_linear_natural.txt,${output_dir}/NW/task1-en-file2_svc_linear_natural.txt,${output_dir}/NW/task1-en-file3_svc_linear_natural.txt,${output_dir}/NW/task1-en-file4_svc_linear_natural.txt,${output_dir}/NW/task1-en-file5_svc_linear_natural.txt,${output_dir}/NW/task1-en-file6_svc_linear_natural.txt,${output_dir}/NW/task1-en-file7_svc_linear_natural.txt"
python3 app/eval/scorer/task1.py --gold_file_path="${gold_data_dir}/English/task1-en-file1.txt,${gold_data_dir}/English/task1-en-file2.txt,${gold_data_dir}/English/task1-en-file3.txt,${gold_data_dir}/English/task1-en-file4.txt,${gold_data_dir}/English/task1-en-file5.txt,${gold_data_dir}/English/task1-en-file6.txt,${gold_data_dir}/English/task1-en-file7.txt" --pred_file_path="${output_dir}/NW/task1-en-file1_svc_rbf_natural.txt,${output_dir}/NW/task1-en-file2_svc_rbf_natural.txt,${output_dir}/NW/task1-en-file3_svc_rbf_natural.txt,${output_dir}/NW/task1-en-file4_svc_rbf_natural.txt,${output_dir}/NW/task1-en-file5_svc_rbf_natural.txt,${output_dir}/NW/task1-en-file6_svc_rbf_natural.txt,${output_dir}/NW/task1-en-file7_svc_rbf_natural.txt"
python3 app/eval/scorer/task1.py --gold_file_path="${gold_data_dir}/English/task1-en-file1.txt,${gold_data_dir}/English/task1-en-file2.txt,${gold_data_dir}/English/task1-en-file3.txt,${gold_data_dir}/English/task1-en-file4.txt,${gold_data_dir}/English/task1-en-file5.txt,${gold_data_dir}/English/task1-en-file6.txt,${gold_data_dir}/English/task1-en-file7.txt" --pred_file_path="${output_dir}/NW/task1-en-file1_random_forest_natural.txt,${output_dir}/NW/task1-en-file2_random_forest_natural.txt,${output_dir}/NW/task1-en-file3_random_forest_natural.txt,${output_dir}/NW/task1-en-file4_random_forest_natural.txt,${output_dir}/NW/task1-en-file5_random_forest_natural.txt,${output_dir}/NW/task1-en-file6_random_forest_natural.txt,${output_dir}/NW/task1-en-file7_random_forest_natural.txt"
