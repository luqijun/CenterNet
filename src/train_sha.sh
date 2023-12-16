python main.py ctcrowd --dataset sha --exp_id SHHA --batch_size 16 --master_batch 15 --lr 1.25e-4 --gpus 0 --num_epochs 500 --val_start 200 --val_intervals 5 --print_iter 30 --metric mae
# --resume --load_model ../exp/ctcrowd/SHHA/saved_logs_2023-12-11-18-23_500ep/model_last.pth
python main.py ctcrowd --dataset sha --exp_id SHHA --batch_size 16 --master_batch 15 --lr 1.25e-4 --gpus 0 --num_epochs 600 --val_start 400 --val_intervals 2 --print_iter 30 --metric mae
