export PYTHONPATH=/root/workspace/Long-Seq-Model
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# >logs/electricity.log

gpu=0
#Electricity
for lr in 4e-4 4e-3
do
  for pred_len in 96 192 336 720
  do
    model=Mamba2_IN
    label_len=192
    seq_len=$label_len
    python -u run_exp.py \
      --is_training 1 \
      --root_path ./dataset/electricity \
      --data_path electricity.csv \
      --model_id electricity'_'Mamba2_IN'_'$label_len'_'$pred_len'_'$lr \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --embed_type 2 \
      --d_state 32 \
      --d_conv 2 \
      --d_layers 3 \
      --dec_in 321 \
      --c_out 321 \
      --batch_size 32 \
      --train_epochs 60 \
      --learning_rate $lr \
      --d_model 32 \
      --lradj type1 \
      --des 'Exp' \
      --gpu $gpu \
      --itr 1 >>logs/electricity'_'IN.log
  done
done

#ETTh1
for lr in 4e-4 4e-3
do
  for pred_len in 96 192 336 720
  do
    model=Mamba2_IN
    label_len=192
    seq_len=$label_len
    python -u run_exp.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small \
      --data_path ETTh1.csv \
      --model_id ETTh1'_'Mamba2'_'IN''_''$label_len'_'$pred_len \
      --model $model \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --embed_type 2 \
      --d_state 16 \
      --d_conv 2 \
      --d_layers 3 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 32 \
      --train_epochs 60 \
      --learning_rate $lr \
      --lradj type1 \
      --des 'Exp' \
      --gpu $gpu \
      --itr 1 >>logs/ETTh1'_'Mamba2'_'IN.log
  done
done

#ETTh2
for lr in 4e-4 4e-3
do
  for pred_len in 96 192 336 720
  do
    model=Mamba2_IN
    label_len=192
    seq_len=$label_len
    python -u run_exp.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small \
      --data_path ETTh2.csv \
      --model_id ETTh2'_'Mamba2'_'IN'_'$label_len'_'$pred_len \
      --model $model \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --embed_type 2 \
      --d_state 16 \
      --d_conv 2 \
      --d_layers 3 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 32 \
      --train_epochs 60 \
      --learning_rate $lr \
      --lradj type1 \
      --des 'Exp' \
      --gpu $gpu \
      --itr 1 >>logs/ETTh2'_'Mamba2'_'IN.log
  done
done

#ETTm1
for lr in 4e-4 4e-3
do
  for pred_len in 96 192 336 720
  do
    model=Mamba2_IN
    label_len=192
    seq_len=$label_len
    python -u run_exp.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small \
      --data_path ETTm1.csv \
      --model_id ETTm1'_'Mamba2'_'IN'_'$label_len'_'$pred_len \
      --model $model \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --embed_type 2 \
      --d_state 16 \
      --d_conv 2 \
      --d_layers 3 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 32 \
      --train_epochs 60 \
      --learning_rate $lr \
      --lradj type1 \
      --des 'Exp' \
      --gpu $gpu \
      --itr 1 >>logs/ETTm1'_'Mamba2'_'IN.log
  done
done

#ETTm2
for lr in 4e-4 4e-3
do
  for pred_len in 96 192 336 720
  do
    model=Mamba2_IN
    label_len=192
    seq_len=$label_len
    python -u run_exp.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'Mamba2'_'IN'_'$label_len'_'$pred_len \
      --model $model \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --embed_type 2 \
      --d_state 16 \
      --d_conv 2 \
      --d_layers 3 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 32 \
      --train_epochs 60 \
      --learning_rate $lr \
      --lradj type1 \
      --des 'Exp' \
      --gpu $gpu \
      --itr 1 >>logs/ETTm2'_'Mamba2'_'IN.log
  done
done

#exchange_rate
for lr in 4e-4 4e-3
do
  for pred_len in 96 192 336 720
  do
    model=Mamba2_IN
    label_len=192
    seq_len=$label_len
    python -u run_exp.py \
      --is_training 1 \
      --root_path ./dataset/exchange'_'rate \
      --data_path exchange_rate.csv \
      --model_id exchange'_'rate'_'Mamba2'_'IN'_'$label_len'_'$pred_len \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --embed_type 2 \
      --d_state 16 \
      --d_conv 4 \
      --d_layers 1 \
      --dec_in 8 \
      --c_out 8 \
      --batch_size 64 \
      --train_epochs 10 \
      --learning_rate 0.001 \
      --lradj type1 \
      --des 'Exp' \
      --gpu $gpu \
      --itr 1 >>logs/exchange'_'rate'_'Mamba2'_'IN.log
  done
done

#Traffic
for lr in 4e-4 4e-3
do
  for pred_len in 96 192 336 720
  do
    model=Mamba2_IN
    label_len=192
    seq_len=$label_len
    python -u run_exp.py \
      --is_training 1 \
      --root_path ./dataset/traffic \
      --data_path traffic.csv \
      --model_id traffic'_'Mamba2'_'IN'_'$label_len'_'$pred_len'_'$lr \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --embed_type 2 \
      --d_state 32 \
      --d_conv 2 \
      --d_layers 3 \
      --dec_in 862 \
      --c_out 862 \
      --batch_size 32 \
      --train_epochs 60 \
      --learning_rate $lr \
      --d_model 64 \
      --lradj type1 \
      --des 'Exp' \
      --gpu $gpu \
      --itr 1 >>logs/traffic'_'Mamba2'_'MS.log
  done
done

#weather
for lr in 4e-4 4e-3
do
  for pred_len in 96 192 336 720
  do
    model=Mamba2_IN
    label_len=192
    seq_len=$label_len
    python -u run_exp.py \
      --is_training 1 \
      --root_path ./dataset/weather \
      --data_path weather.csv \
      --model_id weather'_'Mamba2'_'IN'_'$label_len'_'$pred_len'_'$lr \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --embed_type 2 \
      --d_state 32 \
      --d_conv 2 \
      --d_layers 3 \
      --dec_in 21 \
      --c_out 21 \
      --batch_size 32 \
      --train_epochs 60 \
      --learning_rate $lr \
      --d_model 64 \
      --lradj type1 \
      --des 'Exp' \
      --gpu $gpu \
      --itr 1 >>logs/weather'_'Mamba2'_'IN.log
  done
done