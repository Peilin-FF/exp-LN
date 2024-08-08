export PYTHONPATH=/root/workspace/Long-Seq-Model
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

gpu=0
#Traffic
for if_Regularzation in True False
do
  for lr in 4e-4 4e-3
  do
    for pred_len in 96 192 336 720
    do
      model=Mamba2_LN
      label_len=192
      seq_len=$label_len
      python -u run_exp.py \
        --is_training 1 \
        --root_path ./dataset/traffic \
        --data_path traffic.csv \
        --model_id traffic'_'Mamba2'_'LN'_'$label_len'_'$pred_len'_'$lr \
        --model $model \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --embed_type 2 \
        --d_state 32 \
        --d_conv 2 \
        --d_layers 2 \
        --dec_in 862 \
        --c_out 862 \
        --batch_size 32 \
        --train_epochs 60 \
        --learning_rate $lr \
        --d_model 64 \
        --lradj type8 \
        --des 'Exp' \
        --gpu $gpu \
        --if_Regularzation $if_Regularzation \
        --runname traffic'_'$pred_len'_'$lr'_'$if_Regularzation \
        --itr 1 >>logs/traffic'_'Mamba2'_'LN.log
    done
  done
done

#weather
for if_Regularzation in True False
do
  for lr in 4e-4 4e-3
  do
    for pred_len in 96 192 336 720
    do
      model=Mamba2_LN
      label_len=192
      seq_len=$label_len
      python -u run_exp.py \
        --is_training 1 \
        --root_path ./dataset/weather \
        --data_path weather.csv \
        --model_id weather'_'Mamba2'_'LN'_'$label_len'_'$pred_len'_'$lr \
        --model $model \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --embed_type 2 \
        --d_state 32 \
        --d_conv 2 \
        --d_layers 2 \
        --dec_in 21 \
        --c_out 21 \
        --batch_size 32 \
        --train_epochs 60 \
        --learning_rate $lr \
        --d_model 64 \
        --lradj type8 \
        --des 'Exp' \
        --gpu $gpu \
        --if_Regularzation $if_Regularzation \
        --runname weather'_'$pred_len'_'$lr'_'$if_Regularzation \
        --itr 1 >>logs/weather'_'Mamba2'_'LN.log
    done
  done
done

