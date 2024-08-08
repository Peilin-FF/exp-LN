export PYTHONPATH=/root/workspace/Long-Seq-Model
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# >logs/electricity.log

gpu=0
#ETTm2
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
        --root_path ./dataset/ETT-small \
        --data_path ETTm2.csv \
        --model_id ETTm2'_'Mamba2'_'LN'_'$label_len'_'$pred_len \
        --model $model \
        --data ETTm2 \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --embed_type 2 \
        --d_state 16 \
        --d_conv 2 \
        --d_layers 2 \
        --dec_in 7 \
        --c_out 7 \
        --batch_size 32 \
        --train_epochs 60 \
        --learning_rate $lr \
        --lradj type8 \
        --des 'Exp' \
        --gpu $gpu \
        --if_Regularzation $if_Regularzation \
        --runname ETTm2'_'$pred_len'_'$lr'_'$if_Regularzation \
        --itr 1 >>logs/ETTm2'_'Mamba2'_'LN.log
    done
  done
done

#exchange_rate
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
        --root_path ./dataset/exchange'_'rate \
        --data_path exchange_rate.csv \
        --model_id exchange'_'rate'_'Mamba2'_'LN'_'$label_len'_'$pred_len \
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
        --lradj type8 \
        --des 'Exp' \
        --gpu $gpu \
        --if_Regularzation $if_Regularzation \
        --runname exchange'_'rate'_'$pred_len'_'$lr'_'$if_Regularzation \
        --itr 1 >>logs/exchange'_'rate'_'Mamba2'_'LN.log
    done
  done
done


