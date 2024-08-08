export PYTHONPATH=/root/workspace/Long-Seq-Model
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# >logs/electricity.log

gpu=0
#Electricity
for if_Regularzation in True False
do
  for lr in 4e-3 4e-4
  do
    for pred_len in 96 192 336 720
    do
      model=Mamba2_LN
      label_len=192
      seq_len=$label_len
      python -u run_exp.py \
        --is_training 1 \
        --root_path ./dataset/electricity \
        --data_path electricity.csv \
        --model_id electricity'_'Mamba2_LN'_'$label_len'_'$pred_len'_'$lr \
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
        --dec_in 321 \
        --c_out 321 \
        --batch_size 32 \
        --train_epochs 60 \
        --learning_rate $lr \
        --d_model 32 \
        --lradj type8 \
        --des 'Exp' \
        --gpu $gpu \
        --if_Regularzation $if_Regularzation \
        --runname electricity'_'$pred_len'_'$lr'_'$if_Regularzation \
        --itr 1 >> logs/electricity'_'LN.log
    done
  done
done

#ETTh1
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
        --data_path ETTh1.csv \
        --model_id ETTh1'_'Mamba2'_'LN''_''$label_len'_'$pred_len \
        --model $model \
        --data ETTh1 \
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
        --runname ETTh1'_'$pred_len'_'$lr'_'$if_Regularzation 
        --itr 1 >>logs/ETTh1'_'Mamba2'_'LN.log
    done
  done
done
