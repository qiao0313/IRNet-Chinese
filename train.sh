devices=1
save_name=saved_model

CUDA_VISIBLE_DEVICES=$devices python -u train.py --dataset ./data \
--glove_embed_path ./pretrained_models/sgns.baidubaike.bigram-char \
--cuda \
--epoch 50 \
--loss_epoch_threshold 50 \
--sketch_loss_coefficie 1.0 \
--beam_size 5 \
--seed 42 \
--save ${save_name} \
--embed_size 300 \
--sentence_features \
--column_pointer \
--hidden_size 300 \
--lr_scheduler \
--lr_scheduler_gammar 0.5 \
--att_vec_size 300 > ${save_name}".log"
