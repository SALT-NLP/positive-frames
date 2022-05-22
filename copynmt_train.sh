TASK=data/positive-reframe
onmt_preprocess -train_src $TASK/train-original.txt -train_tgt $TASK/train-reframed.txt -valid_src $TASK/validation-original.txt  -valid_tgt $TASK/validation-reframed.txt -save_data $TASK/train_val_copy -dynamic_dict
CUDA_VISIBLE_DEVICES=0 onmt_train -data $TASK/train_val_copy -save_model model/positive-reframe/model_sgd -gpu_ranks 0 -valid_steps 850 -train_steps 25500 -save_checkpoint_steps 850 -copy_attn -global_attention mlp -word_vec_size 300 -encoder_type brnn -reuse_copy_attn
