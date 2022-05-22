TASK=data/positive-reframe
for f in model/positive-reframe/model_sgd*
do 
    echo '------------------------------------------------------------'
    echo $f
    CUDA_VISIBLE_DEVICES=0 onmt_translate -gpu 0 -model $f -src $TASK/original-test2.txt -tgt $TASK/reframed-test2.txt -replace_unk -output $TASK/reframed-test.pred_copy.txt
    perl OpenNMT-py/tools/multi-bleu.perl $TASK/reframed-test2.txt < $TASK/reframed-test.pred_copy.txt
done