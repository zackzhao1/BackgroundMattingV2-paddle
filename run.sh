# stage1
python ./train.py  \
--dataset-name videomatte240k \
--learning-rate 1e-4 \
--log-train-loss-interval 100 \
--epoch_end 1

# stage2
python ./train.py  \
--dataset-name Distinctions646 \
--learning-rate 5e-5 \
--log-train-loss-interval 20 \
--epoch_end 1000 \
--pretrain stage1.pdparams
