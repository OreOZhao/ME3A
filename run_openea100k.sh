cd src

version="ME3A"
gpus='3'

# EN_DE_100K_V1
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset EN_DE_100K_V1"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --version ${version}"
paras="$paras --neighbor"
paras="$paras --nsp"
paras="$paras --template_id 4"
echo $paras
python -u TEAPreprocess.py $paras
paras="$paras --fold 1"
paras="$paras --gpus ${gpus}"
python -u TEATrain.py $paras


# EN_FR_100K_V1
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset EN_FR_100K_V1"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --version ${version}"
paras="$paras --neighbor"
paras="$paras --nsp"
paras="$paras --template_id 4"
echo $paras

python -u TEAPreprocess.py $paras
paras="$paras --fold 1"
paras="$paras --gpus ${gpus}"
python -u TEATrain.py $paras

