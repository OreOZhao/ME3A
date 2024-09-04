cd src

version="ME3A"
gpus='3'

# fr_en
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset fr_en"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --version ${version}"
paras="$paras --neighbor"
paras="$paras --nsp"
paras="$paras --template_id 4"
paras="$paras --visual"
paras="$paras --prefix 2"
echo $paras
python -u TEAPreprocess.py $paras
paras="$paras --fold 1"
paras="$paras --gpus ${gpus}"
python -u ME3ATrain.py $paras


# ja_en
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset ja_en"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --version ${version}"
paras="$paras --neighbor"
paras="$paras --nsp"
paras="$paras --template_id 4"
paras="$paras --visual"
paras="$paras --prefix 2"
echo $paras

python -u TEAPreprocess.py $paras
paras="$paras --fold 1"
paras="$paras --gpus ${gpus}"
python -u ME3ATrain.py $paras


# zh_en
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset zh_en"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --version ${version}"
paras="$paras --neighbor"
paras="$paras --nsp"
paras="$paras --template_id 4"
paras="$paras --visual"
paras="$paras --prefix 2"
echo $paras

python -u TEAPreprocess.py $paras
paras="$paras --fold 1"
paras="$paras --gpus ${gpus}"
python -u ME3ATrain.py $paras
