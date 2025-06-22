# CSBM, Homophily shift only
# h = 0.8 -> h = 0.2
# d = 5

cd ../../src || exit

gpu=1
seed=0

dataset='csbm'
src_degree=5
tgt_degree=5
src_homophily=0.8
tgt_homophily=0.2
mu=0.03
delta_mu=0.00

# pretrained weight and setting name
weight='homo2hetero'
setting='homo2hetero'

# algorithm
algo='matcha'
base_tta='erm' # 'tent' 't3a' 'adanpc'
gnn='gprgnn'

# hyperparameters for Matcha
ada_lr=1.0
ada_optimizer='sgd'
ada_rounds=50

# hyperparameters for baseTTA
t3a_filter_K=2000
tent_lr=0.1
tent_rounds=10
adanpc_k=10
adanpc_temperature=0.001

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
  --dataset ${dataset} \
  --csbm_src_degree ${src_degree} \
  --csbm_tgt_degree ${tgt_degree} \
  --csbm_src_homophily ${src_homophily} \
  --csbm_tgt_homophily ${tgt_homophily} \
  --csbm_mu ${mu} \
  --csbm_delta_mu ${delta_mu} \
  \
  --algo ${algo} \
  --base_tta ${base_tta} \
  --gnn ${gnn} \
  --ada_lr ${ada_lr} \
  --ada_optimizer ${ada_optimizer} \
  --ada_rounds ${ada_rounds} \
  --t3a_filter_K ${t3a_filter_K} \
  --tent_rounds ${tent_rounds} \
  --tent_lr ${tent_lr} \
  --adanpc_k ${adanpc_k} \
  --adanpc_temperature ${adanpc_temperature} \
  \
  --cuda \
  --seed ${seed} \
  \
  --load_model_path "../weights/csbm/${weight}_seed_${seed}.pkl" # \
  # --save_metrics_path "../results/csbm/${setting}/matcha_${base_tta}.pkl"
