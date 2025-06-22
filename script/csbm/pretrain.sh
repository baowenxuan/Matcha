# CSBM, Run ERM to get the pretrained model (for all settings)

cd ../../src || exit

gpu=1
seed=0
# 8 different settings, correspond to Table 1 in the paper

dataset='csbm'
src_degrees=(5 5 10 2 5 5 10 2)
tgt_degrees=(5 5 2 10 5 5 2 10)
src_homophilys=(0.8 0.2 0.8 0.8 0.8 0.2 0.8 0.8)
tgt_homophilys=(0.2 0.8 0.8 0.8 0.2 0.8 0.8 0.8)
mu=0.03
delta_mus=(0.00 0.00 0.00 0.00 0.02 0.02 0.02 0.02)

weights=('homo2hetero' 'hetero2homo' 'high2low' 'low2high' 'homo2hetero' 'hetero2homo' 'high2low' 'low2high')
settings=('homo2hetero' 'hetero2homo' 'high2low' 'low2high' 'homo2hetero_attr' 'hetero2homo_attr' 'high2low_attr' 'low2high_attr')

for i in {0..3}; do
  {
    CUDA_VISIBLE_DEVICES=${gpu} \
    python main.py \
      --dataset ${dataset} \
      --csbm_src_degree ${src_degrees[i]} \
      --csbm_tgt_degree ${tgt_degrees[i]} \
      --csbm_src_homophily ${src_homophilys[i]} \
      --csbm_tgt_homophily ${tgt_homophilys[i]} \
      --csbm_mu ${mu} \
      --csbm_delta_mu ${delta_mus[i]} \
      \
      --algo erm \
      --gnn gprgnn \
      --tr_rounds 500 \
      --tr_optimizer adam \
      --tr_lr 0.01 \
      \
      --save_model_path "../weights/csbm/${weights[i]}_seed_${seed}.pkl" \
      \
      --cuda \
      --seed ${seed}
  }
done
