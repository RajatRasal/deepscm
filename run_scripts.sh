python3 -m deepscm.experiments.morphomnist.trainer -e SVIExperiment -m ConditionalReversedVISEM --data_dir ./assets/data/morphomnist --default_root_dir . --decoder_type fixed_var --gpus 0

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

python3 -m deepscm.experiments.morphomnist.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised --default_root_dir . --decoder_type fixed_var --gpus 0  --pgm_lr 1e-7 --lr 1e-4  --num_sanity_val_steps 0 --latent_dim 8 --num_svi_particles 4

python3 -m deepscm.experiments.morphomnist.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised_ood_small --default_root_dir . --decoder_type learned_var --gpus 0  --pgm_lr 1e-6 --lr 5e-3 --num_sanity_val_steps 0 --latent_dim 12 --num_svi_particles 8

python3 -m deepscm.experiments.morphomnist.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised_ood_small --default_root_dir . --decoder_type independent_gaussian --gpus 0  --pgm_lr 1e-6 --lr 5e-3 --num_sanity_val_steps 0 --latent_dim 12 --num_svi_particles 8

python3 -m deepscm.experiments.morphomnist.tester -c ./SVIExtensionExperiment/ConditionalClassReversedVISEM/version_40 --data_dir ./assets/data/morphomnist --gpus 0
