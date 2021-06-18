python3 -m deepscm.experiments.morphomnist_binary_labels.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised --default_root_dir .  --gpus 0  --latent_dim 16 --denoising True --terminate_on_nan

python3 -m deepscm.experiments.morphomnist_binary_labels.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised --default_root_dir .  --gpus 0  --latent_dim 8 --denoising True --terminate_on_nan

python3 -m deepscm.experiments.morphomnist_binary_labels.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised --default_root_dir .  --gpus 0  --latent_dim 16 --terminate_on_nan

python3 -m deepscm.experiments.morphomnist_binary_labels.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised --default_root_dir .  --gpus 0  --latent_dim 8 --terminate_on_nan 

python3 -m deepscm.experiments.morphomnist_binary_labels.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised --default_root_dir .  --gpus 0  --latent_dim 8 --num_svi_particles 8 --terminate_on_nan

python3 -m deepscm.experiments.morphomnist_binary_labels.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised --default_root_dir .  --gpus 0  --latent_dim 8 --num_svi_particles 8 --terminate_on_nan

python3 -m deepscm.experiments.morphomnist_binary_labels.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised --default_root_dir .  --gpus 0  --latent_dim 8 --num_svi_particles 32 --terminate_on_nan

python3 -m deepscm.experiments.morphomnist_binary_labels.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist/class_conditional_38_binarised --default_root_dir .  --gpus 0  --latent_dim 4 --num_svi_particles 16 --terminate_on_nan
