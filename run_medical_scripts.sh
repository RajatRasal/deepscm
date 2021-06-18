python3 -m deepscm.experiments.medical_meshes.trainer -e SVIExperiment -m ConditionalVISEM --gpus 0 --default_root_dir ./medical_mesh_experiments --reload_mesh_path False --check_val_every_n_epoch 1 

python3 -m deepscm.experiments.medical_meshes.trainer -e SVIExperiment -m ConditionalVISEM --gpus 0 --default_root_dir ./medical_mesh_experiments --reload_mesh_path True --check_val_every_n_epoch 1 --brain_substructure R_Hipp --img_shape 664 3 --template_path /vol/biomedic3/bglocker/brainshapes/5026976/T1_first-R_Hipp_first.vtk

python3 -m deepscm.experiments.medical_meshes.trainer -e SVIExperiment -m ConditionalVISEM --gpus 0 --default_root_dir ./medical_mesh_experiments --reload_mesh_path True --check_val_every_n_epoch 1 --latent_dim 64 --pgm_lr 1e-5 --terminate_on_nan

python3 -m deepscm.experiments.medical_meshes.trainer -e SVIExperiment -m ConditionalVISEM --gpus 0 --default_root_dir ./medical_mesh_experiments --reload_mesh_path True --check_val_every_n_epoch 5 --latent_dim 64 --pgm_lr 1e-3 --terminate_on_nan --train_batch_size 256

python3 -m deepscm.experiments.medical_meshes.trainer -e SVIExperiment -m ConditionalVISEM --gpus 0 --default_root_dir ./medical_mesh_experiments --reload_mesh_path True --check_val_every_n_epoch 5 --brain_substructure R_Hipp --img_shape 664 3 --template_path /vol/biomedic3/bglocker/brainshapes/5026976/T1_first-R_Hipp_first.vtk --latent_dim 8 --pgm_lr 1e-3 --train_batch_size 256
