# configuration of model hyperparams in training.py

data_path = '/data/chenhaoran/WHUhypspec/data/WHU-Hi-LongKou.mat'
label_path = '/data/chenhaoran/WHUhypspec/data/WHU-Hi-LongKou_gt.mat'
data_key = label_key = 'hyperspectral_data'
output_path = '/home/chenhaoran/LoLA-SpecViT-Model-main/outputs'

target_names = [ 'Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
                    'Narrow-leaf soybean', 'Rice', 'Water',
                    'Roads and houses', 'Mixed weed']

config = {
        'batch_size': 32,                  # Smaller batch size for stability
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'dataset_name': 'LongKou',         # corresponding for dict in line 436.
        'hf_backbone': 'nvidia/GCViT',
        'skip_pretrained': True,
        'lora_rank': 16,
        'lora_alpha': 32,
        'merge_lora_for_inference': False,
        'eval_interval': 5,                # Evaluate every epoch
        'label_smoothing': 0.1,
        'patience': 15,                    # More patience for CPU training
        'use_wandb': False,                # Disabled by default for stability
        'scheduler': {
            'T_0': 10,                     # Shorter warm restarts
            'T_mult': 2,
            'eta_min': 1e-6
        },
        'warmup_epochs': 5,                # Reduced warmup
        'grad_clip': 0.5,
        'lora_dropout': 0.2,
        'fast_eval': True,                # Evaluate on full test set during training
        'eval_batch_size': 32,             # Smaller eval batch size
        'gradient_accumulation_steps': 2,   # Reduced for CPU training
        'test_rate': 0.2,
        'patch_size': 15,                     # Pixel size of image patches.
        'pca_components': 15,

        # memory and processor settings.
        'use_amp': True,                   # disable for CPU.
        'pin_memory': False,               # pin weights in memory down.
        'num_workers': 2,                  # multi-processor settings.
        'max_workers': 4
    }

# Store only for references, without any application in program.
dataset_registries = {
        "LongKou": {
            "data_path": "/data/chenhaoran/WHUhypspec/data/WHU-Hi-LongKou.mat",
            "label_path": "/data/chenhaoran/WHUhypspec/data/WHU-Hi-LongKou_gt.mat",
            "data_key": "hyperspectral_data",
            "label_key": "hyperspectral_data",
            "num_classes": 9,
            "target_names": ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
                           'Narrow-leaf soybean', 'Rice', 'Water',
                           'Roads and houses', 'Mixed weed'],
            "description": "WHU-Hi-LongKou dataset with 9 crop classes"
        },
        "IndianPines": {
            "data_path": "Indian_pines_corrected.mat",
            "label_path": "Indian_pines_gt.mat",
            "data_key": "indian_pines_corrected",
            "label_key": "indian_pines_gt",
            "num_classes": 16,
            "target_names": ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                           'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                           'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                           'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                           'Stone-Steel-Towers'],
            "description": "Indian Pines dataset with 16 land cover classes"
        },
        "PaviaU": {
            "data_path": "PaviaU.mat",
            "label_path": "PaviaU_gt.mat",
            "data_key": "paviaU",
            "label_key": "paviaU_gt",
            "num_classes": 9,
            "target_names": ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                           'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows'],
            "description": "Pavia University dataset with 9 urban classes"
        },
        "PaviaC": {
            "data_path": "PaviaC.mat",
            "label_path": "PaviaC_gt.mat",
            "data_key": "paviaC",
            "label_key": "paviaC_gt",
            "num_classes": 9,
            "target_names": ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks', 'Bitumen',
                           'Tiles', 'Shadows', 'Meadows', 'Bare Soil'],
            "description": "Pavia Center dataset with 9 urban classes"
        },
        "Salinas": {
            "data_path": "data/Salinas_corrected.mat",
            "label_path": "data/Salinas_gt.mat",
            "data_key": "salinas_corrected",
            "label_key": "salinas_gt",
            "num_classes": 16,
            "target_names": ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
                           'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery',
                           'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                           'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                           'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis'],
            "description": "Salinas dataset with 16 agricultural classes"
        },
        "HongHu": {
            "data_path": "/data/chenhaoran/WHUhypspec/data/WHU-Hi-HongHu.mat",
            "label_path": "/data/chenhaoran/WHUhypspec/data/WHU-Hi-HongHu_gt.mat",
            "data_key": "WHU_Hi_HongHu",
            "label_key": "WHU_Hi_HongHu_gt",
            "num_classes": 22,
            "target_names": ['Red roof', 'Road', 'Bare soil', 'Red roof 2', 'Red roof 3',
                           'Gray roof', 'Red roof 4', 'White roof', 'Bright roof', 'Trees',
                           'Grass', 'Red roof 5', 'Red roof 6', 'Red roof 7', 'Red roof 8',
                           'Red roof 9', 'Red roof 10', 'Red roof 11', 'Red roof 12', 'Red roof 13',
                           'Red roof 14', 'Red roof 15'],
            "description": "WHU-Hi-HongHu dataset with 22 urban classes"
        },
        "Qingyun": {
            "data_path": "data/QUH-Qingyun.mat",
            "label_path": "data/QUH-Qingyun_GT.mat", 
            "data_key": "Chengqu",
            "label_key": "ChengquGT",
            "num_classes": 6,
            "target_names": ["Trees", "Concrete building", "Car", "Ironhide building",
                           "Plastic playground", "Asphalt road"],
            "description": "Qingyun dataset with 6 urban classes"
        }
    }