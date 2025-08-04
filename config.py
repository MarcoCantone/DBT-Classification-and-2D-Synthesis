import yaml
import importlib


def load_config(config_file=None):
    if config_file is None:
        return cfg
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def write_config(config, path):
    with open(path, 'w') as file:
        outputs = yaml.dump(config, file)
        return outputs


def get_class_by_path(path):
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def create_object_from_dict(dict):
    cls = get_class_by_path(dict["class"])
    return cls(**dict["params"])


cfg = {
    "DATA_TRANSFORM": [
        {'class': 'customTransform.ToTensorFloat32', 'params': {}},
        {'class': 'torchvision.transforms.Resize', 'params': {"size": [1024, 512], "antialias": True}},
        {'class': 'customTransform.MinMax', 'params': {}},
    ],
    "TARGET_TRANSFORM": [
        {"class": "customTransform.OneHot", "params": {"num_classes": 2}}
    ],
    "DATASET": {
        "class": 'customDatasets.OptimamTomoSurf',
        "params": {
            "dataset_path": "/data/dellascenza/Tomo2Dsynt_federico_patch128_resnet34/"
        }
    },
    "TRAINING": {
        "train_batch_size": 4,
        "test_batch_size": 4,
        "num_workers": 0,
        "epochs": 20,
        "device": "cuda:2",
        "test_ratio": 0.2,
        "resume": False,
        "train_test_split_seed": 42,
        "parallel": None,
        "workspace": "/home/dellascenza/scripts/results_effnetb3_2dsurf_fede/"
    },
    "MODEL": {
        "class": "customModels.efficientnetb3",
        "params": {
            "weights": "DEFAULT",
            "num_classes": 2
        }
    },
    "BACKBONE": {
        "model": {
            "class": "torchvision.models.resnet34_features",
            "params": {
                "weights": "DEFAULT",
            }
        },
        "gray": True,
        "layerToDiscard": 1
    },
    "OPTIMIZER": {
        "class": "torch.optim.SGD",
        "params": {"lr": 0.001, "momentum": 0.9}
    },
    "SCHEDULER": {
        "class": "torch.optim.lr_scheduler.StepLR",
        "params": {"step_size": 5, "gamma": 0.2}
    },
    "LOSS": {
        "class": "torch.nn.CrossEntropyLoss",
        "params": {}
    }
}
