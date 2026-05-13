import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from D_BETA.models.dbeta import DBETA
from HeartLang.QRSTokenizer import QRSTokenizer
from HeartLang.backbone_vqhbr import VqhbrBackbone
from benchmark.models.encoder.clocs import cnn_network_contrastive
from benchmark.models.encoder.random_encoder import RandomEncoder
from benchmark.models.encoder.resnet_merl import MerlResNet18
from benchmark.models.encoder.utils import ECGInterpolator, HeartLangNormalize

_KED_PATH = Path(__file__).resolve().parents[2] / "ECGFM-KED" / "models" / "xresnet1d_101.py"
_KED_SPEC = importlib.util.spec_from_file_location("ked_xresnet1d_101", _KED_PATH)
_KED_MODULE = importlib.util.module_from_spec(_KED_SPEC)
_KED_SPEC.loader.exec_module(_KED_MODULE)
xresnet1d101 = _KED_MODULE.xresnet1d101


def _resolve_weight_file(base_dir, preferred_name, legacy_names=()):
    base = Path(base_dir)
    candidates = [base / preferred_name, *(base / name for name in legacy_names)]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def get_weights_path(model_name, base_dir):
    if model_name == "CLOCS":
        path = _resolve_weight_file(base_dir, "best_weights_clocs")
    elif model_name == "MERL":
        path = _resolve_weight_file(base_dir, "res18_best_encoder.pth")
    elif model_name == "KED":
        path = _resolve_weight_file(base_dir, "ked.pt")
    elif model_name == "HeartLang":
        path = _resolve_weight_file(base_dir, "heart.pth")
    elif model_name == "D_BETA":
        config_path = _resolve_weight_file(base_dir, "dbeta_config.json")
        checkpoint_path = _resolve_weight_file(base_dir, "dbeta_best.pt")
        return config_path, checkpoint_path
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return path


def create_embedding_model(model_name, weights_dir):
    if model_name == "MERL":
        model = MerlResNet18()
        model.load_weights(get_weights_path(model_name, weights_dir))
        return model
    
    elif model_name == "KED":
        base_model = xresnet1d101(
            num_classes=5,
            input_channels=12,
            kernel_size=5,
            ps_head=0.5,
            lin_ftrs_head=[768],
            use_ecgNet_Diagnosis="other",
        )
        checkpoint = torch.load(
            get_weights_path(model_name, weights_dir),
            map_location="cpu",
            weights_only=True,
        )["ecg_model"]
        base_model.load_state_dict(checkpoint)
        base_model.eval()
        return nn.Sequential(ECGInterpolator(100), base_model)

    elif model_name == "HeartLang":
        base_model = VqhbrBackbone(
            seq_len=256,
            time_window=96,
            embed_dim=768,
            depth=12,
            heads=8,
            mlp_dim=1024,
            dropout=0.01,
            emb_dropout=0.01,
            Encoder=True,
        )
        checkpoint = torch.load(get_weights_path(model_name,weights_dir))["model"]
        base_model.load_state_dict(checkpoint, strict=False)
        base_model.eval()

        tokenizer = QRSTokenizer(
            fs=100,                  
            max_len=256,            
            token_len=96,            
            save_path=None,         
            stage="test",
            used_channels=list(range(12)),
        )

        return nn.Sequential(
            HeartLangNormalize(),
            ECGInterpolator(100),
            tokenizer,
            base_model,
        )

    elif model_name == "CLOCS":
        model = cnn_network_contrastive()

        def strip_compile_prefix(state_dict):
            return {
                k.replace("_orig_mod.", ""): v
                for k, v in state_dict.items()
            }
        path = get_weights_path(model_name, weights_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(path, map_location=device, weights_only=True)
        state_dict = strip_compile_prefix(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        assert len(missing) == 0
        return model

    elif model_name == "D_BETA":
        config_path, checkpoint_path = get_weights_path(model_name, weights_dir)
        with open(config_path, "r") as json_file:
            cfg = json.load(json_file)

        cfg = SimpleNamespace(**cfg["model"])
        model = DBETA(cfg)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if "ecg_encoder.mask_emb" in checkpoint["model"].keys():
            del checkpoint["model"]["ecg_encoder.mask_emb"]

        model.load_state_dict(checkpoint["model"], strict=True)

        return model

    elif "Random" in model_name:
        return RandomEncoder(model_name)

    raise ValueError(f"Unknown model type: {model_name}")
