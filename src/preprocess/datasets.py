from src.preprocess.CPSC.data_CPSC import CPSCNPYDataset
from src.preprocess.CSN.data_CSN import CSNNPYDataset
from src.preprocess.ECHO_NEXT.data_ECHO_NEXT import EchoNextNPYDataset
from src.preprocess.PTBXL.data_PTBXL import PTBXLNPYDataset
from src.preprocess.PTBXL_C.data_PTBXL_C import PTBXL_C_NPYDataset


def make_dataset(dataset_name: str, split: str):
    if dataset_name == "CSN":
        return CSNNPYDataset(split=split)
    if dataset_name == "ECHO_NEXT":
        return EchoNextNPYDataset(split=split)
    if dataset_name == "CPSC":
        return CPSCNPYDataset(split=split)
    if dataset_name == "PTBXL_form":
        return PTBXLNPYDataset("form", split=split)
    if dataset_name == "PTBXL_super":
        return PTBXLNPYDataset("diagnostic_class", split=split)
    if dataset_name == "PTBXL_sub":
        return PTBXLNPYDataset("diagnostic_subclass", split=split)
    if dataset_name == "PTBXL_rhythm":
        return PTBXLNPYDataset("rhythm", split=split)
    if dataset_name == "PTBXL_C_form":
        return PTBXL_C_NPYDataset("form", split=split)
    if dataset_name == "PTBXL_C_sub":
        return PTBXL_C_NPYDataset("diagnostic_subclass", split=split)
    if dataset_name == "PTBXL_C_rhythm":
        return PTBXL_C_NPYDataset("rhythm", split=split)

    raise ValueError(f"Unknown dataset: {dataset_name}")
