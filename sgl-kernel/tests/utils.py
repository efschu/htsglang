import torch


def is_sm10x():
    return torch.cuda.get_device_capability() >= (10, 0)


def is_hopper():
    return torch.cuda.get_device_capability() == (9, 0)


def is_datacenter_blackwell():
    """sm100/sm103 (B100/B200/GB200).

    Distinct from consumer Blackwell (sm120/sm121): only the datacenter parts
    lack the classic warp-level IMMA path, which is why they have no INT8
    CUTLASS kernel while sm120 does.
    """
    return torch.cuda.get_device_capability()[0] == 10
