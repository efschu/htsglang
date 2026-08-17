# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/weight_utils.py

"""Utilities for downloading and initializing model weights."""

import collections
import concurrent.futures
import fnmatch
import glob
import hashlib
import itertools
import json
import logging
import os
import re
import struct
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import filelock
import huggingface_hub.constants
import numpy as np
import safetensors.torch
import torch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator
from tqdm.auto import tqdm

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    get_world_group,
)
from sglang.srt.layers.quantization import QuantizationConfig, get_quantization_config
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
)
from sglang.srt.model_loader.ci_weight_validation import (
    ci_download_with_validation_and_retry,
    ci_validate_and_cleanup_local_snapshot,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    BAR_FORMAT,
    find_local_repo_dir,
    is_cpu,
    log_info_on_rank0,
    print_warning_once,
)
from sglang.srt.utils.common import is_cuda_alike
from sglang.utils import is_in_ci

try:
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup
except ImportError:
    SafeTensorsFileLoader = SingleGroup = None

logger = logging.getLogger(__name__)

RUNAI_STREAMER_TENSOR_ATTR = "_sglang_runai_streamer_tensor"


# Matches routed-expert weight keys in both HF-style layouts
# (``...mlp.experts.<N>.{gate,up,down}_proj.weight``) and DeepSeek V4
# layouts (``...ffn.experts.<N>.w{1,2,3}.weight``). ``shared_experts`` is
# excluded because the index segment requires a digit after ``.experts.``.
_ROUTED_EXPERT_KEY_RE = re.compile(
    r"\.experts\.\d+\.(?:w[123]|down_proj|up_proj|gate_proj)\.weight$"
)


def probe_routed_expert_weight_dtype(model_path: str) -> Optional[str]:
    """Return the safetensors dtype string (e.g. ``F8_E4M3``, ``U8``) of one
    routed-expert weight tensor, or ``None`` if the checkpoint is remote or has
    no matching key. Reads only the safetensors header of the relevant shard.
    """
    if not os.path.isdir(model_path):
        return None

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    target_key = None
    target_shard_path = None

    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {}) or {}
        for k, shard in weight_map.items():
            if _ROUTED_EXPERT_KEY_RE.search(k):
                target_key = k
                target_shard_path = os.path.join(model_path, shard)
                break
        if target_key is None:
            return None
    else:
        shards = sorted(Path(model_path).glob("*.safetensors"))
        if not shards:
            return None
        target_shard_path = str(shards[0])

    with open(target_shard_path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))

    if target_key is not None:
        meta = header.get(target_key)
        return meta.get("dtype") if meta else None

    for k, meta in header.items():
        if k == "__metadata__" or not isinstance(meta, dict):
            continue
        if _ROUTED_EXPERT_KEY_RE.search(k):
            return meta.get("dtype")
    return None


# Block size for sequential checkpoint prefetch reads (page cache warming).
_PREFETCH_BLOCK_SIZE = None


def _get_prefetch_block_size() -> int:
    global _PREFETCH_BLOCK_SIZE
    if _PREFETCH_BLOCK_SIZE is None:
        from sglang.srt.environ import envs

        _PREFETCH_BLOCK_SIZE = envs.SGLANG_PREFETCH_BLOCK_SIZE_MB.get() * 1024 * 1024
    return _PREFETCH_BLOCK_SIZE


# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
temp_dir = tempfile.gettempdir()


def get_lock(
    model_name_or_path: str, cache_dir: Optional[str] = None, suffix: str = ""
):
    lock_dir = cache_dir or temp_dir
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + suffix + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu", weights_only=True)
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)

    from safetensors.torch import save_file

    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)

    # check if the tensors are the same
    reloaded = safetensors.torch.load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def replace_prefix(key: str, prefix_mapping: dict[str, str]) -> str:
    for prefix, new_prefix in prefix_mapping.items():
        if key.startswith(prefix):
            key = key.replace(prefix, new_prefix, 1)
    return key


def replace_substrings(key: str, substring_mapping: dict[str, str]) -> str:
    for substr, new_substr in substring_mapping.items():
        if substr in key:
            key = key.replace(substr, new_substr)
    return key


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


# TODO(woosuk): Move this to other place.
def get_quant_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    packed_modules_mapping: Dict[str, List[str]],
    remap_prefix: Dict[str, str] | None = None,
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)

    # GGUF doesn't have config file
    if model_config.quantization == "gguf":
        return quant_cls.from_config({})

    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)
    if hf_quant_config is not None:
        if not isinstance(hf_quant_config, dict):
            hf_quant_config = hf_quant_config.to_dict()

        # For modelopt_mixed, config.json's quantization_config may not
        # contain all runtime metadata. Fall through to the file-based
        # hf_quant_config.json path when the per-layer map or KV-cache
        # quantization metadata is missing.
        modelopt_mixed_config_incomplete = (
            model_config.quantization == "modelopt_mixed"
            and (
                "quantized_layers" not in hf_quant_config
                or (
                    "kv_cache_quant_algo" not in hf_quant_config
                    and "kv_cache_scheme" not in hf_quant_config
                )
            )
        )
        if not modelopt_mixed_config_incomplete:
            hf_quant_config["packed_modules_mapping"] = packed_modules_mapping
            return quant_cls.from_config(hf_quant_config)

    # In case of bitsandbytes/QLoRA, get quant config from the adapter model.
    if model_config.quantization == "bitsandbytes":
        if (
            not load_config.model_loader_extra_config
            or "qlora_adapter_name_or_path" not in load_config.model_loader_extra_config
        ):
            return quant_cls.from_config({"adapter_name_or_path": ""})
        model_name_or_path = load_config.model_loader_extra_config[
            "qlora_adapter_name_or_path"
        ]
    else:
        model_name_or_path = model_config.model_path

    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    # TODO: standardize the handling of online quantization with custom handlenames (mxfp8, quark_mxfp4, etc.)
    if not possible_config_filenames:
        if model_config.quantization == "mxfp8":
            return Fp8Config(use_mxfp8=True, is_checkpoint_fp8_serialized=False)
        if model_config.quantization == "quark_mxfp4":
            return quant_cls(
                online_scheme=model_config.quantization,
                hf_config=model_config.hf_config,
            )
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}"
        )

    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)
        if remap_prefix is not None:
            exclude_modules = [
                replace_prefix(key, remap_prefix)
                for key in config["quantization"]["exclude_modules"]
            ]
            config["quantization"]["exclude_modules"] = exclude_modules
        config["packed_modules_mapping"] = packed_modules_mapping

        if model_config.quantization == "bitsandbytes":
            config["adapter_name_or_path"] = model_name_or_path
        elif model_config.quantization.startswith("modelopt") and (
            config.get("producer", {}).get("name", "").startswith("modelopt")
        ):
            quant_algo = config["quantization"]["quant_algo"]
            if quant_algo is None:
                # (yizhang2077) workaround for nvidia/Llama-4-Maverick-17B-128E-Eagle3
                if model_config.hf_config.architectures[0] != "LlamaForCausalLMEagle3":
                    raise ValueError(
                        f"Invalid quant_config, quantization method: {model_config.quantization},"
                        f"hf architectures: {model_config.hf_config.architectures[0]}. "
                    )
                return None
            elif quant_algo == "FP8" or model_config.quantization == "modelopt_fp8":
                return ModelOptFp8Config.from_config(config)
            elif "FP4" in quant_algo:
                return ModelOptFp4Config.from_config(config)
        return quant_cls.from_config(config)


def _check_index_files_exist(snapshot_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Check if all files listed in safetensors index files actually exist on disk.

    This catches cases where the snapshot directory exists but files are missing
    (e.g., due to incomplete downloads or corrupted cache).

    Args:
        snapshot_dir: Path to the model snapshot directory

    Returns:
        Tuple of (all_exist, error_message)
    """
    index_files = [
        f for f in os.listdir(snapshot_dir) if f.endswith(".safetensors.index.json")
    ]

    if not index_files:
        return True, None  # Not a sharded model

    for index_file in index_files:
        index_path = os.path.join(snapshot_dir, index_file)
        if not os.path.exists(index_path):
            continue
        try:
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            if not weight_map:
                continue
            required_files = set(weight_map.values())
            missing_files = [
                fn
                for fn in required_files
                if not os.path.exists(os.path.join(snapshot_dir, fn))
            ]
            if missing_files:
                return (
                    False,
                    f"Missing {len(missing_files)} file(s) from index {index_file}: "
                    f"{missing_files[:3]}{'...' if len(missing_files) > 3 else ''}",
                )
        except Exception as e:
            logger.warning("Failed to read index file %s: %s", index_file, e)
            continue

    return True, None


def _find_local_hf_snapshot_dir_unlocked(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: List[str],
    revision: Optional[str] = None,
) -> Optional[str]:
    """Find local HF snapshot directory without locking.

    IMPORTANT: Caller MUST hold the model lock before calling this function
    to prevent race conditions during validation and cleanup.

    If the weights are already local, skip downloading and returns the path.
    """
    if os.path.isdir(model_name_or_path):
        return None

    found_local_snapshot_dir = None

    # Check custom cache_dir (if provided)
    if cache_dir:
        try:
            repo_folder = os.path.join(
                cache_dir,
                huggingface_hub.constants.REPO_ID_SEPARATOR.join(
                    ["models", *model_name_or_path.split("/")]
                ),
            )
            rev_to_use = revision
            if not rev_to_use:
                ref_main = os.path.join(repo_folder, "refs", "main")
                if os.path.isfile(ref_main):
                    with open(ref_main) as f:
                        rev_to_use = f.read().strip()
            if rev_to_use:
                rev_dir = os.path.join(repo_folder, "snapshots", rev_to_use)
                if os.path.isdir(rev_dir):
                    found_local_snapshot_dir = rev_dir
        except Exception as e:
            logger.warning(
                "Failed to find local snapshot in custom cache_dir %s: %s",
                cache_dir,
                e,
            )

    # Check default HF cache as well
    if not found_local_snapshot_dir:
        try:
            rev_dir = find_local_repo_dir(model_name_or_path, revision)
            if rev_dir and os.path.isdir(rev_dir):
                found_local_snapshot_dir = rev_dir
        except Exception as e:
            logger.warning("Failed to find local snapshot in default HF cache: %s", e)

    # if local snapshot exists, validate it contains at least one weight file
    # matching allow_patterns before skipping download.
    if found_local_snapshot_dir is None:
        return None

    # Check if snapshot dir exists (might have been cleaned by another process
    # before we acquired the lock)
    if not os.path.isdir(found_local_snapshot_dir):
        return None

    local_weight_files: List[str] = []
    try:
        for pattern in allow_patterns:
            matched_files = glob.glob(os.path.join(found_local_snapshot_dir, pattern))
            for f in matched_files:
                # os.path.exists returns False for broken symlinks.
                if not os.path.exists(f):
                    continue
                local_weight_files.append(f)
    except Exception as e:
        logger.warning(
            "Failed to scan local snapshot %s with patterns %s: %s",
            found_local_snapshot_dir,
            allow_patterns,
            e,
        )
        local_weight_files = []

    # Check for missing files from index (lightweight, for all users)
    # This catches incomplete downloads before they cause cryptic load errors
    if local_weight_files:
        is_complete, error_msg = _check_index_files_exist(found_local_snapshot_dir)
        if not is_complete:
            log_info_on_rank0(
                logger,
                f"Local snapshot incomplete for {model_name_or_path}: {error_msg}. "
                f"Will download missing files.",
            )
            return None  # Triggers snapshot_download() which handles partial downloads

    # Only perform cache validation and cleanup in CI to avoid
    # unnecessary overhead for regular users
    if is_in_ci() and local_weight_files:
        is_valid = ci_validate_and_cleanup_local_snapshot(
            model_name_or_path, found_local_snapshot_dir, local_weight_files
        )
        if not is_valid:
            return None

    if len(local_weight_files) > 0:
        log_info_on_rank0(
            logger,
            f"Found local HF snapshot for {model_name_or_path} at "
            f"{found_local_snapshot_dir}; skipping download.",
        )
        return found_local_snapshot_dir
    else:
        log_info_on_rank0(
            logger,
            f"Local HF snapshot at {found_local_snapshot_dir} has no files matching "
            f"{allow_patterns}; will attempt download.",
        )
        return None


def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: List[str],
    revision: Optional[str] = None,
    ignore_patterns: Optional[Union[str, List[str]]] = None,
    max_retries: int = 3,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (List[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, List[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.
        max_retries (int): Maximum number of download retries if corruption
            is detected. Defaults to 3.

    Returns:
        str: The path to the downloaded model weights.
    """
    # For local paths, no HF operations needed
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    # Use a SINGLE lock for the entire operation (validation + cleanup + download)
    # to prevent race conditions where:
    # 1. Process A validates, finds corruption, deletes corrupted file
    # 2. Process B validates, sees missing file, deletes ENTIRE cache
    # 3. Process A tries to download but cache is gone
    # By using one lock, validation/cleanup and download are atomic.
    with get_lock(model_name_or_path, cache_dir):
        # Check for valid local cache first (validates and cleans up if needed)
        path = _find_local_hf_snapshot_dir_unlocked(
            model_name_or_path, cache_dir, allow_patterns, revision
        )
        if path is not None:
            # Valid local cache found, skip download
            return path

        # In CI, skip HF API calls if we're in offline mode or want to avoid rate limits
        # But we already checked for local cache above, so if we're here we need to download
        if not huggingface_hub.constants.HF_HUB_OFFLINE:
            # Before we download we look at what is available:
            fs = HfFileSystem()
            file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

            # depending on what is available we download different things
            for pattern in allow_patterns:
                matching = fnmatch.filter(file_list, pattern)
                if len(matching) > 0:
                    allow_patterns = [pattern]
                    break

        log_info_on_rank0(logger, f"Using model weights format {allow_patterns}")

        if not is_in_ci():
            # Simple download without validation for non-CI environments
            hf_folder = snapshot_download(
                model_name_or_path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                tqdm_class=DisabledTqdm,
                revision=revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
            return hf_folder
        else:
            # Only perform validation and retry in CI to avoid overhead for regular users
            return ci_download_with_validation_and_retry(
                model_name_or_path=model_name_or_path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                revision=revision,
                max_retries=max_retries,
            )


def download_safetensors_index_file_from_hf(
    model_name_or_path: str,
    index_file: str,
    cache_dir: Optional[str],
    revision: Optional[str] = None,
) -> None:
    """Download hf safetensors index file from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        revision (Optional[str]): The revision of the model.
    """
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        try:
            # Download the safetensors index file.
            hf_hub_download(
                repo_id=model_name_or_path,
                filename=index_file,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        # If file not found on remote or locally, we should not fail since
        # only some models will have index_file.
        except huggingface_hub.utils.EntryNotFoundError:
            logger.debug("No %s found in remote.", index_file)
        except huggingface_hub.utils.LocalEntryNotFoundError:
            logger.debug("No %s found in local cache.", index_file)


# For models like Mistral-7B-v0.3, there are both sharded
# safetensors files and a consolidated safetensors file.
# Passing both of these to the weight loader functionality breaks.
# So, we use the index_file to
# look up which safetensors files should be used.
def filter_duplicate_safetensors_files(
    hf_weights_files: List[str], hf_folder: str, index_file: str
) -> List[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # torch state_dict to safetensors file holding that weight.
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        # NOTE: this is a trick of handling mistral model
        # skip the unsupported consolidated.safetensors file
        if len(hf_weights_files) == 2:
            hf_weights_files.sort()
            if hf_weights_files[0].endswith(
                "consolidated.safetensors"
            ) and hf_weights_files[1].endswith("model.safetensors"):
                return [hf_weights_files[1]]
        return hf_weights_files

    # Iterate through the weight_map (weight_name: safetensors files)
    # to identify weights that we should use.
    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # Filter out any fields that are not found in the index file.
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files


def maybe_add_mtp_safetensors(
    hf_weights_files: List[str], hf_folder: str, index_file: str, hf_config
) -> List[str]:
    """
    Auto-detect and add mtp.safetensors for GLM4Moe MTP/NextN models if:
    1. mtp.safetensors exists in the model directory
    2. mtp.safetensors is NOT in the index (checkpoint packaging bug)
    3. Model architecture is Glm4MoeForCausalLM with num_nextn_predict_layers > 0

    This works around incorrectly packaged FP4 checkpoints like
    baseten-admin/glm-4.7-fp4 where mtp.safetensors exists but
    isn't referenced in model.safetensors.index.json.
    """
    # Only apply for GLM4Moe architecture with nextn layers
    arch = getattr(hf_config, "architectures", [None])[0]
    num_nextn_layers = getattr(
        getattr(hf_config, "text_config", hf_config),
        "num_nextn_predict_layers",
        getattr(hf_config, "num_nextn_predict_layers", 0),
    )
    if not (
        arch
        in [
            "Glm4MoeForCausalLM",
            "Glm4MoeForCausalLMNextN",
            "Glm4MoeLiteForCausalLM",
            "Glm4MoeLiteForCausalLMNextN",
        ]
        and num_nextn_layers > 0
    ):
        return hf_weights_files

    # Check if mtp.safetensors exists and is not already in the file list
    mtp_path = os.path.join(hf_folder, "mtp.safetensors")
    if not os.path.isfile(mtp_path) or mtp_path in hf_weights_files:
        return hf_weights_files

    # mtp.safetensors exists but not in index - this is a bug
    logger.warning(
        f"Found mtp.safetensors but it's not referenced in {index_file}. "
        f"This is a checkpoint packaging bug. Auto-adding it for loading. "
        f"Please report this to the checkpoint provider."
    )

    # Add it to the files list
    return hf_weights_files + [mtp_path]


def filter_files_not_needed_for_inference(hf_weights_files: List[str]) -> List[str]:
    """
    Exclude files that are not needed for inference.

    See https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    """
    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files


def np_cache_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str],
    hf_folder: str,
    hf_weights_files: List[str],
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model np files.

    Will dump the model weights to numpy files if they are not already dumped.
    """
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    # Convert the model weights from torch tensors to numpy arrays for
    # faster loading.
    np_folder = os.path.join(hf_folder, "np")
    os.makedirs(np_folder, exist_ok=True)
    weight_names_file = os.path.join(np_folder, "weight_names.json")
    # Use file lock to prevent multiple processes from
    # dumping the same model weights to numpy at the same time.
    with get_lock(model_name_or_path, cache_dir):
        if not os.path.exists(weight_names_file):
            weight_names: List[str] = []
            for bin_file in tqdm(
                hf_weights_files,
                desc="Loading np_cache checkpoint shards",
                disable=not enable_tqdm,
                bar_format=BAR_FORMAT,
                position=tqdm._get_free_pos(),
            ):
                state = torch.load(bin_file, map_location="cpu", weights_only=True)
                for name, param in state.items():
                    param_path = os.path.join(np_folder, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())
                    weight_names.append(name)
            with open(weight_names_file, "w") as f:
                json.dump(weight_names, f)

    with open(weight_names_file) as f:
        weight_names = json.load(f)

    for name in weight_names:
        param_path = os.path.join(np_folder, name)
        with open(param_path, "rb") as f:
            param = np.load(f)
        yield name, torch.from_numpy(param)


def _prefetch_checkpoint_file(file_path: str) -> None:
    """Prefetch a checkpoint file into the OS page cache.

    Reads the file sequentially in 16 MB blocks so the kernel caches its pages
    before workers load the same file via mmap.
    """
    with open(file_path, "rb") as f:
        while f.read(_get_prefetch_block_size()):
            pass


def _prefetch_all_checkpoints(
    sorted_files: List[str],
    num_threads: int = 4,
) -> None:
    """Start prefetching checkpoint files into page cache in a background thread.

    When multiple ranks on the same node load the same checkpoint (e.g.
    DP-attention), each rank independently mmaps the same files, causing
    redundant NFS/Lustre reads. By distributing the prefetch across ranks
    (each rank reads 1/Nth of the shards), the total network I/O is reduced
    from N * checkpoint_size to 1 * checkpoint_size, with subsequent
    mmap accesses hitting the shared OS page cache.

    The prefetch runs in a background thread so that loading can start
    immediately and benefit from pages that have already been cached,
    rather than blocking until all files are prefetched. This pipelining
    naturally adapts to any RAM size — even if the full checkpoint does
    not fit in page cache, the prefetch thread stays ahead of the loader.
    """
    import threading
    import time

    if num_threads < 1:
        raise ValueError("weight loader prefetch num_threads must be >= 1")

    # Use node-local rank so that each node independently prefetches the
    # full checkpoint into its own page cache. Global rank would split files
    # across nodes, but page cache is not shared across nodes.
    if torch.distributed.is_initialized():
        world_group = get_world_group()
        local_rank = world_group.local_rank
        local_world_size = world_group.local_size or world_group.world_size
    else:
        local_rank = 0
        local_world_size = 1

    my_files = sorted_files[local_rank::local_world_size]
    total_for_rank = len(my_files)

    logger.info(
        "Rank %d: prefetching %d/%d checkpoint shards into page cache "
        "(background, %d local ranks sharing the work, %d threads per rank)...",
        local_rank,
        total_for_rank,
        len(sorted_files),
        local_world_size,
        num_threads,
    )

    def _prefetch_all() -> None:
        completed = 0
        next_log_pct = 10

        def record_complete() -> None:
            nonlocal completed, next_log_pct

            completed += 1
            if total_for_rank > 0 and next_log_pct <= 100:
                pct = 100 * completed / total_for_rank
                while pct >= next_log_pct and next_log_pct <= 100:
                    logger.info(
                        "Rank %d: prefetching checkpoint files: %d%% (%d/%d)",
                        local_rank,
                        next_log_pct,
                        completed,
                        total_for_rank,
                    )
                    next_log_pct += 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            file_iter = iter(my_files)
            pending: Dict[concurrent.futures.Future, str] = {}

            for path in itertools.islice(file_iter, num_threads):
                pending[executor.submit(_prefetch_checkpoint_file, path)] = path

            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    path = pending.pop(future)
                    try:
                        future.result()
                    except Exception:
                        logger.warning(
                            "Failed to prefetch checkpoint file %r.",
                            path,
                            exc_info=True,
                        )
                    finally:
                        record_complete()

                    next_path = next(file_iter, None)
                    if next_path is not None:
                        pending[
                            executor.submit(_prefetch_checkpoint_file, next_path)
                        ] = next_path

    def _run_prefetch() -> None:
        start = time.perf_counter()
        _prefetch_all()
        elapsed = time.perf_counter() - start
        logger.info(
            "Rank %d: prefetching checkpoint files into page cache "
            "finished in %.2fs",
            local_rank,
            elapsed,
        )

    threading.Thread(target=_run_prefetch, daemon=True).start()


#: O_DIRECT needs the buffer, the offset and the length aligned. 4096 covers
#: every ashift/recordsize combination this rig runs; a larger block only costs
#: a bigger tail, never correctness.
_DIRECT_ALIGN = 4096
_DIRECT_BLOCK = 1 << 20


def read_file_direct(path: str) -> bytes:
    """Read `path` with O_DIRECT, bypassing the page cache entirely.

    WHY THIS EXISTS (#738). On ZFS both eviction-advice rungs are no-ops:
    `posix_fadvise(DONTNEED)` was measured dead by #408, and its replacement
    `MADV_PAGEOUT` was measured dead here on 2026-08-17 -- it returns rc=0,
    reporting the advice ACCEPTED, and freed 529 of 136,781 pages (0.4%). A
    silent success is the worst shape: the flag looked wired and did nothing.

    You cannot evict what you never cached, so this attacks the cause instead
    of the symptom. Measured on this pool (OpenZFS 2.3.4, real direct I/O):
    an aligned 1.20 GB read moved the page cache by ZERO pages, against a
    baseline where a buffered read of the same shard adds ~293k.

    STRICTLY OPT-IN and byte-identical: the returned bytes are exactly what
    `open(path,'rb').read()` returns. Only the path they take into memory
    differs.

    THE TAIL IS READ BUFFERED, deliberately. O_DIRECT requires aligned length;
    the final partial block cannot satisfy that. Reading it through the normal
    path caches at most `_DIRECT_ALIGN`-rounded bytes -- kilobytes against
    gigabytes -- and keeps the function total rather than refusing files whose
    size is not a multiple of the block.
    """
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    o_direct = getattr(os, "O_DIRECT", 0o40000)
    size = os.path.getsize(path)
    whole = (size // _DIRECT_BLOCK) * _DIRECT_BLOCK
    out = bytearray()
    raw = ctypes.create_string_buffer(_DIRECT_BLOCK + _DIRECT_ALIGN)
    base = (ctypes.addressof(raw) + _DIRECT_ALIGN - 1) & ~(_DIRECT_ALIGN - 1)
    fd = os.open(path, os.O_RDONLY | o_direct)
    try:
        while len(out) < whole:
            n = libc.read(fd, ctypes.c_void_p(base), ctypes.c_size_t(_DIRECT_BLOCK))
            if n <= 0:
                err = ctypes.get_errno()
                raise OSError(err, f"O_DIRECT read failed on {path} at {len(out)}")
            out += ctypes.string_at(base, n)
    finally:
        os.close(fd)
    if size > whole:  # the unalignable tail, through the ordinary path
        with open(path, "rb") as f:
            f.seek(whole)
            out += f.read()
    return bytes(out)


def _drop_file_cache_after_load(
    path: str, extents: Optional[Sequence[Tuple[int, int]]] = None
) -> None:
    """Release checkpoint pages after weights have been copied out.

    Opt-in via ``--weight-loader-drop-cache-after-load``
    (``server_args.weight_loader_drop_cache_after_load``), used to avoid CPU
    OOM in RL.

    Whole-file drop, not incremental: unlike the GGUF stream
    (``gguf_shards.ConsumedPageDropper``), every tensor from ``path`` has
    already been read AND copied into its destination parameter by the
    caller of the weights iterator by the time this runs -- there is no
    reader still working behind us, so there is no consumed/unconsumed
    watermark to keep; the whole file is safe to advise in one shot.

    ``posix_fadvise(POSIX_FADV_DONTNEED)`` alone -- the previous
    implementation -- is a measured no-op on this rig's ZFS pool:
    ``invalidate_mapping_pages()`` does not free a page that is still
    mapped, and on ZFS it does not free it even after the mapping goes away
    either (see the ladder rationale in the ``gguf_shards.py`` module
    comment).

    ``extents``: ``(address, nbytes)`` of every tensor the call site read as
    a live, zero-copy ``safetensors.safe_open`` mmap view, captured *before*
    the mapping had a chance to be torn down. These are used directly:
    ``MADV_PAGEOUT`` walks the page table of the mapping it is given, so it
    needs addresses from a mapping that is still alive. A separate,
    freshly-opened mapping of the same file created *after the fact* was
    tried and measured NOT to work here -- with the original
    ``safe_open``-backed mapping still referenced elsewhere (which its
    lifetime does not rule out), reclaim through an unrelated second mapping
    left the page cache untouched. Reusing the live mapping's own addresses,
    the same principle ``gguf_shards.ConsumedPageDropper`` uses, is what
    actually frees the pages (measured: 100% resident -> 0% resident on this
    rig's ZFS pool).

    When no live mapping is available -- ``disable_mmap=True`` reads the
    whole file into a fresh buffer with no page-cache-backed view to advise
    at all, or the platform has no ``madvise`` -- this falls back to plain
    ``posix_fadvise(DONTNEED)`` on the whole file: the pre-#408 mechanism,
    a measured no-op on this rig's ZFS pool but still the best available
    option with nothing live to advise.

    No additional utilization ceiling, safety margin, or rounding is applied
    on top of what is advised: this reclaims pages that have already been
    fully read and copied out, it does not discard anything a caller has not
    consumed yet.
    """
    from sglang.srt.model_loader.gguf_shards import (
        _page_cache_advice_available,
        drop_page_cache_range,
    )

    if extents and _page_cache_advice_available():
        for address, nbytes in extents:
            if nbytes <= 0:
                continue
            # Each tensor is advised as its own live range: address is the
            # tensor's own data pointer, so no file-offset math is needed,
            # and map_len == nbytes bounds it exactly.
            drop_page_cache_range(path, 0, nbytes, address=address, map_len=nbytes)
        return

    # No live mapping to advise -- fall back to plain fadvise on the whole
    # file, exactly the pre-#408 mechanism.
    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        return
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
        posix_fadvise(fd, 0, 0, dontneed)
    except OSError as e:
        logger.debug("Failed to drop file cache for %s: %s", path, e)
    finally:
        if fd is not None:
            os.close(fd)


def safetensors_weights_iterator(
    hf_weights_files: List[str],
    disable_mmap: bool = False,
    direct_io: bool = False,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    if prefetch and not disable_mmap:
        _prefetch_all_checkpoints(
            sorted(hf_weights_files), num_threads=prefetch_num_threads
        )

    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=BAR_FORMAT,
        position=tqdm._get_free_pos(),
    ):
        if disable_mmap:
            if direct_io:
                # #738: bypass the page cache instead of trying to evict it
                # afterwards. Both advice rungs are no-ops on ZFS, and
                # MADV_PAGEOUT reports rc=0 while freeing 0.4% -- a silent
                # success. Bytes are identical; only the route differs.
                result = safetensors.torch.load(read_file_direct(st_file))
            else:
                with open(st_file, "rb") as f:
                    result = safetensors.torch.load(f.read())
            for name in sorted(result.keys()):
                yield name, result[name]
            if drop_cache_after_load and not direct_io:
                # The whole file went through a plain read() into a fresh
                # buffer, not a memory map -- there is no live mapping to
                # hand the ladder, so this falls back to plain fadvise.
                # Pointless after a direct read: nothing was cached.
                _drop_file_cache_after_load(st_file)
        else:
            extents: List[Tuple[int, int]] = []
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    if drop_cache_after_load:
                        extents.append(
                            (tensor.data_ptr(), tensor.numel() * tensor.element_size())
                        )
                    yield name, tensor
                # Still inside the `with` block on purpose: the mapping the
                # extents above point into is only guaranteed alive up to
                # `f`'s own __exit__, so the drop has to happen before it,
                # not after (see _drop_file_cache_after_load's docstring).
                if drop_cache_after_load:
                    _drop_file_cache_after_load(st_file, extents)


def fastsafetensors_weights_iterator(
    hf_weights_files: List[str],
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the weights in the model safetensor files
    using fastsafetensor library to accelerate loading via GPU Direct Storage (if available).
    """
    if SafeTensorsFileLoader is None:
        raise ImportError(
            "Please install fastsafetensors via `pip install fastsafetensors`"
        )

    if torch.distributed.is_initialized():
        pg = torch.distributed.group.WORLD
    else:
        pg = SingleGroup()

    try:
        rank = pg.rank()
    except Exception:
        rank = 0

    device = torch.device(f"cuda:{rank}")

    weight_files_sub_lists = [
        hf_weights_files[i : i + pg.size()]
        for i in range(0, len(hf_weights_files), pg.size())
    ]

    _BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for f_list in tqdm(
        weight_files_sub_lists,
        desc="Loading safetensors using Fastsafetensor loader",
        disable=False,
        bar_format=_BAR_FORMAT,
    ):
        loader = SafeTensorsFileLoader(pg, device)
        rank_file_map = {i: [f] for i, f in enumerate(f_list)}
        loader.add_filenames(rank_file_map)
        try:
            fb = loader.copy_files_to_device()
            try:
                keys = list(fb.key_to_rank_lidx.keys())
                for k in keys:
                    t = fb.get_tensor(k)
                    yield k, t
            finally:
                pass
        finally:
            loader.close()


def multi_thread_safetensors_weights_iterator(
    hf_weights_files: List[str],
    max_workers: int,
    disable_mmap: bool = False,
    direct_io: bool = False,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model safetensor files."""
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    def _load_file(st_file: str):
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                result = {k: f.get_tensor(k) for k in f.keys()}

        return st_file, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_file, st_file) for st_file in hf_weights_files]

        if enable_tqdm:
            futures_iter = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(hf_weights_files),
                desc="Multi-thread loading shards",
                disable=not enable_tqdm,
                bar_format=BAR_FORMAT,
            )
        else:
            futures_iter = concurrent.futures.as_completed(futures)

        for future in futures_iter:
            st_file, state_dict = future.result()
            extents: Optional[List[Tuple[int, int]]] = None
            if drop_cache_after_load and not disable_mmap:
                # `_load_file`'s `with safe_open(...)` block has already
                # exited by the time we get `state_dict` back, but its
                # tensors are zero-copy views that keep the underlying
                # mapping alive for as long as `state_dict` is -- captured
                # here, before `del state_dict` below lets it go.
                extents = [
                    (t.data_ptr(), t.numel() * t.element_size())
                    for t in state_dict.values()
                ]
            for name, param in state_dict.items():
                yield name, param
            if drop_cache_after_load:
                _drop_file_cache_after_load(st_file, extents)
            del state_dict


def buffered_multi_thread_safetensors_weights_iterator(
    hf_weights_files: List[str],
    max_workers: int,
    disable_mmap: bool = False,
    direct_io: bool = False,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Multi-threaded safetensor loader with bounded memory via a sliding window.

    At most (max_workers + 1) shard files are in-flight at any time:
    max_workers loading concurrently + 1 prefetched and ready to yield.
    Peak CPU RAM ≈ (max_workers + 2) × shard_file_size.
    """
    if prefetch and not disable_mmap:
        _prefetch_all_checkpoints(
            sorted(hf_weights_files), num_threads=prefetch_num_threads
        )
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    def _load_file(st_file: str):
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                result = {k: f.get_tensor(k) for k in f.keys()}
        return result

    # Sliding window: max_workers loading + 1 prefetched.
    buffer_size = max_workers + 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        file_iter = iter(hf_weights_files)
        pending: collections.deque = collections.deque()

        # Seed the buffer.
        for st_file in itertools.islice(file_iter, buffer_size):
            pending.append((st_file, executor.submit(_load_file, st_file)))

        with tqdm(
            total=len(hf_weights_files),
            desc="Multi-thread loading shards",
            disable=not enable_tqdm,
            bar_format=BAR_FORMAT,
            position=tqdm._get_free_pos(),
        ) as pbar:
            while pending:
                st_file, future = pending.popleft()
                state_dict = future.result()
                del future  # let GC reclaim the Future's internal result

                # Replenish: submit the next file to keep the buffer full.
                next_file = next(file_iter, None)
                if next_file is not None:
                    pending.append((next_file, executor.submit(_load_file, next_file)))

                extents: Optional[List[Tuple[int, int]]] = None
                if drop_cache_after_load and not disable_mmap:
                    # `_load_file`'s `with safe_open(...)` block has already
                    # exited by the time we get `state_dict` back, but its
                    # tensors are zero-copy views that keep the underlying
                    # mapping alive for as long as `state_dict` is -- capture
                    # the live addresses before `del state_dict` lets it go.
                    extents = [
                        (t.data_ptr(), t.numel() * t.element_size())
                        for t in state_dict.values()
                    ]
                for name in sorted(state_dict.keys()):
                    yield name, state_dict[name]
                if drop_cache_after_load:
                    _drop_file_cache_after_load(st_file, extents)
                del state_dict
                pbar.update(1)


def _load_pt_file(bin_file: str) -> dict:
    """Load a PyTorch checkpoint file, handling legacy tar format.

    PyTorch 2.6 changed the default of weights_only from False to True.
    Legacy tar format files cannot be loaded with weights_only=True.
    This function tries weights_only=True first, then falls back to False
    for legacy tar format files from trusted sources (HuggingFace Hub).
    """
    try:
        return torch.load(bin_file, map_location="cpu", weights_only=True)
    except RuntimeError as e:
        if "legacy .tar format" in str(e):
            logger.warning(
                "Loading %s with weights_only=False (legacy tar format)",
                os.path.basename(bin_file),
            )
            return torch.load(bin_file, map_location="cpu", weights_only=False)
        raise


def pt_weights_iterator(
    hf_weights_files: List[str],
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm,
        bar_format=BAR_FORMAT,
        position=tqdm._get_free_pos(),
    ):
        state = _load_pt_file(bin_file)
        yield from state.items()
        del state


def multi_thread_pt_weights_iterator(
    hf_weights_files: List[str],
    max_workers: int,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model bin/pt files."""
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_load_pt_file, bin_file) for bin_file in hf_weights_files
        ]

        if enable_tqdm:
            futures_iter = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(hf_weights_files),
                desc="Multi-thread loading pt checkpoint shards",
                disable=not enable_tqdm,
                bar_format=BAR_FORMAT,
            )
        else:
            futures_iter = concurrent.futures.as_completed(futures)

        for future in futures_iter:
            state = future.result()
            yield from state.items()


def get_gguf_extra_tensor_names(
    gguf_file: str, gguf_to_hf_name_map: Dict[str, str]
) -> List[str]:
    from sglang.srt.model_loader.gguf_shards import gguf_tensor_names

    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    # Union over the whole split set: a tensor that lives on part 3 is not an
    # "extra" (i.e. absent) tensor just because part 1 does not hold it.
    exact_gguf_keys = gguf_tensor_names(gguf_file)
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [gguf_to_hf_name_map[key] for key in extra_keys]


def gguf_quantized_name(name: str, leaf: str) -> str:
    """Rename the trailing ``.weight`` of a mapped tensor name to *leaf*.

    Only the FINAL path segment is rewritten. The previous spelling was
    ``name.replace("weight", <leaf>)``, which rewrites EVERY occurrence, so a
    module whose own name contains "weight" was renamed along with its leaf:
    ``...attn.indexer.weights_proj.weight`` became
    ``...attn.indexer.qweights_proj.qweight``, a parameter no model has
    (#391 wall 16). That tensor is F32 in the published DeepSeek V4 export and
    F32 tensors never reach this rename, which is the only reason the defect
    was invisible rather than fatal.

    A name that does not end in ``.weight`` is returned unchanged, matching the
    old behaviour for the bare parameters (``...attn_sink``, ``...hc_ffn_fn``,
    ``...gate.tid2eid``) that carry no ``weight`` substring at all.
    """
    head, sep, tail = name.rpartition(".")
    if tail != "weight":
        return name
    return f"{head}{sep}{leaf}"


# #647: HF-name suffixes that are ALWAYS plain dense parameters, never
# GGUF-quantized linears -- so an unquantized tensor destined for one of them
# must keep its ".weight" name instead of being renamed to ".qweight".
#
# A table rather than a chain of endswith() in one family's adapter: the
# defect is in the GENERIC iterator, so every family and the no-adapter path
# hit it, and DeepSeek V4 had already grown a bespoke repair for its own gate
# (gguf_deepseek4.py:366-375) that no other model benefited from.
#
# Membership alone is not sufficient to fire -- see is_dense_gguf_target. A
# module that genuinely IS quantized keeps the quantized path even if its name
# matches, because the type check fails.
GGUF_DENSE_PARAM_SUFFIXES: Tuple[str, ...] = (
    # MoE router gate. Built ReplicatedLinear(..., quant_config=None), so it
    # owns a ".weight" and nothing else; a ".qweight" emitted for it matches
    # no parameter and is dropped silently, leaving the gate uninitialised and
    # the MoE routing on garbage.
    ".gate.weight",
    # MoE shared-expert gate. Same hazard, different suffix: the HF name ends
    # in "_gate.weight", never ".gate.weight", so the entry above does not
    # reach it. Built torch.nn.Linear(hidden, 1) on the CUDA path
    # (qwen2_moe.py:467) -- no quant method, hence no ".qweight" either.
    # Found on metal (#656, 2026-08-12): a Qwen3.6-35B-A3B GGUF boot with
    # NEXTN speculation refused outright, because the checkpoint stores
    # blk.<mtp>.ffn_gate_inp_shexp.weight in BF16 and the renamed tensor
    # reached no parameter, leaving the draft model's gate unloaded.
    ".shared_expert_gate.weight",
)


def is_dense_gguf_target(hf_name: str, weight_type) -> bool:
    """Whether this tensor must bypass the ``.qweight`` rename (#647).

    Two conditions, both required:

    * the destination is a known dense module (``GGUF_DENSE_PARAM_SUFFIXES``);
    * the stored GGML type is genuinely unquantized -- F32, F16 or BF16, the
      layer's own ``UNQUANTIZED_TYPES`` set. The iterator historically tested
      only ``!= "F32"``, which is narrower than that set by exactly F16 and
      BF16, and that gap is the whole of #647.

    Requiring the type check keeps a genuinely quantized tensor on the
    quantized path even if its name matches the table, so the rule cannot
    misfire on a model that really does quantize the module.
    """
    if not any(hf_name.endswith(suffix) for suffix in GGUF_DENSE_PARAM_SUFFIXES):
        return False
    return getattr(weight_type, "name", None) in ("F32", "F16", "BF16")


def gguf_dense_payload(tensor: torch.Tensor, weight_type) -> torch.Tensor:
    """Reinterpret the raw payload gguf-py returns for a dense F16/BF16 tensor.

    gguf-py hands F16/BF16 back as ``uint8`` with the last dimension doubled;
    a view is enough, no conversion. A tensor that already carries a float
    dtype (F32, or a future gguf-py that decodes BF16 itself) is passed
    through unchanged.

    Generalised from ``gguf_deepseek4._bytes_to_bf16``, which did this for one
    family's router gate only.
    """
    if tensor.dtype != torch.uint8:
        return tensor
    name = getattr(weight_type, "name", None)
    if name == "BF16":
        return tensor.contiguous().view(torch.bfloat16)
    if name == "F16":
        return tensor.contiguous().view(torch.float16)
    return tensor


def gguf_quant_weights_iterator(
    gguf_file: str, gguf_to_hf_name_map: Dict[str, str]
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the quant weights in the model gguf files and convert
    them to torch tensors
    """

    import gguf

    from sglang.srt.model_loader.gguf_mxfp4_repack import (
        log_gguf_repack_plan,
        repacked_gguf_bytes,
        repacked_gguf_type,
    )
    from sglang.srt.model_loader.gguf_shards import (
        declared_tensor_count,
        make_stream_page_dropper,
        resolve_gguf_shard_paths,
    )

    # A split export is several files; an unsplit one resolves to a single-entry
    # list and this reads exactly like the one-reader version it replaces. Both
    # passes below run over the CONCATENATION in shard order, which is the order
    # a merged file would present, rather than per-shard pass1+pass2.
    shard_paths = resolve_gguf_shard_paths(gguf_file)
    readers = [gguf.GGUFReader(path, "r") for path in shard_paths]
    all_tensors = []
    # (shard index, index within that shard) per tensor, so the page-cache
    # dropper below can name the region a tensor occupies in its own file.
    tensor_origin: List[Tuple[int, int]] = []
    for shard_index, reader in enumerate(readers):
        for tensor_index, tensor in enumerate(reader.tensors):
            all_tensors.append(tensor)
            tensor_origin.append((shard_index, tensor_index))

    expected_tensors = declared_tensor_count(gguf_file)
    if expected_tensors is not None and len(all_tensors) != expected_tensors:
        raise RuntimeError(
            f"GGUF split export {gguf_file}: the {len(shard_paths)} parts hold "
            f"{len(all_tensors)} tensors but declare split.tensors.count="
            f"{expected_tensors}. Loading them would produce a model with "
            "silently missing weights."
        )

    # #391: types with no GGUF kernel that a value-exact load-time repack turns
    # into types that have one (MXFP4 -> Q5_0) are converted HERE, at the two
    # points below where a tensor's type marker and its payload leave the
    # reader. Everything downstream -- the per-expert split of the stacked
    # ffn_*_exps tensors included -- sees only the repacked type, so no kernel,
    # type bucket or adapter needs to know the source type existed. The repack
    # is not free in bytes; this states the cost once, before it is paid.
    log_gguf_repack_plan(all_tensors)

    # MoE expert weight name patterns
    MOE_WEIGHT_PATTERNS = {
        "ffn_gate_exps": "gate_proj",  # gate projection
        "ffn_up_exps": "up_proj",  # up projection
        "ffn_down_exps": "down_proj",  # down projection
    }

    # First pass: yield weight types
    for tensor in all_tensors:
        tensor_name = tensor.name
        weight_type = repacked_gguf_type(tensor.tensor_type, tensor_name)

        # Check if this is a MoE expert weight (packed format)
        is_moe_weight = any(
            pattern in tensor_name for pattern in MOE_WEIGHT_PATTERNS.keys()
        )

        if is_moe_weight:
            # MoE weights need special handling - extract layer_id and weight type
            # Format: blk.{layer_id}.ffn_gate_exps.weight
            import re

            match = re.match(r"blk\.(\d+)\.(ffn_\w+_exps)\.weight", tensor_name)
            if match:
                layer_id = int(match.group(1))
                weight_pattern = match.group(2)
                hf_weight_name = MOE_WEIGHT_PATTERNS.get(weight_pattern)

                if hf_weight_name and weight_type.name != "F32":
                    # Yield weight type for each expert
                    weight = tensor.data
                    num_experts = weight.shape[0]
                    for expert_id in range(num_experts):
                        hf_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{hf_weight_name}.qweight_type"
                        yield hf_name, torch.tensor(weight_type)
        elif tensor_name in gguf_to_hf_name_map:
            # Normal weight handling
            name = gguf_to_hf_name_map[tensor_name]

            # #647: a dense module gets no type marker either. The marker is
            # named after the ".qweight" that is not going to exist, so
            # emitting it would leave a second orphan alongside the first.
            if weight_type.name != "F32" and not is_dense_gguf_target(
                name, weight_type
            ):
                weight_type_name = gguf_quantized_name(name, "qweight_type")
                yield weight_type_name, torch.tensor(weight_type)

    # Second pass: yield actual weights.
    #
    # #391: this is the only place in the loader that faults GGUF payload pages
    # in, so it is also the only place that can give them back. `dropper` marks
    # each tensor's byte range droppable ONE ITERATION AFTER the loop has passed
    # it -- i.e. only once the consumer has come back for the next item, which
    # means the copy out of the mapping (torch.tensor / repacked_gguf_bytes)
    # has certainly completed. Skipped tensors are marked too: the stream has
    # passed them and nothing downstream will read their payload.
    #
    # Under TP each rank runs its own dropper over the same files. That is not a
    # cross-rank hazard: madvise only unmaps the calling process's PTEs, and the
    # following fadvise can only free pages no process still maps -- a rank that
    # is behind keeps its own pages until it passes them itself.
    dropper = make_stream_page_dropper(shard_paths, readers)
    previous_origin: Optional[Tuple[int, int]] = None
    try:
        for tensor, origin in zip(all_tensors, tensor_origin):
            if previous_origin is not None:
                dropper.consume(*previous_origin)
            previous_origin = origin

            weight = tensor.data
            tensor_name = tensor.name
            source_type = tensor.tensor_type
            weight_type = repacked_gguf_type(source_type, tensor_name)

            # Check if this is a MoE expert weight (packed format)
            is_moe_weight = any(
                pattern in tensor_name for pattern in MOE_WEIGHT_PATTERNS.keys()
            )

            if is_moe_weight:
                # MoE weights: split packed format into individual expert weights
                import re

                match = re.match(r"blk\.(\d+)\.(ffn_\w+_exps)\.weight", tensor_name)
                if match:
                    layer_id = int(match.group(1))
                    weight_pattern = match.group(2)
                    hf_weight_name = MOE_WEIGHT_PATTERNS.get(weight_pattern)

                    if hf_weight_name:
                        # Packed format: [num_experts, ...]
                        num_experts = weight.shape[0]
                        for expert_id in range(num_experts):
                            # Repack per expert, not per stacked tensor: a Q5_0
                            # block is self-contained and the split runs on whole
                            # blocks, so one expert's slice repacks to a valid
                            # payload on its own -- and the transient buffer stays
                            # one expert wide instead of one layer wide.
                            expert_weight = repacked_gguf_bytes(
                                source_type, weight[expert_id], tensor_name
                            )

                            if weight_type.name != "F32":
                                hf_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{hf_weight_name}.qweight"
                            else:
                                hf_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{hf_weight_name}.weight"

                            yield hf_name, torch.tensor(expert_weight)
            elif tensor_name in gguf_to_hf_name_map:
                # Normal weight handling
                name = gguf_to_hf_name_map[tensor_name]

                # #647: an unquantized tensor bound for a dense module keeps
                # its ".weight" name and is handed over as real values, not
                # as the raw uint8 payload with a doubled last dimension.
                if is_dense_gguf_target(name, weight_type):
                    yield name, gguf_dense_payload(
                        torch.tensor(
                            repacked_gguf_bytes(source_type, weight, tensor_name)
                        ),
                        weight_type,
                    )
                else:
                    if weight_type.name != "F32":
                        name = gguf_quantized_name(name, "qweight")
                    param = torch.tensor(
                        repacked_gguf_bytes(source_type, weight, tensor_name)
                    )
                    yield name, param

        if previous_origin is not None:
            dropper.consume(*previous_origin)
    finally:
        dropper.close()


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise


def row_parallel_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    """Load weights that are row-parallelized."""
    tp_rank = get_parallel().tp_rank
    shard_dim = 0 if param.dim() != 1 else None

    if shard_dim is not None:
        shard_size = param.data.shape[shard_dim]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)

    return default_weight_loader(param, loaded_weight)


LoaderFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sharded_weight_loader(shard_axis: int) -> LoaderFunction:
    """Create a weight loader that shards the weights along the given axis"""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        from sglang.srt.distributed.utils import tp_loaded_shard_start

        tp_rank = get_parallel().attn_tp_rank

        shard_size = param.data.shape[shard_axis]
        # Even TP: tp_rank * shard_size, as before. Uneven TP
        # (--rank-tp-ratio): prefix-sum offset from the shard plan; the
        # optional `tp_units` attribute on the parameter (e.g. GDN k
        # heads) marks the dimension's indivisible unit count.
        start_idx = tp_loaded_shard_start(
            loaded_weight.shape[shard_axis],
            get_parallel().attn_tp_size,
            tp_rank,
            shard_size,
            getattr(param, "tp_units", None),
        )

        if (
            is_cpu()
            and (
                loaded_weight.size(0) % get_parallel().tp_size != 0
                or loaded_weight.size(0) < get_parallel().tp_size * shard_size
            )
            and loaded_weight.dim() == 1
        ):
            param_data = param.data  # view copy on param for uneven padding
            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data,
                loaded_weight,
                0,  # param_data_start
                start_idx,
                shard_axis,
                shard_size,
            )
            return default_weight_loader(param_data, loaded_weight)
        else:
            loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
            return default_weight_loader(param, loaded_weight)

    return loader


def composed_weight_loader(
    loader: LoaderFunction, fn: Callable[[torch.Tensor], torch.Tensor]
) -> LoaderFunction:
    """Create a weight loader that post-processes the weights after loading"""

    def composed_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        loader(param, loaded_weight)
        param.data.copy_(fn(param))
        return

    return composed_loader


def runai_safetensors_weights_iterator(
    hf_weights_files: List[str], is_distributed: bool = False, device: str = "cpu"
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    from runai_model_streamer import SafetensorsStreamer

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    device = device if is_distributed and is_cuda_alike() else "cpu"

    with SafetensorsStreamer() as streamer:
        streamer.stream_files(
            hf_weights_files,
            device=device,
            is_distributed=is_distributed,
        )
        total_tensors = sum(
            len(tensors_meta)
            for tensors_meta in streamer.files_to_tensors_metadata.values()
        )

        tensor_iter = tqdm(
            streamer.get_tensors(),
            total=total_tensors,
            desc="Loading safetensors using Runai Model Streamer",
            bar_format=BAR_FORMAT,
            disable=not enable_tqdm,
            mininterval=2,
        )

        for name, tensor in tensor_iter:
            setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
            yield name, tensor


def set_runai_streamer_env(load_config: LoadConfig):
    if load_config.model_loader_extra_config:
        extra_config = load_config.model_loader_extra_config

        if "concurrency" in extra_config and isinstance(
            extra_config.get("concurrency"), int
        ):
            os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(
                extra_config.get("concurrency")
            )

        if "memory_limit" in extra_config and isinstance(
            extra_config.get("memory_limit"), int
        ):
            os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = str(
                extra_config.get("memory_limit")
            )

    runai_streamer_s3_endpoint = os.getenv("RUNAI_STREAMER_S3_ENDPOINT")
    aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    if runai_streamer_s3_endpoint is None and aws_endpoint_url is not None:
        os.environ["RUNAI_STREAMER_S3_ENDPOINT"] = aws_endpoint_url


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.

    We use per-parameter random seed, so that dummy weights are consistent,
    even if the model is partitioned across multiple devices. When the seed
    is fixed, the random values generated by this function only depends on
    the parameter's number of elements and its data type.
    """
    for param in model.state_dict().values():
        if torch.is_floating_point(param):
            generator = torch.Generator(device=param.data.device)
            generator.manual_seed(seed)
            # Tensor subclasses such as MXFP8 wrappers expose a low-bit raw
            # storage dtype through `.data`, but their wrapper `uniform_` also
            # updates side tensors such as block scales.
            if torch.finfo(param.dtype).bits < 16:
                # uniform_ doesn't support < 16-bit datatypes (FP8)
                dtype = param.data.dtype
                tmp_param = param.data.to(torch.float16)
                tmp_param = tmp_param.uniform_(low, high, generator=generator).to(dtype)
                param.data.copy_(tmp_param)
            else:
                param.uniform_(low, high, generator=generator)


def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> Optional[str]:
    """Remap the name of FP8 k/v_scale parameters.

    This function handles the remapping of FP8 k/v_scale parameter names.
    It detects if the given name ends with a suffix and attempts to remap
    it to the expected name format in the model. If the remapped name is not
    found in the params_dict, a warning is printed and None is returned.

    Args:
        name (str): The original loaded checkpoint parameter name.
        params_dict (dict): Dictionary containing the model's named parameters.

    Returns:
        str: The remapped parameter name if successful, or the original name
             if no remapping is needed.
        None: If the remapped name is not found in params_dict.
    """
    if name.endswith(".kv_scale"):
        print_warning_once(
            "DEPRECATED. Found kv_scale in the checkpoint. "
            "This format is deprecated in favor of separate k_scale and "
            "v_scale tensors and will be removed in a future release. "
            "Functionally, we will remap kv_scale to k_scale and duplicate "
            "k_scale to v_scale"
        )
        # NOTE: we remap the deprecated kv_scale to k_scale
        remapped_name = name.replace(".kv_scale", ".attn.k_scale")
        if remapped_name not in params_dict:
            print_warning_once(
                f"Found kv_scale in the checkpoint (e.g. {name}), "
                "but not found the expected name in the model "
                f"(e.g. {remapped_name}). kv_scale is "
                "not loaded."
            )
            return None
        return remapped_name

    possible_scale_names = [".k_scale", ".v_scale"]
    # Patterns where modelopt stores scales under k_proj/v_proj
    # but the model expects them under attn (RadixAttention)
    modelopt_attn_prefixes = [".self_attn.", ".mixer."]
    for scale_name in possible_scale_names:
        if name.endswith(scale_name):
            # Check if this is a modelopt-style scale under k_proj/v_proj
            matched_prefix = None
            for attn_prefix in modelopt_attn_prefixes:
                if f"{attn_prefix}{scale_name[1]}_proj{scale_name}" in name:
                    matched_prefix = attn_prefix
                    break

            if matched_prefix is not None:
                remapped_name = name.replace(
                    f"{matched_prefix}{scale_name[1]}_proj{scale_name}",
                    f"{matched_prefix}attn{scale_name}",
                )
            else:
                remapped_name = name.replace(scale_name, f".attn{scale_name}")
            if remapped_name not in params_dict:
                print_warning_once(
                    f"Found {scale_name} in the checkpoint (e.g. {name}), "
                    "but not found the expected name in the model "
                    f"(e.g. {remapped_name}). {scale_name} is "
                    "not loaded."
                )
                return None
            return remapped_name

    quark_scale_names = {
        ".q_proj.output_scale": ".attn.q_scale",
        ".k_proj.output_scale": ".attn.k_scale",
        ".v_proj.output_scale": ".attn.v_scale",
        "self_attn.prob_output_scale": ".attn.prob_scale",
    }
    for quark_scale_name, sglang_scale_name in quark_scale_names.items():
        if name.endswith(quark_scale_name):
            return name.replace(quark_scale_name, sglang_scale_name)

    # If there were no matches, return the untouched param name
    return name


# Adapted from https://github.com/vllm-project/vllm/blob/68ad4e3a8d8a66fb2a43be57471ee13a8bec4ec0/vllm/model_executor/layers/quantization/schema.py
class KVCacheQuantSchema(BaseModel):
    dtype: str
    # Each key is a TP rank. Each value is a dictionary mapping a TP rank's
    # layer indices to their per-tensor KV cache scaling factor.
    # TODO: Consider pulling this and its validation methods out into its
    # own schema class (tricky as its members are variable)
    scaling_factor: Dict[int, Dict[int, float]]

    @model_validator(mode="after")
    def check_is_fp8(self) -> "KVCacheQuantSchema":
        assert self.dtype == "float8_e4m3fn", (
            "Loaded scaling factors intended for KV cache dtype = "
            f"{self.dtype} rather than float8_e4m3fn!"
        )
        return self

    @model_validator(mode="after")
    def check_tp_ranks(self, info: ValidationInfo) -> "KVCacheQuantSchema":
        context = info.context
        if context:
            tp_size = context["tp_size"]
            num_hidden_layers = context["num_hidden_layers"]
            assert len(self.scaling_factor) == tp_size, (
                f"Loaded dictionary has TP size {len(self.scaling_factor)} "
                f"but LLM engine is currently running with TP size {tp_size}."
            )
            for tp_rank, layer_maps in self.scaling_factor.items():
                assert len(layer_maps) == num_hidden_layers, (
                    f"KV cache scales map for TP rank {tp_rank} is malformed. "
                    f"Expected {num_hidden_layers} layers, got "
                    f"{len(layer_maps)}."
                )
            for i in range(tp_size):
                assert i in self.scaling_factor, (
                    f"KV cache scales map for TP rank {i} not found."
                )
        return self

    @model_validator(mode="after")
    def check_current_rank(self, info: ValidationInfo) -> "KVCacheQuantSchema":
        context = info.context
        if context:
            tp_rank = context["tp_rank"]
            num_hidden_layers = context["num_hidden_layers"]
            layer_scales_map = self.scaling_factor[tp_rank]
            for i in range(num_hidden_layers):
                assert i in layer_scales_map, (
                    f"Could not find KV cache scales for layer {i} in "
                    f"TP rank {tp_rank}."
                )
        return self


class QuantParamSchema(BaseModel):
    # TODO: Generalize and extend with more fields
    # (e.g. weights/activations params) once functionality is enabled
    model_config = ConfigDict(protected_namespaces=())
    model_type: Optional[str]
    kv_cache: KVCacheQuantSchema

    @model_validator(mode="after")
    def check_model_type(self, info: ValidationInfo) -> "QuantParamSchema":
        context = info.context
        if context:
            model_type = context.get("model_type", None)
            if model_type is not None:
                assert model_type == self.model_type, (
                    f"Model type is {model_type} but loaded "
                    f"scaling factors belonging to different "
                    f"model type {self.model_type}!"
                )
        return self


def kv_cache_scales_loader(
    filename: str,
    tp_rank: int,
    tp_size: int,
    num_hidden_layers: int,
    model_type: Optional[str],
) -> Iterable[Tuple[int, float]]:
    """
    A simple utility to read in KV cache scaling factors that have been
    previously serialized to disk. Used by the model to populate the appropriate
    KV cache scaling factors. The serialization should represent a dictionary
    whose keys are the TP ranks and values are another dictionary mapping layers
    to their KV cache scaling factors.
    """
    try:
        with open(filename) as f:
            context = {
                "model_type": model_type,
                "num_hidden_layers": num_hidden_layers,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
            schema_dct = json.load(f)
            schema = QuantParamSchema.model_validate(schema_dct, context=context)
            layer_scales_map = schema.kv_cache.scaling_factor[tp_rank]
            return layer_scales_map.items()
    except FileNotFoundError:
        logger.error("File or directory '%s' not found.", filename)
    except json.JSONDecodeError:
        logger.error("Error decoding JSON in file '%s'.", filename)
    except Exception:
        logger.error("An error occurred while reading '%s'.", filename)
    # This section is reached if and only if any of the excepts are hit
    # Return an empty iterable (list) => no KV cache scales are loaded
    # which ultimately defaults to 1.0 scales
    logger.warning(
        "Defaulting to KV cache scaling factors = 1.0 for all "
        "layers in TP rank %d as an error occurred during loading.",
        tp_rank,
    )
    return []


def get_actual_shard_size(shard_size, weight_start, weight_end):
    if weight_end < weight_start:
        return 0

    return min(shard_size, weight_end - weight_start)


def reset_param_data_if_needed(param_data, dim, start, length):
    if length == 0:
        return

    assert length > 0, f"Length should be positive, but got {length}"

    param_data.narrow(dim, start, length).zero_()
    return


def narrow_padded_param_and_loaded_weight(
    param_data,
    loaded_weight,
    param_data_start,
    weight_start,
    dim,
    shard_size,
    narrow_weight=True,
):
    actual_shard_size = get_actual_shard_size(
        shard_size, weight_start, loaded_weight.size(dim)
    )

    if narrow_weight:
        if actual_shard_size > 0:
            loaded_weight = loaded_weight.narrow(dim, weight_start, actual_shard_size)
        else:
            # No real data to load; create a dummy tensor filled with zeros
            loaded_weight = torch.zeros_like(
                param_data.narrow(dim, param_data_start, actual_shard_size)
            )

    # [Note] Reset padded weights to zero.
    # If the actual shard size is less than the shard size, we need to reset
    # the padded param_data to zero and then copy the loaded_weight into it.
    reset_param_data_if_needed(
        param_data,
        dim,
        param_data_start + actual_shard_size,
        shard_size - actual_shard_size,
    )

    param_data = param_data.narrow(dim, param_data_start, actual_shard_size)

    return param_data, loaded_weight


def pad_loaded_weight(loaded_weight, output_dim, output_sizes):
    # This function is for padding zeros when loaded_weight is less than output_sizes.
    # Most cases, sum(output_sizes) = loaded_weight.size(output_dim),
    # while in some TP cases like TP6, output_sizes will be padded, thus loaded_weight needs padding.
    total_output_size = sum(output_sizes)
    raw_output_size = loaded_weight.size(output_dim)
    if total_output_size > raw_output_size:
        loaded_weight_pad = []
        weight_split_size = [
            int(output_size / total_output_size * raw_output_size)
            for output_size in output_sizes
        ]
        assert sum(weight_split_size) == raw_output_size, (
            f"Padding the loaded weight failed due to sizes are not divisible cleanly from {output_sizes} to {raw_output_size}"
        )

        split_weight = loaded_weight.split_with_sizes(weight_split_size, dim=output_dim)
        for i, output_size in enumerate(output_sizes):
            pad_size = output_size - weight_split_size[i]
            target_pad_shape = list(loaded_weight.size())
            target_pad_shape[output_dim] = pad_size
            pad_tensor = torch.zeros(target_pad_shape).to(loaded_weight.dtype)
            loaded_weight_pad.append(
                torch.cat([split_weight[i], pad_tensor], dim=output_dim)
            )
        return torch.cat(loaded_weight_pad, dim=output_dim)
    else:
        return loaded_weight


#: Draft parameters the target hands over AFTER the load, so the draft
#: checkpoint legitimately never carries them: `set_embed_and_head`,
#: `set_embed_and_head_modules` and `set_lm_head_from_target` all replace these
#: modules (or their `.weight`) with the target's once both models exist.
_TARGET_PROVIDED_DRAFT_PARAM_FRAGMENTS = (
    "embed_tokens.",
    "lm_head.",
    "embed_tokens_",
)


#: Draft classes already reported as unchecked, so the notice is one line per
#: class per process rather than one per rank per load.
_DRAFT_LOAD_UNCHECKED_SEEN: set = set()


def _warn_draft_load_unchecked(
    model: torch.nn.Module, model_path: str, reason: str
) -> None:
    """Say out loud that the draft-load completeness check did NOT run.

    The failure this guard exists for -- a draft that loads "successfully" and
    then proposes noise, with an accept rate near zero as the only symptom --
    is silent. A guard that silently declines to run therefore reproduces the
    exact condition it was written to remove, which is what audit #505 found:
    the hoist to the loader was meant to make the check "a property of loading
    A DRAFT, not of one model class", and most draft classes were skipping it.
    """
    name = type(model).__name__
    if name in _DRAFT_LOAD_UNCHECKED_SEEN:
        return
    _DRAFT_LOAD_UNCHECKED_SEEN.add(name)
    logger.warning(
        "draft-load completeness NOT checked for %s (%s): %s. An unwritten "
        "parameter would leave the drafter on uninitialised memory, whose only "
        "symptom is a speculative accept rate near zero. Have this class's "
        "load_weights return the set of parameter names it filled to enable "
        "the check (#505-A1-01).",
        name,
        model_path or "<no path>",
        reason,
    )


def raise_on_unloaded_draft_parameters(
    model: torch.nn.Module,
    loaded_params: Optional[Iterable[str]],
    *,
    model_path: str = "",
) -> None:
    """Turn silently skipped DRAFT parameters into a named error.

    A checkpoint name that resolves to nothing is skipped by design (rotary
    caches, foreign namespaces). A PARAMETER that nothing ever wrote is a
    different thing: the drafter then runs on ``torch.empty``, proposes
    near-random tokens, and the only symptom is an accept rate that looks like
    a weak drafter rather than an unloaded one -- accept 1.005 both times this
    happened.

    #290 installed exactly this check, but inside ``DFlashDraftModel``, so it
    covered one architecture. #318 hit the same wall from the other side: a
    GPTQ-Marlin MTP skeleton fed a dense checkpoint dropped 7 projections per
    layer behind a per-NAME ``logger.warning_once`` -- deduplicated, on the
    checkpoint-name side, and never counted against the parameters that were
    supposed to be filled. Hoisting the check to the loader makes it a property
    of loading A DRAFT, not of one model class.

    Only models whose ``load_weights`` REPORTS the parameters it filled are
    checked; a model that returns ``None`` is left exactly as it was -- but NOT
    silently (#514/#505-A1-01). Audit #505 measured this function's real reach
    and found it skipping most draft classes, with the skip indistinguishable
    from a pass. The unchecked state is still allowed -- retro-fitting every
    draft class's ``load_weights`` to return its set is a per-class change with
    real risk -- but it is no longer invisible.
    """
    if loaded_params is None or not isinstance(
        loaded_params, (set, frozenset, list, tuple)
    ):
        _warn_draft_load_unchecked(
            model,
            model_path,
            "its load_weights returns None instead of the set of parameters it "
            "filled, so completeness cannot be checked",
        )
        return
    loaded = {name for name in loaded_params if isinstance(name, str)}
    if not loaded:
        # Nothing reported at all: a load that filled everything through a
        # different mechanism is indistinguishable from one that filled
        # nothing, so this stays out of the raising business.
        _warn_draft_load_unchecked(
            model,
            model_path,
            "its load_weights reported an EMPTY set of filled parameters, "
            "which a load that fills everything by another mechanism cannot be "
            "told apart from a load that filled nothing",
        )
        return

    missing = sorted(
        name
        for name in dict(model.named_parameters())
        if name not in loaded
        and not any(
            fragment in name for fragment in _TARGET_PROVIDED_DRAFT_PARAM_FRAGMENTS
        )
    )
    if not missing:
        return

    shown = missing[:8]
    message = (
        f"Draft checkpoint {model_path or '(unknown path)'} left "
        f"{len(missing)} parameter(s) of {type(model).__name__} unloaded: "
        f"{shown}{' ...' if len(missing) > len(shown) else ''}. "
        "The checkpoint's tensor names do not reach these parameters, so the "
        "draft model would run on uninitialized weights and propose noise "
        "(the symptom is an accept length of ~1.0, not a load error). The two "
        "known causes are a quantization mismatch in either direction: a "
        "packed checkpoint loaded into a model built WITHOUT a quantization "
        "config (the stream carries `*.qweight`, the skeleton has dense "
        "`*.weight`), or a model built WITH one over a checkpoint that stores "
        "the draft block dense (the skeleton expects "
        "`qweight`/`qzeros`/`scales`/`g_idx`, the file has plain `*.weight`). "
        "Set --speculative-draft-model-quantization explicitly to pin the "
        "side that is wrong."
    )
    from sglang.srt.environ import envs

    if envs.SGLANG_ALLOW_UNLOADED_DRAFT_PARAMS.get():
        logger.error("%s (SGLANG_ALLOW_UNLOADED_DRAFT_PARAMS=1: continuing)", message)
        return
    raise ValueError(message)
