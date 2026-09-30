"""Launch the inference server."""

import asyncio
import os
import sys
import warnings

from flliper.srt.server_args import prepare_server_args
from flliper.srt.utils import kill_process_tree
from flliper.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()


def run_server(server_args):
    """Run the server based on the gRPC flags and server_args.encoder_only."""
    from flliper._compat_boot import link_state_dir   # rename transition: rig-state dir (compat_shims)

    link_state_dir()
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.smg_grpc_mode or server_args.grpc_mode:
            from flliper.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from flliper.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.smg_grpc_mode:
        # Legacy SMG gRPC server (--smg-grpc-mode, or the deprecated --grpc-mode
        # which __post_init__ folds into smg_grpc_mode). The native Rust gRPC
        # server is a separate path, enabled by --grpc-port, that starts
        # alongside the default HTTP server below.
        from flliper.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        try:
            from flliper.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'flliper[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from flliper.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m flliper.launch_server' is still supported, but "
        "'flliper serve' is the recommended entrypoint.\n"
        "  Example: flliper serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    from flliper.srt.plugins import load_plugins

    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
