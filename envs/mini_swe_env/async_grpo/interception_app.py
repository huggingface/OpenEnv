"""Entrypoint for Space-hosted interception control plane.

Run inside the HF Space process that should expose interception endpoints:

    PYTHONPATH=src:envs python -m mini_swe_env.async_grpo.interception_app
"""

from __future__ import annotations

import asyncio
import logging
import signal

from .control_plane import SWEAsyncControlPlane, SWEAsyncControlPlaneConfig


_log = logging.getLogger(__name__)


async def run_forever(config: SWEAsyncControlPlaneConfig | None = None) -> None:
    cfg = config or SWEAsyncControlPlaneConfig.from_env()
    plane = SWEAsyncControlPlane(config=cfg)
    await plane.start()

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        if not stop_event.is_set():
            _log.info("swe_async_control_plane_shutdown_requested")
            stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    finally:
        await plane.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run_forever())


if __name__ == "__main__":
    main()
