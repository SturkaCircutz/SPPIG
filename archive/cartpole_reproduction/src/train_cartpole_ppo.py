from __future__ import annotations

from cartpole.ppo import train as _train
from cartpole.ppo.train import *  # noqa: F401,F403


_load_ppo_runtime = _train._load_ppo_runtime


def main() -> None:
    original_load_ppo_runtime = _train._load_ppo_runtime
    _train._load_ppo_runtime = _load_ppo_runtime
    try:
        _train.main()
    finally:
        _train._load_ppo_runtime = original_load_ppo_runtime


if __name__ == "__main__":
    main()
