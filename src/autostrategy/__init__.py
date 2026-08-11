"""autostrategy: 面向个人投资者的本地开源量化策略平台."""

from pathlib import Path

from dotenv import load_dotenv

__version__ = "0.1.0"
__all__ = ["__version__"]


def _load_local_dotenv() -> None:
    """Load the nearest .env file (cwd upwards) without overriding real env vars."""
    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        env_file = directory / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            return


_load_local_dotenv()
