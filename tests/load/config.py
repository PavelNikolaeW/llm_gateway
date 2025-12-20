"""Load testing configuration."""
from dataclasses import dataclass
from enum import Enum


class LoadProfile(Enum):
    """Load test profiles."""

    SMOKE = "smoke"
    LOAD = "load"
    STRESS = "stress"
    SOAK = "soak"


@dataclass
class LoadConfig:
    """Load test configuration."""

    users: int
    spawn_rate: int
    run_time: str
    host: str = "http://localhost:8000"


PROFILES: dict[LoadProfile, LoadConfig] = {
    LoadProfile.SMOKE: LoadConfig(
        users=5,
        spawn_rate=1,
        run_time="30s",
    ),
    LoadProfile.LOAD: LoadConfig(
        users=50,
        spawn_rate=5,
        run_time="5m",
    ),
    LoadProfile.STRESS: LoadConfig(
        users=200,
        spawn_rate=20,
        run_time="10m",
    ),
    LoadProfile.SOAK: LoadConfig(
        users=30,
        spawn_rate=2,
        run_time="1h",
    ),
}


def get_locust_command(profile: LoadProfile, headless: bool = True) -> str:
    """Generate locust command for given profile.

    Args:
        profile: Load test profile
        headless: Whether to run without web UI

    Returns:
        Locust command string
    """
    config = PROFILES[profile]
    cmd = f"locust -f tests/load/locustfile.py"
    cmd += f" -u {config.users}"
    cmd += f" -r {config.spawn_rate}"
    cmd += f" -t {config.run_time}"
    cmd += f" --host {config.host}"

    if headless:
        cmd += " --headless"

    return cmd
