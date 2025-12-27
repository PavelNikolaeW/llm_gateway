"""Client for fetching data from omnimap-back API."""

from dataclasses import dataclass

import httpx

from src.config.logging import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class BackendUser:
    """User data from omnimap-back."""

    id: int
    username: str
    email: str
    is_active: bool
    is_staff: bool


class BackendClient:
    """Client for omnimap-back API calls."""

    def __init__(self, token: str | None = None):
        """Initialize client with optional JWT token.

        Args:
            token: JWT access token for authenticated requests
        """
        self._token = token
        self._users_cache: list[BackendUser] | None = None

    async def get_users(
        self,
        page: int = 1,
        page_size: int = 200,
    ) -> list[BackendUser]:
        """Fetch users from omnimap-back.

        Args:
            page: Page number (1-based)
            page_size: Number of users per page (max 200)

        Returns:
            List of BackendUser objects
        """
        if not self._token:
            logger.warning("No token provided for backend API call")
            return []

        try:
            headers = {"Authorization": f"Bearer {self._token}"}
            params = {"page": page, "page_size": min(page_size, 200)}

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    settings.backend_users_url,
                    headers=headers,
                    params=params,
                )

            if response.status_code != 200:
                logger.warning(
                    f"Failed to fetch users: {response.status_code}",
                    extra={"status_code": response.status_code},
                )
                return []

            data = response.json()
            users = []

            for user_data in data.get("results", []):
                users.append(
                    BackendUser(
                        id=user_data["id"],
                        username=user_data["username"],
                        email=user_data.get("email", ""),
                        is_active=user_data.get("is_active", True),
                        is_staff=user_data.get("is_staff", False),
                    )
                )

            return users

        except httpx.TimeoutException:
            logger.error("Timeout fetching users from backend")
            return []
        except httpx.RequestError as e:
            logger.error(f"Error fetching users: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected error fetching users: {e}")
            return []

    async def get_all_users(self) -> list[BackendUser]:
        """Fetch all users from omnimap-back (handles pagination).

        Returns:
            List of all BackendUser objects
        """
        if self._users_cache is not None:
            return self._users_cache

        all_users: list[BackendUser] = []
        page = 1

        while True:
            users = await self.get_users(page=page, page_size=200)
            if not users:
                break
            all_users.extend(users)
            if len(users) < 200:
                break
            page += 1

        self._users_cache = all_users
        return all_users

    def clear_cache(self) -> None:
        """Clear the users cache."""
        self._users_cache = None


# Global client instance (token will be set per-request in admin views)
_backend_client: BackendClient | None = None


def get_backend_client(token: str | None = None) -> BackendClient:
    """Get backend client instance.

    Args:
        token: JWT token for authentication

    Returns:
        BackendClient instance
    """
    global _backend_client
    if _backend_client is None or token:
        _backend_client = BackendClient(token=token)
    return _backend_client
