"""Format migration system for .usn files.

Supports evolving the .usn binary format across versions by providing
a registry of migration functions that transform data from one version
to another.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from usn.exceptions import VersionError
from usn.serialization.format_spec import FORMAT_VERSION

# Type alias for migration functions
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]


class FormatMigrator:
    """Handles format version migrations for .usn files.

    Maintains a registry of migration functions that can transform
    loaded data from older format versions to newer ones. Migrations
    are applied sequentially (e.g., v1→v2→v3).

    Example:
        migrator = FormatMigrator()

        def migrate_v1_to_v2(data):
            data["metadata"]["format_version"] = 2
            # Apply transformations...
            return data

        migrator.register_migration(1, 2, migrate_v1_to_v2)
        updated_data = migrator.migrate(old_data, from_version=1, to_version=2)
    """

    def __init__(self) -> None:
        """Initialize with an empty migration registry."""
        self._migrations: dict[tuple[int, int], MigrationFn] = {}

    @property
    def registered_migrations(self) -> list[tuple[int, int]]:
        """Return all registered migration version pairs."""
        return sorted(self._migrations.keys())

    def register_migration(self, from_version: int, to_version: int, fn: MigrationFn) -> None:
        """Register a migration function for a specific version transition.

        Args:
            from_version: Source format version.
            to_version: Target format version (must be from_version + 1).
            fn: Callable that takes a data dict and returns a transformed
                data dict for the new version.

        Raises:
            ValueError: If to_version is not exactly from_version + 1,
                or if a migration for this pair is already registered.
        """
        if to_version != from_version + 1:
            raise ValueError(
                f"Migrations must be sequential: expected to_version={from_version + 1}, "
                f"got {to_version}. Register one step at a time."
            )

        if to_version > FORMAT_VERSION:
            raise ValueError(
                f"Cannot register migration to version {to_version}: "
                f"maximum supported version is {FORMAT_VERSION}."
            )

        key = (from_version, to_version)
        if key in self._migrations:
            raise ValueError(
                f"Migration from v{from_version} to v{to_version} is already registered."
            )

        self._migrations[key] = fn

    def migrate(self, data: dict[str, Any], from_version: int, to_version: int) -> dict[str, Any]:
        """Migrate data from one format version to another.

        Applies migrations sequentially (v1→v2→v3...) until the target
        version is reached.

        Args:
            data: The loaded .usn data dictionary to migrate.
            from_version: The current format version of the data.
            to_version: The desired target format version.

        Returns:
            The migrated data dictionary.

        Raises:
            VersionError: If no migration path exists between the versions.
            ValueError: If from_version >= to_version (no migration needed).
        """
        if from_version == to_version:
            return data

        if from_version > to_version:
            raise ValueError(
                f"Cannot downgrade from version {from_version} to {to_version}. "
                "Only forward migrations are supported."
            )

        if to_version > FORMAT_VERSION:
            raise VersionError(to_version, FORMAT_VERSION)

        # Apply migrations sequentially
        current_data = data
        for version in range(from_version, to_version):
            key = (version, version + 1)
            if key not in self._migrations:
                raise VersionError(version, FORMAT_VERSION)
            migration_fn = self._migrations[key]
            current_data = migration_fn(current_data)

        return current_data

    def can_migrate(self, from_version: int, to_version: int) -> bool:
        """Check if a complete migration path exists between two versions.

        Args:
            from_version: Source format version.
            to_version: Target format version.

        Returns:
            True if all intermediate migration steps are registered.
        """
        if from_version >= to_version:
            return from_version == to_version

        for version in range(from_version, to_version):
            if (version, version + 1) not in self._migrations:
                return False

        return True
