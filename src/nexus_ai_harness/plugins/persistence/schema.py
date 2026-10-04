from __future__ import annotations

from collections.abc import Callable

SCHEMA_VERSION = 1
"""The snapshot format version this release writes and the newest it can read."""

Migration = Callable[[dict], dict]
"""Upgrades a snapshot from one version to the next, returning the new dict."""

_MIGRATIONS: dict[int, Migration] = {}


class SnapshotVersionError(ValueError):
    """A snapshot whose schema version this release cannot read."""


def register_migration(from_version: int) -> Callable[[Migration], Migration]:
    """Register the function that upgrades a snapshot from ``from_version``.

    Each migration takes a snapshot at ``from_version`` and returns it at
    ``from_version + 1``; :func:`migrate` chains them to reach
    :data:`SCHEMA_VERSION`. Bump ``SCHEMA_VERSION`` and register a migration
    whenever the snapshot format changes.

    Args:
        from_version: The version the decorated function upgrades from.

    Returns:
        A decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If a migration for ``from_version`` is already registered.
    """

    def register(fn: Migration) -> Migration:
        if from_version in _MIGRATIONS:
            raise ValueError(f"a migration from v{from_version} is already registered")
        _MIGRATIONS[from_version] = fn
        return fn

    return register


def snapshot_version(data: dict) -> int:
    """Return a snapshot's schema version; files without one are version 1.

    Raises:
        SnapshotVersionError: If the version is not a positive integer.
    """
    version = data.get("version", 1)
    # bool is an int subclass, but ``true`` is never a meaningful version.
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SnapshotVersionError(f"invalid snapshot version: {version!r}")
    return version


def migrate(data: dict) -> dict:
    """Upgrade a snapshot to :data:`SCHEMA_VERSION`.

    Loading fails rather than silently dropping data it doesn't understand.

    Args:
        data: A snapshot at any supported version; it is not modified.

    Returns:
        The snapshot at the current version.

    Raises:
        SnapshotVersionError: If the snapshot is newer than this release
            supports, its version is invalid, or no migration path exists.
    """
    version = snapshot_version(data)
    if version > SCHEMA_VERSION:
        raise SnapshotVersionError(
            f"snapshot version {version} is newer than the supported version "
            f"{SCHEMA_VERSION}; upgrade nexus-ai-harness to load it"
        )
    out = dict(data)
    while version < SCHEMA_VERSION:
        step = _MIGRATIONS.get(version)
        if step is None:
            raise SnapshotVersionError(f"no migration from snapshot version {version}")
        out = step(out)
        version += 1
    out["version"] = version
    return out
