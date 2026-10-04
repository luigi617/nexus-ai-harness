from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# Upper bound for the one-off bubblewrap probe, so a wedged helper can't hang init.
_PROBE_TIMEOUT = 10.0


class IsolationUnavailable(RuntimeError):
    """An OS isolation backend that was required or requested cannot be used."""


class IsolationWarning(RuntimeWarning):
    """A sandbox is running commands without kernel-enforced isolation."""


@dataclass(frozen=True)
class IsolationSpec:
    """What a backend must enforce for one sandboxed command.

    Attributes:
        root: The resolved workspace root; the only persistent writable tree.
        tmpdir: The sandbox's private scratch dir, exposed as ``$TMPDIR``.
        allow_network: Whether the process may reach the network.
    """

    root: Path
    tmpdir: Path
    allow_network: bool


class IsolationBackend(ABC):
    """A strategy that wraps a command so the OS confines what it can do.

    A backend turns a plain ``argv`` into one that runs under an OS-level
    confinement helper (for example ``sandbox-exec`` or ``bwrap``). It must
    confine writes to :attr:`IsolationSpec.root` and :attr:`IsolationSpec.tmpdir`
    and deny network unless :attr:`IsolationSpec.allow_network` is set.
    Subclass it to plug in a custom mechanism and pass an instance to
    :class:`~nexus_ai_harness.plugins.sandbox.WorkspaceSandbox`.

    Attributes:
        name: A short identifier reported as the sandbox's isolation level.
        enforced: Whether this backend actually applies kernel-level isolation.
    """

    name: str = ""
    enforced: bool = True

    @abstractmethod
    def available(self) -> bool:
        """Report whether the backend can run on this host right now."""

    @abstractmethod
    def wrap(self, argv: Sequence[str], spec: IsolationSpec) -> list[str]:
        """Return ``argv`` wrapped so it runs under this backend's confinement.

        Args:
            argv: The command to run; ``argv[0]`` is the program.
            spec: The confinement the wrapped command must be held to.

        Returns:
            The argument vector to hand to the OS in place of ``argv``.
        """


class NoIsolation(IsolationBackend):
    """A passthrough backend that runs commands without OS-level confinement.

    Path confinement, command policy, the scrubbed environment, and the timeout
    still apply; only write confinement and network denial are not enforced.
    """

    name = "none"
    enforced = False

    def available(self) -> bool:
        """Always ``True``: running unconfined needs no helper."""
        return True

    def wrap(self, argv: Sequence[str], spec: IsolationSpec) -> list[str]:
        """Return ``argv`` unchanged."""
        return list(argv)


class SandboxExecBackend(IsolationBackend):
    """Confinement via the macOS ``sandbox-exec`` (Seatbelt) profile language.

    The profile allows everything by default, then narrows file writes to the
    root, the scratch dir, and ``/dev`` (for stdio), and denies network unless
    allowed. Reads are not restricted.
    """

    name = "sandbox-exec"

    def available(self) -> bool:
        """Report whether this is macOS and ``sandbox-exec`` is on ``PATH``."""
        return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None

    def wrap(self, argv: Sequence[str], spec: IsolationSpec) -> list[str]:
        """Prefix ``argv`` with ``sandbox-exec -p <profile>``."""
        executable = shutil.which("sandbox-exec") or "sandbox-exec"
        return [executable, "-p", self.profile(spec), *argv]

    def profile(self, spec: IsolationSpec) -> str:
        """Build the SBPL profile string enforcing ``spec``."""
        lines = [
            "(version 1)",
            "(allow default)",
            "(deny file-write*)",
            f'(allow file-write* (subpath "{sbpl_escape(spec.root)}"))',
            f'(allow file-write* (subpath "{sbpl_escape(spec.tmpdir)}"))',
            '(allow file-write* (subpath "/dev"))',
        ]
        if not spec.allow_network:
            lines.append("(deny network*)")
        return "\n".join(lines)


class BubblewrapBackend(IsolationBackend):
    """Confinement via Linux ``bwrap`` (bubblewrap) namespaces.

    The whole filesystem is bind-mounted read-only, with fresh ``/dev`` and
    ``/proc`` and writable binds of only the root and the scratch dir. The
    network namespace is unshared when network is denied, and the child gets its
    own PID and IPC namespaces, a new session (so it can't inject input into the
    caller's terminal), and dies with its parent.

    Reads are not restricted, except under the masked directories (``/tmp``,
    ``/run``, a real ``/var/run``, and ``$XDG_RUNTIME_DIR``), which are replaced
    by private, empty tmpfs mounts. Masking them gives the command a writable
    private ``/tmp`` and hides the host's Unix sockets (D-Bus, Docker, ssh and
    gpg agents, X11): a read-only mount does not stop ``connect()`` on a socket
    file, so a reachable socket would let a command escape write confinement and
    network denial through a host service. Host content under those directories
    is invisible to the command; re-expose specific paths read-only with
    ``expose``. Top-level symlinks in ``/run`` that point outside it (such as
    NixOS's ``/run/current-system``) are recreated so programs reached through
    them still resolve.

    Unix sockets elsewhere on the host (for example under ``$HOME`` or
    ``/var/lib``) are not masked; deny the clients that use them via command
    policy, or pass a custom backend, if that matters for your deployment.

    Availability is probed once per ``bwrap`` binary by actually starting a
    trivial sandbox with the same layout, because ``bwrap`` is often installed
    but unusable (for example in containers that forbid unprivileged user
    namespaces).

    Attributes:
        executable: An explicit ``bwrap`` path; ``None`` looks it up on ``PATH``.
        expose: Host paths bound read-only on top of the masked directories.
    """

    name = "bubblewrap"

    def __init__(
        self,
        executable: str | None = None,
        *,
        expose: Sequence[str | Path] = (),
    ) -> None:
        """Create the backend.

        Args:
            executable: Path to ``bwrap``; ``None`` resolves it from ``PATH``.
            expose: Host paths (for example a virtualenv under ``/tmp``) to bind
                read-only inside the sandbox even though a mask hides them.
        """
        self.executable = executable
        # bwrap only runs on Linux, so these must stay POSIX-style even if the
        # host is Windows, where str(Path(...)) would use backslashes instead.
        self.expose = tuple(Path(path).as_posix() for path in expose)

    def _resolve(self) -> str | None:
        return self.executable or shutil.which("bwrap")

    def available(self) -> bool:
        """Report whether this is Linux and a working ``bwrap`` is present."""
        if not sys.platform.startswith("linux"):
            return False
        executable = self._resolve()
        return executable is not None and _bwrap_works(executable)

    def wrap(self, argv: Sequence[str], spec: IsolationSpec) -> list[str]:
        """Prefix ``argv`` with a ``bwrap`` invocation enforcing ``spec``."""
        executable = self._resolve() or "bwrap"
        masks = masked_dirs()
        options = bwrap_args(
            spec, masks=masks, symlinks=escaping_symlinks(masks), expose=self.expose
        )
        return [executable, *options, "--", *argv]


# Always masked: /tmp for a private scratch area, /run for host service sockets.
_BASE_MASKS = ("/tmp", "/run")

# Usually a symlink to /run; masked separately only when it is a real directory.
_VAR_RUN = Path("/var/run")


def masked_dirs() -> tuple[str, ...]:
    """Return the host directories bubblewrap hides behind private tmpfs mounts.

    ``/tmp`` and ``/run`` are always masked. ``/var/run`` is added when it is a
    real directory rather than the usual symlink to ``/run``, and
    ``$XDG_RUNTIME_DIR`` when it is a directory outside the other masks.
    """
    masks = list(_BASE_MASKS)
    if _VAR_RUN.is_dir() and not _VAR_RUN.is_symlink():
        masks.append(str(_VAR_RUN))
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if (
        runtime
        and os.path.isabs(runtime)
        and Path(runtime).is_dir()
        and not any(_is_within(runtime, mask) for mask in masks)
    ):
        masks.append(runtime)
    return tuple(masks)


def escaping_symlinks(masks: Sequence[str]) -> list[tuple[str, str]]:
    """List ``(target, link)`` for top-level symlinks in ``masks`` leaving them.

    A mask would otherwise hide pointer links such as NixOS's
    ``/run/current-system``, whose targets hold programs on ``PATH``. Links into
    a masked dir are skipped, since their targets are hidden anyway, and
    ``/tmp`` is not scanned, since it is meant to start empty.
    """
    links: list[tuple[str, str]] = []
    for mask in masks:
        if mask == "/tmp":
            continue  # a private scratch area; host links there stay hidden
        try:
            entries = list(os.scandir(mask))
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue
            try:
                target = os.readlink(entry.path)
            except OSError:
                continue
            # Windows prefixes an absolute symlink target with \\?\, which
            # would otherwise defeat the plain-string comparison below.
            target = target.removeprefix("\\\\?\\")
            resolved = os.path.normpath(os.path.join(mask, target))
            if not any(_is_within(resolved, other) for other in masks):
                links.append((target, entry.path))
    return links


def _is_within(path: str, parent: str) -> bool:
    return Path(path) == Path(parent) or Path(parent) in Path(path).parents


def bwrap_args(
    spec: IsolationSpec,
    *,
    masks: Sequence[str] = _BASE_MASKS,
    symlinks: Sequence[tuple[str, str]] = (),
    expose: Sequence[str] = (),
) -> list[str]:
    """Build the ``bwrap`` options (without program or ``--``) for ``spec``.

    Mount order matters: later mounts shadow earlier ones, so the masks come
    first, then the preserved symlinks and read-only ``expose`` binds, and the
    writable binds last, keeping a root or scratch dir that lives under a masked
    directory (such as ``/tmp``) visible and writable.

    Args:
        spec: The confinement to enforce.
        masks: Directories replaced by private, empty tmpfs mounts.
        symlinks: ``(target, link)`` pairs recreated inside the masks.
        expose: Host paths bound read-only on top of the masks; missing paths
            are skipped.

    Returns:
        The option list to place between ``bwrap`` and ``--``.
    """
    root, tmpdir = str(spec.root), str(spec.tmpdir)
    args = ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
    for mask in masks:
        args += ["--tmpfs", mask]
    for target, link in symlinks:
        args += ["--symlink", target, link]
    for path in expose:
        args += ["--ro-bind-try", path, path]
    args += [
        "--bind", tmpdir, tmpdir,
        "--bind", root, root,
        "--unshare-pid",
        "--unshare-ipc",
        "--die-with-parent",
        "--new-session",
        "--chdir", root,
    ]  # fmt: skip
    if not spec.allow_network:
        args.append("--unshare-net")
    return args


@cache
def _bwrap_works(executable: str) -> bool:
    """Report whether ``executable`` can start a sandbox with our layout."""
    with tempfile.TemporaryDirectory(prefix="nexus-bwrap-probe-") as scratch:
        path = Path(scratch)
        spec = IsolationSpec(root=path, tmpdir=path, allow_network=False)
        probe = [executable, *bwrap_args(spec), "--", "true"]
        try:
            completed = subprocess.run(
                probe,
                capture_output=True,
                timeout=_PROBE_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
    return completed.returncode == 0


def sbpl_escape(path: Path) -> str:
    r"""Escape ``path`` for use inside an SBPL double-quoted string literal.

    SBPL (TinyScheme) escapes with backslash, so backslashes must be doubled
    before quotes are escaped — otherwise a path byte like ``\"`` would
    terminate the literal early and let trailing bytes parse as directives.
    """
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


# Tried in order when isolation is auto-selected; each gates itself on platform.
_AUTO_BACKENDS: tuple[type[IsolationBackend], ...] = (
    SandboxExecBackend,
    BubblewrapBackend,
)

_NAMED_BACKENDS: dict[str, type[IsolationBackend]] = {
    NoIsolation.name: NoIsolation,
    SandboxExecBackend.name: SandboxExecBackend,
    BubblewrapBackend.name: BubblewrapBackend,
}


def select_backend(
    isolation: IsolationBackend | str = "auto",
    *,
    require: bool = False,
) -> IsolationBackend:
    """Choose the isolation backend for a sandbox.

    ``"auto"`` picks the first available platform backend (``sandbox-exec`` on
    macOS, ``bubblewrap`` on Linux) and falls back to :class:`NoIsolation` when
    none is usable. A named backend or a backend instance is used as given, and
    must be available.

    Args:
        isolation: ``"auto"``, a backend name (``"sandbox-exec"``,
            ``"bubblewrap"``, ``"none"``), or an :class:`IsolationBackend`.
        require: If ``True``, the result must enforce isolation.

    Returns:
        The backend to wrap commands with.

    Raises:
        ValueError: If ``isolation`` is an unknown name, or ``"none"`` (or
            another non-enforcing backend) is combined with ``require``.
        IsolationUnavailable: If an explicitly chosen backend is not available,
            or ``require`` is set and auto-selection found no usable backend.
    """
    if isinstance(isolation, str):
        if isolation == "auto":
            return _auto_backend(require=require)
        try:
            backend = _NAMED_BACKENDS[isolation]()
        except KeyError:
            names = ", ".join(sorted(["auto", *_NAMED_BACKENDS]))
            raise ValueError(
                f"unknown isolation backend {isolation!r}; expected one of {names}"
            ) from None
    else:
        backend = isolation
    if require and not backend.enforced:
        raise ValueError(
            f"isolation backend {backend.name!r} does not enforce isolation, "
            "but require_isolation is set"
        )
    if not backend.available():
        raise IsolationUnavailable(
            f"isolation backend {backend.name!r} is not available on {sys.platform}"
        )
    return backend


def _auto_backend(*, require: bool) -> IsolationBackend:
    for backend_type in _AUTO_BACKENDS:
        backend = backend_type()
        if backend.available():
            return backend
    if require:
        raise IsolationUnavailable(
            f"no OS isolation backend is available on {sys.platform}; {_install_hint()}"
        )
    return NoIsolation()


def _install_hint() -> str:
    if sys.platform.startswith("linux"):
        return "install bubblewrap (bwrap) and allow unprivileged user namespaces"
    if sys.platform == "darwin":
        return "sandbox-exec was not found on PATH"
    return "this platform has no supported backend"
