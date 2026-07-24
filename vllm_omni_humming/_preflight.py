"""Shared preflight checks — verify that the private vLLM-Omni symbols this plugin
patches still exist with the expected signature, and fail LOUD (naming the symbol
and the installed version) when they don't, instead of degrading silently.

The distinction this module encodes:

  * host module not importable YET  -> legitimate; the caller soft-skips (returns
    False). This happens by design in the processes / entry points where vLLM-Omni
    has not been imported at the moment ``register()`` runs.
  * host module imported, but a symbol we patch is MISSING or has an unexpected
    signature -> a genuine version incompatibility -> raise
    ``UnsupportedVllmOmniError``.

Only the second case is fatal. Callers therefore import their host module under a
narrow ``ImportError`` guard (soft-skip) and, once it is in hand, call the
``require_*`` helpers here (loud). Silently no-op'ing a moved symbol would let the
server come up and then load humming checkpoints incorrectly (wrong dtype/shape ->
corrupted images), which is strictly worse than refusing to start.
"""
import importlib.metadata
import inspect
import logging

logger = logging.getLogger("vllm_omni_humming")

_TARGET = "0.24.x"  # the vLLM-Omni line this plugin's patch points are verified against
_MISSING = object()


class UnsupportedVllmOmniError(RuntimeError):
    """The installed vLLM-Omni is outside the range vllm-omni-humming was verified
    against: a private symbol this plugin patches is absent or has changed
    signature. Raised instead of silently skipping the patch."""


def omni_version() -> str:
    """Best-effort installed vLLM-Omni version string ("unknown" if undeterminable)."""
    for dist in ("vllm-omni", "vllm_omni"):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - never let version lookup mask the real error
            break
    return "unknown"


def _fail(symbol: str, detail: str) -> "UnsupportedVllmOmniError":
    """Build (and log at CRITICAL) an UnsupportedVllmOmniError for ``symbol``.

    Returns the exception so callers can ``raise _fail(...)`` at the point of the
    violation (keeps the traceback anchored to the failing patch).
    """
    msg = (
        f"unsupported vLLM-Omni version: installed {omni_version()}, "
        f"vllm-omni-humming targets {_TARGET}. "
        f"Expected upstream symbol `{symbol}` {detail}. "
        f"This plugin patches private vLLM-Omni internals; that symbol has moved or "
        f"changed shape, so serving would load humming checkpoints incorrectly. "
        f"Refusing to continue — pin `vllm-omni=={_TARGET}` or upgrade vllm-omni-humming."
    )
    logger.critical("vllm-omni-humming: %s", msg)
    return UnsupportedVllmOmniError(msg)


def require_attr(obj, name: str, where: str):
    """Return ``getattr(obj, name)``; raise ``UnsupportedVllmOmniError`` if absent."""
    val = getattr(obj, name, _MISSING)
    if val is _MISSING:
        raise _fail(f"{where}.{name}", "to exist, but it is absent")
    return val


def require_params(func, params, where: str) -> None:
    """Raise ``UnsupportedVllmOmniError`` unless ``func``'s signature accepts every
    name in ``params`` by keyword.

    A ``**kwargs``-accepting signature is treated as satisfying the check: upstream
    may forward through ``**kwargs`` and we cannot disprove the contract, so we do
    not raise a false positive on it.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError) as e:
        raise _fail(where, f"to be introspectable, but signature() failed ({e})")
    names = set(sig.parameters)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return
    missing = [p for p in params if p not in names]
    if missing:
        raise _fail(where, f"to accept parameter(s) {missing}, but its signature is {sig}")
