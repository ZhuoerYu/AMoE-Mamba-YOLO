"""Pure reporting helpers for DualRoute pilot artifacts."""

from __future__ import annotations


def routing_snapshot(model) -> list[dict[str, object]]:
    """Return one record per real DMS selective-scan shell."""
    records = []
    for name, module in model.named_modules():
        # DMSXSSBlock exposes its nested shell's routing as a convenience property.
        # Persist only the actual shell to avoid counting every neck decision twice.
        if module.__class__.__name__ != "DMSSelectiveShell":
            continue
        routing = getattr(module, "last_routing", None)
        if not routing:
            continue
        records.append(
            {
                "module": name,
                "selected_indices": routing["selected_indices"].cpu().tolist(),
                "entropy": routing["entropy"].cpu().tolist(),
                "expert_calls": list(routing["expert_calls"]),
                "real_selective_scan": routing["real_selective_scan"],
                "scan_backend": routing["scan_backend"],
            }
        )
    return records
