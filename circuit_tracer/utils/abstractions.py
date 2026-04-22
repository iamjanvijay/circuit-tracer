"""Graph abstractions computed at dump time.

Each abstraction takes the fully built ``Model`` (post-pruning, with node
``influence`` / ``raw_influence`` already populated) and returns a new ``Model``
that represents a higher-level view of the same attribution. The original
``Model`` is never mutated.

Abstractions live alongside the base graph as ``{slug}__{name}.json`` files;
the base graph ``{slug}.json`` is always the ``none`` abstraction.
"""

from __future__ import annotations

from typing import Callable

from circuit_tracer.frontend.graph_models import Metadata, Model, Node, QParams

NONE = "none"
TOKEN_LEVEL = "token_level"


def _clone_metadata(meta: Metadata, *, abstraction: str) -> Metadata:
    data = meta.model_dump()
    data["abstraction"] = abstraction
    return Metadata(**data)


def _clone_qparams(qparams: QParams) -> QParams:
    return QParams(**qparams.model_dump())


def _token_level(model: Model) -> Model:
    """Token-level abstraction.

    Restricts focus to token nodes. Each token node's ``raw_influence`` is the
    aggregate contribution over all paths from that token to the target logit
    (already computed via the power-series influence in ``graph.py``). The
    abstracted graph has one direct edge per token to the target logit, with
    edge weight equal to that aggregate, making "which input position drove
    the generated token" readable at a glance.

    ``influence`` on each token node is **recomputed over the token set only**:
    tokens are ranked by ``|raw_influence|`` descending, and each token's
    ``influence`` is set to the cumulative share of total token |raw_influence|
    up to and including its rank (``Σ_{i≤rank} |raw[i]| / Σ_total`` ∈ [0, 1]).
    The base-graph ``influence`` is computed over the full graph (features +
    errors + tokens), so token-only cumulative needs this separate pass.
    """
    target = next((n for n in model.nodes if n.is_target_logit), None)
    if target is None:
        return model

    token_nodes = [n for n in model.nodes if n.feature_type == "embedding"]

    # Sort indices by |raw_influence| desc; ties broken by ctx_idx asc for stability.
    def mag(n: Node) -> float:
        return abs(n.raw_influence) if n.raw_influence is not None else 0.0

    total = sum(mag(n) for n in token_nodes)
    order = sorted(range(len(token_nodes)), key=lambda i: (-mag(token_nodes[i]), token_nodes[i].ctx_idx))
    cum_by_id: dict[str, float] = {}
    running = 0.0
    for rank, i in enumerate(order):
        running += mag(token_nodes[i])
        cum_by_id[token_nodes[i].node_id] = (running / total) if total > 0 else 0.0

    kept_nodes: list[Node] = []
    links: list[dict] = []
    for n in token_nodes:
        data = n.model_dump()
        data["influence"] = cum_by_id[n.node_id]
        kept_nodes.append(Node(**data))
        weight = n.raw_influence if n.raw_influence is not None else 0.0
        links.append({"source": n.node_id, "target": target.node_id, "weight": weight})
    kept_nodes.append(target)

    return Model(
        metadata=_clone_metadata(model.metadata, abstraction=TOKEN_LEVEL),
        qParams=_clone_qparams(model.qParams),
        nodes=kept_nodes,
        links=links,
    )


_REGISTRY: dict[str, Callable[[Model], Model]] = {
    TOKEN_LEVEL: _token_level,
}


def available() -> list[str]:
    """Names of all abstractions, with ``none`` first."""
    return [NONE] + sorted(_REGISTRY.keys())


def apply(model: Model, name: str) -> Model:
    """Apply an abstraction by name. ``none`` returns a metadata-tagged clone."""
    if name == NONE:
        return Model(
            metadata=_clone_metadata(model.metadata, abstraction=NONE),
            qParams=_clone_qparams(model.qParams),
            nodes=list(model.nodes),
            links=list(model.links),
        )
    if name not in _REGISTRY:
        raise ValueError(f"unknown abstraction {name!r}; available: {available()}")
    return _REGISTRY[name](model)
