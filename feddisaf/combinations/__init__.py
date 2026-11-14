"""Only base-method + FedDisAF combinations are public execution modes."""

COMBINATION_NAMES = ("FedAvg", "FedNova", "FedProx", "SCAFFOLD", "FedALA", "FedFTG", "FedPop")


def build_combination(config, attention_aggregator=None):
    """Build a combination; the name always denotes a method with FedDisAF enabled."""
    from .fedavg import FedAvgWithFedDisAF
    from .fednova import FedNovaWithFedDisAF
    from .fedprox import FedProxWithFedDisAF
    from .scaffold import SCAFFOLDWithFedDisAF
    from .fedala import FedALAWithFedDisAF
    from .fedftg import FedFTGWithFedDisAF
    from .fedpop import FedPOPWithFedDisAF
    from ..legacy_attention import ServerAttentionAggregator

    registry = dict(zip(COMBINATION_NAMES, (
        FedAvgWithFedDisAF, FedNovaWithFedDisAF, FedProxWithFedDisAF,
        SCAFFOLDWithFedDisAF, FedALAWithFedDisAF, FedFTGWithFedDisAF, FedPOPWithFedDisAF,
    )))
    name = config.get("base_algorithm", "FedAvg")
    if name not in registry:
        raise ValueError("Unknown FedDisAF combination: " + str(name))
    if config.get("feddisaf_enabled", True) is not True:
        raise ValueError("This archive provides FedDisAF combinations only")
    if attention_aggregator is None:
        attention_aggregator = ServerAttentionAggregator(config)
    return registry[name](config, attention_aggregator)


__all__ = ["COMBINATION_NAMES", "build_combination"]
