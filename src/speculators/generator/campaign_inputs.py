"""Pinned trajectory recipes and campaign metadata preparation."""

from speculators.bank.artifacts import write_json
from speculators.bank.config import BankConfig
from speculators.bank.hub import prepare
from speculators.generator.campaign_config import DOMAINS


def campaign_banks(cfg):
    banks = {}
    if cfg.training_recipe is not None:
        recipe = BankConfig.load(cfg.training_recipe)
        if recipe.hub is None or recipe.hub.trajectories_revision == "main":
            raise ValueError("Training recipe requires pinned trajectories metadata")
        for domain in DOMAINS:
            local = recipe.model_copy(deep=True)
            local.dataset.category = domain
            local.output_root = cfg.input_root / domain
            banks[domain] = local
        return banks
    for path in [cfg.bank_config, *cfg.additional_bank_configs]:
        bank = BankConfig.load(path).model_copy(deep=True)
        if bank.hub is None or bank.hub.trajectories_revision == "main":
            raise ValueError("Campaign requires pinned consumer bank configs")
        domain = bank.dataset.category
        if domain in banks:
            raise ValueError(f"Duplicate campaign category: {domain}")
        bank.hub.mode = "consumer"
        bank.hub.selection_root = cfg.input_root / "selection"
        bank.execution.activation_local_dir = None
        bank.output_root = cfg.input_root / domain
        banks[domain] = bank
    if set(banks) != set(DOMAINS):
        raise ValueError(f"Campaign requires categories {DOMAINS}")
    first = next(iter(banks.values()))
    for bank in banks.values():
        if (
            bank.hub.trajectories_repo,
            bank.hub.trajectories_revision,
        ) != (
            first.hub.trajectories_repo,
            first.hub.trajectories_revision,
        ):
            raise ValueError("Campaign recipes must share pinned trajectory sources")
    return banks


def prepare_inputs(cfg):
    banks = campaign_banks(cfg)
    cfg.input_root.mkdir(parents=True, exist_ok=True)
    for domain, bank in banks.items():
        manifest = prepare(bank, "hidden")
        for number in range(6):
            name = f"{domain}-{number:05d}"
            subset = next((s for s in manifest["subsets"] if s["id"] == name), None)
            if subset is None:
                raise ValueError(f"Missing campaign subset {name}")
            for split, count in (
                ("train", cfg.expected_train_examples),
                ("validation", cfg.expected_validation_examples),
            ):
                if len(subset[f"{split}_indices"]) != count:
                    raise ValueError(f"Unexpected {split} count for {name}")
        write_json(cfg.input_root / f"bank_{domain}.json", bank.model_dump(mode="json"))
    return banks
