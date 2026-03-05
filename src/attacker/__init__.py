from ._attacker_interface import AttackerInterface
from .lira import LiraOfflineAttacker, LiraOnlineAttacker
from .rmia import RmiaOfflineAttacker, RmiaOnlineAttacker


ATTACKER_HUB = {
    "lira_offline": LiraOfflineAttacker,
    "lira_online": LiraOnlineAttacker,
    "rmia_offline": RmiaOfflineAttacker,
    "rmia_online": RmiaOnlineAttacker,
}


def load_attacker(
    keyword: str,
    config: dict,
    target_model,
    **attacker_kwargs,
) -> AttackerInterface:
    return ATTACKER_HUB[keyword](config, target_model, **attacker_kwargs)
