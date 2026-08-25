from src.llm_mia.plotting import GROUP_SOURCE, GROUP_VARIANT


def test_lira_attack_generated_groups_have_plot_styles() -> None:
    expected_sources = {
        "gen_from_squad_train": "squad_train",
        "gen_from_squad_validation": "squad_validation",
        "gen_from_trivia_validation": "trivia_validation",
    }
    for group, source in expected_sources.items():
        assert GROUP_SOURCE[group] == source
        assert GROUP_VARIANT[group] == "lora_generated"
