from src.boundaries import boundaries
from src.config import (
    Department,
    Language,
    TextStyle,
    get_price_distribution_weights,
)
from src.critic import Critic
from src.dataset_builder import format_for_training
from src.seed_generator import SeedGenerator, SeedSpec
from src.verbalizer import Verbalizer


class FakeLLM:
    def __init__(self, responses=None, fail_batch=False, fail_on_call=None):
        self.responses = list(responses or [])
        self.fail_batch = fail_batch
        self.fail_on_call = fail_on_call
        self.calls = 0

    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=512):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("simulated API failure")
        if self.responses:
            return self.responses.pop(0)
        return "Title: Dell UltraSharp Monitor\nDescription: A 27-inch 4K monitor."

    def generate_batch(
        self, prompts, system_prompt=None, temperature=0.7, max_tokens=512, batch_size=32
    ):
        if self.fail_batch:
            raise RuntimeError("simulated batch failure")
        return [self.generate(p, system_prompt, temperature, max_tokens) for p in prompts]


def _seed(seed_id=0):
    return SeedGenerator(seed=42).generate_seeds(1)[0]


def test_seed_spec_round_trips_through_json_friendly_dict():
    seed = _seed()
    loaded = SeedSpec.from_dict(seed.to_dict())

    assert loaded.department is seed.department
    assert loaded.language is seed.language
    assert loaded.text_style is seed.text_style
    assert loaded.gl_account is seed.gl_account
    assert loaded.to_dict() == seed.to_dict()


def test_loaded_seed_builds_prompt_without_value_error():
    loaded = SeedSpec.from_dict(_seed().to_dict())
    verbalizer = Verbalizer(llm_client=FakeLLM())

    prompt = verbalizer._build_prompt(loaded)
    assert "EUR" in prompt
    assert loaded.language.value in prompt


def test_parse_plain_text_strips_labels_instead_of_dropping_lines():
    title, description = Verbalizer._parse_plain_text(
        "Title: Dell UltraSharp Monitor\nDescription: A 27-inch 4K monitor.", 0
    )
    assert title == "Dell UltraSharp Monitor"
    assert description == "A 27-inch 4K monitor."


def test_parse_plain_text_accepts_line_number_prefixes():
    title, description = Verbalizer._parse_plain_text(
        "Line 1: USB-C hub\nLine 2: Seven-port hub with HDMI output.", 0
    )
    assert title == "USB-C hub"
    assert description == "Seven-port hub with HDMI output."


def test_parse_plain_text_keeps_single_line_fallback():
    title, description = Verbalizer._parse_plain_text("Only one line", 7)
    assert title == description == "Only one line"


def test_verbalize_batch_survives_batch_and_row_failures():
    verbalizer = Verbalizer(llm_client=FakeLLM(fail_batch=True, fail_on_call=2))
    seeds = SeedGenerator(seed=3).generate_seeds(3)

    results = verbalizer.verbalize_batch(seeds, show_progress=False, batch_size=2)

    assert len(results) == len(seeds)
    assert results[0]["description"] == "A 27-inch 4K monitor."
    assert "generation_error" in results[1]
    assert results[2]["description"] == "A 27-inch 4K monitor."


def test_verbalize_one_records_generation_errors():
    verbalizer = Verbalizer(llm_client=FakeLLM(fail_on_call=1))
    result = verbalizer.verbalize_one(_seed())
    assert "generation_error" in result
    assert result["description"] == "Amazon Business item"


def test_price_weights_follow_the_supplied_threshold():
    weights_800 = get_price_distribution_weights("IT_HARDWARE", 20, threshold=800.0)
    weights_1000 = get_price_distribution_weights("IT_HARDWARE", 20, threshold=1000.0)

    core_800 = {i for i, weight in enumerate(weights_800) if weight == 5.0}
    core_1000 = {i for i, weight in enumerate(weights_1000) if weight == 5.0}
    assert core_800 and core_1000
    assert core_800 != core_1000
    assert min(core_1000) > min(core_800)
    # Non-IT categories still sample uniformly.
    assert set(get_price_distribution_weights("OFFICE_CONSUMABLES", 4, threshold=800.0)) == {1.0}


def test_critic_rejects_generation_errors_without_calling_the_llm():
    llm = FakeLLM()
    critic = Critic(llm_client=llm)
    record = {
        "vendor": "Amazon",
        "department": "IT_OPS",
        "net_amount_eur": 750.0,
        "description": "monitor",
        "gl_account": "GL_6220_IT_PERIPHERALS_OPEX",
        "generation_error": "simulated failure",
    }

    verified = critic.verify_batch([record], show_progress=False)

    assert verified[0]["predicted_gl"] == "ERROR"
    assert verified[0]["matches_label"] is False
    assert llm.calls == 0

    single = critic.verify_one(record.copy())
    assert single["predicted_gl"] == "ERROR"
    assert llm.calls == 0


def test_counterfactual_pairs_cross_the_discovered_it_threshold():
    boundary = boundaries("IT_HARDWARE")[0]
    low, high = SeedGenerator(seed=1).generate_counterfactual_pairs(1)[0]

    assert low.base_category == high.base_category == "IT_HARDWARE"
    assert low.net_amount_eur < boundary.amount <= high.net_amount_eur


def test_legacy_format_does_not_leak_price_or_department_and_keeps_seed_zero():
    record = {
        "vendor": "Amazon",
        "department": "IT_OPS",
        "net_amount_eur": 799.0,
        "description": "Dell monitor for 799.00 EUR",
        "gl_account": "GL_6220_IT_PERIPHERALS_OPEX",
        "seed_id": 0,
    }
    formatted = format_for_training(record)

    assert "[NET_AMT]" not in formatted["text"]
    assert "[DEPT]" not in formatted["text"]
    assert "799" not in formatted["text"]
    assert formatted["metadata"]["seed_id"] == 0


def test_enum_fields_are_coerced_from_strings():
    assert SeedSpec(
        seed_id=1,
        base_category="IT_HARDWARE",
        department="IT_OPS",
        vendor="Amazon",
        language="DE",
        text_style="FULL_MARKETPLACE_TITLE",
        net_amount_eur="850.0",
        gl_account="GL_1400_IT_HARDWARE_CAPEX",
    ).department is Department.IT_OPS
    assert SeedSpec(
        seed_id=1,
        base_category="IT_HARDWARE",
        department="IT_OPS",
        vendor="Amazon",
        language="DE",
        text_style="FULL_MARKETPLACE_TITLE",
        net_amount_eur=850.0,
        gl_account="GL_1400_IT_HARDWARE_CAPEX",
    ).language is Language.DE
    assert SeedSpec(
        seed_id=1,
        base_category="IT_HARDWARE",
        department="IT_OPS",
        vendor="Amazon",
        language="DE",
        text_style="FULL_MARKETPLACE_TITLE",
        net_amount_eur=850.0,
        gl_account="GL_1400_IT_HARDWARE_CAPEX",
    ).text_style is TextStyle.FULL_MARKETPLACE_TITLE
