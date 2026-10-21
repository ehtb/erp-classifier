"""
Configuration module for ERP GL Account Classifier.

The GL taxonomy maps Amazon Business invoices to 8 accounts. The vendor name
(Amazon) tells you nothing; the classifier learns from department, amount, and
item description together.

The accounting rule that matters: IT hardware under 800 EUR goes to GL_6220
(operating expense). At 800 EUR or above, it goes to GL_1400 (capital expenditure).
That threshold is why we oversample prices between 650 and 950 EUR when generating
training data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple


class GLAccount(str, Enum):
    """8 GL account classes for Amazon Business multi-category spend."""

    GL_6220_IT_PERIPHERALS_OPEX = "GL_6220_IT_PERIPHERALS_OPEX"
    GL_1400_IT_HARDWARE_CAPEX = "GL_1400_IT_HARDWARE_CAPEX"
    GL_6400_OFFICE_CONSUMABLES = "GL_6400_OFFICE_CONSUMABLES"
    GL_6410_FACILITY_PANTRY = "GL_6410_FACILITY_PANTRY"
    GL_6150_MARKETING_PROMO = "GL_6150_MARKETING_PROMO"
    GL_6310_WORKPLACE_SAFETY_ERGONOMICS = "GL_6310_WORKPLACE_SAFETY_ERGONOMICS"
    GL_6250_TECH_BOOKS_TRAINING = "GL_6250_TECH_BOOKS_TRAINING"
    GL_6450_CLEANING_MAINTENANCE = "GL_6450_CLEANING_MAINTENANCE"


class Department(str, Enum):
    """Department codes."""

    IT_OPS = "IT_OPS"
    ENG = "ENG"
    MARKETING = "MARKETING"
    HR = "HR"
    FACILITIES = "FACILITIES"


class Language(str, Enum):
    """Language variants for descriptions."""

    EN = "EN"
    DE = "DE"
    NL = "NL"


class TextStyle(str, Enum):
    """Text style variants for realism."""

    FULL_MARKETPLACE_TITLE = "FULL_MARKETPLACE_TITLE"
    SAP_SGTXT_ABBREVIATED = "SAP_SGTXT_ABBREVIATED"
    ASIN_SKU_HEAVY = "ASIN_SKU_HEAVY"
    GENERIC_AMBIGUOUS = "GENERIC_AMBIGUOUS"


@dataclass
class AccountingRule:
    """Defines business logic for GL classification."""

    gl_account: GLAccount
    base_category: str
    threshold_rule: str | None = None

    def applies(self, amount_eur: float, category: str) -> bool:
        """Check if this rule applies to the given context."""
        if self.base_category != category:
            return False

        if self.threshold_rule is None:
            return True

        # Parse threshold rule: "< 800" or ">= 800"
        if "< 800" in self.threshold_rule:
            return amount_eur < 800.0
        elif ">= 800" in self.threshold_rule:
            return amount_eur >= 800.0

        return True


# Accounting rules with explicit IT hardware threshold.
# The 800 EUR line decides whether IT purchases are expensed or capitalized.
# Everything else maps to a single GL account per category.
ACCOUNTING_RULES = [
    AccountingRule(
        gl_account=GLAccount.GL_6220_IT_PERIPHERALS_OPEX,
        base_category="IT_HARDWARE",
        threshold_rule="< 800"  # Below threshold: operating expense
    ),
    AccountingRule(
        gl_account=GLAccount.GL_1400_IT_HARDWARE_CAPEX,
        base_category="IT_HARDWARE",
        threshold_rule=">= 800"  # At or above: capital expenditure
    ),
    AccountingRule(
        gl_account=GLAccount.GL_6400_OFFICE_CONSUMABLES,
        base_category="OFFICE_CONSUMABLES"
    ),
    AccountingRule(
        gl_account=GLAccount.GL_6410_FACILITY_PANTRY,
        base_category="FACILITY_PANTRY"
    ),
    AccountingRule(
        gl_account=GLAccount.GL_6150_MARKETING_PROMO,
        base_category="MARKETING_PROMO"
    ),
    AccountingRule(
        gl_account=GLAccount.GL_6310_WORKPLACE_SAFETY_ERGONOMICS,
        base_category="WORKPLACE_SAFETY"
    ),
    AccountingRule(
        gl_account=GLAccount.GL_6250_TECH_BOOKS_TRAINING,
        base_category="TECH_TRAINING"
    ),
    AccountingRule(
        gl_account=GLAccount.GL_6450_CLEANING_MAINTENANCE,
        base_category="CLEANING_MAINTENANCE"
    ),
]


def get_gl_account(base_category: str, amount_eur: float) -> GLAccount:
    """
    Apply accounting rules to determine GL account.

    Critical rule: IT hardware < 800 EUR -> GL_6220 (OPEX)
                   IT hardware >= 800 EUR -> GL_1400 (CAPEX)
    """
    for rule in ACCOUNTING_RULES:
        if rule.applies(amount_eur, base_category):
            return rule.gl_account

    raise ValueError(f"No GL account rule matched for category={base_category}, amount={amount_eur}")


# Vendor name variants (realistic Amazon Business invoice variations)
VENDOR_VARIANTS = [
    "Amazon EU S.a.r.l.",
    "AMZN Mktp DE",
    "Amazon.nl",
    "Amazon Business EU",
    "Amazon Media EU S.à r.l.",
    "AMAZON EU",
    "Amazon Services Europe",
]


# Category -> Department affinity matrix (realistic business context)
CATEGORY_DEPT_AFFINITY: Dict[str, List[Department]] = {
    "IT_HARDWARE": [Department.IT_OPS, Department.ENG],
    "OFFICE_CONSUMABLES": [Department.IT_OPS, Department.ENG, Department.MARKETING, Department.HR],
    "FACILITY_PANTRY": [Department.FACILITIES, Department.HR],
    "MARKETING_PROMO": [Department.MARKETING],
    "WORKPLACE_SAFETY": [Department.FACILITIES, Department.HR],
    "TECH_TRAINING": [Department.IT_OPS, Department.ENG],
    "CLEANING_MAINTENANCE": [Department.FACILITIES],
}


# Price ranges per category (EUR)
PRICE_RANGES: Dict[str, Tuple[float, float]] = {
    "IT_HARDWARE": (45.0, 2500.0),  # Wide range to cover threshold testing
    "OFFICE_CONSUMABLES": (5.0, 150.0),
    "FACILITY_PANTRY": (10.0, 300.0),
    "MARKETING_PROMO": (25.0, 500.0),
    "WORKPLACE_SAFETY": (15.0, 250.0),
    "TECH_TRAINING": (20.0, 120.0),
    "CLEANING_MAINTENANCE": (8.0, 180.0),
}


# Price sampling weights. We oversample around 800 EUR for IT hardware to test
# whether the model learned the threshold. Without oversampling, too few training
# examples land near the boundary to verify it reliably.
def get_price_distribution_weights(category: str, num_bins: int = 20) -> List[float]:
    """
    Generate price sampling weights with oversampling for IT hardware threshold region.

    For IT_HARDWARE: sample 5x as often between 650-950 EUR (centered on the 800 EUR
    threshold). This gives us enough examples near the boundary to check whether the
    model makes the right call.

    For other categories: sample uniformly across the price range.
    """
    if category != "IT_HARDWARE":
        return [1.0] * num_bins

    min_price, max_price = PRICE_RANGES[category]
    bin_size = (max_price - min_price) / num_bins

    weights = []
    for i in range(num_bins):
        bin_start = min_price + i * bin_size
        bin_end = bin_start + bin_size
        bin_mid = (bin_start + bin_end) / 2

        # Oversample near the 800 EUR threshold
        if 650 <= bin_mid <= 950:
            weights.append(5.0)  # 5x in the core region
        elif 550 <= bin_mid <= 1050:
            weights.append(2.0)  # 2x in the wider band
        else:
            weights.append(1.0)  # Normal sampling elsewhere

    return weights


# Label to index mapping
LABEL2ID = {label.value: idx for idx, label in enumerate(GLAccount)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

NUM_LABELS = len(GLAccount)


# Model configuration
MODEL_CONFIG = {
    "base_model": "distilbert-base-uncased",  # Lightweight, good for small datasets
    # Alternative options:
    # "answerdotai/ModernBERT-base"  # 2x faster than DeBERTa, state-of-the-art (Dec 2024)
    # "microsoft/mdeberta-v3-base"  # Best accuracy, slower
    # "bert-base-uncased"            # Established baseline
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}


# Data generation configuration
DATA_GEN_CONFIG = {
    "total_samples": 10000,  # Scaled from 4000 to 10000
    "test_samples": 160,  # For dry-run testing
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "counterfactual_pairs": 150,  # Scaled from 50 to 150 (proportional to 10k samples)
    "corruption_ratio": 0.3,  # 30% of records get ERP corruption
    "random_seed": 42,
}
