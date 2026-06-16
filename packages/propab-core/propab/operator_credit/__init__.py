"""Operator credit assignment — Experience → Replay → Counterfactuals → Priors."""

from propab.operator_credit.bandit import OperatorBandit
from propab.operator_credit.campaign_corpus import CampaignCorpus, ingest_trajectory_file
from propab.operator_credit.campaign_era import (
    CampaignEraPartition,
    GoldCorpus,
    EraId,
)
from propab.operator_credit.campaign_family_dag import CampaignFamilyDAG
from propab.operator_credit.counterfactual_replay import CounterfactualSpec, run_counterfactual_suite
from propab.operator_credit.credit_cycle import OperatorCreditReport, run_operator_credit_cycle
from propab.operator_credit.db_trace_loader import (
    CampaignDBBundle,
    campaign_ids_from_trajectory,
    extract_traces_from_db_bundle,
    load_bundles_from_db,
    load_bundles_from_trajectory_file,
)
from propab.operator_credit.difference_rewards import DifferenceRewardLedger, OperatorCredit
from propab.operator_credit.hierarchical_credit import CreditLevel, HierarchicalCreditLedger
from propab.operator_credit.operator_bench import OperatorBenchSuite, run_operator_bench_suite
from propab.operator_credit.operator_dag import OperatorDAG
from propab.operator_credit.operator_priors import OperatorPriors
from propab.operator_credit.operator_registry import OPERATOR_FAMILIES, OperatorFamily, OperatorRegistry
from propab.operator_credit.operator_statistics import OperatorStatistics
from propab.operator_credit.operator_trace import NodeOperatorTrace, OperatorTraceLedger
from propab.operator_credit.search_state_v3 import SearchStateV3

__all__ = [
    "OPERATOR_FAMILIES",
    "CampaignCorpus",
    "CampaignDBBundle",
    "CampaignEraPartition",
    "CampaignFamilyDAG",
    "EraId",
    "GoldCorpus",
    "CounterfactualSpec",
    "CreditLevel",
    "DifferenceRewardLedger",
    "HierarchicalCreditLedger",
    "NodeOperatorTrace",
    "OperatorBandit",
    "OperatorBenchSuite",
    "OperatorCredit",
    "OperatorCreditReport",
    "OperatorDAG",
    "OperatorFamily",
    "OperatorPriors",
    "OperatorRegistry",
    "OperatorStatistics",
    "OperatorTraceLedger",
    "SearchStateV3",
    "campaign_ids_from_trajectory",
    "ingest_trajectory_file",
    "extract_traces_from_db_bundle",
    "load_bundles_from_db",
    "load_bundles_from_trajectory_file",
    "run_counterfactual_suite",
    "run_operator_bench_suite",
    "run_operator_credit_cycle",
]
