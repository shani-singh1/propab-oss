"""Layer 0.5 — offline replay, simulation, and microbenchmarks (fixes.md)."""

from propab.layer05.analyst_replay import AnalystReplayResult, replay_analyst_on_history
from propab.layer05.campaign_replay import CampaignReplayResult, replay_campaign_snapshots
from propab.layer05.microbenchmarks import run_all_benchmarks, simulator_bench
from propab.layer05.policy_offline_eval import OfflinePolicyEvalResult, evaluate_policy_offline
from propab.layer05.replay_state import ReplayState, SearchState
from propab.layer05.hybrid_simulator import SIM_V2, simulate_search_hybrid
from propab.layer05.search_simulator import SimulationResult, simulate_search
from propab.layer05.simulator_calibration import CalibrationReport, run_calibration_cycle
from propab.layer05.simulator_registry import SIM_V1, SimulatorRegistry
from propab.layer05.simulation_fitness_ledger import (
    SimulationFitnessLedger,
    SimulationFitnessRecord,
)
from propab.layer05.simulator_bench import SimulatorBenchResult, run_simulator_bench
from propab.layer05.trajectories import (
    BranchingTrajectory,
    CampaignTrajectories,
    ClosureTrajectory,
    EntropyTrajectory,
    ThemeSaturationTrajectory,
)

__all__ = [
    "AnalystReplayResult",
    "BranchingTrajectory",
    "CampaignReplayResult",
    "CampaignTrajectories",
    "ClosureTrajectory",
    "EntropyTrajectory",
    "OfflinePolicyEvalResult",
    "ReplayState",
    "SearchState",
    "SimulationFitnessLedger",
    "SimulationFitnessRecord",
    "SimulationResult",
    "SimulatorBenchResult",
    "ThemeSaturationTrajectory",
    "evaluate_policy_offline",
    "replay_analyst_on_history",
    "replay_campaign_snapshots",
    "run_all_benchmarks",
    "run_simulator_bench",
    "SIM_V1",
    "SIM_V2",
    "CalibrationReport",
    "SimulatorRegistry",
    "run_calibration_cycle",
    "simulate_search",
    "simulate_search_hybrid",
    "simulator_bench",
]
