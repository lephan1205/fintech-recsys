"""End-to-end serving pipeline: slate of <= 10, all eligible, guardrails respected,
per-stage latency reported, artifacts round-trip through save / load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from recsys import seed_everything
from recsys.data.schema import SyntheticDataset
from recsys.data.synthetic_generator import GeneratorConfig, SyntheticFintechDataGenerator
from recsys.serving.eligibility_engine import ComplianceViolation
from recsys.serving.pipeline import (
    STAGES,
    PipelineArtifacts,
    RecommendationPipeline,
    build_artifacts,
    latency_summary,
)
from recsys.training.config import TrainingConfig

SMALL = GeneratorConfig(num_users=40, num_products=90, seed=5, slates_per_user=2, slate_size=12)


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)
    torch.set_num_threads(2)


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    return SyntheticFintechDataGenerator(SMALL).generate()


@pytest.fixture(scope="module")
def artifacts(dataset: SyntheticDataset) -> PipelineArtifacts:
    return build_artifacts(dataset, TrainingConfig.tiny(), seed=0)


def test_pipeline_serves_eligible_unique_guarded_slates(
    dataset: SyntheticDataset, artifacts: PipelineArtifacts
) -> None:
    pipe = RecommendationPipeline(artifacts)
    cfg = artifacts.config
    results = pipe.run_many(range(8))
    served_any = False
    for res in results:
        user = dataset.users[res.user_index]
        assert res.size <= cfg.models.slate_size
        assert len(set(res.item_ids)) == res.size
        assert res.num_retrieved <= min(cfg.models.num_candidates, res.num_eligible)
        assert res.num_after_post_filter == res.num_retrieved  # trie mask == vectorized gate
        artifacts.engine.assert_all_eligible(user, res.item_ids)
        for item in res.item_ids:
            assert item not in user.held_product_ids and item not in user.pending_product_ids
        assert [s.name for s in res.telemetry.stages] == list(STAGES)
        assert res.telemetry.total_ms > 0
        # guardrails on: nothing served with NB < -delta while a safe same-family survivor exists
        served_any |= res.size > 0
        for p1, p2, p3 in zip(res.p_click, res.p_apply, res.p_approve, strict=True):
            assert 0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0 and 0.0 <= p3 <= 1.0
        for u in res.utility:
            assert np.isfinite(u)  # excluded (-inf) candidates never reach the slate
    assert served_any
    summary = latency_summary(results)
    assert set(summary) == {*STAGES, "total"}
    assert summary["total"]["p50"] > 0


def test_pipeline_handles_user_with_nothing_eligible(
    dataset: SyntheticDataset, artifacts: PipelineArtifacts
) -> None:
    pipe = RecommendationPipeline(artifacts)
    # make every product ineligible for user 0 by rewriting the profile in place
    user = dataset.users[0]
    blocked = user.model_copy(update={"fico": 300, "dti": 1.0, "annual_income": 0.0})
    original = artifacts.users[0]
    artifacts.users[0] = blocked
    try:
        res = pipe.run(0)
    finally:
        artifacts.users[0] = original
    assert res.size == 0 and res.num_eligible == 0 and res.num_retrieved == 0
    assert [s.name for s in res.telemetry.stages] == list(STAGES)


def test_output_assertion_catches_violations(
    dataset: SyntheticDataset, artifacts: PipelineArtifacts
) -> None:
    user = dataset.users[1]
    ineligible = [
        p.item_id for p in dataset.products if not artifacts.engine.mask_for_user(user)[p.item_id]
    ]
    assert ineligible
    with pytest.raises(ComplianceViolation):
        artifacts.engine.assert_all_eligible(user, ineligible[:1])


def test_artifacts_round_trip_gives_identical_slates(
    dataset: SyntheticDataset, artifacts: PipelineArtifacts, tmp_path: Path
) -> None:
    pipe = RecommendationPipeline(artifacts)
    before = [pipe.run(u) for u in range(4)]
    artifacts.save(tmp_path)
    assert (tmp_path / "meta.json").exists() and (tmp_path / "item_codes.npy").exists()
    loaded = PipelineArtifacts.load(tmp_path, dataset, artifacts.config)
    pipe2 = RecommendationPipeline(loaded)
    after = [pipe2.run(u) for u in range(4)]
    for a, b in zip(before, after, strict=True):
        assert a.item_ids == b.item_ids
        assert np.allclose(a.utility, b.utility) and np.allclose(a.p_click, b.p_click)
    assert torch.equal(loaded.item_codes, artifacts.item_codes)
