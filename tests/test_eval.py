from clora.eval import PromptSpec, bootstrap_ci, default_toy_embedders, evaluate
from clora.toy_model import build_toy_model
from clora.train import TrainConfig, train


def test_bootstrap_ci_basic():
    result = bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_resamples=200, seed=0)
    assert result.mean == 1.0
    assert result.ci_low <= result.mean <= result.ci_high


def test_bootstrap_ci_empty():
    result = bootstrap_ci([], n_resamples=100, seed=0)
    assert result.mean != result.mean  # NaN


def test_evaluate_runs_end_to_end():
    model = build_toy_model(channels=8, cond_dim=8, depth=1, seed=0)
    loras = {
        "a": train(TrainConfig(rank=2, alpha=2.0, steps=2, channels=8, cond_dim=8, depth=1, seed=1)),
        "b": train(TrainConfig(rank=2, alpha=2.0, steps=2, channels=8, cond_dim=8, depth=1, seed=2)),
    }
    prompts = [PromptSpec(prompt="a cat", seed=0), PromptSpec(prompt="a dog", seed=1)]
    embedders = default_toy_embedders(channels=8, seq_len=4, cond_len=4, cond_dim=8)

    result = evaluate(model, loras, prompts, embedders=embedders, bootstrap_n=50)

    assert set(result.identity_interference) == {"a", "b"}
    assert set(result.style_interference) == {"a", "b"}
    for score in result.identity_interference.values():
        assert score.ci_low <= score.mean <= score.ci_high
    assert result.prompt_adherence.mean == result.prompt_adherence.mean  # not NaN
