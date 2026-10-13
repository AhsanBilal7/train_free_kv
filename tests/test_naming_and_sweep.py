"""Tests of the run-folder names (scripts/run_naming.py) and the sweep expansion (scripts/sweep.py)."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import run_naming  # noqa: E402
import sweep  # noqa: E402

CORE = dict(mode="causal_lookahead", model_tag="Mistral-7B-Instruct-v0.2", budget="128",
            chunk_size="4", num_particles="4", lookahead_len="8", lambda_lookahead="1.0",
            lookahead_temperature="1.0", beta="1.0", gamma="1.3", lambda_d="0.0",
            normalize_typicality="0", robust_alpha="0.0", robust_topk_beta="1.0",
            lambda_k="0.0", lambda_v="0.0", budget_tau="0.0", score_objective="projected")


def test_core_run_name_matches_its_config():
    """The core configuration gives the name of its config file and parses back."""
    name = run_naming.build_run_name(**CORE)
    assert name == ("lorekv-ent-unif-topk_B_128_chunk_4_M_4_T_8_lL_1.0_lT_1.0_nrm_0"
                    "_alpha_0.0_kappa_1.0")
    p = run_naming.parse_run_name(name)
    assert (p["cell"], p["brule"], p["sel"], p["obj"]) == ("ent", "unif", "topk", "projected")
    assert (p["b"], p["g"], p["lD"]) == (1.0, 1.3, 0.0)


def test_identity_follows_the_numbers_not_the_file():
    """The run name comes from the hyperparameter values."""
    llama_dis = dict(CORE, model_tag="Llama-3.1-8B-Instruct", gamma="1.0", lambda_d="0.3")
    assert run_naming.build_run_name(**llama_dis).startswith("lorekv-dis-unif-topk-llama_B_128")
    ablation = dict(CORE, score_objective="attention", budget_tau="1.0")
    assert run_naming.build_run_name(**ablation).startswith("lorekv-ent-adap-topk_")
    assert run_naming.build_run_name(**ablation).endswith("_obj_attention")


def test_baselines_and_legacy_names_parse():
    """Baseline names and older ``_red_`` names parse."""
    assert run_naming.build_run_name(mode="andpro", model_tag="Qwen2.5-14B-Instruct",
                                     budget=256, chunk_size=4) == "andpro_Qwen2.5-14B-Instruct_256_chunk_4"
    p = run_naming.parse_run_name("critical_Llama-3.1-8B-Instruct_512_chunk_4_red_0.0")
    assert (p["mode"], p["budget"]) == ("critical", "512")


def test_every_config_parses_and_names_its_runs():
    """Each config in ``config/`` gives run names that parse."""
    cfg_dir = os.path.join(REPO, "config")
    for fn in sorted(os.listdir(cfg_dir)):
        s = sweep.Settings(sweep.load_conf(os.path.join(cfg_dir, fn)))
        rows = list(sweep.combinations(s))
        assert rows, fn
        tag = os.path.basename(s.get("MODEL_PATH"))
        for row in rows:
            row["RUN_NAME_PREFIX"] = s.get("RUN_NAME_PREFIX") or None
            name = sweep.run_name(row, tag, "", "/nonexistent")
            parsed = run_naming.parse_run_name(name)
            assert parsed is not None, name
            if row["MODE"] == "causal_lookahead" and not row["RUN_NAME_PREFIX"] \
                    and row["SCORE_OBJECTIVE"] == "projected":
                assert name.split("_B_")[0] == fn[:-5].replace("-qwen35", "-Qwen3.5-9B") \
                    .replace("-gemma4", "-gemma-4-12B-it"), (fn, name)


def test_environment_overrides_config(monkeypatch):
    """Environment variables replace the config values."""
    monkeypatch.setenv("BUDGET", "128 512")
    monkeypatch.setenv("NUM_PARTICLES", "1 4")
    s = sweep.Settings(sweep.load_conf(os.path.join(REPO, "config", "lorekv-ent-unif-topk.conf")))
    rows = list(sweep.combinations(s))
    assert [(r["BUDGET"], r["NUM_PARTICLES"]) for r in rows] == \
        [("128", "1"), ("128", "4"), ("512", "1"), ("512", "4")]
