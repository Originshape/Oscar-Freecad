"""Model capabilities: resolution order, host classes, the seed table, learning and persistence."""

import dataclasses
import os
import sys
import threading
import time
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import models, params, provider_probe  # noqa: E402


def _profile(provider="openai_compat", base_url="", model=None, **extra):
    """A Profile with a model name unique to the test, so persisted / mirrored learning never leaks between tests."""
    return params.Profile(name="t", provider=provider, base_url=base_url,
                          model=model or f"m-{uuid.uuid4().hex[:10]}", **extra)


def _forget(profile):
    key = models.caps_key(profile)
    with models._MIRROR_LOCK:
        models._LEARNED.pop(key, None)
        models._RUNTIME.pop(key, None)
        models._PROBE_FAILED.pop(key, None)
    params.set_model_caps(key, {})


class _Cleanup(unittest.TestCase):
    def forget_on_cleanup(self, profile):
        self.addCleanup(_forget, profile)
        return profile


class ModelCapsTest(unittest.TestCase):
    def test_defaults_are_the_contract_defaults(self):
        caps = models.ModelCaps()
        self.assertEqual((caps.tools, caps.reasoning, caps.reasoning_control), ("native", "none", "none"))
        self.assertIsNone(caps.vision)   # UNKNOWN: the provider stays optimistic
        self.assertEqual((caps.context_window, caps.max_output), (32768, 4096))
        self.assertTrue(caps.sampling_ok and caps.tool_choice_ok and caps.parallel_calls and caps.tool_args_stream)
        self.assertTrue(caps.tool_choice_required_with_reasoning)
        self.assertFalse(caps.prompt_cache or caps.reasoning_replay)
        self.assertEqual(caps.source, "default")

    def test_frozen_and_replaceable(self):
        caps = models.ModelCaps()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            caps.tools = "text"
        self.assertEqual(dataclasses.replace(caps, tools="text").tools, "text")

    def test_dict_roundtrip(self):
        caps = models.ModelCaps(tools="text", vision=False, context_window=8192, source="learned")
        self.assertEqual(models.ModelCaps.from_dict(caps.to_dict()), caps)

    def test_from_dict_ignores_unknown_keys_and_bad_values(self):
        caps = models.ModelCaps.from_dict({
            "tools": "bogus", "vision": "yes", "context_window": -5, "max_output": "lots", "sampling_ok": "maybe",
            "reasoning": "stream", "nonsense": 1, "parallel_calls": "false", "prompt_cache": True})
        self.assertEqual(caps.tools, "native")
        self.assertIsNone(caps.vision)
        self.assertEqual((caps.context_window, caps.max_output), (32768, 4096))
        self.assertTrue(caps.sampling_ok)
        self.assertEqual(caps.reasoning, "stream")
        self.assertFalse(caps.parallel_calls)     # "false" text is accepted as a bool
        self.assertTrue(caps.prompt_cache)

    def test_from_dict_accepts_none_and_non_dict(self):
        self.assertEqual(models.ModelCaps.from_dict(None), models.ModelCaps())
        self.assertEqual(models.ModelCaps.from_dict("junk"), models.ModelCaps())

    def test_numeric_coercion_boundaries(self):
        f = models._clean_fields
        self.assertEqual(f({"context_window": "8192"}), {"context_window": 8192})
        self.assertEqual(f({"context_window": 8192.9}), {"context_window": 8192})
        self.assertEqual(f({"context_window": 0}), {})
        self.assertEqual(f({"context_window": 100_000_001}), {})
        self.assertEqual(f({"context_window": True}), {})   # a bool is not a number here
        self.assertEqual(f({"max_output": float("inf")}), {})
        self.assertEqual(f({"vision": None}), {"vision": None})
        self.assertEqual(f({"vision": 1}), {})


class CapsKeyTest(unittest.TestCase):
    def test_key_format_and_trailing_slash(self):
        p = params.Profile(provider="openai_compat", base_url="http://localhost:11434/v1/", model="qwen2.5:7b")
        self.assertEqual(models.caps_key(p), "openai_compat|http://localhost:11434/v1|qwen2.5:7b")

    def test_bare_profile_attributes_read_as_empty(self):
        bare = type("P", (), {"base_url": "https://x.test", "model": "m"})()
        self.assertEqual(models.caps_key(bare), "|https://x.test|m")
        self.assertEqual(models.caps_key(object()), "||")

    def test_non_string_attributes_are_ignored(self):
        weird = type("P", (), {"provider": 3, "base_url": None, "model": ["x"]})()
        self.assertEqual(models.caps_key(weird), "||")


class HostClassTest(unittest.TestCase):
    CASES = [
        ("anthropic", "https://api.anthropic.com", "anthropic"),
        ("anthropic", "", "anthropic"),
        ("openai_compat", "", "openai"),
        ("", "", "openai"),
        ("openai_compat", "https://api.openai.com/v1", "openai"),
        ("openai_compat", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini"),
        ("openai_compat", "https://openrouter.ai/api/v1", "openrouter"),
        ("openai_compat", "https://api.z.ai/api/paas/v4", "zai"),
        ("openai_compat", "https://open.bigmodel.cn/api/paas/v4", "zai"),
        ("openai_compat", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen"),
        ("openai_compat", "https://api.deepseek.com", "deepseek"),
        ("openai_compat", "https://api.groq.com/openai/v1", "groq"),
        ("openai_compat", "https://api.x.ai/v1", "xai"),
        ("openai_compat", "https://api.together.xyz/v1", "together"),
        ("openai_compat", "https://api.fireworks.ai/inference/v1", "fireworks"),
        ("openai_compat", "https://api.mistral.ai/v1", "mistral"),
        ("openai_compat", "https://api.cerebras.ai/v1", "cerebras"),
        ("openai_compat", "https://api.perplexity.ai", "perplexity"),
        ("openai_compat", "http://localhost:11434/v1", "ollama"),
        ("openai_compat", "localhost:11434/v1", "ollama"),
        ("openai_compat", "http://ollama.lan:8080/v1", "ollama"),
        ("openai_compat", "http://localhost:1234/v1", "lmstudio"),
        ("openai_compat", "http://my-lm-studio-box/v1", "lmstudio"),
        ("openai_compat", "https://llm.example.com/v1", "custom"),
        ("openai_compat", "http://host:notaport/v1", "custom"),
        ("some_other", "", "custom"),
        ("originshape", "", "originshape"),                          # the account: its server is not a base URL
        ("originshape", "http://localhost:11434/v1", "originshape"),  # a hand-edited URL changes nothing
    ]

    def test_host_classes(self):
        for provider, url, expected in self.CASES:
            with self.subTest(provider=provider, url=url):
                self.assertEqual(models.host_class(_profile(provider, url)), expected)

    def test_host_comes_from_the_url_not_the_model_name(self):
        self.assertEqual(models.host_class(_profile("openai_compat", "https://llm.example.com/v1", "gpt-5")), "custom")

    def test_uppercase_host_is_normalized(self):
        self.assertEqual(models.host_class(_profile("openai_compat", "https://API.OPENAI.COM/v1")), "openai")


class SeedTableTest(unittest.TestCase):
    def caps(self, provider, base_url, model):
        return models.resolve_caps(_profile(provider, base_url, model))

    def test_claude_models(self):
        for model in ("claude-sonnet-4-5", "claude-opus-4-1-20250805", "claude-haiku-4-5", "claude-fable-1"):
            with self.subTest(model=model):
                c = self.caps("anthropic", "https://api.anthropic.com", model)
                self.assertTrue(c.vision)
                self.assertEqual((c.reasoning, c.reasoning_control), ("stream", "anthropic_budget"))
                self.assertEqual((c.context_window, c.max_output), (200000, 64000))
                self.assertTrue(c.prompt_cache and c.reasoning_replay)
                self.assertFalse(c.tool_choice_required_with_reasoning)
                self.assertEqual(c.source, "table")

    def test_the_originshape_aliases_have_no_rows_of_their_own(self):
        # Their capabilities come from the server's /v1/models (oscar.auth.account), never from this table.
        for alias in ("origin1", "origin1-super", "oscar-default", "oscar-fast", "oscar-unknown"):
            with self.subTest(alias=alias):
                self.assertEqual(self.caps("originshape", "", alias).source, "default")
        self.assertEqual(self.caps("openai_compat", "https://llm.example.com/v1", "oscar-default").source, "default")

    def test_claude_3_has_vision_and_cache_but_no_thinking(self):
        c = self.caps("anthropic", "", "claude-3-5-sonnet-20241022")
        self.assertTrue(c.vision and c.prompt_cache)
        self.assertEqual(c.reasoning_control, "none")
        self.assertEqual(c.max_output, 8192)

    def test_unknown_anthropic_model_keeps_native_tools_and_unknown_vision(self):
        c = self.caps("anthropic", "", "mystery-model")
        self.assertEqual((c.tools, c.source), ("native", "default"))
        self.assertIsNone(c.vision)

    def test_openai_reasoning_models_reject_sampling(self):
        for model in ("o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"):
            with self.subTest(model=model):
                c = self.caps("openai_compat", "https://api.openai.com/v1", model)
                self.assertEqual((c.reasoning, c.reasoning_control), ("hidden", "openai_effort"))
                self.assertFalse(c.sampling_ok)
                self.assertEqual((c.context_window, c.max_output), (400000, 32768))

    def test_openai_gpt4_family_windows(self):
        self.assertEqual(self.caps("openai_compat", "https://api.openai.com/v1", "gpt-4.1-mini").context_window, 1000000)
        c = self.caps("openai_compat", "https://api.openai.com/v1", "gpt-4o")
        self.assertEqual(c.context_window, 128000)
        self.assertTrue(c.vision and c.sampling_ok)

    def test_vendor_prefix_matches_on_the_model_part(self):
        c = self.caps("openai_compat", "https://api.openai.com/v1", "openai/gpt-5")
        self.assertEqual(c.reasoning_control, "openai_effort")


    def test_gemini(self):
        c = self.caps("openai_compat", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-pro")
        self.assertEqual((c.reasoning_control, c.context_window, c.max_output), ("gemini_effort", 1000000, 65536))

    def test_zai_glm_is_text_only_with_thinking_switch(self):
        c = self.caps("openai_compat", "https://api.z.ai/api/paas/v4", "glm-4.6")
        self.assertIs(c.vision, False)
        self.assertEqual(c.reasoning_control, "zai_thinking")
        self.assertEqual(self.caps("openai_compat", "https://api.z.ai/api/paas/v4", "glm-5").reasoning_control, "zai_thinking")
        # an older GLM is not in the table: unknown, optimistic
        self.assertIsNone(self.caps("openai_compat", "https://api.z.ai/api/paas/v4", "glm-4").vision)

    def test_deepseek_qwen_and_xai_rows(self):
        c = self.caps("openai_compat", "https://api.deepseek.com", "deepseek-reasoner")
        self.assertEqual((c.reasoning, c.context_window, c.max_output), ("stream", 128000, 8192))
        self.assertEqual(self.caps("openai_compat", "https://api.deepseek.com", "deepseek-chat").reasoning, "none")
        q = self.caps("openai_compat", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen3-max")
        self.assertEqual((q.reasoning_control, q.context_window), ("qwen_enable_thinking", 131072))
        x = self.caps("openai_compat", "https://api.x.ai/v1", "grok-3-mini")
        self.assertEqual(x.reasoning_control, "xai_effort")
        self.assertEqual(self.caps("openai_compat", "https://api.x.ai/v1", "grok-4").reasoning_control, "none")

    def test_ollama_known_families_are_native_unknown_are_text(self):
        for model in ("llama3.1:8b", "qwen2.5:7b", "qwen3:4b", "mistral-nemo", "command-r", "hermes3", "granite3.1"):
            with self.subTest(model=model):
                c = self.caps("openai_compat", "http://localhost:11434/v1", model)
                self.assertEqual(c.tools, "native")
                self.assertFalse(c.tool_args_stream)
                self.assertEqual(c.context_window, 4096)
        c = self.caps("openai_compat", "http://localhost:11434/v1", "tinyllama")
        self.assertEqual(c.tools, "text")
        self.assertFalse(c.tool_args_stream)

    def test_lmstudio_streams_tool_arguments_until_learned_otherwise(self):
        # migration: LM Studio documents that function names and arguments stream "in pieces" in delta.tool_calls
        # (https://lmstudio.ai/docs/developer/openai-compat/tools); a buffering server is still learned at runtime
        c = self.caps("openai_compat", "http://localhost:1234/v1", "any")
        self.assertTrue(c.tool_args_stream)
        self.assertEqual(c.tools, "native")

    def test_openrouter_and_custom_hosts_keep_defaults_for_unrecognised_models(self):
        # host-aware: "anthropic/claude-..." through OpenRouter is NOT the anthropic row (different wire, different caps)
        c = self.caps("openai_compat", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4")
        self.assertEqual((c.source, c.reasoning_control, c.prompt_cache), ("default", "none", False))
        self.assertIsNone(c.vision)
        self.assertEqual(self.caps("openai_compat", "https://llm.example.com/v1", "x").tools, "native")

    def test_every_table_row_is_well_formed(self):
        for host, pattern, fields in models.MODEL_CAP_TABLE:
            with self.subTest(host=host, pattern=pattern):
                self.assertIsInstance(host, str)
                models._regex(pattern)   # compiles
                cleaned = models._clean_fields(fields)
                self.assertEqual(cleaned, fields)   # a seed row never carries an invalid value
        self.assertEqual(models.MODEL_CAP_TABLE[-1][0], "*")


class ResolutionOrderTest(_Cleanup):
    def make(self):
        p = _profile("openai_compat", "https://llm.example.com/v1")
        self.forget_on_cleanup(p)
        return p

    def seed_api(self, profile, fields):
        params.set_model_caps(models.caps_key(profile), {
            "api": fields, "api_ts": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def test_defaults_when_nothing_matches(self):
        c = models.resolve_caps(self.make())
        self.assertEqual(c, models.ModelCaps())

    def test_api_layer_beats_the_table(self):
        p = _profile("anthropic", "", "claude-sonnet-4-5")
        self.forget_on_cleanup(p)
        self.seed_api(p, {"context_window": 123456})
        c = models.resolve_caps(p)
        self.assertEqual(c.context_window, 123456)
        self.assertEqual(c.max_output, 64000)        # untouched table field survives
        self.assertEqual(c.source, "api")

    def test_learned_beats_api_and_override_beats_learned(self):
        p = self.make()
        self.seed_api(p, {"context_window": 1000, "tools": "native", "vision": True})
        models.learn(p, context_window=2000, vision=False)
        c = models.resolve_caps(p)
        self.assertEqual((c.context_window, c.vision, c.tools), (2000, False, "native"))
        self.assertEqual(c.source, "learned")
        p.caps_override = {"context_window": 3000, "tools": "text"}
        c = models.resolve_caps(p)
        self.assertEqual((c.context_window, c.vision, c.tools), (3000, False, "text"))
        self.assertEqual(c.source, "override")

    def test_each_layer_overrides_only_the_fields_it_names(self):
        p = _profile("openai_compat", "http://localhost:11434/v1", "tinyllama")
        self.forget_on_cleanup(p)
        p.caps_override = {"max_output": 777}
        models.learn(p, vision=False)
        c = models.resolve_caps(p)
        self.assertEqual(c.tools, "text")            # table
        self.assertEqual(c.context_window, 4096)     # table
        self.assertFalse(c.vision)                   # learned
        self.assertEqual(c.max_output, 777)          # override

    def test_probe_argument_replaces_the_stored_api_layer_for_one_call(self):
        p = self.make()
        self.seed_api(p, {"context_window": 1000})
        self.assertEqual(models.resolve_caps(p, probe={"context_window": 5000}).context_window, 5000)
        self.assertEqual(models.resolve_caps(p).context_window, 1000)

    def test_invalid_values_in_any_layer_are_skipped(self):
        p = self.make()
        self.seed_api(p, {"context_window": "huge", "tools": 7, "vision": True})
        p.caps_override = {"max_output": -1, "reasoning_control": "magic"}
        c = models.resolve_caps(p)
        self.assertEqual((c.context_window, c.tools, c.vision, c.max_output, c.reasoning_control),
                         (32768, "native", True, 4096, "none"))

    def test_the_source_field_cannot_be_forged_by_a_layer(self):
        p = self.make()
        p.caps_override = {"source": "override"}
        self.assertEqual(models.resolve_caps(p).source, "default")

    def test_corrupted_settings_never_raise(self):
        p = self.make()
        key = models.caps_key(p)
        for junk in ({"learned": "x", "api": 5, "api_ts": 9}, {"learned": [1], "api": None}):
            params.set_model_caps(key, junk)
            self.assertEqual(models.resolve_caps(p), models.ModelCaps())
            self.assertEqual(models.learned_fields(p), {})
            self.assertEqual(models.probe_metadata(p), {})

    def test_resolution_survives_a_missing_settings_store(self):
        p = self.make()
        with mock.patch.object(models, "_params", return_value=None):
            self.assertEqual(models.resolve_caps(p), models.ModelCaps())
            models.learn(p, vision=False)   # mirror only
            self.assertIs(models.resolve_caps(p).vision, False)

    def test_a_profile_without_attributes_resolves_to_defaults(self):
        self.assertEqual(models.resolve_caps(object()), models.ModelCaps())


class LearnTest(_Cleanup):
    def make(self):
        p = _profile("openai_compat", "https://llm.example.com/v1")
        self.forget_on_cleanup(p)
        return p

    def test_learn_persists_in_the_settings_file(self):
        p = self.make()
        models.learn(p, vision=False, tools="text", context_window=8192)
        stored = params.get_model_caps(models.caps_key(p))
        self.assertEqual(stored["learned"], {"vision": False, "tools": "text", "context_window": 8192})

    def test_learning_survives_a_restart(self):
        p = self.make()
        models.learn(p, vision=False)
        with models._MIRROR_LOCK:
            models._LEARNED.pop(models.caps_key(p), None)   # simulate a new process: only the file remains
        self.assertIs(models.resolve_caps(p).vision, False)

    def test_learn_merges_instead_of_replacing(self):
        p = self.make()
        models.learn(p, vision=False)
        models.learn(p, tool_choice_ok=False)
        learned = models.learned_fields(p)
        self.assertEqual((learned["vision"], learned["tool_choice_ok"]), (False, False))

    def test_invalid_caps_values_are_dropped_but_valid_ones_kept(self):
        p = self.make()
        models.learn(p, tools="bogus", max_output=2048, source="override")
        learned = models.learned_fields(p)
        self.assertEqual(learned, {"max_output": 2048})

    def test_request_level_markers_are_stored_but_are_not_caps_fields(self):
        p = self.make()
        models.learn(p, stream_options_ok=False, max_tokens_cap=1024, note="x", junk=[1, 2])
        learned = models.learned_fields(p)
        self.assertEqual(learned, {"stream_options_ok": False, "max_tokens_cap": 1024, "note": "x"})
        self.assertEqual(models.resolve_caps(p), models.ModelCaps())   # markers never leak into ModelCaps

    def test_empty_learn_is_a_noop(self):
        p = self.make()
        models.learn(p)
        models.learn(p, tools="bogus")
        self.assertEqual(params.get_model_caps(models.caps_key(p)), {})

    def test_a_failing_settings_write_still_updates_the_mirror(self):
        p = self.make()
        with mock.patch.object(params, "update_model_caps", side_effect=OSError("disk full")):
            models.learn(p, vision=False)
        self.assertIs(models.resolve_caps(p).vision, False)
        self.assertEqual(params.get_model_caps(models.caps_key(p)), {})

    def test_learning_is_per_model_and_per_base_url(self):
        a = self.make()
        b = _profile("openai_compat", "https://llm.example.com/v1")
        c = _profile("openai_compat", "https://other.example.com/v1", a.model)
        for p in (b, c):
            self.forget_on_cleanup(p)
        models.learn(a, vision=False)
        self.assertIsNone(models.resolve_caps(b).vision)
        self.assertIsNone(models.resolve_caps(c).vision)

    def test_concurrent_learning_loses_no_update(self):
        p = self.make()
        names = ["stream_options_ok", "parallel_tool_calls_ok", "options_ok", "thinking_ok", "reasoning_ok",
                 "output_config_ok", "tool_choice_ok", "think_ok"]
        threads = [threading.Thread(target=models.learn, args=(p,), kwargs={n: False}) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        stored = params.get_model_caps(models.caps_key(p))["learned"]
        self.assertEqual(set(stored), set(names))

    def test_learned_fields_returns_a_copy(self):
        p = self.make()
        models.learn(p, vision=False)
        models.learned_fields(p)["vision"] = True
        self.assertIs(models.resolve_caps(p).vision, False)


class LiteAndProbeTest(_Cleanup):
    def test_lite_prompt_boundaries(self):
        self.assertFalse(models.lite_prompt_wanted(models.ModelCaps()))
        self.assertTrue(models.lite_prompt_wanted(models.ModelCaps(context_window=16384)))
        self.assertFalse(models.lite_prompt_wanted(models.ModelCaps(context_window=16385)))
        self.assertTrue(models.lite_prompt_wanted(models.ModelCaps(tools="text")))
        self.assertFalse(models.lite_prompt_wanted(models.ModelCaps(tools="none")))
        self.assertTrue(models.lite_prompt_wanted(models.ModelCaps(parallel_calls=False)))

    def test_probe_metadata_without_a_stored_layer_is_empty(self):
        p = _profile()
        self.assertEqual(models.probe_metadata(p), {})

    def test_probe_metadata_returns_a_fresh_stored_api_layer(self):
        p = _profile()
        self.forget_on_cleanup(p)
        params.set_model_caps(models.caps_key(p), {
            "api": {"context_window": 9000, "bogus": 1}, "api_ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.assertEqual(models.probe_metadata(p), {"context_window": 9000})

    def test_probe_metadata_ignores_a_stale_or_malformed_stamp(self):
        p = _profile()
        self.forget_on_cleanup(p)
        key = models.caps_key(p)
        old = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 25 * 3600))
        params.set_model_caps(key, {"api": {"context_window": 9000}, "api_ts": old})
        self.assertEqual(models.probe_metadata(p), {})
        params.set_model_caps(key, {"api": {"context_window": 9000}, "api_ts": "yesterday"})
        self.assertEqual(models.probe_metadata(p), {})
        future = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 3600))
        params.set_model_caps(key, {"api": {"context_window": 9000}, "api_ts": future})
        self.assertEqual(models.probe_metadata(p), {})

    def test_probe_metadata_never_raises(self):
        self.assertEqual(models.probe_metadata(None), {})
        with mock.patch.object(models, "_stored_entry", side_effect=RuntimeError("boom")):
            self.assertEqual(models.probe_metadata(_profile()), {})


class ParamsAccessorsTest(_Cleanup):
    """The settings-file side of capability persistence (``params.get_model_caps`` / ``set_model_caps``)."""

    def test_get_returns_a_deep_copy(self):
        key = f"k|u|{uuid.uuid4().hex}"
        self.addCleanup(params.set_model_caps, key, {})
        params.set_model_caps(key, {"learned": {"vision": False}})
        got = params.get_model_caps(key)
        got["learned"]["vision"] = True
        self.assertIs(params.get_model_caps(key)["learned"]["vision"], False)

    def test_missing_and_removed_entries_read_as_empty(self):
        key = f"k|u|{uuid.uuid4().hex}"
        self.assertEqual(params.get_model_caps(key), {})
        params.set_model_caps(key, {"learned": {"a": 1}})
        params.set_model_caps(key, {})
        self.assertEqual(params.get_model_caps(key), {})

    def test_update_is_an_atomic_read_modify_write_and_none_removes(self):
        key = f"k|u|{uuid.uuid4().hex}"
        self.addCleanup(params.set_model_caps, key, {})
        params.update_model_caps(key, lambda e: {**e, "n": 1})
        params.update_model_caps(key, lambda e: {**e, "m": 2})
        self.assertEqual(params.get_model_caps(key), {"n": 1, "m": 2})
        params.update_model_caps(key, lambda e: None)
        self.assertEqual(params.get_model_caps(key), {})

    def test_default_state_carries_the_key_and_old_files_gain_it(self):
        state = params._read_state()
        self.assertIsInstance(state["model_caps"], dict)


# --------------------------------------------------------------------------------------------------------------
# Tightened vision learning, the runtime layer, metadata probes


class UnambiguousImageRejectionTest(_Cleanup):
    """A size / base64 error must never persist ``vision=False`` (a review finding)."""

    UNAMBIGUOUS = (
        "this model does not support image input", "Image input is not supported by this model",
        "unknown variant `image_url`, expected `text` at line 1 column 98",
        "{\"error\":\"this model is missing data required for image input\"}",
        "No endpoints found that support image input", "messages.content.type is invalid, allowed values: ['text']",
        "Model does not support images. Please use a vision model.", "400: text-only model",
    )
    AMBIGUOUS = (
        "Invalid base64 image data", "image exceeds 5 MB maximum", "Invalid image: could not decode the payload",
        "image dimensions must not exceed 8000 pixels", "unsupported image media type: image/webp",
        "the request is invalid", "", "messages[2].content[1]: invalid value",
    )

    def test_classifier(self):
        for text in self.UNAMBIGUOUS:
            self.assertTrue(models.unambiguous_image_rejection(text), text)
        for text in self.AMBIGUOUS:
            self.assertFalse(models.unambiguous_image_rejection(text), text)
        self.assertFalse(models.unambiguous_image_rejection(None))

    def test_ambiguous_evidence_does_not_persist_vision_false(self):
        p = self.forget_on_cleanup(_profile())
        models.learn(p, vision=False, evidence="Invalid base64 image data")
        self.assertNotIn("vision", models.learned_fields(p))
        self.assertIsNone(models.resolve_caps(p).vision)
        self.assertEqual(params.get_model_caps(models.caps_key(p)), {})

    def test_unambiguous_evidence_persists_it(self):
        p = self.forget_on_cleanup(_profile())
        models.learn(p, vision=False, evidence="this model does not support image input")
        self.assertIs(models.resolve_caps(p).vision, False)
        self.assertIs(params.get_model_caps(models.caps_key(p))["learned"]["vision"], False)

    def test_without_evidence_the_call_is_taken_at_face_value(self):
        p = self.forget_on_cleanup(_profile())
        models.learn(p, vision=False)
        self.assertIs(models.resolve_caps(p).vision, False)

    def test_other_fields_in_the_same_call_survive_a_dropped_vision(self):
        p = self.forget_on_cleanup(_profile())
        models.learn(p, vision=False, tools="text", evidence="image too large")
        self.assertEqual(models.learned_fields(p), {"tools": "text"})

    def test_vision_true_and_other_fields_ignore_the_evidence_filter(self):
        p = self.forget_on_cleanup(_profile())
        models.learn(p, vision=True, evidence="whatever")
        self.assertIs(models.resolve_caps(p).vision, True)


class RuntimeLayerTest(_Cleanup):
    def test_the_runtime_layer_sits_between_learned_and_override(self):
        p = self.forget_on_cleanup(_profile())
        models.learn(p, context_window=8000)
        models.set_runtime_caps(p, context_window=16000)
        self.assertEqual(models.resolve_caps(p).context_window, 16000)
        p.caps_override = {"context_window": 12000}
        self.assertEqual(models.resolve_caps(p).context_window, 12000)

    def test_it_is_never_persisted_and_can_be_cleared(self):
        p = self.forget_on_cleanup(_profile())
        models.set_runtime_caps(p, context_window=16000)
        self.assertEqual(params.get_model_caps(models.caps_key(p)), {})
        models.clear_runtime_caps(p)
        self.assertEqual(models.resolve_caps(p).context_window, 32768)
        models.clear_runtime_caps(p)          # clearing twice is fine

    def test_invalid_values_and_empty_calls_are_ignored(self):
        p = self.forget_on_cleanup(_profile())
        models.set_runtime_caps(p, context_window="lots", tools="x")
        models.set_runtime_caps(p)
        self.assertEqual(models.resolve_caps(p), models.ModelCaps())


class ServerRootTest(unittest.TestCase):
    def test_roots(self):
        for base, want in (("http://localhost:11434/v1", "http://localhost:11434"), ("http://h:1/v1/", "http://h:1"),
                           ("http://h:1/v1/chat/completions", "http://h:1"), ("http://h:1/api/chat", "http://h:1"),
                           ("http://h:1", "http://h:1"), ("", ""), ("https://x.example/api/v1", "https://x.example")):
            self.assertEqual(provider_probe.server_root(base), want, base)


OPENROUTER = {"data": [
    {"id": "vendor/tooly", "context_length": 100000, "architecture": {"input_modalities": ["text", "image"]},
     "top_provider": {"context_length": 90000, "max_completion_tokens": 8192},
     "supported_parameters": ["tools", "tool_choice", "temperature", "reasoning", "include_reasoning"]},
    {"id": "vendor/plain", "context_length": 32000, "architecture": {"input_modalities": ["text"]},
     "top_provider": {"max_completion_tokens": None}, "supported_parameters": ["temperature", "max_tokens"]},
    {"id": "vendor/notool-nochoice", "context_length": 8000, "supported_parameters": ["tools", "temperature"]},
    {"id": "vendor/o-like", "context_length": 200000, "supported_parameters": ["tools", "tool_choice", "reasoning"]},
    {"id": "vendor/bare", "context_length": 4096}, "junk", {"no_id": 1}]}


class ProbeParsersTest(unittest.TestCase):
    def test_openrouter_full_entry(self):
        got = provider_probe.parse_openrouter(OPENROUTER, "vendor/tooly")["fields"]
        self.assertEqual(got, {"context_window": 90000, "max_output": 8192, "tools": "native", "tool_choice_ok": True,
                               "sampling_ok": True, "reasoning": "stream", "reasoning_control": "openrouter_reasoning",
                               "vision": True})

    def test_openrouter_without_tools_means_text_mode_and_text_only_means_no_vision(self):
        got = provider_probe.parse_openrouter(OPENROUTER, "vendor/plain")["fields"]
        self.assertEqual((got["tools"], got["vision"], got["context_window"]), ("text", False, 32000))
        self.assertNotIn("max_output", got)                       # None is not a number
        self.assertNotIn("reasoning_control", got)

    def test_openrouter_tool_choice_and_sampling_flags(self):
        got = provider_probe.parse_openrouter(OPENROUTER, "vendor/notool-nochoice")["fields"]
        self.assertEqual((got["tool_choice_ok"], got["sampling_ok"]), (False, True))
        got = provider_probe.parse_openrouter(OPENROUTER, "vendor/o-like")["fields"]
        self.assertEqual(got["sampling_ok"], False)
        self.assertNotIn("vision", got)                            # modalities not stated: stays unknown

    def test_openrouter_only_states_what_the_entry_states(self):
        self.assertEqual(provider_probe.parse_openrouter(OPENROUTER, "vendor/bare"), {"fields": {"context_window": 4096}})

    def test_openrouter_variant_suffix_falls_back_and_unknown_models_are_empty(self):
        self.assertEqual(provider_probe.parse_openrouter(OPENROUTER, "vendor/plain:free")["fields"]["context_window"], 32000)
        for payload in (OPENROUTER, {}, {"data": "x"}, None, []):
            self.assertEqual(provider_probe.parse_openrouter(payload, "missing/model"), {})

    SHOW = {"capabilities": ["completion", "tools", "vision", "thinking"],
            "model_info": {"general.architecture": "qwen2", "qwen2.context_length": 32768, "qwen2.block_count": 28}}
    PS = {"models": [{"name": "qwen3:8b", "model": "qwen3:8b", "context_length": 4096, "size": 1}]}

    def test_ollama_show_capabilities_and_the_trained_maximum(self):
        got = provider_probe.parse_ollama(self.SHOW, None, "qwen3:8b")
        self.assertEqual(got["fields"], {"tools": "native", "vision": True, "reasoning": "stream",
                                         "reasoning_control": "ollama_effort"})
        self.assertEqual(got["extra"], {"context_max": 32768})
        self.assertNotIn("context_window", got["fields"])         # the trained maximum is NEVER the window

    def test_ollama_ps_gives_the_active_window_of_a_loaded_model(self):
        got = provider_probe.parse_ollama(self.SHOW, self.PS, "qwen3:8b")
        self.assertEqual(got["fields"]["context_window"], 4096)
        self.assertEqual(got["extra"], {"context_max": 32768, "loaded": True})
        got = provider_probe.parse_ollama(self.SHOW, self.PS, "other:1b")
        self.assertNotIn("context_window", got["fields"])         # not loaded: nothing is claimed

    def test_ollama_latest_tag_equivalence(self):
        ps = {"models": [{"name": "llama3.2:latest", "context_length": 2048}]}
        self.assertEqual(provider_probe.parse_ollama(None, ps, "llama3.2")["fields"], {"context_window": 2048})
        self.assertEqual(provider_probe.parse_ollama(None, ps, "LLAMA3.2:latest")["fields"], {"context_window": 2048})

    def test_ollama_capabilities_without_tools_means_text_and_no_vision(self):
        got = provider_probe.parse_ollama({"capabilities": ["completion"]}, None, "m")
        self.assertEqual(got["fields"], {"tools": "text", "vision": False})

    def test_ollama_old_servers_without_capabilities_state_nothing(self):
        self.assertEqual(provider_probe.parse_ollama({"model_info": {}}, {"models": []}, "m"), {})
        self.assertEqual(provider_probe.parse_ollama(None, None, "m"), {})
        self.assertEqual(provider_probe.parse_ollama("x", [], "m"), {})

    LMSTUDIO = {"data": [
        {"id": "qwen2.5-7b-instruct", "type": "llm", "state": "loaded", "max_context_length": 32768,
         "loaded_context_length": 4096, "capabilities": ["tool_use"]},
        {"id": "llava", "type": "vlm", "state": "not-loaded", "max_context_length": 4096, "loaded_context_length": 4096},
        {"id": "embed", "type": "embeddings"}]}

    def test_lmstudio_loaded_model(self):
        got = provider_probe.parse_lmstudio(self.LMSTUDIO, "qwen2.5-7b-instruct")
        self.assertEqual(got["fields"], {"vision": False, "context_window": 4096, "tools": "native"})
        self.assertEqual(got["extra"], {"context_max": 32768})

    def test_lmstudio_unloaded_model_claims_no_window(self):
        got = provider_probe.parse_lmstudio(self.LMSTUDIO, "llava")
        self.assertEqual(got["fields"], {"vision": True})
        self.assertEqual(provider_probe.parse_lmstudio(self.LMSTUDIO, "nope"), {})
        self.assertEqual(provider_probe.parse_lmstudio(None, "x"), {})

    def test_anthropic_model_entry(self):
        got = provider_probe.parse_anthropic({"id": "claude-x", "max_input_tokens": 200000, "max_tokens": 64000,
                                              "capabilities": {"image_input": {"supported": True}}})
        self.assertEqual(got["fields"], {"context_window": 200000, "max_output": 64000, "vision": True})
        self.assertEqual(provider_probe.parse_anthropic({"id": "claude-x", "display_name": "X"}), {})
        self.assertEqual(provider_probe.parse_anthropic({"max_input_tokens": "lots", "capabilities": {"image_input": {"supported": "yes"}}}), {})
        self.assertEqual(provider_probe.parse_anthropic(None), {})


class FakeHttp:
    """Replaces ``provider_probe.http_json``: ``routes`` maps ``(method, url)`` to a payload (None = HTTP error)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, *, headers=None, body=None, timeout=8.0):
        self.calls.append((method, url, dict(headers or {}), body, timeout))
        if (method, url) in self.routes:
            payload = self.routes[(method, url)]
            return (200, payload) if payload is not None else (500, None)
        return 0, None


class ProbeMetadataTest(_Cleanup):
    def patch(self, routes):
        fake = FakeHttp(routes)
        patcher = mock.patch.object(provider_probe, "http_json", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def ollama(self, model="qwen3:8b"):
        return self.forget_on_cleanup(_profile("openai_compat", "http://localhost:11434/v1", model))

    def test_ollama_probe_stores_the_api_layer_and_resolve_caps_sees_it(self):
        fake = self.patch({("POST", "http://localhost:11434/api/show"): ProbeParsersTest.SHOW,
                           ("GET", "http://localhost:11434/api/ps"): ProbeParsersTest.PS})
        p = self.ollama()
        got = models.probe_metadata(p)
        self.assertEqual(got["tools"], "native")
        self.assertEqual(got["context_window"], 4096)
        self.assertEqual([c[:2] for c in fake.calls], [("POST", "http://localhost:11434/api/show"), ("GET", "http://localhost:11434/api/ps")])
        self.assertEqual(fake.calls[0][3], {"model": "qwen3:8b"})
        stored = params.get_model_caps(models.caps_key(p))
        self.assertEqual(stored["api"]["context_max"], 32768)               # the extras live next to the caps fields
        self.assertRegex(stored["api_ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")
        caps = models.resolve_caps(p)
        self.assertEqual((caps.vision, caps.reasoning_control, caps.source), (True, "ollama_effort", "api"))
        self.assertEqual(models.stored_api_extra(p, "context_max"), 32768)

    def test_a_fresh_stored_layer_is_returned_without_any_request(self):
        fake = self.patch({})
        p = self.ollama()
        params.set_model_caps(models.caps_key(p), {"api": {"context_window": 7000}, "api_ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.assertEqual(models.probe_metadata(p), {"context_window": 7000})
        self.assertEqual(fake.calls, [])

    def test_refresh_forces_a_new_request_and_a_stale_layer_is_refreshed(self):
        routes = {("GET", "http://localhost:11434/api/ps"): {"models": [{"name": "qwen3:8b", "context_length": 8192}]}}
        fake = self.patch(routes)
        p = self.ollama()
        key = models.caps_key(p)
        old = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 25 * 3600))
        params.set_model_caps(key, {"api": {"context_window": 1000}, "api_ts": old})
        self.assertEqual(models.probe_metadata(p)["context_window"], 8192)
        calls = len(fake.calls)
        models.probe_metadata(p)
        self.assertEqual(len(fake.calls), calls)                              # fresh now
        models.probe_metadata(p, refresh=True)
        self.assertGreater(len(fake.calls), calls)

    def test_a_failed_probe_returns_empty_keeps_the_old_layer_and_is_remembered_for_a_minute(self):
        fake = self.patch({})
        p = self.ollama()
        old = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 25 * 3600))
        params.set_model_caps(models.caps_key(p), {"api": {"context_window": 1000}, "api_ts": old})
        self.assertEqual(models.probe_metadata(p), {})
        asked = len(fake.calls)
        self.assertEqual(models.probe_metadata(p), {})
        self.assertEqual(len(fake.calls), asked)                              # the negative cache spared the second timeout
        self.assertEqual(models.resolve_caps(p).context_window, 1000)         # the stale layer still applies
        self.assertGreater(len(fake.calls) if models.probe_metadata(p, refresh=True) == {} else 0, asked)

    def test_hosts_without_a_probe_cost_no_request(self):
        fake = self.patch({})
        for base in ("https://api.openai.com/v1", "https://llm.example.com/v1", "https://api.groq.com/openai/v1"):
            p = self.forget_on_cleanup(_profile("openai_compat", base, "gpt-4o"))
            self.assertEqual(models.probe_metadata(p, refresh=True), {})
        self.assertEqual(fake.calls, [])

    def test_openrouter_probe(self):
        fake = self.patch({("GET", "https://openrouter.ai/api/v1/models"): OPENROUTER})
        p = self.forget_on_cleanup(_profile("openai_compat", "https://openrouter.ai/api/v1", "vendor/tooly"))
        got = models.probe_metadata(p)
        self.assertEqual((got["context_window"], got["vision"], got["tools"]), (90000, True, "native"))
        self.assertEqual(fake.calls[0][2].get("Authorization"), None)          # the public listing needs no key

    def test_lmstudio_probe(self):
        self.patch({("GET", "http://localhost:1234/api/v0/models"): ProbeParsersTest.LMSTUDIO})
        p = self.forget_on_cleanup(_profile("openai_compat", "http://localhost:1234/v1", "qwen2.5-7b-instruct"))
        self.assertEqual(models.probe_metadata(p)["context_window"], 4096)
        self.assertEqual(models.stored_api_extra(p, "context_max"), 32768)

    def test_anthropic_probe_needs_a_key_and_sends_the_headers(self):
        url = "https://api.anthropic.com/v1/models/claude-sonnet-4-5"
        fake = self.patch({("GET", url): {"max_input_tokens": 200000, "max_tokens": 64000}})
        keyless = self.forget_on_cleanup(_profile("anthropic", "https://api.anthropic.com", "claude-sonnet-4-5"))
        self.assertEqual(models.probe_metadata(keyless), {})
        self.assertEqual(fake.calls, [])
        keyed = self.forget_on_cleanup(_profile("anthropic", "https://api.anthropic.com", "claude-sonnet-4-5", api_key="sk-test"))
        self.assertEqual(models.probe_metadata(keyed, refresh=True), {"context_window": 200000, "max_output": 64000})
        self.assertEqual((fake.calls[0][2]["x-api-key"], fake.calls[0][2]["anthropic-version"]), ("sk-test", "2023-06-01"))

    def test_anthropic_model_names_with_odd_characters_are_not_put_into_a_url(self):
        fake = self.patch({})
        p = self.forget_on_cleanup(_profile("anthropic", "https://api.anthropic.com", "x/../../admin?y=1", api_key="k"))
        self.assertEqual(models.probe_metadata(p, refresh=True), {})
        self.assertEqual(fake.calls, [])

    def test_a_probe_that_raises_is_swallowed(self):
        p = self.ollama()
        with mock.patch.object(provider_probe, "probe", side_effect=RuntimeError("boom")):
            self.assertEqual(models.probe_metadata(p, refresh=True), {})

    def test_the_probe_never_overrides_the_users_override_or_learned_facts(self):
        self.patch({("POST", "http://localhost:11434/api/show"): ProbeParsersTest.SHOW})
        p = self.ollama()
        models.learn(p, tools="text")
        p.caps_override = {"vision": False}
        models.probe_metadata(p)
        caps = models.resolve_caps(p)
        self.assertEqual((caps.tools, caps.vision), ("text", False))

    def test_http_json_never_raises_and_maps_errors_to_status_zero_or_a_code(self):
        class Resp:
            def __init__(self, status, payload=None, boom=False):
                self.status_code, self._payload, self._boom = status, payload, boom

            def json(self):
                if self._boom:
                    raise ValueError("not json")
                return self._payload

        import requests

        with mock.patch.object(requests, "get", return_value=Resp(200, {"a": 1})):
            self.assertEqual(provider_probe.http_json("GET", "http://x"), (200, {"a": 1}))
        with mock.patch.object(requests, "get", return_value=Resp(404)):
            self.assertEqual(provider_probe.http_json("GET", "http://x"), (404, None))
        with mock.patch.object(requests, "get", return_value=Resp(200, boom=True)):
            self.assertEqual(provider_probe.http_json("GET", "http://x"), (0, None))
        with mock.patch.object(requests, "get", side_effect=requests.ConnectionError("down")):
            self.assertEqual(provider_probe.http_json("GET", "http://x"), (0, None))
        with mock.patch.object(requests, "post", return_value=Resp(200, {"b": 2})) as post:
            self.assertEqual(provider_probe.http_json("POST", "http://x", body={"model": "m"}), (200, {"b": 2}))
        self.assertEqual(post.call_args.kwargs["json"], {"model": "m"})


if __name__ == "__main__":
    unittest.main()
