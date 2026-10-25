"""Provider wire shapes pinned against OFFICIAL DOCUMENTATION.

Every test names the documentation page its expectation comes from. Nothing here touches the network: bodies are built
with ``_body`` and 400 adaptation is driven with ``_adapt_to_rejection`` and the exact error texts the vendors document.

Sources: platform.claude.com/docs/en/build-with-claude/{extended-thinking,thinking,thinking-troubleshooting,effort,
prompt-caching}, .../agents-and-tools/tool-use/define-tools, .../api/handling-stop-reasons, .../api/models/retrieve;
docs.z.ai chat-completion; openrouter.ai/docs/use-cases/reasoning-tokens; api-docs.deepseek.com thinking_mode;
alibabacloud.com model-studio deep-thinking; github.com/ollama/ollama docs/api.md + docs.ollama.com openai-compatibility;
lmstudio.ai openai-compat tools; the OpenAI OpenAPI spec.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import test_providers as tp  # noqa: E402
from oscar import models, provider_probe, provider_reasoning as pr, providers  # noqa: E402
from oscar.provider_stream import stop_info  # noqa: E402

HIST = [providers.Message(role="user", content="hi")]
TOOL = tp.TOOL


def _loop_with_thinking():
    """An open tool loop whose assistant turn replays a signed thinking block."""
    turn = providers.Message(role="assistant", provider_blocks=[
        {"type": "thinking", "thinking": "t", "signature": "sig"},
        {"type": "tool_use", "id": "t1", "name": "create_object", "input": {}}],
        tool_calls=[providers.ToolCall("t1", "create_object")])
    return [providers.Message(role="user", content="go"), turn,
            providers.Message(role="user", tool_results=[providers.ToolResult("t1", "create_object", "ok")])]


class ClaudeIdentityTest(unittest.TestCase):
    def test_version_parsing(self):
        cases = {
            "claude-opus-4-5-20251101": ("opus", 4, 5), "claude-opus-4-20250514": ("opus", 4, 0),
            "claude-sonnet-4-6": ("sonnet", 4, 6), "claude-sonnet-5": ("sonnet", 5, 0),
            "claude-sonnet-5-5": ("sonnet", 5, 5), "anthropic/claude-sonnet-5.5": ("sonnet", 5, 5),
            "claude-fable-5-1": ("fable", 5, 1), "claude-mythos-preview": ("mythos", 0, 0),
            "claude-haiku-4-5": ("haiku", 4, 5), "CLAUDE-OPUS-4-8": ("opus", 4, 8),
        }
        for model, want in cases.items():
            with self.subTest(model=model):
                self.assertEqual(pr.claude_version(model), want)
        for junk in ("", "gpt-5", "claude-3-5-sonnet-20241022", None):
            with self.subTest(junk=junk):
                self.assertIsNone(pr.claude_version(junk))

    def test_forced_tool_choice_is_rejected_only_by_the_four_documented_models(self):
        for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"):
            self.assertFalse(pr.anthropic_forced_choice_ok(model), model)
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-sonnet-4-5", "claude-fable-5", "", "x"):
            self.assertTrue(pr.anthropic_forced_choice_ok(model), model)

    def test_which_models_can_be_switched_off(self):
        # thinking-troubleshooting table: "disabled" is a 400 on Opus 5.5 / Fable / Mythos 5.x; Sonnet 5.5 takes between_tools
        self.assertEqual(pr.anthropic_thinking_off("claude-opus-4-8"), "disabled")
        self.assertEqual(pr.anthropic_thinking_off("claude-sonnet-4-6"), "disabled")
        self.assertEqual(pr.anthropic_thinking_off("claude-sonnet-5"), "disabled")
        self.assertEqual(pr.anthropic_thinking_off("claude-opus-5"), "disabled")
        self.assertEqual(pr.anthropic_thinking_off("claude-sonnet-5-5"), "between_tools")
        for model in ("claude-opus-5-5", "claude-fable-5-1", "claude-fable-5", "claude-mythos-5", "claude-mythos-5-1"):
            self.assertEqual(pr.anthropic_thinking_off(model), "", model)
        self.assertEqual(pr.anthropic_thinking_off("unknown"), "disabled")


class AdaptiveFragmentTest(unittest.TestCase):
    def test_effort_levels_use_adaptive_plus_output_config(self):
        # effort page: output_config.effort low|medium|high (no beta header); thinking page: type "adaptive"
        for level in ("low", "medium", "high"):
            self.assertEqual(pr.reasoning_fragment("anthropic_adaptive", level, model="claude-opus-4-8"),
                             {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}})

    def test_adaptive_never_carries_a_budget(self):
        frag = pr.reasoning_fragment("anthropic_adaptive", "high", model="claude-sonnet-5", max_tokens=64000)
        self.assertNotIn("budget_tokens", frag["thinking"])

    def test_off_spelling_per_model(self):
        off = lambda m: pr.reasoning_fragment("anthropic_adaptive", "off", model=m)   # noqa: E731
        self.assertEqual(off("claude-opus-4-7"), {"thinking": {"type": "disabled"}})
        self.assertEqual(off("claude-sonnet-5-5"), {"thinking": {"type": "between_tools"}})
        for model in ("claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5"):
            self.assertEqual(off(model), {"output_config": {"effort": "low"}}, model)

    def test_auto_and_junk_send_nothing(self):
        self.assertEqual(pr.reasoning_fragment("anthropic_adaptive", None, model="claude-opus-4-8"), {})
        self.assertEqual(pr.reasoning_fragment("anthropic_adaptive", "auto", model="claude-opus-4-8"), {})

    def test_output_config_is_a_reasoning_field_the_adaptation_may_drop(self):
        self.assertIn("output_config", pr.REASONING_FIELDS)


class AnthropicTableTest(unittest.TestCase):
    def caps(self, model):
        p = type("P", (), {"provider": "anthropic", "base_url": "https://api.anthropic.com", "model": model,
                           "caps_override": {}})()
        return models.resolve_caps(p)

    def test_scheme_per_model_family(self):
        adaptive = ("claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6", "claude-sonnet-5",
                    "claude-sonnet-5-5", "claude-opus-5", "claude-opus-5-5", "claude-fable-5", "claude-fable-5-1",
                    "claude-mythos-5", "claude-mythos-preview")
        budget = ("claude-opus-4-5-20251101", "claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1-20250805",
                  "claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-fable-1")
        for model in adaptive:
            with self.subTest(model=model):
                c = self.caps(model)
                self.assertEqual(c.reasoning_control, "anthropic_adaptive")
                self.assertTrue(c.vision and c.prompt_cache and c.reasoning_replay)
        for model in budget:
            with self.subTest(model=model):
                c = self.caps(model)
                self.assertEqual(c.reasoning_control, "anthropic_budget")
                self.assertFalse(c.tool_choice_required_with_reasoning)   # manual thinking forbids forced tool use

    def test_sonnet_55_output_ceiling_is_documented(self):
        self.assertEqual(self.caps("claude-sonnet-5-5").max_output, 128000)       # effort page: "128,000, the model's maximum"
        self.assertEqual(self.caps("claude-sonnet-5").max_output, 64000)

    def test_claude_3_has_no_thinking_control(self):
        self.assertEqual(self.caps("claude-3-5-sonnet-20241022").reasoning_control, "none")


class AnthropicAdaptiveBodyTest(tp._ProviderCase):
    def test_adaptive_models_get_adaptive_thinking_and_effort_not_a_budget(self):
        body = self.anthropic("claude-opus-4-8")._body("s", HIST, [], reasoning="medium")
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "medium"})
        self.assertNotIn("temperature", body)
        self.assertEqual(body["max_tokens"], 8192)          # thinking tokens count toward max_tokens: untouched

    def test_budget_models_keep_the_manual_scheme_and_its_rules(self):
        body = self.anthropic("claude-sonnet-4-5")._body("s", HIST, [], reasoning="high")
        self.assertEqual(body["thinking"], {"type": "enabled", "budget_tokens": 7168})   # >= 1024 and < max_tokens
        self.assertNotIn("output_config", body)

    def test_no_reasoning_request_sends_neither_field(self):
        body = self.anthropic("claude-sonnet-5")._body("s", HIST, [])
        self.assertNotIn("thinking", body)
        self.assertNotIn("output_config", body)

    def test_off_on_a_model_that_cannot_be_disabled_sends_low_effort_only(self):
        body = self.anthropic("claude-fable-5-1")._body("s", HIST, [], reasoning="off")
        self.assertNotIn("thinking", body)
        self.assertEqual(body["output_config"], {"effort": "low"})

    def test_off_inside_an_open_loop_keeps_thinking_on_at_low_effort(self):
        # thinking page: "you can't toggle thinking in the middle of a turn" (it silently disables thinking)
        body = self.anthropic("claude-opus-4-8")._body("s", _loop_with_thinking(), [], reasoning="off")
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "low"})
        fresh = self.anthropic("claude-opus-4-8")._body("s", HIST, [], reasoning="off")
        self.assertEqual(fresh["thinking"], {"type": "disabled"})

    def test_cache_control_sits_on_the_system_block_and_the_last_tool(self):
        # prompt-caching page: breakpoints on tools / system / messages blocks, at most 4 per request
        body = self.anthropic("claude-opus-4-8")._body("sys", HIST, [TOOL, tp.TOOL2])
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", body["tools"][0])
        self.assertEqual(body["tools"][-1]["cache_control"], {"type": "ephemeral"})


class AnthropicForcedChoiceTest(tp._ProviderCase):
    def choice(self, model, **opts):
        return self.anthropic(model)._body("s", HIST, [TOOL], **opts).get("tool_choice")

    def test_the_four_models_never_get_a_forced_choice(self):
        for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"):
            with self.subTest(model=model):
                self.assertIsNone(self.choice(model, tool_choice="required"))
                self.assertIsNone(self.choice(model, tool_choice="create_object", reasoning="off"))
                self.assertEqual(self.choice(model, tool_choice="none"), {"type": "none"})   # "none" is always allowed

    def test_adaptive_thinking_allows_a_forced_choice_elsewhere(self):
        self.assertEqual(self.choice("claude-opus-5", tool_choice="required", reasoning="medium"), {"type": "any"})
        self.assertEqual(self.choice("claude-opus-4-8", tool_choice="create_object", reasoning="high"),
                         {"type": "tool", "name": "create_object"})

    def test_manual_thinking_still_downgrades_a_forced_choice(self):
        self.assertIsNone(self.choice("claude-sonnet-4-5", tool_choice="required", reasoning="medium"))
        self.assertEqual(self.choice("claude-sonnet-4-5", tool_choice="required"), {"type": "any"})


class AnthropicThinkingAdaptationTest(tp._ProviderCase):
    ENABLED = ('"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and '
               '"output_config.effort" to control thinking behavior.')

    def test_enabled_rejection_switches_the_model_to_adaptive(self):
        p = self.anthropic("claude-sonnet-4-5", )
        body = p._body("s", HIST, [], reasoning="medium")
        self.assertEqual(body["thinking"]["type"], "enabled")
        self.assertTrue(p._adapt_to_rejection(self.ENABLED, body))
        again = p._body("s", HIST, [], reasoning="medium")
        self.assertEqual(again["thinking"], {"type": "adaptive"})
        self.assertEqual(again["output_config"], {"effort": "medium"})
        self.assertEqual(providers.AnthropicProvider(p.profile)._caps().reasoning_control, "anthropic_adaptive")   # learned for good

    def test_enabled_rejection_is_not_retried_when_already_adaptive(self):
        p = self.anthropic("claude-opus-4-8")
        body = p._body("s", HIST, [], reasoning="medium")
        self.assertFalse(p._adapt_anthropic_thinking(self.ENABLED.lower(), body))   # the documented remedy does not apply

    def test_adaptive_rejection_switches_back_to_the_manual_budget(self):
        p = self.anthropic("claude-opus-4-8")
        body = p._body("s", HIST, [], reasoning="low")
        self.assertTrue(p._adapt_to_rejection("adaptive thinking is not supported on this model", body))
        self.assertEqual(p._body("s", HIST, [], reasoning="low")["thinking"], {"type": "enabled", "budget_tokens": 1024})

    def test_disabled_rejection_turns_off_into_low_effort(self):
        p = self.anthropic("claude-opus-4-8")
        body = p._body("s", HIST, [], reasoning="off")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        text = ('"thinking.type.disabled" is not supported for this model. Use "thinking.type.adaptive" and '
                '"output_config.effort" to control thinking behavior.')
        self.assertTrue(p._adapt_to_rejection(text, body))
        again = p._body("s", HIST, [], reasoning="off")
        self.assertNotIn("thinking", again)
        self.assertEqual(again["output_config"], {"effort": "low"})
        self.assertFalse(p._adapt_anthropic_thinking(text.lower(), again))   # the documented remedy is used up: no endless retry

    def test_signature_binding_error_strips_replayed_thinking_once(self):
        p = self.anthropic("claude-opus-5-5")
        loop = _loop_with_thinking()
        body = p._body("s", loop, [], reasoning="medium")
        self.assertEqual(body["messages"][1]["content"][0]["type"], "thinking")
        text = ("messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different "
                "conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\".")
        self.assertTrue(p._adapt_to_rejection(text, body))
        stripped = p._body("s", loop, [], reasoning="medium")
        kinds = [b["type"] for b in stripped["messages"][1]["content"]]
        self.assertEqual(kinds, ["tool_use"])
        self.assertNotIn("thinking", stripped)                       # and no thinking is requested any more
        self.assertFalse(p._adapt_to_rejection(text, stripped))      # a second identical failure is surfaced, not looped

    def test_modified_block_error_is_handled_the_same_way(self):
        p = self.anthropic("claude-sonnet-4-5")
        body = p._body("s", _loop_with_thinking(), [], reasoning="low")
        text = "`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified"
        self.assertTrue(p._adapt_to_rejection(text, body))
        self.assertEqual([b["type"] for b in p._body("s", _loop_with_thinking(), [])["messages"][1]["content"]], ["tool_use"])

    def test_unrelated_thinking_text_is_not_misread(self):
        p = self.anthropic("claude-sonnet-4-5")
        body = p._body("s", HIST, [], reasoning="low")
        self.assertFalse(p._adapt_anthropic_thinking("please keep thinking about it", body))


class StopReasonDocsTest(unittest.TestCase):
    def test_documented_anthropic_and_zai_reasons(self):
        info = lambda raw: stop_info(raw, None, "openai")   # noqa: E731
        self.assertEqual(info("model_context_window_exceeded")["reason"], "max_tokens")
        self.assertTrue(info("model_context_window_exceeded")["truncated"])
        self.assertEqual(info("sensitive")["reason"], "content_filter")
        self.assertEqual(info("network_error")["reason"], "error")
        self.assertEqual(info("refusal")["reason"], "refusal")
        self.assertEqual(info("end_turn")["reason"], "end_turn")
        self.assertEqual(info("tool_use")["reason"], "tool_use")
        self.assertEqual(info("something_new")["reason"], "unknown")
        self.assertEqual(info("model_context_window_exceeded")["raw"], "model_context_window_exceeded")



class HostRulesDocsTest(tp._ProviderCase):
    def test_zai_temperature_is_clamped_to_its_documented_range(self):
        base = "https://api.z.ai/api/paas/v4"
        for given, want in ((1.5, 1.0), (-0.5, 0.0), (0.6, 0.6)):
            with self.subTest(given=given):
                body = self.openai("glm-4.6", base_url=base, temperature=given)._body("s", HIST, [])
                self.assertEqual(body["temperature"], want)
        other = self.openai("gpt-4o", temperature=1.5)._body("s", HIST, [])
        self.assertEqual(other["temperature"], 1.5)             # only Z.ai documents the 0..1 range

    def test_zai_row_documents_auto_only_tool_choice_and_buffered_arguments(self):
        c = models.resolve_caps(self.profile("openai_compat", "https://api.z.ai/api/paas/v4", "glm-4.6"))
        self.assertFalse(c.tool_choice_ok)
        self.assertFalse(c.tool_args_stream)
        body = self.openai("glm-4.6", base_url="https://api.z.ai/api/paas/v4")._body("s", HIST, [TOOL], tool_choice="required")
        self.assertNotIn("tool_choice", body)

    def test_zai_thinking_shape(self):
        self.assertEqual(pr.reasoning_fragment("zai_thinking", "high"), {"thinking": {"type": "enabled"}})
        self.assertEqual(pr.reasoning_fragment("zai_thinking", "off"), {"thinking": {"type": "disabled"}})

    def test_ollama_openai_endpoint_never_gets_tool_choice(self):
        # docs.ollama.com openai-compatibility: tool_choice is NOT supported
        p = self.openai("llama3.1", base_url="http://localhost:11434/v1")
        body = p._body("s", HIST, [TOOL], tool_choice="required")
        self.assertNotIn("tool_choice", body)
        self.assertEqual(body["tools"][0]["function"]["name"], "create_object")

    def test_lmstudio_streams_tool_arguments_by_default(self):
        c = models.resolve_caps(self.profile("openai_compat", "http://localhost:1234/v1", "any-model"))
        self.assertTrue(c.tool_args_stream)

    def test_deepseek_reasoner_replays_reasoning_content_with_tools(self):
        # thinking_mode guide: with `tools` every earlier reasoning_content must be passed back (HTTP 400 otherwise)
        p = self.openai("deepseek-reasoner", base_url="https://api.deepseek.com/v1")
        self.assertTrue(p._caps().reasoning_replay)
        turn = providers.Message(role="assistant", content="", tool_calls=[providers.ToolCall("c1", "create_object")],
                                 provider_blocks=[{"type": "reasoning_content", "text": "because"}])
        result = providers.Message(role="user", tool_results=[providers.ToolResult("c1", "create_object", "ok")])
        body = p._body("s", [providers.Message(role="user", content="go"), turn, result], [TOOL])
        assistant = next(m for m in body["messages"] if m["role"] == "assistant")
        self.assertEqual(assistant["reasoning_content"], "because")
        self.assertEqual(assistant["tool_calls"][0]["id"], "c1")

    def test_other_hosts_do_not_receive_reasoning_content(self):
        p = self.openai("gpt-4o")
        turn = providers.Message(role="assistant", content="x", provider_blocks=[{"type": "reasoning_content", "text": "r"}])
        body = p._body("s", [providers.Message(role="user", content="go"), turn], [])
        self.assertNotIn("reasoning_content", next(m for m in body["messages"] if m["role"] == "assistant"))

    def test_openai_reasoning_models_use_max_completion_tokens(self):
        # OpenAPI spec: max_tokens is deprecated in favour of max_completion_tokens; o-series / gpt-5 reject max_tokens
        body = self.openai("gpt-5")._body("s", HIST, [])
        self.assertIn("max_completion_tokens", body)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("temperature", body)
        self.assertEqual(body["stream_options"], {"include_usage": True})


class VendorFragmentPinTest(unittest.TestCase):
    def frag(self, control, effort, **kw):
        return pr.reasoning_fragment(control, effort, **kw)

    def test_openrouter_reasoning_object(self):
        # openrouter.ai/docs/use-cases/reasoning-tokens: top-level `reasoning` {effort | enabled}; `reasoning_effort` is not a field
        self.assertEqual(self.frag("openrouter_reasoning", "high"), {"reasoning": {"effort": "high"}})
        self.assertEqual(self.frag("openrouter_reasoning", "off"), {"reasoning": {"enabled": False}})
        self.assertNotIn("reasoning_effort", self.frag("openrouter_reasoning", "low"))

    def test_dashscope_enable_thinking_is_top_level(self):
        # model-studio deep-thinking: enable_thinking (extra_body of the SDK == top level of the JSON body)
        self.assertEqual(self.frag("qwen_enable_thinking", "high", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
                         {"enable_thinking": True})
        self.assertEqual(self.frag("qwen_enable_thinking", "off", base_url="https://dashscope-intl.aliyuncs.com/x"),
                         {"enable_thinking": False})

    def test_ollama_openai_endpoint_accepts_reasoning_effort(self):
        # docs.ollama.com openai-compatibility lists reasoning_effort
        self.assertEqual(self.frag("ollama_effort", "medium"), {"reasoning_effort": "medium"})

    def test_openai_chat_effort_levels(self):
        for level in ("low", "medium", "high"):                       # the OpenAPI enum
            self.assertEqual(self.frag("openai_effort", level, model="o3"), {"reasoning_effort": level})


class AnthropicProbeDocsTest(unittest.TestCase):
    def payload(self, adaptive, enabled, supported=True):
        return {"id": "m", "max_input_tokens": 200000, "max_tokens": 64000, "capabilities": {
            "image_input": {"supported": True},
            "thinking": {"supported": supported, "types": {"adaptive": {"supported": adaptive},
                                                           "enabled": {"supported": enabled}}}}}

    def test_models_api_selects_the_thinking_scheme(self):
        parse = provider_probe.parse_anthropic
        self.assertEqual(parse(self.payload(True, False))["fields"]["reasoning_control"], "anthropic_adaptive")
        self.assertEqual(parse(self.payload(True, True))["fields"]["reasoning_control"], "anthropic_adaptive")
        self.assertEqual(parse(self.payload(False, True))["fields"]["reasoning_control"], "anthropic_budget")
        self.assertEqual(parse(self.payload(False, False, supported=False))["fields"]["reasoning_control"], "none")
        fields = parse(self.payload(True, False))["fields"]
        self.assertEqual((fields["context_window"], fields["max_output"], fields["vision"]), (200000, 64000, True))

    def test_missing_capabilities_state_nothing(self):
        self.assertNotIn("reasoning_control", provider_probe.parse_anthropic({"max_tokens": 10}).get("fields", {}))
        self.assertEqual(provider_probe.parse_anthropic("junk"), {})


if __name__ == "__main__":
    unittest.main()
