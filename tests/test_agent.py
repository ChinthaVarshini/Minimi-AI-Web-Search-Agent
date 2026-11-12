import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent import (
    MAX_MEMORY_MESSAGES,
    WEB_SEARCH_TOOL,
    add_search_results,
    build_messages,
    run_agent,
    validate_citations,
)


class AgentHelperTests(unittest.TestCase):
    def test_build_messages_keeps_recent_context_for_follow_ups(self):
        history = [
            {"role": "user", "content": f"Question {index}"}
            for index in range(MAX_MEMORY_MESSAGES + 2)
        ]

        messages = build_messages(history)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(len(messages), MAX_MEMORY_MESSAGES + 1)
        self.assertEqual(messages[1]["content"], "Question 2")
        self.assertEqual(messages[-1]["content"], f"Question {MAX_MEMORY_MESSAGES + 1}")

    def test_search_results_get_stable_citations_and_deduplicate_urls(self):
        sources = []
        source_ids = {}
        results = [
            {
                "url": "https://example.com/article",
                "title": "Article",
                "content": "Useful details.",
            },
            {
                "url": "https://example.com/article",
                "title": "Duplicate",
                "content": "Additional details.",
            },
            {
                "url": "javascript:alert(1)",
                "title": "Invalid link",
                "content": "Ignore this result.",
            },
        ]

        context = add_search_results(results, sources, source_ids)

        self.assertEqual(len(sources), 1)
        self.assertIn("[1] Article", context)
        self.assertIn("Additional details.", context)
        self.assertNotIn("javascript:", context)

    def test_validate_citations_removes_unknown_references(self):
        sources = [{"id": 1, "url": "https://example.com", "title": "Source"}]

        answer = validate_citations("A supported claim [1]. An unsupported one [8].", sources)

        self.assertIn("[1]", answer)
        self.assertNotIn("[8]", answer)

    def test_validate_citations_adds_references_when_model_omits_them(self):
        sources = [{"id": 1, "url": "https://example.com", "title": "Source"}]

        answer = validate_citations("A concise answer.", sources)

        self.assertIn("Sources: [1]", answer)

    @patch("agent.search_web")
    @patch("agent.OpenAI")
    def test_agent_uses_only_web_search_and_preserves_follow_up_context(
        self, openai_class, search
    ):
        def chunk(content=None, tool_calls=None):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=content, tool_calls=tool_calls)
                    )
                ]
            )

        tool_delta_start = SimpleNamespace(
            index=0,
            id="call_1",
            function=SimpleNamespace(name="web_search", arguments='{"query":"'),
        )
        tool_delta_end = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name=None, arguments='weather in Tokyo now"}'),
        )
        client = openai_class.return_value
        client.chat.completions.create.side_effect = [
            iter([chunk(tool_calls=[tool_delta_start]), chunk(tool_calls=[tool_delta_end])]),
            iter([chunk(content="Tokyo is sunny [1].")]),
        ]
        search.return_value = [
            {
                "url": "https://example.com/weather",
                "title": "Tokyo weather",
                "content": "Current conditions.",
            }
        ]
        history = [
            {"role": "user", "content": "What is the weather in Tokyo?"},
            {"role": "assistant", "content": "Checking the latest forecast."},
            {"role": "user", "content": "What about right now?"},
        ]
        reset = Mock()

        answer, sources = run_agent(
            history,
            openai_api_key="test-key",
            tavily_api_key="test-search-key",
            model="test-model",
            on_reset=reset,
        )

        self.assertEqual(answer, "Tokyo is sunny [1].")
        self.assertEqual(sources[0]["url"], "https://example.com/weather")
        search.assert_called_once_with("weather in Tokyo now", "test-search-key")
        reset.assert_called_once()
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call.kwargs["tools"] == WEB_SEARCH_TOOL for call in calls))
        self.assertTrue(all(call.kwargs["parallel_tool_calls"] is False for call in calls))
        second_request = calls[1].kwargs["messages"]
        self.assertTrue(any(message["role"] == "tool" for message in second_request))
        self.assertIn("What about right now?", [m["content"] for m in second_request])


if __name__ == "__main__":
    unittest.main()
