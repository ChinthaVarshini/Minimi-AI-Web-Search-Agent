import unittest
from unittest.mock import Mock, patch

import requests

from search import MAX_RESULTS, REQUEST_TIMEOUT, search_web


class SearchTests(unittest.TestCase):
    @patch("search.requests.post")
    def test_search_bounds_results_and_rejects_non_web_urls(self, post):
        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "url": "https://example.com/page",
                    "title": "Example",
                    "content": "A" * 1_500,
                },
                {
                    "url": "javascript:alert(1)",
                    "title": "Invalid",
                    "content": "Not a web citation.",
                },
            ]
        }
        post.return_value = response

        results = search_web("  a focused question  ", "test-key")

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]["content"]), 1_200)
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["query"], "a focused question")
        self.assertEqual(post.call_args.kwargs["json"]["max_results"], MAX_RESULTS)
        self.assertEqual(post.call_args.kwargs["timeout"], REQUEST_TIMEOUT)

    def test_search_requires_query_and_key_without_network_requests(self):
        with patch("search.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "query cannot be empty"):
                search_web("  ", "test-key")
            with self.assertRaisesRegex(ValueError, "TAVILY_API_KEY"):
                search_web("a question", "")
        post.assert_not_called()

    @patch("search.requests.post")
    def test_search_surfaces_provider_http_errors(self, post):
        post.return_value.raise_for_status.side_effect = requests.HTTPError("provider error")

        with self.assertRaises(requests.HTTPError):
            search_web("a question", "test-key")


if __name__ == "__main__":
    unittest.main()
