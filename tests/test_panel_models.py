import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402  (the header every test that imports oscar must carry)

isolation.isolate_process()   # sandboxed settings/projects/logs BEFORE importing oscar

from oscar.panel_models import catalog_models, model_availability, selected_profile  # noqa: E402


class ModelCatalogTest(unittest.TestCase):
    def test_keeps_any_number_of_provider_models_and_deduplicates(self):
        self.assertEqual(catalog_models(["one", "two", "two", "", None, "three"]),
                         ["one", "two", "three"])

    def test_does_not_claim_unverified_or_unlisted_model_is_ready(self):
        self.assertEqual(model_availability("local-model", ["other"]), "Not listed by provider")
        self.assertEqual(model_availability("local-model", [], "missing credentials"),
                         "Not verified — missing credentials")
        self.assertEqual(model_availability("listed", ["listed"]), "Listed by provider")

    def test_profile_lookup_uses_configured_names(self):
        class Profile:
            def __init__(self, name):
                self.name = name

        profiles = [Profile("local"), Profile("remote")]
        self.assertIs(selected_profile(profiles, "remote"), profiles[1])
        self.assertIsNone(selected_profile(profiles, "missing"))


if __name__ == "__main__":
    unittest.main()
