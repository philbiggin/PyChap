"""Tests for pychap.elements (van der Waals radii lookup)."""

import unittest

from pychap.elements import (
    DEFAULT_VDW_RADIUS,
    guess_element_from_name,
    vdw_radius_for_element,
    vdw_radius_for_name,
)


class TestElementGuessing(unittest.TestCase):
    def test_common_protein_atom_names(self):
        self.assertEqual(guess_element_from_name("CA"), "C")  # alpha carbon, not calcium
        self.assertEqual(guess_element_from_name("N"), "N")
        self.assertEqual(guess_element_from_name("O"), "O")
        self.assertEqual(guess_element_from_name("CB"), "C")
        self.assertEqual(guess_element_from_name("OG1"), "O")
        self.assertEqual(guess_element_from_name("SG"), "S")

    def test_names_with_leading_digits(self):
        self.assertEqual(guess_element_from_name("1HB"), "H")
        self.assertEqual(guess_element_from_name("2HG1"), "H")

    def test_unrecognised_name_falls_back_to_carbon(self):
        self.assertEqual(guess_element_from_name("ZZZZZ"), "C")


class TestVdwRadiusLookup(unittest.TestCase):
    def test_known_elements(self):
        self.assertAlmostEqual(vdw_radius_for_element("C"), 1.70)
        self.assertAlmostEqual(vdw_radius_for_element("O"), 1.52)
        self.assertAlmostEqual(vdw_radius_for_element("N"), 1.55)
        self.assertAlmostEqual(vdw_radius_for_element("S"), 1.80)

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(vdw_radius_for_element(" c "), vdw_radius_for_element("C"))

    def test_unknown_element_returns_default(self):
        self.assertEqual(vdw_radius_for_element("Xx"), DEFAULT_VDW_RADIUS)

    def test_vdw_radius_for_name_matches_element_lookup(self):
        self.assertAlmostEqual(vdw_radius_for_name("CA"), vdw_radius_for_element("C"))
        self.assertAlmostEqual(vdw_radius_for_name("OD1"), vdw_radius_for_element("O"))


if __name__ == "__main__":
    unittest.main()
