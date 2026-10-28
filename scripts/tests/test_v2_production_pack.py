"""Canonical packing preserves its default command and supports explicit input retention."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
import v2_production_pack as owner


class PackingInputs(unittest.TestCase):
    def test_no_prune_is_explicit_and_other_quantization_and_dial_options_are_unchanged(self):
        commands = []
        with patch.object(owner, 'strict_training_groups'), \
                patch.object(owner, 'dense_bake', return_value=Path('/invented/dense')), \
                patch.object(owner, 'sha', return_value='invented'), \
                patch.object(owner, 'run', side_effect=lambda command, log: commands.append(command)):
            legacy = owner.pack_production(Path('/invented/input'), Path('/invented/out'), Path('/invented/anchor'), True)
            retained = owner.pack_production(Path('/invented/input'), Path('/invented/out'), Path('/invented/anchor'), True, True)
        self.assertNotIn('--no-prune', commands[0])
        self.assertEqual(commands[0], [x for x in commands[1] if x != '--no-prune'])
        self.assertIn('--protect-last', commands[1])
        self.assertEqual(commands[1][commands[1].index('--dtype') + 1], 'f16')
        self.assertEqual(commands[1][commands[1].index('--zerobias-bulk') + 1], '0')
        self.assertNotIn('input_pruning', legacy)
        self.assertEqual(retained['input_pruning'], 'disabled; preserve derived-input positions')


if __name__ == '__main__':
    unittest.main()
