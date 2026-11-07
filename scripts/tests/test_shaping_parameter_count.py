from pathlib import Path
import sys, unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_parameter_count import counts


class ParameterTests(unittest.TestCase):
    def test_max_pair_count_matches_distinct_symbols_in_expanded_layer(self):
        # Two inputs, four first units, one max pair, eight second units.
        # Repeated symbolic labels represent tied entries in the expanded model.
        first_weights = [f'w{i}-{j}' for i in range(2) for j in range(4)]
        first_biases = [None, 'b1', 'b2', 'b3']
        intermediate = [[f'm{group}-{j}' for j in range(8)] for group in (0, 0, 1, 2)]
        mid_biases = [f'c{j}' for j in range(8)]
        output = [f'o{j}' for j in range(8)]
        symbols = first_weights+first_biases+sum(intermediate, [])+mid_biases+output+[None]
        self.assertEqual(counts(2, 4, 1)['trainable_scalar_entries'], len(set(symbols)-{None}))
        self.assertLess(counts(2, 4, 1)['trainable_scalar_entries'], len(symbols))

    def test_diagnostic_bias_and_learned_temperature_are_actual_parameters(self):
        head = dict(schema='monotone-head-v1', family='silu_diagnostic', first_width=4,
                    max_pairs=0, widths=[8], unrestricted_first=True,
                    analytic=dict(learn_temperature=False))
        fixed = counts(2, 4, 0, head)
        self.assertEqual(fixed['components']['output_biases'], 1)
        head['analytic']['learn_temperature'] = True
        self.assertEqual(counts(2, 4, 0, head)['trainable_scalar_entries'], fixed['trainable_scalar_entries']+1)
        head['max_pairs'] = 1
        with self.assertRaises(ValueError): counts(2, 4, 1, head)

    def test_unused_product_gate_and_fixed_knots_do_not_count(self):
        term = dict(column=0, damage=13, reference=14, separate=False,
                    knots=[.25, 1.], slopes=[1., 1., 1.], gate=[0., .5, 1.])
        shape = dict(schema='zensim-input-shaping-v1', trainable=True, terms=[term])
        base = counts(2, 4, 1)['trainable_scalar_entries']
        self.assertEqual(counts(2, 4, 1, shaping=shape)['trainable_scalar_entries'], base+3)
        term['separate'] = True
        self.assertEqual(counts(2, 4, 1, shaping=shape)['trainable_scalar_entries'], base+6)
        shape['trainable'] = False
        result = counts(2, 4, 1, shaping=shape)
        self.assertEqual(result['trainable_scalar_entries'], base)
        self.assertEqual(result['frozen_response_shape_values'], 6)

    def test_refuses_shape_column_aliasing(self):
        term = dict(column=0, damage=13, reference=None, separate=False,
                    knots=[.25, 1.], slopes=[1., 1., 1.], gate=[0., .5, 1.])
        with self.assertRaisesRegex(ValueError, 'columns'):
            counts(2, 4, 1, shaping=dict(schema='zensim-input-shaping-v1', trainable=True, terms=[term, term]))

    def test_logical_shape_ids_are_not_compact_model_positions(self):
        term = dict(column=2344, damage=719, reference=712, separate=False,
                    knots=[.25, 1.], slopes=[1., 1., 1.], gate=[0., .5, 1.])
        shape = dict(schema='zensim-input-shaping-v1', trainable=True, terms=[term])
        result = counts(2, 4, 1, shaping=shape)
        self.assertEqual(result['components']['damage_slopes'], 3)


if __name__ == '__main__': unittest.main()
