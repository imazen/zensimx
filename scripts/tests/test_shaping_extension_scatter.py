import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_scatter_prepare import registered_scatter_cells


class ExtensionScatterTests(unittest.TestCase):
    def test_registered_extensions_join_primary_records_without_dropping_primary(self):
        primary=[dict(id='primary',stage='inner')];extension=dict(id='extension',stage='exposure_extension')
        old=dict(cell=primary[0],panel={'saved':True});new=dict(cell=extension,panel={'saved':True})
        snapshot=dict(records=[old],extension_records=[new])
        with patch('shaping_stable_export_prepare.registered_extensions',return_value=[extension]) as full:
            self.assertEqual(registered_scatter_cells(Path('/study'),primary,snapshot),[old,new])
        full.assert_called_once_with(Path('/study'),tuple(primary))

    def test_unregistered_extension_or_outer_candidate_and_duplicate_records_refuse(self):
        primary=[dict(id='primary',stage='inner')];extension=dict(id='extension',stage='exposure_extension')
        with patch('shaping_stable_export_prepare.registered_extensions',return_value=[extension]):
            with self.assertRaisesRegex(ValueError,'registered inner scope'):
                registered_scatter_cells(Path('/study'),primary,dict(records=[],extension_records=[dict(cell=dict(extension,role='invented'),panel=True)]))
        outer=dict(id='outer',stage='outer')
        with self.assertRaisesRegex(ValueError,'registered inner scope'):
            registered_scatter_cells(Path('/study'),[outer],dict(records=[dict(cell=outer,panel=True)]))
        row=dict(cell=primary[0],panel=True)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            registered_scatter_cells(Path('/study'),primary,dict(records=[row,row]))


if __name__=='__main__':unittest.main()
