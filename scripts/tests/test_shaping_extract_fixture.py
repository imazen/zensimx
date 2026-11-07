from pathlib import Path
import hashlib,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_extract_fixture import pixel_hash


class PixelTests(unittest.TestCase):
    def test_chunk_boundary_matches_scalar_integer_definition(self):
        size=257
        for salt in (1,6):
            raw=bytearray()
            for i in range(size*size):
                k=(((i+salt*7919)*2654435761)&((1<<64)-1))%65521
                raw.extend((k%256,(k//3)%256,(k//7)%256))
            self.assertEqual(pixel_hash(size,salt),hashlib.sha256(raw).hexdigest())


if __name__=='__main__':unittest.main()
