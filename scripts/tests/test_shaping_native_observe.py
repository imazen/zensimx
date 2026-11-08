import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_native_observe import observe


class NativeObservationTests(unittest.TestCase):
    def test_scoped_actual_native_pid_and_snapshot_fd_are_bound_without_mutating_run(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);scope=root/'study';run=scope/'runs/cell/attempt-001';fit=run/'fit';(fit/'checkpoints').mkdir(parents=True)
            (run/'RUN.json').write_text(json.dumps(dict(cell=dict(id='cell'))))
            (fit/'checkpoints/ckpt_epoch000.bin').write_bytes(b'preserved')
            proc=root/'proc/42';proc.mkdir(parents=True);(proc/'comm').write_text('zensim_mlp_trai\n')
            (proc/'cmdline').write_bytes(chr(0).join(['trainer','--out',str(fit/'best.bin'),'']).encode())
            (proc/'stat').write_text('42 (zensim_mlp_trai) R '+' '.join(['0']*18+['1234']))
            (proc/'cgroup').write_text('0::/test-scope\n');cap=root/'cgroups/test-scope';cap.mkdir(parents=True)
            (cap/'memory.max').write_text('2147483648\n');(cap/'memory.swap.max').write_text('0\n')
            state=root/'shared-training-snapshots'/('a'*64);state.mkdir(parents=True);(state/'.lock').write_bytes(b'')
            (proc/'fd').mkdir();(proc/'fd/5').symlink_to(state/'.lock')
            before=(run/'RUN.json').read_bytes();record=observe([scope],proc=root/'proc',cgroups=root/'cgroups')
            self.assertEqual(len(record['native_jobs']),1);row=record['native_jobs'][0]
            self.assertEqual(row['pid'],42);self.assertEqual(row['state'],'R');self.assertEqual(row['start_ticks'],1234)
            self.assertEqual(row['snapshot_directory'],str(state));self.assertEqual(row['memory_max'],'2147483648')
            self.assertEqual(row['checkpoint_count'],1);self.assertEqual((run/'RUN.json').read_bytes(),before)
            self.assertEqual(observe([root/'other'],proc=root/'proc',cgroups=root/'cgroups')['native_jobs'],[])


if __name__=='__main__':unittest.main()
