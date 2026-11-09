import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from shaping_progress import panel_values,average,paired_delta,comparison_rows,read_saved_panel,dashboard_claim
from shaping_execution import live_status
from shaping_completion import sha


class DashboardTests(unittest.TestCase):
    def test_plain_and_gzip_scalar_documents_preserve_exact_numbers(self):
        import gzip
        from shaping_scatter_prepare import read_points
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);value={'predicted':[0.,-1.2345678901234567,100.00000000000001],'outer_results_read':False}
            payload=json.dumps(value).encode();plain=root/'points.json';compressed=root/'points.json.gz'
            plain.write_bytes(payload);compressed.write_bytes(gzip.compress(payload,mtime=0))
            self.assertEqual(read_points(plain),value);self.assertEqual(read_points(compressed),value)
            compressed.write_bytes(gzip.compress(b' '*(16*1024**2+1),mtime=0))
            with self.assertRaisesRegex(ValueError,'sixteen MiB'):read_points(compressed)

    def panel(self,value=.5):
        summary={k:dict(pooled=value,within=value,rows=10,sources=2)
                 for k in ['kadid','tid2013','konfig','cid22_a25','cid22_ssim2']}
        summary['cid22_ssim2'].update(pooled=.99,within=.99)
        return dict(summary=summary,selection_score=value)

    def record(self,seed,value,family='relu',partition='inner-0-0',reported=True):
        cell=dict(id=family+partition+str(seed),stage='inner',family=family,partition=partition,setting=0,seed=seed)
        packed=panel_values(self.panel(value)) if value is not None else panel_values(self.panel(None))
        return dict(cell=cell,trained=True,state='trained',original_failed_gate=False,investigation=False,
                    panel=dict(validation=packed,train=packed,active=8) if reported else None)

    def test_human_composite_excludes_teacher(self):
        p=panel_values(self.panel());self.assertEqual(p['human'],.5)
        self.assertEqual(p['datasets']['cid22_ssim2']['mean'],.99)

    def test_incomplete_and_undefined_seed_groups_have_no_aggregate(self):
        for values in [[.8,.9],[.8,.9,None]]:
            row=average(values,3);self.assertIsNone(row['mean']);self.assertIsNone(row['sd'])
        self.assertAlmostEqual(average([.1,.2,.3],3)['sd'],.1)
        self.assertIsNone(average([.5],1)['sd'])

    def test_pending_seed_is_not_dropped_from_comparison(self):
        records=[self.record(0,.8),self.record(1,.9),self.record(2,.7,reported=False)]
        row=comparison_rows(records)[0]
        self.assertEqual((row['reported'],row['required']),(2,3));self.assertIsNone(row['human']['mean'])

    def test_extension_roles_exposures_and_optimizer_choices_remain_separate(self):
        records=[]
        for role,epochs,choice in [('primary',[480],'base'),('fresh_head',[240],'base'),
                                   ('primary',[480],'constant_lr')]:
            for seed in range(3):
                r=self.record(seed,.8);r['cell'].update(stage='exposure_extension',role=role,
                    phase_epochs=epochs,optimizer_choice=choice,total_pair_draws=sum(epochs)*50000)
                records.append(r)
        # A primary-budget result must not complete any longer-exposure cohort.
        records.append(self.record(0,.95))
        rows=comparison_rows(records);extensions=[r for r in rows if r['stage']=='exposure_extension']
        self.assertEqual(len(extensions),3)
        self.assertTrue(all(r['required']==3 and r['reported']==3 and r['delta'] is None for r in extensions))
        self.assertEqual({r['total_pair_draws'] for r in extensions},{12000000,24000000})
        records[0]['panel']=None
        rows=comparison_rows(records)
        partial=next(r for r in rows if r['stage']=='exposure_extension' and r['role']=='primary' and r['optimizer_choice']=='base')
        self.assertEqual((partial['required'],partial['reported']),(3,2))
        self.assertIsNone(partial['human']['mean'])

    def test_undefined_pair_excludes_entire_setting_across_inner_folds(self):
        records=[self.record(s,.8,family='product_joint_identity') for s in range(3)]
        records += [self.record(s,None if s==0 else .8,family='product_joint_identity',partition='inner-0-1') for s in range(3)]
        rows=comparison_rows(records)
        self.assertTrue(all(r['ineligible'] for r in rows))
        self.assertAlmostEqual(rows[0]['human']['mean'],.8)
        self.assertIsNone(rows[1]['human']['mean'])

    def test_delta_requires_same_seeds_and_no_deterministic_pseudoreplication(self):
        left=[self.record(s,.8) for s in range(3)];right=[self.record(s,.7) for s in range(3)]
        self.assertAlmostEqual(paired_delta(left,right)['mean'],.1)
        self.assertIsNone(paired_delta(left,right[:2]))
        self.assertIsNone(paired_delta([self.record(None,.8)],[self.record(None,.7)]))

    def test_report_drift_identity_and_outer_exposure_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            p=Path(d)/'REPORT.json';cell=dict(id='cell');row=dict(cell=cell,plan_sha256='plan',
                outer_results_read=False,train=self.panel(),inner_validation=self.panel())
            p.write_text(json.dumps(row));pin=sha(p);self.assertEqual(read_saved_panel(p,pin,cell,'plan')['validation']['human'],.5)
            with self.assertRaisesRegex(ValueError,'identity differs'):read_saved_panel(p,pin,dict(id='other'),'plan')
            p.write_text(json.dumps({**row,'extra':'changed'}))
            with self.assertRaisesRegex(ValueError,'report drift'):read_saved_panel(p,pin,cell,'plan')
            p.write_text(json.dumps({**row,'outer_results_read':True}))
            with self.assertRaisesRegex(ValueError,'outer exposure'):read_saved_panel(p,sha(p),cell,'plan')

    def test_live_observation_can_be_read_without_writing_status_or_stdout(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);p=root/'LIVE_STATUS.json';p.write_text('existing owner data')
            with contextlib.redirect_stdout(io.StringIO()) as output:row=live_status(root,persist=False,emit=False)
            self.assertEqual(row['live'],0);self.assertEqual(p.read_text(),'existing owner data')
            self.assertEqual(output.getvalue(),'')

    def test_second_publisher_cannot_replace_owned_view(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);first=dashboard_claim(root)
            try:
                with self.assertRaisesRegex(ValueError,'another statistics publisher'):dashboard_claim(root)
            finally:first.close()
            dashboard_claim(root).close()


if __name__=='__main__':unittest.main()
