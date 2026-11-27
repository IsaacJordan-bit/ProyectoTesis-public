"""Replay of P22 from saved results. No clustering, embeddings or resampling."""
from pathlib import Path
import hashlib,io,json,zipfile
import pandas as pd
import numpy as np

RUN_ID='2639d8e76465de84146d39074a41604cf15d126de25999bb86f7d65162c4fcd8'
BANK_ID='68682213c8913476a11a1ef997136dbcb80ec50eeedb4f466cb8dd8deae8b981'

def require(ok,message):
    if not ok:raise ValueError(message)

class SavedExploration:
    def __init__(self,archive):
        self.archive=Path(archive)
        expected=json.loads((self.archive.parent/'SHA256.json').read_text())[self.archive.name]
        require(hashlib.sha256(self.archive.read_bytes()).hexdigest()==expected,'Saved P22 archive changed')
        with zipfile.ZipFile(self.archive) as z:
            self.data={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
        self.plan=self.json('PLAN.json')
        self.audit=self.json('BANK_AUDIT.json')
        self.summary=self.json('SUMMARY_COMPLETE.json')
        self.tables={Path(n).stem:self.csv(n) for n in self.data if n.startswith('tables/') and n.endswith('.csv')}
        self.verified=False

    def json(self,name):return json.loads(self.data[name])
    def csv(self,name):return pd.read_csv(io.BytesIO(self.data[name]))

    def verify(self):
        run=self.json('RUN.json')
        for name in ['RUN.json','COMPUTATION_COMPLETE.json','SUMMARY_COMPLETE.json']:
            require(self.json(name)['contract_id']==RUN_ID,'Wrong analysis identity: '+name)
        digest=hashlib.sha256(json.dumps(self.plan,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        require(digest==RUN_ID,'Scientific plan does not match run identity')
        require(self.audit['bank_id']==BANK_ID==self.plan['bank_id'],'Wrong controlled bank')
        require(hashlib.sha256(self.data['code/p22_contextual.py']).hexdigest()==self.plan['code_sha256'],'Scientific source altered')
        expected=set(run['words']);require(len(expected)==640,'Unexpected cohort')
        frames={name:[] for name in ['word_geometry','heldout_probes','controls','text_alignment']}
        seen=set();receipts=0
        for name in self.data:
            if not name.startswith('words/') or not name.endswith('.receipt.json'):continue
            receipt=self.json(name);path=name.replace('.receipt.json','.json')
            require(hashlib.sha256(self.data[path]).hexdigest()==receipt['sha256'],'Word checkpoint altered: '+path)
            r=self.json(path);word=r['word']
            require(word==receipt['word'] and word not in seen,'Duplicate or mismatched word')
            require(r['contract_id']==receipt['contract_id']==RUN_ID,'Mismatched word analysis')
            require(r['category']==self.audit['label_map'][word],'Mismatched category')
            seen.add(word);receipts+=1
            key=dict(word=word,category=r['category'],in_common=word in self.audit['common_words'],in_full=word in self.audit['full_words'])
            frames['word_geometry'].append(dict(key,**r['geometry']))
            frames['heldout_probes'].extend(dict(key,**p) for p in r['probes'])
            frames['controls'].extend(dict(key,**p) for p in r['controls'])
            frames['text_alignment'].append(dict(key,**r['text']))
        require(seen==expected,'Incomplete word checkpoints')
        # The archived summary CSVs must reproduce their source checkpoints.
        for name,rows in frames.items():
            new=pd.DataFrame(rows);old=self.csv(name+'.csv')
            require(set(new.columns)==set(old.columns),'Unexpected columns: '+name)
            cols=sorted(new.columns)
            # String representation distinguishes absent/boolean/status fields while
            # numeric comparison retains tolerance for CSV float formatting.
            keys=[k for k in ('word','seed','training_fraction','k','control','repetition') if k in cols]
            if not keys:keys=['word']
            new=new[cols].sort_values(keys,kind='stable').reset_index(drop=True)
            old=old[cols].sort_values(keys,kind='stable').reset_index(drop=True)
            pd.testing.assert_frame_equal(new,old,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        g=self.csv('word_geometry.csv');p=self.csv('heldout_probes.csv');t=self.csv('text_alignment.csv')
        require(p.status.value_counts().to_dict()==self.summary['probe_status_counts'],'Probe status counts differ')
        require(t.status.value_counts().to_dict()==self.summary['text_status_counts'],'Text status counts differ')
        np.testing.assert_allclose(g.total_variance,g.within_P_variance+g.between_P_variance,rtol=1e-10,atol=1e-10)
        # Reproduce category-balanced point estimates, retaining all available words.
        for row in self.tables['geometry_summary'].itertuples():
            subset=g if row.scope=='COHORT' else g[g.in_common]
            val=subset.groupby('category')[row.metric].mean().mean()
            np.testing.assert_allclose(val,row.estimate,rtol=0,atol=1e-12)
        pp=self.tables['word_averaged_probes']
        for row in self.tables['heldout_summary'].itertuples():
            if row.status!='OK':continue
            subset=pp[(pp.k==row.k)&(pp.training_fraction==row.training_fraction)]
            if row.scope=='COMMON':subset=subset[subset.in_common]
            val=subset.groupby('category')[row.metric].mean().mean()
            np.testing.assert_allclose(val,row.estimate,rtol=0,atol=1e-12)
        self.verified=True
        return dict(status='VERIFIED',contract_id=RUN_ID,bank_id=BANK_ID,word_receipts=receipts,
                    words=self.audit['words'],occurrences=self.audit['occurrences'],
                    prototypes=self.audit['prototypes'],common_words=len(self.audit['common_words']),
                    probe_status=self.summary['probe_status_counts'],text_status=self.summary['text_status_counts'],
                    checkpoint_tables_reconciled=True,summary_point_estimates_reconciled=True,
                    raw_embeddings_rechecked=False,bootstrap_intervals_recomputed=False,new_experiments=0)

    def plot(self,output_root,dpi=220,formats=('png','pdf','svg'),show=True):
        from figuras_exploratorio import make_figures
        require(self.verified,'Call verify() before plotting')
        require('png' in formats and set(formats)<=set(('png','pdf','svg')),'Include png; supported formats: png, pdf, svg')
        root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
        # Only short, known paths are materialized. Original absolute paths are provenance.
        names=['SUMMARY_COMPLETE.json','word_geometry.csv','text_alignment.csv',
               'tables/heldout_summary.csv','tables/occurrence_reference_summary.csv']
        for name in names:
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(self.data[name])
        paths=make_figures(dict(root=root,config=self.plan['scientific_config']),dpi=dpi,formats=formats)
        if show:
            from IPython.display import display,Image
            for p in paths:display(Image(filename=str(p)))
        return paths
