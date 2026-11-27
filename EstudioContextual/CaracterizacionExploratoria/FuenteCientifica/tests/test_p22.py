"""Synthetic correctness checks. These are not empirical thesis results."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p22_contextual as C


class ScientificChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiter = threadpool_limits(limits=1)

    def test_group_split_keeps_sources_and_duplicate_vectors_together(self):
        X = np.arange(24., dtype=float).reshape(8, 3)+1
        X[3] = X[4]
        meta = pd.DataFrame({"source_doc_key":["a","a","b","b","c","d","e","f"]})
        groups = C.connected_groups(meta, X)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[4])  # transitive source/duplicate union
        train, test = C.group_split(groups, 123, .2)
        self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertEqual(set(train) | set(test), set(range(8)))

    def test_identical_contexts_are_kept_together(self):
        X = np.eye(4)
        meta = pd.DataFrame({"source_doc_key":list("abcd")})
        groups = C.connected_groups(meta, X, ["same  TEXT", "same text", "else", "other"])
        self.assertEqual(groups[0], groups[1])

    def test_raw_reconstruction_keeps_orthogonal_test_error(self):
        rng=np.random.default_rng(16)
        train=rng.normal(size=(12,30))
        test=rng.normal(size=(9,30))
        mean=train.mean(0)
        _,_,basis=np.linalg.svd(train-mean,full_matrices=False)
        basis=basis[:11]
        centers=(train-mean)[:3]@basis.T
        projected=(test-mean)@basis.T
        outside=np.square(test-mean).sum(1)-np.square(projected).sum(1)
        raw_centers=mean+centers@basis
        np.testing.assert_allclose(C.nearest(projected,centers)[1]+outside,
                                   C.nearest(test,raw_centers)[1],rtol=1e-12,atol=1e-12)

    def test_exact_variance_decomposition_and_one_prototype(self):
        rng=np.random.default_rng(3)
        X=rng.normal(size=(100,8))+2
        P=np.stack([X[:50].mean(0),X[50:].mean(0)])
        A=np.stack([X.mean(0),rng.normal(size=8)+2,rng.normal(size=8)+2,rng.normal(size=8)+2])
        refs=C.unit_rows(A)
        g=C.geometry_metrics(X,P,np.repeat([0,1],50),np.array([50,50]),A,refs,np.array([1,1,0,0],bool),0,2)
        self.assertAlmostEqual(g["total_variance"],g["within_P_variance"]+g["between_P_variance"],places=11)
        q=C.geometry_metrics(X,X.mean(0,keepdims=True),np.zeros(100,int),np.array([100]),A,refs,np.array([1,1,0,0],bool),0,2)
        self.assertAlmostEqual(q["frozen_P_retained_fraction"],0,places=12)

    def test_random_partition_expectation_is_not_a_fitted_effect(self):
        rng=np.random.default_rng(4)
        X=rng.normal(size=(40,6));a=X.mean(0)
        V=np.square(X-a).sum()/40
        observed=[]
        for _ in range(2000):
            ix=rng.permutation(40)
            chunks=[ix[:5],ix[5:17],ix[17:]]
            observed.append(sum(len(c)*np.square(X[c].mean(0)-a).sum() for c in chunks)/40/V)
        self.assertAlmostEqual(np.mean(observed),(3-1)/(40-1),delta=.004)

    def probe_cfg(self):
        cfg=C.Config(split_seeds=(0,1),probe_k=(1,2),training_fractions=(1.,),
                     control_repetitions=2,bootstrap_repetitions=50,kmeans_n_init=2)
        return C.clean_json(C.asdict(cfg))

    def test_heldout_baseline_is_train_mean_and_controls_are_paired(self):
        rng=np.random.default_rng(123)
        X=np.r_[rng.normal(-5,.2,(50,6)),rng.normal(5,.2,(50,6))]+1
        groups=np.arange(100)
        cfg=self.probe_cfg()
        rows,controls=C.heldout_probes(X,groups,"synthetic_mixture",cfg)
        for row in rows:
            train,test=C.group_split(groups,C.seed_for("split","synthetic_mixture",row["seed"]),.2)
            expected=np.square(X[test]-X[train].mean(0)).sum(1).mean()
            self.assertAlmostEqual(row["A_test_mse"],expected,places=10)
            if row["k"]==1:
                self.assertAlmostEqual(row["relative_gain"],0,places=10)
                self.assertAlmostEqual(row["O_train_test_mse"],C.loss(X[test],X[train]),places=10)
                self.assertEqual(row["O_train_codebook_size"],len(train))
            else:
                self.assertGreater(row["relative_gain"],.9)
                self.assertEqual(row["control_status"],"OK")
                self.assertGreater(row["gain_minus_random"],.5)
                self.assertGreater(row["stability_ARI"],.9)
        self.assertEqual(len(controls),4)

    def test_insufficient_groups_remain_explicitly_nonevaluable(self):
        X=np.random.default_rng(2).normal(size=(100,8))+1
        rows,controls=C.heldout_probes(X,np.zeros(100,int),"one_record",self.probe_cfg())
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r["status"]=="NOT_EVALUABLE" for r in rows))
        self.assertEqual(controls,[])

    def test_bootstrap_weights_categories_equally(self):
        df=pd.DataFrame({"word":["a","b","c","d"],"category":["x","y","y","y"],"value":[10.,0,0,0]})
        r=C.stratified_summary(df,"value",100)
        self.assertAlmostEqual(r["estimate"],5.)
        df.loc[df.category.eq("x"),"value"]=np.nan
        self.assertEqual(C.stratified_summary(df,"value",100)["status"],"NOT_EVALUABLE_MISSING_CATEGORY")

    def test_pairwise_text_control_and_missing_status(self):
        rng=np.random.default_rng(17)
        X=np.r_[rng.normal(-2,.1,(25,4)),rng.normal(2,.1,(25,4))]
        meta=pd.DataFrame({"source_doc_key":[str(i) for i in range(50)]})
        texts=[f"target orchard fruit harvest sweet apple sentence {i}" if i<25 else
               f"target software computer screen processor device sentence {i}" for i in range(50)]
        result=C.text_alignment(X,meta,texts,"target",np.arange(50),19)
        self.assertEqual(result["status"],"OK")
        self.assertGreater(result["rho_minus_null"],.5)
        self.assertEqual(C.text_alignment(X,meta,None,"target",np.arange(50),19)["status"],"NOT_AVAILABLE")

    def test_corrupt_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/"words/abc.json"
            run=dict(root=str(root),contract_id="test",words=["word"])
            C.write_json(path,{"word":"word","value":1})
            C.write_json(path.with_suffix(".receipt.json"),dict(word="word",contract_id="test",sha256=C.sha(path)))
            self.assertEqual(len(C.completed(run)),1)
            C.write_json(path,{"word":"word","value":2})
            with self.assertRaises(C.AuditError):C.completed(run)


if __name__=="__main__":
    unittest.main()
