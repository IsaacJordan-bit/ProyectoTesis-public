"""End-to-end synthetic fixture. No generated values are empirical thesis data."""
from pathlib import Path
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import p22_contextual as C


def fixture(root):
    bank=root/"synthetic_bank";folder=bank/"inputs/apo";folder.mkdir(parents=True)
    ref=root/"synthetic_reference";ref.mkdir()
    words=[f"word_{i}" for i in range(8)]
    rng=np.random.default_rng(12345)
    # Production schema, including dimension and canonical occurrence count.
    basis=rng.normal(size=(4,768))
    A=[];P=[];O=[];units=[];meta=[];texts=[]
    for ai,word in enumerate(words):
        labels=(np.arange(100)//5)%2
        latent=rng.normal(scale=.25,size=(100,4));latent[:,0]+=(labels*2-1)*2
        X=latent@basis+3+ai*.1
        A.append(X.mean(0));O.extend(X)
        for pi in range(2):
            P.append(X[labels==pi].mean(0))
            units.append(dict(word=word,unit_id=f"{word}:{pi}",unit_index=2*ai+pi,sense_id=pi,support=int((labels==pi).sum())))
        for j in range(100):
            row=dict(word=word,occ_id=100*ai+j,source_doc_key=f"source_{ai}_{j//2}",word_pos_original=j,
                     a_index=ai,p_index=2*ai+int(labels[j]),sense_id=int(labels[j]),unit_id=f"{word}:{labels[j]}")
            meta.append(row)
            text=(f"{word} orchard fruit harvest apple sweet branch season " if labels[j]==0 else
                  f"{word} computer screen software processor digital device ")+str(j)
            texts.append(dict(occ_id=row["occ_id"],word=word,source_doc_key=row["source_doc_key"],
                              word_pos_original=j,context_window=text))
    for name,array in (("A",A),("P",P),("O",O)):np.save(folder/(name+".npy"),np.asarray(array,dtype=np.float64))
    C.write_json(folder/"words.json",words)
    pd.DataFrame(meta).to_parquet(folder/"occurrences.parquet",index=False)
    pd.DataFrame(units).to_parquet(folder/"units.parquet",index=False)
    protocol=dict(n_words=8,positions_per_word=100,fixture="SYNTHETIC_ONLY")
    C.write_json(bank/"protocol.json",protocol);C.write_json(ref/"bank_protocol.json",protocol)
    identity="synthetic_only_not_a_research_bank"
    C.write_json(ref/"bank_identity.json",dict(run_id=identity))
    C.write_json(ref/"geometry.json",dict(neighbors={"category_A":words[:4],"category_B":words[4:]}))
    pd.DataFrame({"word":words}).to_parquet(ref/"full.parquet",index=False)
    pd.DataFrame({"word":words[::2]}).to_parquet(ref/"common.parquet",index=False)
    C.write_json(folder/"done.json",dict(identity=C.digest(dict(run=identity,stage="CANONICAL_APO")),
                   files={p.name:C.sha(p) for p in folder.iterdir() if p.is_file()}))
    source=root/"synthetic_texts.parquet"
    pd.DataFrame(texts).to_parquet(source,row_group_size=120,index=False)
    return bank,ref,source


def main():
    keep=os.environ.get("P22_SMOKE_OUTPUT")
    context=tempfile.TemporaryDirectory() if not keep else None
    root=Path(keep or context.name);root.mkdir(parents=True,exist_ok=True)
    bank,ref,source=fixture(root)
    C.REFERENCE=ref  # Test-only reference; production notebook never changes it.
    cfg=C.Config(bank=bank,output_parent=root/"output",source_parquet=source,
                 split_seeds=(0,1),probe_k=(1,2),training_fractions=(.5,1.),
                 control_repetitions=2,text_permutations=19,bootstrap_repetitions=50,
                 nearest_neighbors=3,kmeans_n_init=2,max_workers=2,max_hours_per_call=.1,
                 reserve_gib=1.)
    before={str(p):C.sha(p) for p in bank.rglob("*") if p.is_file()}
    run=C.prepare(cfg)
    pilot=C.run_analysis(run,pilot=True)
    assert pilot["completed_words"]==4
    full=C.run_analysis(run)
    assert full["status"]=="COMPLETE"
    repeated=C.run_analysis(run)
    assert repeated["new_words"]==0
    tables=C.summarize(run)
    assert len(tables["geometry"])==8
    assert tables["probes"].status.eq("OK").all()
    assert tables["text"].status.eq("OK").all()
    figures=C.make_figures(run)
    assert len(figures)==6
    bundle=C.export_review(run)
    assert bundle.is_file()
    assert before=={str(p):C.sha(p) for p in bank.rglob("*") if p.is_file()}
    # Hash audit must reject even a one-byte source change.
    with (bank/"inputs/apo/A.npy").open("ab") as f:f.write(b"x")
    try:
        C.audit_bank(cfg)
    except C.AuditError:
        pass
    else:
        raise AssertionError("Modified canonical input was not rejected")
    print("SYNTHETIC SMOKE PASS: authenticated fixture, text identity, spawned workers, pilot reuse, resume, summaries, six figure families, export, source immutability and tamper detection.")
    if context:context.cleanup()


if __name__=="__main__":main()
