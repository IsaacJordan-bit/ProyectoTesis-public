"""Execute actual notebook code cells sequentially on a synthetic fixture.

This avoids requiring a socket-enabled Jupyter kernel in a restricted test host.
Only the configuration cell is replaced with small synthetic paths/settings.
It tests cell logic and process spawning, not the Jupyter UI or native Windows.
"""
from pathlib import Path
import contextlib
import io
import json
import os
import sys
import tempfile

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from smoke_pipeline import fixture


def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        bank,ref,source=fixture(root)
        nb=json.loads((HERE/"P22_Contextual_Properties.ipynb").read_text(encoding="utf-8"))
        cfg=f'''BANK=Path({str(bank)!r})
SOURCE_PARQUET=Path({str(source)!r})
C.REFERENCE=Path({str(ref)!r})
CFG=C.Config(bank=BANK,output_parent=Path({str(root/'output')!r}),source_parquet=SOURCE_PARQUET,
 split_seeds=(0,1),probe_k=(1,2,4,8),training_fractions=(.25,.5,1.),control_repetitions=2,
 text_permutations=19,bootstrap_repetitions=50,nearest_neighbors=3,kmeans_n_init=2,
 max_workers=2,reserve_gib=1.,max_hours_per_call=.2)
'''
        namespace={"__name__":"__main__"}
        old=Path.cwd();os.chdir(HERE)
        try:
            executed=[]
            for i,cell in enumerate(nb["cells"]):
                if cell["cell_type"]!="code":continue
                source_code="".join(cell["source"])
                if "CFG = C.Config(" in source_code:source_code=cfg
                print(f"Executing notebook code cell {i}",flush=True)
                # Display objects and large tables need not clutter the test log.
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(source_code,f"notebook-cell-{i}","exec"),namespace)
                executed.append(i)
            assert namespace["STATUS"]["status"]=="COMPLETE"
            assert len(namespace["TABLES"]["geometry"])==8
            assert len(namespace["FIGURES"])==6
            assert namespace["REVIEW_BUNDLE"].is_file()
            print("NOTEBOOK CELL EXECUTION PASS:",executed)
        finally:
            os.chdir(old)


if __name__=="__main__":main()
