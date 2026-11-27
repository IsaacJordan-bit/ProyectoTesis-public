"""Optional command-line entrypoint; the notebook is the recommended entrypoint."""
from pathlib import Path
import argparse, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e6_p20 as p
from p20_config import CONFIG

def main():
    parser=argparse.ArgumentParser(description='Controlled P20 A/P/O experiment')
    parser.add_argument('stage',choices=['preflight','prepare','pilot','run','counts','empirical','models','figures','finish','all'])
    args=parser.parse_args()
    if args.stage=='preflight':print(p.preflight(CONFIG).to_string(index=False));return
    c=p.prepare(CONFIG)
    if args.stage=='prepare':return
    if args.stage in ('pilot','all'):print(p.run(c,pilot=True))
    if args.stage in ('run','all'):print(p.run(c))
    if args.stage in ('counts','all'):print(p.aggregate(c))
    if args.stage in ('empirical','all'):print(p.empirical(c))
    if args.stage in ('models','all'):print(p.statistics(c))
    if args.stage in ('figures','all'):p.show(c)
    if args.stage in ('finish','all'):p.finalize(c);p.export_review(c)

if __name__=='__main__':main()
