import pandas as pd

def imdbDataframe(srcpath):
	videonames = os.listdir(srcpath)
	for videoname in videonames:
		filenames = os.listdir(srcpath+'/'+videoname)
		

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess GOT10K DATASET')
    parser.add_argument('-src',     '--datapath',       type=str,       required=True,         help='got10k')
    parser.add_argument('-cwd',     '--rootpath',       type=str,       required=True,         help=os.getcwd())
    args = parser.parse_args()