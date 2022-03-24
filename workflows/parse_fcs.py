import fcsparser, sys

path = sys.argv[1]
meta, data = fcsparser.parse(path, reformat_meta=True)

data.to_csv(path+'.csv')
