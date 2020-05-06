import argparse


def mergefiles(directories, out_path):
    documents = []
    for directory in directories:
        print("- reading {}".format(directory))
        file = open(directory, mode="r", encoding='utf-8')
        documents.append(file.read().strip())
        file.close()
    out = '\n'.join(documents)
    with open(out_path, mode='x', encoding='utf-8') as out_file:
        out_file.write(out)


def lowerdoc(path,out):
    f=open(path,'r',encoding='utf-8')
    doc=f.read().strip().lower().replace('baykan','bakan').replace('\'','').replace('"','').replace('-','').replace(':','').replace('  ',' ')
    with open(out,'x',encoding='utf-8') as ofile:
        ofile.write(doc)

    f.close()
if __name__ == "__main__":

    lowerdoc('../data/xaa.txt','../data/data.txt')
    exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', '--list',dest='files', nargs='+', help='List of files to merge', required=True)
    parser.add_argument('--out', action='store', dest='out',help='Out path with .txt at the end')
    opt = parser.parse_args()

    mergefiles(opt.files,opt.out)