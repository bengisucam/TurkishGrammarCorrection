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
    mergefiles('../data/', opt.out)
