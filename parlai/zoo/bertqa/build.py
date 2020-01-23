import parlai.core.build_data as build_data
import os


def download(datapath, version='v1.0'):
    dpath = os.path.join(datapath, 'models', 'bertqa')

    #if not build_data.built(dpath, version):
    #    print('[downloading bertqa models: ' + dpath + ']')
        # if build_data.built(dpath):
        #     # An older version exists, so remove these outdated files.
        #     build_data.remove_dir(dpath)
        # build_data.make_dir(dpath)

        # # Download the data.
        # fnames = ['bert-base-uncased.tar.gz', 'bert-base-uncased-vocab.txt']
        # for fname in fnames:
        #     url = 'https://s3.amazonaws.com/models.huggingface.co/bert/' + fname
        #     build_data.download(url, dpath, fname)

        # Mark the data as built.
        # build_data.mark_done(dpath, version)