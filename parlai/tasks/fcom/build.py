#!/usr/bin/env python3

# Copyright AIRBUS

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'fcom')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname1 = 'train-v{}.json'.format(version)
        fname2 = 'dev-v{}.json'.format(version)
        # url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
        # build_data.download(url + fname1, dpath, fname1)s
        # build_data.download(url + fname2, dpath, fname2)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
