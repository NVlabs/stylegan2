# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""List of pre-trained StyleGAN2 networks located on Google Drive."""

import pickle
import dnnlib
import dnnlib.tflib as tflib

#----------------------------------------------------------------------------
# StyleGAN2 Google Drive root: https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7

gdrive_urls = {
    'gdrive:networks/stylegan2-car-config-a.pkl':                           'https://drive.google.com/uc?id=1MhZpQAqgxKTz22u_urk0HSXA-BOLMCLV',
    'gdrive:networks/stylegan2-car-config-b.pkl':                           'https://drive.google.com/uc?id=1MirO1UBmfF4c-aZDDrfyknOj8iO8Qvb2',
    'gdrive:networks/stylegan2-car-config-c.pkl':                           'https://drive.google.com/uc?id=1MlFg5VVajuPyPkFt3f1HGiJ6OBWAPdaJ',
    'gdrive:networks/stylegan2-car-config-d.pkl':                           'https://drive.google.com/uc?id=1MpM83SpDgitOab_icAWU12D5P2ZpCHFl',
    'gdrive:networks/stylegan2-car-config-e.pkl':                           'https://drive.google.com/uc?id=1MpsFaO0BFo3qhor0MN0rnPFQCr_JpqLm',
    'gdrive:networks/stylegan2-car-config-f.pkl':                           'https://drive.google.com/uc?id=1MutzVf8XjNo6TUg03a6CUU_2Vlc0ltbV',
    'gdrive:networks/stylegan2-cat-config-a.pkl':                           'https://drive.google.com/uc?id=1MvGHMNicQjhOdGs94Zs7fw6D9F7ikJeO',
    'gdrive:networks/stylegan2-cat-config-f.pkl':                           'https://drive.google.com/uc?id=1MyowTZGvMDJCWuT7Yg2e_GnTLIzcSPCy',
    'gdrive:networks/stylegan2-church-config-a.pkl':                        'https://drive.google.com/uc?id=1N2g_buEUxCkbb7Bfpjbj0TDeKf1Vrzdx',
    'gdrive:networks/stylegan2-church-config-f.pkl':                        'https://drive.google.com/uc?id=1N3iaujGpwa6vmKCqRSHcD6GZ2HVV8h1f',
    'gdrive:networks/stylegan2-ffhq-config-a.pkl':                          'https://drive.google.com/uc?id=1MR3Ogs9XQlupSF_al-nGIAh797Cp5nKA',
    'gdrive:networks/stylegan2-ffhq-config-b.pkl':                          'https://drive.google.com/uc?id=1MW5O1rxT8CsPfJ9i7HF6Xr0qD8EKw5Op',
    'gdrive:networks/stylegan2-ffhq-config-c.pkl':                          'https://drive.google.com/uc?id=1MWfZdKNqWHv8h2K708im70lx0MDcP6ow',
    'gdrive:networks/stylegan2-ffhq-config-d.pkl':                          'https://drive.google.com/uc?id=1MbdyjloQxe4pdAUnad-M08EZBxeYAIOr',
    'gdrive:networks/stylegan2-ffhq-config-e.pkl':                          'https://drive.google.com/uc?id=1Md448HIgwM5eCdz39vk-m5pRbJ3YqQow',
    'gdrive:networks/stylegan2-ffhq-config-f.pkl':                          'https://drive.google.com/uc?id=1Mgh-jglZjgksupF0XLl0KzuOqd1LXcoE',
    'gdrive:networks/stylegan2-horse-config-a.pkl':                         'https://drive.google.com/uc?id=1N4lnXL3ezv1aeQVoGY6KBen185MTvWOu',
    'gdrive:networks/stylegan2-horse-config-f.pkl':                         'https://drive.google.com/uc?id=1N55ZtBhEyEbDn6uKBjCNAew1phD5ZAh-',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl':        'https://drive.google.com/uc?id=1NuS7MSsVcP17dgPX_pLMPtIf5ElcE3jJ',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl':      'https://drive.google.com/uc?id=1O7BD5yqSk87cjVQcOlLEGUeztOaC-Cyw',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl':        'https://drive.google.com/uc?id=1O2NjtullNlymC3ZOUpULCeMtvkCottnn',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl':      'https://drive.google.com/uc?id=1OMe7OaicfJn8KUT2ZjwKNxioJJZz5QrI',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl':    'https://drive.google.com/uc?id=1OpogMnDdehK5b2pqBbvypYvm3arrhCtv',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl':      'https://drive.google.com/uc?id=1OZjZD4-6B7W-WUlsLqXUHoM0XnPPtYQb',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl':        'https://drive.google.com/uc?id=1O7CVde1j-zh7lMX-gXGusRRSpY-0NY8L',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl':      'https://drive.google.com/uc?id=1OCJ-OZZ_N-_Qay6ZKopQFe4M_dAy54eS',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl':        'https://drive.google.com/uc?id=1OAPFAJYcJTjYHLP5Z29KlkWIOqB8goOk',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl':       'https://drive.google.com/uc?id=1N8wMCQ5j8iQKwLFrQl4T4gJtY_9wzigu',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl':     'https://drive.google.com/uc?id=1NRhA2W87lx4DQg3KpBT8QuH5a3RzqSXd',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl':       'https://drive.google.com/uc?id=1NBvTUYqzx6NZfXgmdOSyg-2PdrksEj8U',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl':     'https://drive.google.com/uc?id=1NhyfG5h9mbA400nUqejpOVyEouxbKeMx',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl':   'https://drive.google.com/uc?id=1Ntq-RrbSjZ-gxbRL46BoNrEygbsDkNrB',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl':     'https://drive.google.com/uc?id=1NkJi8o9pDRNCOlv-nYmlM4rvhB27UVc5',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl':       'https://drive.google.com/uc?id=1NdlwIO2nvQCfwyY-a-111B3aZQlZGrk8',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl':     'https://drive.google.com/uc?id=1Nheaxsq08HsTn2gTDlBydv90M818NeJk',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl':       'https://drive.google.com/uc?id=1Nfe0O5M-4654w0_5xvnSf-ng07vXIFBR',
}

#----------------------------------------------------------------------------

def get_path_or_url(path_or_gdrive_path):
    return gdrive_urls.get(path_or_gdrive_path, path_or_gdrive_path)

#----------------------------------------------------------------------------

_cached_networks = dict()

def load_networks(path_or_gdrive_path):
    path_or_url = get_path_or_url(path_or_gdrive_path)
    if path_or_url in _cached_networks:
        return _cached_networks[path_or_url]

    if dnnlib.util.is_url(path_or_url):
        stream = dnnlib.util.open_url(path_or_url, cache_dir='.stylegan2-cache')
    else:
        stream = open(path_or_url, 'rb')

    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
    _cached_networks[path_or_url] = G, D, Gs
    return G, D, Gs

#----------------------------------------------------------------------------
