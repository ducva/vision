from fastai.core import *
from fastai.datasets import URLs, Config


def is_tar(url: str):
    return url.endswith('.tar.gz')


def is_gzip(url: str):
    return url.endswith('.tgz')


def modelpath4file(filename):
    """
    Returns URLs.MODEL path if file exists. Otherwise returns config path"
    :param filename:
    :return:
    """
    local_path = URLs.LOCAL_PATH / 'models' / filename
    if local_path.exists():
        return local_path
    else:
        return Config.model_path() / filename


def datapath4file(filename):
    "Returns URLs.DATA path if file exists. Otherwise returns config path"
    local_path = URLs.LOCAL_PATH / 'data' / filename
    if local_path.exists():
        return local_path
    else:
        return Config.data_path() / filename


def url2name(url: str):
    return url.split('/')[-1]


def _url2path(url: str, is_data=True):
    return datapath4file(f'{url2name(url)}') if is_data else modelpath4file(f'{url2name(url)}')


def download_data(url: str, fname: PathOrStr = None, is_data: bool = True):
    """
    Download `url` to destination `fname`
    :param url:
    :param fname:
    :param is_data:
    :return:
    """
    fname = Path(ifnone(fname, _url2path(url, is_data)))
    os.makedirs(fname.parent, exist_ok=True)
    if not fname.exists():
        print(f'Downloading {url}')
        download_url(f'{url}', fname)
    else:
        print(f'Data exists: {fname}')
    return fname


def untar_data(url: str, fname: PathOrStr = None, dest: PathOrStr = None, data=True):
    """
    Download `url` if it doesn't exist to `fname` and un-tgz to folder `dest`
    :param url:
    :param fname:
    :param dest:
    :param data:
    :return:
    """
    dest = Path(ifnone(dest, _url2path(url, data)))
    if not dest.exists():
        fname = download_data(url, fname=fname, is_data=data)
        mode = 'r:gz' if is_gzip(url) else 'r:bz2'
        tarfile.open(fname, mode).extractall(dest.parent)
    else:
        print('Data existed')
    return dest
