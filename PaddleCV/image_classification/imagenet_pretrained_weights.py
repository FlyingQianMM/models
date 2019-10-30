import os
import requests
import os.path as osp
import tarfile
import shutil
import tqdm
import hashlib

DOWNLOAD_RETRY_LIMIT = 3

pretrained_weights_urls = {
    "ResNet18": "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar"
}

fc_vars = {
    "ResNet18": ["fc_0.b_0", "fc_0.w_0"]
}

def check_md5sum(fullname, md5sum=None):
    if md5sum is None:
        return True
    print("File {} md5sum checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        print("File {} md5 check failed, {}(calc) != {}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True

def decompress(fname):
    print("Decompress {}...".format(fname))
    fpath = '/'.join(fname.split('/')[:-1])
    fpath_tmp = osp.join(fpath, 'tmp')
    if osp.isdir(fpath_tmp):
        shutil.rmtree(fpath_tmp)
        os.makedirs(fpath_tmp)

    if fname.find('.tar') >= 0:
        with tarfile.open(fname) as tf:
            tf.extractall(path=fpath_tmp)
    elif fname.find('.zip') >= 0:
        with zipfile.ZipFile(fname) as zf:
            zf.extractall(path=fpath_tmp)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    for f in os.listdir(fpath_tmp):
        src_dir = osp.join(fpath_tmp, f)
        dst_dir = osp.join(fpath, f)
        move_and_merge_tree(src_dir, dst_dir)

    shutil.rmtree(fpath_tmp)
    os.remove(fname)
    return dst_dir

def move_and_merge_tree(src, dst):
    """
    Move src directory to dst, if dst is already exists,
    merge src to dst
    """
    if not osp.exists(dst):
        shutil.move(src, dst)
    else:
        for fp in os.listdir(src):
            src_fp = osp.join(src, fp)
            dst_fp = osp.join(dst, fp)
            if osp.isdir(src_fp):
                if osp.isdir(dst_fp):
                    _move_and_merge_tree(src_fp, dst_fp)
                else:
                    shutil.move(src_fp, dst_fp)
            elif osp.isfile(src_fp) and \
                    not osp.isfile(dst_fp):
                shutil.move(src_fp, dst_fp)

def get_pretrained_weights(model_name, path, md5sum=None):
    assert model_name in pretrained_weights_urls, "{} is not a valid model name"
    if not osp.exists(path):
        os.makedirs(path) 

    url = pretrained_weights_urls[model_name]
    fname = url.split('/')[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and check_md5sum(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))
        
        print("Downloading {} from {}".format(fname, url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                for chunk in tqdm.tqdm(
                        req.iter_content(chunk_size=1024),
                        total=(int(total_size) + 1023) // 1024,
                        unit='KB'):
                    f.write(chunk)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

    dst_dir = decompress(fullname)

    return dst_dir

def remove_fc_vars(pretrained_weights_dir, model_name):
    vars = fc_vars[model_name]
    for var in vars:
        weight_file = osp.join(pretrained_weights_dir, var)
        os.remove(weight_file)
