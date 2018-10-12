import os

def dirs(path):
    return sorted([x for x in os.listdir(path)
                   if os.path.isdir(path + '/' + x)])

def check_isnp(path, isnp):
    ds = dirs(path)

    if isnp not in dirs(path):
        raise FileNotFoundError('isnp {} not found in {}'.format(isnp, path))
    
