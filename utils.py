import os
import tqdm
import multiprocessing as mp

def run_parrlell_on_dir(dir, func):
    files_path = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for file in files:
            files_path.append(f'{dir}/{file}')
    with mp.Pool(mp.cpu_count() - 1) as pool:
        return list(tqdm.tqdm(pool.imap_unordered(func, files_path), total=len(files_path), desc=func.__name__))