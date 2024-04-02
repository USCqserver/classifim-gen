import datetime
import random

def get_lambdas(max_log2_resolution=6, job_id=0, num_jobs=1, random_order=False):
    """Returns a list of (lambda0, lambda1) tuples to be processed by this job.
    """
    res = []
    for log2_resolution in range(max_log2_resolution + 1):
        resolution = 2**log2_resolution
        cur_res = [
            (lambda0i / resolution, lambda1i / resolution)
            for lambda0i in range(resolution)
            for lambda1i in range(resolution)
            if resolution == 1 or lambda0i % 2 == 1 or lambda1i % 2 == 1]
        res.extend(cur_res)
    job_id = job_id % num_jobs
    res = res[job_id::num_jobs]
    if random_order:
        random.shuffle(res)
    return res
