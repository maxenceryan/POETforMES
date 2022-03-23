import time

timing_log = {}

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        
        if method.__name__  in timing_log:
            timing_log[method.__name__] += int((te - ts) * 1000)
        else:
            timing_log[method.__name__] = int((te - ts) * 1000)
        # if 'log_time' in kw:
        #     name = kw.get('log_name', method.__name__.upper())
        #     kw['log_time'][name] = int((te - ts) * 1000)
        # else:
        #     print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

def print_it():
    print("\nAggregated timing log...")
    logs = [(k,v) for k,v in timing_log.items()]
    logs.sort(key=lambda x: x[1], reverse=True)
    for log in logs:
        method, time = log
        print('%2.2f ms \t %r' % (time, method))
        
def reset():
    global timing_log
    timing_log = {}