import time

def tic():
    globals()['tt'] = time.clock()

def toc(end=False):
    interv = time.clock() - globals()['tt']
    return interv
    # print '\nElapsed time:%.8f seconds\n' % (time.clock()-globals()['tt'])
    # if end:
    #     exit()