import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import subprocess

# consolidated parameter space
from param_space import *

def parse_time(log, phrase, token_idx=-2):
    remove_chars = ['%', '(', ')', ',']
    for line in log.split('\n'):
        if phrase in line:
            line = filter(lambda i: i not in remove_chars, line)
            if not line:
                print('WARNING: no %s found' % phrase)
            return float(line.split()[token_idx])


show_figs = False
plot= False

# nvprof, times, etc.
kernels = ['convolution']
metrics = ['GFlops']
results = np.zeros((len(kernels), len(metrics)) )

params = ''
for ki, kernel in enumerate( kernels ):
        for mi, metric in enumerate( metrics ):
                cmd=''
                try:
                    log = subprocess.check_output('egrep %s prof-%s-%s.txt | awk \'{print $2}\''
                            % (kernel, metric), stderr=subprocess.STDOUT, shell=True )
                    slog = log.decode('ascii')
                    remove_chars = [' ', '\t', '\n']
                    # print(slog)
                    if len(slog) == 0:
                        print('Error running command: ' + '"' + str(cmd) + '"' + ' see above shell error')
                    else:
                        slog
                        slog = filter(lambda i: i not in remove_chars, slog)
                        # if slog is None: assert(False)
                        # print(float( slog ))
                        # print('success\n')
                        results[ki][pi][mi][ai] = float(slog)
                except subprocess.CalledProcessError as e:
                    print(cmd)
                    print(e.output)
                    print('Error running command: ' + '"' + str(e.cmd) + '"' + ' see above shell error')
                    print('Return code: ' + str(e.returncode))
                    continue
    # PLOTTING
    if plot:
        for ci in caches:
            for pi, policy in enumerate( policies ):
                plt.plot(assocs, results[ki][pi][ci + 0 - 1] / results[ki][pi][ci + 1 - 1], label=policy)
            plt.xlabel('associativity')
            plt.xticks(assocs)
            plt.ylabel('hits / misses')
            title = 'Kernel `%s`: Cache hit to miss ratio %s' % ( kernel , 'L%d' % ci)
            plt.title(title)
            # plt.xlim(max(block_sizes) + 5, min(block_sizes) -5)
            plt.legend()
            plt.tight_layout()
            # fig_output_path = '%s.png' % (title.replace(' ', '_'), datetime.now().strftime(FORMAT))
            fig_output_path = '%s.png' % title.replace(' ', '_')
            print(fig_output_path)
            plt.savefig(fig_output_path)
            if show_figs:
                plt.show()
            plt.close()


