#!/usr/bin/env python

import sys
import subprocess

def main(argv):
    count = 0
    for line in sys.stdin:         # Iterate on lines
        input = line.split(' ')
        try:
            if len(input) == 1:#2:    # Apply rule 1
                #id = input[0]
                url = input[0][0:-1]
                count += 1
                args = ("./make_feature",url)
                popen = subprocess.Popen(args, stdout=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read()
                print '666'+'\t'+url+'\t'+output

        except "end of file":
            return None

if __name__ == '__main__':
    main(sys.argv)
