import os, sys
import glob


def main():
    countdir = './pressures_counts'
    dlst = glob.glob('%s/*.log' % countdir)

    countimelst = []
    realtimelst = []
    for ts in dlst:
        if os.path.isfile(ts):
            of = open(ts)
            for line in of.readlines():
                # ===---===spend time(Chinese GOP):1.69881272315979
                if line.find('spend time') != -1:
                    thistime = line.split(':', 1)[1]
                    countimelst.append(thistime) 
                elif line.find('==time') != -1:
                    thistime = line.split(':', 1)[1]
                    realtimelst.append(thistime) 
            of.close()

    avgtime = 0
    avgcount = 0
    for ts in countimelst:
        avgtime += float(ts)
        avgcount += 1

    avgT = avgtime / avgcount
    print('avg time(%s):' % avgcount, avgT)

    avgtime = 0
    avgcount = 0
    for ts in realtimelst:
        avgtime += float(ts)
        avgcount += 1

    avgT = avgtime / avgcount
    print('real time(%s):' % avgcount, avgT)

if __name__ == '__main__': main()
