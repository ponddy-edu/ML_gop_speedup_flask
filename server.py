from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs
import threading
import time, os, shutil
from collections import deque
import socket

minscores = 60
prjDir = 'mixedZhuyin'
utt='UTT'
spk='SPK'
mapoov=' 2'

idToPhone = {}
idToWord = {}
wordToId = {}
wordToPhones = {}

def init():
    os.environ['LD_LIBRARY_PATH'] = 'kaldi_rt/lib'
    modeldir = prjDir+'/exp/nnet3'
    os.system("kaldi_rt/bin/ivector-extract-online2 --config=0.conf ark:0.s2u :12345:scp:%lld/feats.scp 'ark:|kaldi_rt/bin/copy-feats --compress=true ark:- ark,scp:%lld/ivec.ark,%lld/ivec.scp'&")

    with open('splice.conf', 'w') as f:
      with open(modeldir+'/extractor/splice_opts', 'r') as f1:
        for opt in f1.read().split():
          f.write(opt+'\n')

    with open('0.conf', 'w') as f:
      f.write('--cmvn-config=%s/extractor/online_cmvn.conf\n' % modeldir)
      f.write('--ivector-period=10\n')
      f.write('--splice-config=splice.conf\n')
      f.write('--lda-matrix=%s/extractor/final.mat\n' % modeldir)
      f.write('--global-cmvn-stats=%s/extractor/global_cmvn.stats\n' % modeldir)
      f.write('--diag-ubm=%s/extractor/final.dubm\n' % modeldir)
      f.write('--ivector-extractor=%s/extractor/final.ie\n' % modeldir)
      f.write('--num-gselect=5\n')
      f.write('--min-post=0.025\n')
      f.write('--posterior-scale=0.1\n')
      f.write('--max-remembered-frames=1000\n')
      f.write('--max-count=0\n')

    with open('0.u2s', 'w') as f:
      f.write('%s %s\n' % (utt, spk))

    with open('0.s2u', 'w') as f:
      f.write('%s %s\n' % (spk, utt))

    with open(prjDir+'/lang/phones.txt', 'r') as f:
        for line in f:
            line = line.strip()
            phone, ID = line.split()
            idToPhone[ID] = phone
            
    with open(prjDir+'/lang/words.txt', 'r') as f:
        for line in f:
            line = line.strip()
            word, ID = line.split()
            idToWord[ID] = word
            wordToId[word]=' '+ID

    with open(prjDir+'/dict/lexicon.txt', 'r') as f:
        for line in f:
            line = line.strip()
            tokens = line.split()
            wordToPhones.setdefault(tokens[0], []).append(tokens[1:])

def compute(wav, txt, dir):
    wavscp = dir + '/wav.scp'
    txtark = dir + '/txt.ark'
    with open(wavscp, 'w') as f:
      f.write('%s /data/%s\n' % (utt, wav))

    with open(txtark, 'w') as f:
      f.write(utt)
      for word in txt.split():
        try:
          f.write(wordToId[word])
        except:
          f.write(mapoov)

    os.system("kaldi_rt/bin/compute-mfcc-feats --verbose=2 --config=%s/conf/mfcc_hires.conf scp,p:%s ark:- | kaldi_rt/bin/copy-feats --compress=true ark:- ark,scp:%s/feats.ark,%s/feats.scp" % (prjDir, wavscp, dir, dir))

    os.system("kaldi_rt/bin/compute-cmvn-stats --spk2utt=ark:0.s2u scp:%s/feats.scp ark,scp:%s/cmvn.ark,%s/cmvn.scp" % (dir, dir, dir))

    sk=socket.socket()
    sk.connect(('localhost', 12345))
    sk.send(int(dir).to_bytes(8, 'little'))
    sk.recv(1)
    sk.close()

    os.system("kaldi_rt/bin/compute-dnn-gop --use-gpu=no %s/exp/nnet3/tdnn_sp/tree %s/exp/nnet3/tdnn_sp/final.mdl %s/lang/L.fst 'ark,s,cs:kaldi_rt/bin/apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:0.u2s scp:%s/cmvn.scp scp:%s/feats.scp ark:- |' scp:%s/ivec.scp ark:%s ark,t:%s/gop ark,t:%s/align ark,t:%s/phoneme_ll" % (prjDir, prjDir, prjDir, dir, dir, dir, txtark, dir, dir, dir))

def parseGOPOutput(gopFilePath):
    def GOPscore(p, a=0.01, b=2):
        return 100/(1+(p/a)**b)  

    def findRank(cpls, phoneID):
        for i, val in enumerate(cpls):
            ll, ph = val
            if ph == phoneID:
                return i+1

    def matchWord(words, pairs):
        pair = [x for x in list(pairs.copy()) if x[0] not in ['SPN', 'SIL', 'sil']]
        def helper(ws, ps):
            ans = []
            if len(ws) == 0:
                return len(ps) == 0, [[]]
            #print(ps)
            for phoneSeq in wordToPhones[ws[0]]:
                n = len(phoneSeq)
                if len(ps) < n:
                    continue
                candidate = [ps[j][0] for j in range(n)]
                #if not cmpPhoneSeq(phoneSeq, candidate):
                if phoneSeq != candidate:
                    #print(phoneSeq, candidate)
                    continue
                #print(ws[1:], ps[n:])
                flag, subans = helper(ws[1:], ps[n:])
                #print('subans:', subans)
                if flag:
                    for sa in subans:
                        sa.append((ws[0], ps[:n]))
                    ans.extend(subans)
            return len(ans) > 0, ans
        flag, res = helper(words, pair)
        if not flag:
            print('matching error!')
        return res

    GOPData = {}
    with open(gopFilePath, 'r') as f:
        i = 0
        for line in f:
            line = line.strip()
            if line[0].isalpha():
                data = line[line.find('[')+1:line.find(']')].split()
                if i == 0:
                    GOPData['phoneGOPs'] = list(map(float, data))
                elif i == 1:
                    GOPData['words'] = list(map(lambda x: idToWord[x], data))
                elif i == 2:
                    GOPData['phones'] = list(map(lambda x: idToPhone[x], data))
                else:
                    tmp = list(map(float, data))
                    intervals = []
                    for j, val in enumerate(tmp):
                        if j == 0:
                            intervals.append((0.0, val))
                        else:
                            intervals.append((tmp[j-1], val))
                    GOPData['intervals'] = intervals

                i = (i+1)%4
            else:
                phoneID = int(line[:line.find('[')])
                cpls = list(map(float, line[line.find('[')+1:line.find(']')].split()))
                cpls = sorted([(val, i+1) for i, val in enumerate(cpls)], reverse=True)
                rank = findRank(cpls, phoneID)
                rr = float(rank-1)/float(len(cpls))
                GOPData.setdefault('phoneRRs', []).append(rr)
                GOPData.setdefault('predictPhone', []).append(idToPhone[str(cpls[0][1])])

    pairs = deque(list(zip(GOPData['phones'], GOPData['phoneGOPs'], GOPData['phoneRRs'], GOPData['intervals'], GOPData['predictPhone'])))
    wordGOP = []
    wb = matchWord(GOPData['words'], pairs)
    if len(wb) == 0:
        print('Error!')
        return wordGOP

    total_GOPScore = 0
    total_RRScore = 0

    total_word = 0
    total_correct_RR = 0
    total_correct_GOP = 0

    wb = wb[0][::-1]
    for w, phoneSeq in wb:
        phBuf, phGOPBuf, phRRBuf, intervalBuf, predPh = list(zip(*phoneSeq))
        phGOPScore = list(map(GOPscore, phGOPBuf))
        phRRScore = list(map(GOPscore, phRRBuf))
        cur = {}
        cur['word'] = w
        cur['phone'] = '_'.join(phBuf)
        cur['rawGOP'] = phGOPBuf
        cur['rawRR'] = phRRBuf
        cur['rawGOPScores'] = phGOPScore
        cur['rawRRScores'] = phRRScore
        cur['GOPScore'] = min(phGOPScore)
        cur['RRScore'] = min(phRRScore)
        cur['phoneIntervals'] = intervalBuf
        cur['intervals'] = (min(x[0] for x in intervalBuf), max(x[1] for x in intervalBuf))
        cur['predPhone'] = predPh

        # 計算每個聲母、韻母分數，必須大於等於 minscores=60
        RR_status = "true"
        for sv in phRRScore:
            if sv < minscores: RR_status = "false"
        if RR_status == "true": 
            total_correct_RR += 1
        cur['RR_Result'] = RR_status

        total_RRScore += cur['RRScore']

        # 計算每個聲母、韻母分數，必須大於等於 minscores=60
        GOP_status = "true"
        for sv in phGOPScore:
            if sv < minscores: GOP_status = "false"        
        cur['GOP_Result'] = GOP_status
        if GOP_status == "true":
            total_correct_GOP += 1

        total_GOPScore += cur['GOPScore']
        
        wordGOP.append(cur)
        total_word += 1  

    avg_gopscore = int(total_GOPScore / total_word)
    avg_rrscore = int(total_RRScore / total_word)
    #print('avg_gopscore', avg_gopscore)
    #print('avg_rrscore', avg_rrscore)

    gopdic = {'RR_scores': avg_rrscore, 
              'RR_total': total_word, 'RR_correct': total_correct_RR,
              "GOP_scores": avg_gopscore,
              'GOP_total': total_word, 'GOP_correct': total_correct_GOP,
              'parts': wordGOP}
    #print('gopdic', gopdic)

    wordCTM = []
    pairs = deque(list(zip(GOPData['phones'], GOPData['phoneGOPs'], GOPData['phoneRRs'], GOPData['intervals'], GOPData['predictPhone'])))
    wb = matchWord(GOPData['words'], pairs)
    if len(wb) == 0:
        print('Error!')
        return wordCTM
    wb = wb[0][::-1]
    for w, phoneSeq in wb:
        phBuf, phGOPBuf, phRRBuf, intervalBuf, predPh = list(zip(*phoneSeq))
        phGOPScore = list(map(GOPscore, phGOPBuf))
        phRRScore = list(map(GOPscore, phRRBuf))
        cur = {}
        cur['word'] = w
        cur['beg'] = min(x[0] for x in intervalBuf)
        cur['end'] = max(x[1] for x in intervalBuf)
        wordCTM.append(cur)
    return gopdic, wordCTM

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        tm1 = time.time()
        self.send_response(200)
        self.end_headers()
        dir = str(threading.get_ident())
        os.mkdir(dir)

        try:
          qs = parse_qs(self.path[self.path.index('?')+1:])
          compute(qs['wav'][0], qs['txt'][0], dir)
          gopParsed, ctmParsed = parseGOPOutput(dir+'/gop')
          self.wfile.write(str(gopParsed).encode())
          self.wfile.write(b'\n')
          self.wfile.write(str(ctmParsed).encode())
        except:
          self.wfile.write(b'Failed')

        shutil.rmtree(dir)
        fin = '\n===---===spend time(Chinese GOP):%f\n' % (time.time()-tm1)
        self.wfile.write(fin.encode())

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

if __name__ == '__main__':
    init()
    ThreadedHTTPServer(('', 8085), Handler).serve_forever()
