from http.server import HTTPServer, BaseHTTPRequestHandler
from flask import Flask, request, jsonify
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs
import threading
import time, os, shutil
from collections import deque
import socket
import base64
import tempfile
import subprocess
from flask_apiexceptions import (JSONExceptionHandler, ApiException, ApiError, api_exception_handler)


debugpslog = True
app = Flask(__name__)
ext = JSONExceptionHandler(app)
ext.register(code_or_exception=ApiException, handler=api_exception_handler)
# scoremode : angel(天使) / rigorous(嚴謹)
minscores_rigorous = 80   # 80
minscores_angel = 60      # 60

prjDir = 'mixedZhuyin'
utt='UTT'
spk='SPK'
mapoov=' 2'

idToPhone = {}
idToWord = {}
wordToId = {}
wordToPhones = {}
phoneToZhuyin = {}
zhuyinToPinyin = {}
vowelIDs = set()
consonantIDs = set()

def init():
    os.environ['LD_LIBRARY_PATH'] = 'kaldi_rt/lib'
    modeldir = prjDir+'/exp/nnet3'
    os.system("kaldi_rt/bin/ivector-extract-online2 --config=0.conf ark:0.s2u :scp:%lld/feats.scp 'ark:|kaldi_rt/bin/copy-feats --compress=true ark:- ark,scp:%lld/ivec.ark,%lld/ivec.scp'&")

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
            if int(ID) < 2:
                vowelIDs.add(ID)
                consonantIDs.add(ID)
            elif int(ID) >= 40:
                continue
            elif any(v in phone for v in 'aeiou'):
                vowelIDs.add(ID)
            else:
                consonantIDs.add(ID)
            
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

    with open(prjDir+'/lang/phoneToZhuyin.txt', 'r') as f:
        for line in f:
            line = line.strip()
            phone, zhuyin = line.split()
            phoneToZhuyin[phone] = zhuyin

    with open(prjDir+'/lang/zhuyinToPinyin.txt', 'r') as f:
        for line in f:
            line = line.strip()
            tokens = line.split()
            zhuyinToPinyin[tokens[0]] = tokens[1]


def compute(wav, txt, dir):
    wavscp = dir + '/wav.scp'
    txtark = dir + '/txt.ark'
    with open(wavscp, 'w') as f:
        f.write('%s ./kaldi_wavs/%s\n' % (utt, wav))
    with open(txtark, 'w') as f:
        f.write(utt)
        for word in txt.split():
            try:
                f.write(wordToId[word])
            except:
                f.write(mapoov)
    pslog('SSSSSSSSSSSSSSS--1')
    cmd = "kaldi_rt/bin/compute-mfcc-feats --verbose=2 --config=%s/conf/mfcc_hires.conf scp,p:%s ark:- | kaldi_rt/bin/copy-feats --compress=true ark:- ark,scp:%s/feats.ark,%s/feats.scp" % (prjDir, wavscp, dir, dir)
    os.system(cmd)
    pslog('SSSSSSSSSSSSSSS--2')
    cmd = "kaldi_rt/bin/compute-cmvn-stats --spk2utt=ark:0.s2u scp:%s/feats.scp ark,scp:%s/cmvn.ark,%s/cmvn.scp" % (dir, dir, dir)
    os.system(cmd)
    pslog('SSSSSSSSSSSSSSS--3') 
    sk=socket.socket()
    sk.connect(('localhost', 12345))
    sk.send(int(dir).to_bytes(8, 'little'))
    sk.recv(1)
    sk.close()
    pslog('SSSSSSSSSSSSSSS--4')
    cmd = "kaldi_rt/bin/compute-dnn-gop --use-gpu=no %s/exp/nnet3/tdnn_sp/tree %s/exp/nnet3/tdnn_sp/final.mdl %s/lang/L.fst 'ark,s,cs:kaldi_rt/bin/apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:0.u2s scp:%s/cmvn.scp scp:%s/feats.scp ark:- |' scp:%s/ivec.scp ark:%s ark,t:%s/gop ark,t:%s/align ark,t:%s/phoneme_ll" % (prjDir, prjDir, prjDir, dir, dir, dir, txtark, dir, dir, dir)
    os.system(cmd)
    print('done computing gop.')


def parseGOPOutput(gopFilePath, topN=3, scoremode='angel', ans=None, a=1, b=2):
    # =============================== begin of utility function definitions ========================== #
    def GOPscore(p, adjust=False, a=0.01, b=2):
        f = lambda x: 10*(x**0.5)
        score = 100/(1+(p/a)**b)
        return f(score) if adjust else score

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

    def compareZhuyin(ans, preds):        
        def phoneToZhuyinWithIndex(phSeq, diffs):
            retZhuyin = ''
            zhuyinDiffs = []
            for i, ph in enumerate(phSeq):
                curZhuyin = phoneToZhuyin.get(ph, '*')
                if i in diffs:
                    zhuyinDiffs.extend([len(retZhuyin)+j for j in range(len(curZhuyin))])
                retZhuyin += curZhuyin
            return retZhuyin, zhuyinDiffs

        ret, ans_diffs, pred_diffs = [], [], []
        if len(ans) == len(preds):
            for i, val in enumerate(ans):
                curPreds = [phoneToZhuyin.get(x, x) for x in preds[i]]
                curVal = phoneToZhuyin.get(val, val)
                if curVal in curPreds:
                    ret.append(val)
                else:
                    choices = [x for x in preds[i] if x not in ['SPN']]
                    topChoice = choices[0] if len(choices)>0 else 'sil'
                    ret.append(topChoice)
                    ans_diffs.append(i)
                    pred_diffs.append(i)
        else:
            # find LCS first
            m, n = len(ans), len(preds)
            L = [[0 for _ in range(m+1)] for _ in range(n+1)]
            for i in range(n+1):
                for j in range(m+1):
                    if i == 0 or j == 0:
                        L[i][j] = 0
                    elif ans[j-1] in preds[i-1]:
                        L[i][j] = L[i-1][j-1]+1
                    else:
                        L[i][j] = max(L[i-1][j], L[i][j-1])
            i, j = n, m
            while i > 0 and j > 0:
                if ans[j-1] in preds[i-1]:
                    ret.append(ans[j-1])
                    j -= 1
                    i -= 1
                elif L[i-1][j] > L[i][j-1]: # deletion
                    choices = [x for x in preds[i-1] if x not in ['SPN']]
                    topChoice = choices[0] if len(choices)>0 else 'sil'
                    ret.append(topChoice)
                    pred_diffs.append(i-1)
                    i -= 1
                else: # insertion
                    ans_diffs.append(j-1)
                    j -= 1
            ret = ret[::-1]
            pred_diffs = pred_diffs[::-1]
            ans_diffs = ans_diffs[::-1]

        total, error = len(ans), max(len(ans_diffs), len(pred_diffs))
        predZhuyin, predDiffs = phoneToZhuyinWithIndex(ret, pred_diffs)
        zhuyinAns, ansDiffs = phoneToZhuyinWithIndex(ans, ans_diffs)
        singleZhuyinToPinyin = {phoneToZhuyin[k]: k for k in phoneToZhuyin}
        predPinyinBest = zhuyinToPinyin.get(predZhuyin, '+'.join(singleZhuyinToPinyin.get(x, x) for x in predZhuyin))
        
        return {'zhuyinAns': zhuyinAns, 
                'ansZhuyinDiffIndex': ansDiffs,
                'ansPhone': ans, 'ansPhoneDiffIndex': ans_diffs,
                'predZhuyinBest': predZhuyin, 
                'predZhuyinDiffIndex': predDiffs,
                'predPhoneBest': ret, 'predPhoneDiffIndex': pred_diffs,
                'predPinyinBest': predPinyinBest,
                'predPhoneError': (error, total)}

    def scoreUtt(gopParsed):
        for i, row in enumerate(gopParsed['parts']):
            correctPhSeq = gopParsed['parts'][i]['phone'].split('_')
            phSeqLength = len(correctPhSeq)
            diagnosis = compareZhuyin(correctPhSeq, gopParsed['parts'][i]['predPhone'])            
            gopParsed['parts'][i].update(diagnosis)

    # ======================== end of utility function definitions ================================= #
#     print('argument a = %s, b = %s' % (a, b))
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
                
                curPhone = idToPhone[str(phoneID)]
                isVowel = lambda x: any(v in x for v in 'aeiou')
                
                raw_cpls = list(map(float, line[line.find('[')+1:line.find(']')].split()))
                raw_cpls = sorted([(val, i+1) for i, val in enumerate(raw_cpls)], reverse=True)
                cpls = [x for x in raw_cpls if str(x[1]) in vowelIDs] if isVowel(curPhone) else [x for x in raw_cpls if str(x[1]) in consonantIDs]
                
                rank = findRank(cpls, phoneID)
                rank = max(rank-topN+1, 1)
                rr = float(rank-1)/float(len(cpls)-topN+1)
                GOPData.setdefault('phoneRRs', []).append(rr)
                topNpred = tuple([idToPhone[str(cpls[j][1])] for j in range(topN)])
                GOPData.setdefault('predictPhone', []).append(topNpred)

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
    if scoremode == 'rigorous':
        minscores = minscores_rigorous
    else:
        minscores = minscores_angel    
    wb = wb[0][::-1]
    for w, phoneSeq in wb:
        phBuf, phGOPBuf, phRRBuf, intervalBuf, predPh = list(zip(*phoneSeq))
        phGOPScore = list(map(lambda x: GOPscore(x, adjust=True, a=a, b=b), phGOPBuf))
        phRRScore = list(map(lambda x: GOPscore(x, adjust=True), phRRBuf))
        cur = {}
        cur['word'] = w
        cur['pinyin_notone'] = w
        cur['phone'] = '_'.join(phBuf)
        cur['rawGOP'] = phGOPBuf
        cur['rawRR'] = phRRBuf
        cur['rawGOPScores'] = phGOPScore
        cur['rawRRScores'] = phRRScore
        cur['GOPScore'] = sum(phGOPScore)/len(phGOPScore)
        cur['RRScore'] = sum(phRRScore)/len(phRRScore)
        cur['phoneIntervals'] = intervalBuf
        cur['intervals'] = (min(x[0] for x in intervalBuf), max(x[1] for x in intervalBuf))
        cur['predPhone'] = predPh

        # 計算每個聲母、韻母分數，必須大於等於 minscores=60
        RR_status = "true" if cur['RRScore'] >= minscores else "false"
        if RR_status == "true": 
            total_correct_RR += 1
        cur['RR_Result'] = RR_status

        total_RRScore += cur['RRScore']
        cur['start_time'] = min(x[0] for x in intervalBuf)
        cur['end_time'] = max(x[1] for x in intervalBuf)

        # 計算每個聲母、韻母分數，必須大於等於 minscores=60
        GOP_status = "true" if cur['GOPScore'] >= minscores else "false"    
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
              'RR_scores_avg' : avg_rrscore,
              'RR_correct_avg': avg_rrscore,              
              'GOP_scores': avg_gopscore,
              'GOP_total': total_word, 'GOP_correct': total_correct_GOP,
              'parts': wordGOP,
              'scoremode' : scoremode, 'arg_a': a, 'arg_b': b,
              'minscores' : minscores}
    scoreUtt(gopdic)
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


def pslog(msg, msg1=''):
    if debugpslog:
        wf = open('/tmp/logs.txt', 'a')
        ## 2014.12.22 加上讓tranking.pslog 加上debug時間功能
        if os.environ.get('pslogobtime', ''):
            import time
            ptime = time.time()
            wf.write('%s--%s:%s\n' % (ptime, msg, msg1))
        else:
            wf.write('%s:%s\n' % (msg, msg1))
        wf.close()


def handle_uploaded_file_base64(imgstring, UPLOAD_PATH='./kaldi_wavs/'):
    import base64
    if not os.path.isdir(UPLOAD_PATH):
        os.mkdir(UPLOAD_PATH)
    timeseries = int(round(time.time() * 1000000000))
    wavfilename = 'upload_%s.wav' % (timeseries)
    fileabspath = '%s%s' % (UPLOAD_PATH, wavfilename)    
    imgdata = base64.b64decode(imgstring)
    with open(fileabspath, "wb") as fh:
        fh.write(imgdata)    
    return fileabspath


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
          try:
              scoremode = qs['scoremode'][0]
          except: scoremode = 'angel'
          try:
              a, b = float(qs['a'][0]), float(qs['b'][0])
          except: a, b = 1, 2
          gopParsed, ctmParsed = parseGOPOutput(dir+'/gop', topN=3, scoremode=scoremode, a=a, b=b)
          gopstring = 'GOP-json:%s' % str(gopParsed)
          self.wfile.write(str(gopstring).encode())
          self.wfile.write(b'\n')
          ctmstring = 'CTM-json:%s' % str(ctmParsed)
          self.wfile.write(str(ctmstring).encode())
        except:
          self.wfile.write(b'Failed')

#         shutil.copy2(dir+'/gop', '/models/kaldi_wavs/st001_gop.txt')
        shutil.rmtree(dir)
        fin = '\n===---===spend time(Chinese GOP):%f\n' % (time.time()-tm1)
        self.wfile.write(fin.encode())


@app.route('/predict', methods=['POST'])
def predict():
    pslog('==================START==================')
    # dir = str(threading.get_ident())
    # 路徑長度一定要15，否則會有問題
    tmpdir = '1' + str(int(round(time.time() * 10000000000)))[-14:]
    try:
        os.mkdir(tmpdir) 
    except: pass
    pslog('tmpdir', tmpdir) 
    error_gop = ApiError(code='gopscore', message='process can not complete gop score.')
    if request.method == 'POST':
        qs = request.form
        # word = request.form.get('word')
        # print('qs', qs)
        try:
            pslog('P--1')
            base64_file_path = '' 
            pslog('qs.keys()', qs.keys())  
            if qs.get('base64'):
                pslog('cond--1', qs.get('txt'))
                pslog('cond--base', qs.get('base64')[0:1000])
                base64_file_path = handle_uploaded_file_base64(qs.get('base64'))
                pslog('base64_file_path', base64_file_path)
                fh, ft = os.path.split(base64_file_path)
                pslog('ft', ft)
                compute(ft, qs.get('txt'), tmpdir)            
            else:
                pslog('cond-2', qs.get('txt'))
                compute(qs.get('wav'), qs.get('txt'), tmpdir)
            pslog('P--2')        
            try:
                scoremode = qs.get('scoremode', 'angel')
            except: scoremode = 'angel'
            pslog('P--4')        
            try:
                a, b = float(qs.get('a')), float(qs.get('b'))
            except: a, b = 1, 2
            pslog('gop-a', a)
            pslog('gop-b', b)
            pslog('scoremode', scoremode)
            gopParsed, ctmParsed = parseGOPOutput(dir+'/gop', topN=3, scoremode=scoremode, a=a, b=b)
            # gopstring = 'GOP-json:%s' % str(gopParsed)
            # print('gopstring', gopstring)
            pslog('P--5')
            # self.wfile.write(str(gopstring).encode())
            ctmstring = 'CTM-json:%s' % str(ctmParsed)
            # 移除base64 生成的檔案
            if os.path.isfile(base64_file_path):                
                try:
                    pslog('remove file', base64_file_path)
                    os.remove(base64_file_path)
                except: pass
            shutil.rmtree(tmpdir)
            return jsonify({"error": False, "gop": gopParsed, "ctm": ctmParsed, "err_msg": None})
        except:
            shutil.rmtree(tmpdir)
            raise ApiException(status_code=422, error=error_gop)        
        return jsonify({"error": True, "err_msg": "Gop json can not generate!"})


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=8505, debug=False)    
