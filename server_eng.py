from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs
import threading
import time, os, shutil
from collections import deque
import functools
import socket
import traceback

# scoremode : angel(天使) / rigorous(嚴謹)
minscores_rigorous = 80   # 80
minscores_angel = 60      # 60

prjDir = 'ponddyEng'
utt='UTT'
spk='SPK'
mapoov=' 2'

vowels = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'ER', 'EY', 'IH', 'IY', 'IX', 'OW', 'OY', 'UH', 'UW'])

idToPhone = {}
idToWord = {}
wordToId = {}
wordToPhones = {}
phoneToIPA = {}
ipaToCMU = {}
vowelIDs, consonantIDs = set(), set()

ExceptList = [('ɪ', 'iː'),]

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
            if int(ID) <= 2:
                vowelIDs.add(ID)
                consonantIDs.add(ID)
            elif int(ID) >= 87:
                continue
            elif any(v in phone for v in 'AEIOU'):
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

    with open(prjDir+'/lang/phoneToIPA.txt', 'r') as f:
        for line in f:
            line = line.strip()
            phone, IPA = line.split()
            phoneToIPA[phone] = IPA

    with open(prjDir+'/lang/ipaToCMU.txt', 'r') as f:
        for line in f:
            line = line.strip()
            ipa, cmu = line.split('\t')
            ipaToCMU[ipa] = cmu.split()

def compute(wav, txt, dir, ipa_ans=None):
    wavscp = dir + '/wav.scp'
    txtark = dir + '/txt.ark'
    with open(wavscp, 'w') as f:
      f.write('%s /models/kaldi_wavs/%s\n' % (utt, wav))

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
    print('done computing gop.')

def parseGOPOutput(gopFilePath, ipa_ans=None, cmu_ans = None, topN=3, scoremode='angel', a=1, b=2, word_model=False):
    # ============================== Begin of utility function definition ============================= #
    def convertIPAToCMU(ipa):
        ipaToPhone = {phoneToIPA[x]: x for x in phoneToIPA}
        ipa = ipa.replace('ˈ', '')
        ipa = ipa.replace('ˌ', '')
        def dfs(s):
            if len(s) == 0:
                return True, []
            for n in range(min(3, len(s)), 0, -1):
                if s[:n] in ipaToPhone:
                    subAns = dfs(s[n:])
                    if subAns[0]:
                        return True, [ipaToPhone[s[:n]]]+subAns[1]
            return False, []
        ret = dfs(ipa)
        if ret[0]:
            return ret[1]

    def GOPscore(p, adjust=False, a=0.01, b=2):
        f = lambda x: 10*(x**0.5)
        score = 100/(1+(p/a)**b)
        return f(f(score)) if adjust else f(score)

    def findRankEng(cpls, phoneID, idToPhone):
        #print(cpls)
        targetPhone = idToPhone[str(phoneID)]
        for i, val in enumerate(cpls):
            ll, ph = val
            ph = idToPhone[str(ph)]
            #print('targetPhone = ', targetPhone, '; curPhone = ', ph)
            if len(targetPhone) == 3 or targetPhone in vowels:
                if targetPhone[:2] == ph[:2]:
                    #print('targetPhone = ', targetPhone, '; curPhone = ', ph)
                    return i+1
            #if ph == phoneID
            if targetPhone == ph:
                return i+1
            
    consonants = set(['p', 'b', 't', 'd', 'k', 'g', 'ʧ', 'ʤ', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j'])
    vowels = ['ɔ', 'ɑ', 'i', 'u', 'e', 'ɪ', 'ʊ', 'ʌ', 'ə', 'æ', 'a', 'o', 'ɜ', 'ː']

    def convertToIPA(phs_in, stressed=False):
        phs = phs_in[:]
    
        def isVowel(ph):
            for v in vowels:
                if ph.startswith(v):
                    return True
            return False
        def vowelCnts():
            ret = 0
            i = 0
            while i<len(ipas):
                if isVowel(ipas[i]):
                    ret += 1
                    if i+1 < len(ipas) and isVowel(ipas[i+1]):
                        i += 1
                i += 1
            return ret
        def util(ph):
            if ph.endswith('1'):
                return ("ˈ", phoneToIPA.get(ph, '*'))
            elif ph.endswith('2'):
                return ("ˌ", phoneToIPA.get(ph, '*'))
            elif ph.endswith('0'):
                return (phoneToIPA.get(ph, '*'))
            else:
                return phoneToIPA.get(ph, '*')
        two_cons = ['bl', 'br', 'kl', 'kr', 'fr', 'gl', 'gr', 'pl', 'pr', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw',
                    'tw', 'θr', 'dr', 'tr', 'ʃr', 'sf', 'fl']
        tri_cons = ['skw', 'str', 'skr', 'spr']
        ipas = [x for sub in (map(util, phs)) for x in sub]
        v_cnts = vowelCnts()
        for i, c in enumerate(ipas):
            if c in "ˈˌ":
                if i > 2 and ''.join(ipas[i-3:i]) in tri_cons:
                    ipas[i-3:i+1] = ipas[i:i+1]+ipas[i-3:i]
                elif i > 1 and ''.join(ipas[i-2:i]) in two_cons:
                    ipas[i-2:i+1] = ipas[i:i+1]+ipas[i-2:i]
                elif i > 0 and ipas[i-1] in consonants:
                    ipas[i-1:i+1] = ipas[i:i+1]+ipas[i-1:i]
        if v_cnts == 1 and ("ˈ" in ipas or "ˌ" in ipas):
            if "ˈ" in ipas:
                ipas.remove("ˈ")
            if "ˌ" in ipas:
                ipas.remove("ˌ")
        if not stressed:
            ipas = [c for c in ipas if c not in "ˈˌ"]
        return '%s'%(''.join(ipas))

    def compareIPA2(ans, preds, ph_cmu=None):
        def phoneToIPAWithIndex(phSeq, diffs):
            retIPA = ''
            ipaDiffs = []
            for i, ph in enumerate(phSeq):
                curIPA = phoneToIPA.get(ph, '*')
                if i in diffs:
                    ipaDiffs.extend([len(retIPA)+j for j in range(len(curIPA))])
                retIPA += curIPA
            return retIPA, ipaDiffs

        ans_cmu = ipaToCMU.get(ans, convertIPAToCMU(ans)) if ph_cmu is None else ph_cmu
        if ans_cmu is None:
            return
        ret, ans_diffs, pred_diffs = [], [], []
        if len(ans_cmu) == len(preds):
            for i, (x, y) in enumerate(zip(ans_cmu, preds)):
                if x in y:
                    ret.append(x)
                elif any(v in x for v in 'AEIOU'):
                    skip = False
                    for yy in y:
                        if yy[:2] == x[:2]:
                            ret.append(yy)
                            skip = True
                            break
                    if not skip:
                        ret.append(y[0])
                        ans_diffs.append(i)
                        pred_diffs.append(i)
                else:
                    ret.append(y[0])
                    ans_diffs.append(i)
                    pred_diffs.append(i)
        else:
            # find LCS first
            m, n = len(ans_cmu), len(preds)
            L = [[0 for _ in range(m+1)] for _ in range(n+1)]
            for i in range(n+1):
                for j in range(m+1):
                    if i == 0 or j == 0:
                        L[i][j] = 0
                    elif ans_cmu[j-1] in preds[i-1]:
                        L[i][j] = L[i-1][j-1]+1
                    else:
                        L[i][j] = max(L[i-1][j], L[i][j-1])
            i, j = n, m
            while i > 0 and j > 0:
                if ans_cmu[j-1] in preds[i-1]:
                    ret.append(ans_cmu[j-1])
                    j -= 1
                    i -= 1
                elif L[i-1][j] > L[i][j-1]: # deletion
                    choices = [x for x in preds[i-1] if x not in ['SPN', 'TS', 'IX']]
                    topChoice = choices[0] if len(choices)>0 else 'SIL'
    #                 if allowExcept and checkException(topChoice, val):
    #                     continue
                    ret.append(topChoice)
                    pred_diffs.append(i-1)
                    i -= 1
                else: # insertion
                    ans_diffs.append(j-1)
                    j -= 1
            
            while i>0 and len(ret) != len(ans_cmu):
                choices = [x for x in preds[i-1] if x not in ['SPN', 'TS', 'IX']]
                topChoice = choices[0] if len(choices)>0 else 'SIL'
                ret.append(topChoice)
                pred_diffs.append(i-1)
                i -= 1
            while j>0:
                ans_diffs.append(j-1)
                j -= 1
            ret = ret[::-1]
            pred_diffs = pred_diffs[::-1]
            ans_diffs = ans_diffs[::-1]

        total, error = len(ans_cmu), max(len(ans_diffs), len(pred_diffs))
        predIPA, predDiffs = phoneToIPAWithIndex(ret, pred_diffs)
        ipa_ans_no_stress, ansDiffs = phoneToIPAWithIndex(ans_cmu, ans_diffs)

        ipa_ans_no_stress = ''.join([phoneToIPA[x] for x in ans_cmu])
        if ipa_ans_no_stress == ''.join(predIPA):
            error = 0
        return {'ipaAns': ans, 
                'ipaAnsNoStress': ipa_ans_no_stress,
                'ipaAnsList': [phoneToIPA[ph] for ph in ans_cmu], 
                'ansIpaDiffIndex': ansDiffs,
                'ansCMU': ans_cmu, 'ansCMUDiffIndex': ans_diffs,
                'predPhoneBestIPA': predIPA, 
                'predIpaDiffIndex': predDiffs,
                'predPhoneBestCMU': ret, 'predPhoneDiffIndex': pred_diffs,
                'predIPABestList': [phoneToIPA.get(ph, '*') for ph in ret], 
                'predPhoneError': (error, total),
                'wordScore': (1-(error/total))*100}

    def matchWord(words_in, pairs):
        words = [x for x in words_in if x != '<SPOKEN_NOISE>']
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

    def scoreUtt(gopParsed, ans=None):
        if ans is None:
            for i, row in enumerate(gopParsed['parts']):
                ph = convertToIPA(row['phone'].split('_'))
                phSeqLength = len(row['phone'].split('_'))
                diagnosis = compareIPA2(ph, row['predPhone'])
                gopParsed['parts'][i].update(diagnosis)
        else:
            j = 0
            for i, row in enumerate(gopParsed['parts']):
        #         print(row, ans[j])
                while row['word'] != ans[j][0]:
                    j += 1
                ph = convertToIPA(row['phone'].split('_'))
                phSeqLength = len(row['phone'].split('_'))

                diagnosis = compareIPA2(ans[j][1], row['predPhone'])
                if diagnosis is None:
                    return {'error': 'fail to decode IPA on word %s, please check if your IPA is correct!' % row['word']}
                #print(diagnosis)
                gopParsed['parts'][i].update(diagnosis)
                j += 1
        return gopParsed

    # ================================ End of utility function definitions =========================== #

    print('Begin of parseGOPOutput')

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
                phoneID = line[:line.find('[')].strip()
                curPhone = idToPhone[phoneID]
                isVowel = lambda x: any(v in x for v in 'AEIOU')

                raw_cpls = list(map(float, line[line.find('[')+1:line.find(']')].split()))
                raw_cpls = sorted([(val, i+1) for i, val in enumerate(raw_cpls)], reverse=True)
                cpls = [x for x in raw_cpls if str(x[1]) in vowelIDs] if isVowel(curPhone) else [x for x in raw_cpls if str(x[1]) in consonantIDs]

                rank = findRankEng(cpls, phoneID, idToPhone)
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
    
    j = 0
#     print('cmu_ans = ', cmu_ans)
    for w, phoneSeq in wb:
        phBuf, phGOPBuf, phRRBuf, intervalBuf, predPh = list(zip(*phoneSeq))
        phGOPScore = list(map(lambda x: GOPscore(x, adjust=word_model, a=a, b=b), phGOPBuf))
        phRRScore = list(map(lambda x: GOPscore(x, adjust=word_model), phRRBuf))
        cur = {}
        cur['word'] = w
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

        if cmu_ans is not None:
            while cur['word'] != cmu_ans[j][0]:
                j += 1
            phSeqLength = len(cur['phone'].split('_'))
            
            cur_ipa = convertToIPA(cmu_ans[j][1])
            diagnosis = compareIPA2(cur_ipa, cur['predPhone'], cmu_ans[j][1])
            if diagnosis is None:
                return {'error': 'fail to decode IPA on word %s, please check if your IPA is correct!' % row['word']}
            cur.update(diagnosis)
            j += 1
        elif ipa_ans is None:
            diagnosis_list = []
            for ph_cmu in wordToPhones[w]:
                if len(ph_cmu) != len(cur['predPhone']):
                    continue
                ph = convertToIPA(ph_cmu)
                phSeqLength = len(cur['phone'].split('_'))
                diagnosis = compareIPA2(ph, cur['predPhone'], ph_cmu=ph_cmu)
                diagnosis_list.append(diagnosis)
            diagnosis_list.sort(key=lambda x: x['wordScore'])
            cur.update(diagnosis_list[-1])
        else:
            while cur['word'] != ipa_ans[j][0]:
                j += 1
            phSeqLength = len(cur['phone'].split('_'))

            diagnosis = compareIPA2(ipa_ans[j][1], cur['predPhone'])
            if diagnosis is None:
                return {'error': 'fail to decode IPA on word %s, please check if your IPA is correct!' % row['word']}
            cur.update(diagnosis)
            j += 1

#        # exclude error prone words: A, THE
#         if cur['word'] in ['A', 'THE']:
#             cur['wordScore'] = 100

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
              'parts': wordGOP,
              'scoremode' : scoremode, 'arg_a': a, 'arg_b': b,
              'minscores' : minscores}
    wordScoresList = [x['wordScore'] for x in gopdic['parts']]
#     wordScoresList = [x['GOPScore'] for x in gopdic['parts']]
    gopdic['sentScores'] = round(sum(wordScoresList)/len(wordScoresList), 2)
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
    wf = open('/tmp/logs.txt', 'a')
    ## 2014.12.22 加上讓tranking.pslog 加上debug時間功能
    if os.environ.get('pslogobtime', ''):
        import time
        ptime = time.time()
        wf.write('%s--%s:%s' % (ptime, msg, msg1))
    else:
        wf.write('%s:%s' % (msg, msg1))
    wf.close()


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
          ipa_ans, cmu_ans = None, None
          word_model = False
          if 'ipa_ans' in qs:
            ipa = qs['ipa_ans'][0]
            # 將ipa輸入的:ɡ 統一換成 g
            if ipa.find('ɡ') != -1:
                ipa = ipa.replace('ɡ', 'g')            
            # self.wfile.write(str(ipa).encode())
            ipa_ans = list(zip(qs['txt'][0].split(), ipa.split()))
          if 'cmu_ans' in qs:
            cmu = qs['cmu_ans'][0].split()
            # self.wfile.write(str(ipa).encode())
            cmu = [x.split('_') for x in cmu]
            cmu_ans = list(zip(qs['txt'][0].split(), cmu))
          if 'word_model' in qs:
            word_model = bool(qs['word_model'][0])
          try:
              scoremode = qs['scoremode'][0]
          except: scoremode = 'angel'           
          try:
              a, b = float(qs['a'][0]), float(qs['b'][0])
          except: a, b = 1, 2
          gopParsed, ctmParsed = parseGOPOutput(dir+'/gop', ipa_ans=ipa_ans, cmu_ans=cmu_ans, scoremode=scoremode, a=a, b=b, word_model=word_model)
          gopstring = 'GOP-json:%s' % str(gopParsed)
          self.wfile.write(str(gopstring).encode())
          self.wfile.write(b'\n')
          ctmstring = 'CTM-json:%s' % str(ctmParsed)
          self.wfile.write(str(ctmstring).encode())
          
        except Exception as e:
          errMsg = traceback.format_exc()
          #self.wfile.write(b'Failed')
          self.wfile.write(str.encode(errMsg))          

        shutil.rmtree(dir)
        fin = '\n===---===spend time(Chinese GOP):%f\n' % (time.time()-tm1)
        self.wfile.write(fin.encode())

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

if __name__ == '__main__':
    init()
    ThreadedHTTPServer(('', 8506), Handler).serve_forever()
