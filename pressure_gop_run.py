import os, sys
import glob

# tri2
cmds_tri2 = """
# 他們都是大學生麻
python3 compute_ctm_gop.py --wavPath /home/ubuntu/trained_kaldi_models/001.wav --txt "ta men dou shi da xue sheng ma" --prjDir "mixedZhuyin"
# 在 教授的指導 下 ，我終於完成了 我的博士論文
python3 compute_gop.py --wavPath /home/ubuntu/trained_kaldi_models/ch_002.wav --txt "zai jiao shou de zhi dao xia wo zhong yu wan cheng le wo de bo shi lun wen" --prjDir mixedZhuyin
# 茶 有益於 減低血脂， 算是 健康的 飲料
python3 compute_gop.py --wavPath /home/ubuntu/trained_kaldi_models/ch_003.wav --txt "cha you yi yu jian di xue zhi suan shi jian kang de yin liao" --prjDir mixedZhuyin
# 後天 跟 明天 一樣 冷
python3 compute_gop.py --wavPath /home/ubuntu/trained_kaldi_models/ch_004.wav --txt "hou tian gen ming tian yi yang leng" --prjDir mixedZhuyin
# 可以使一直這麼健健康康的
python3 compute_gop.py --wavPath /home/ubuntu/trained_kaldi_models/ch_005.wav --txt "ke yi shi yi zhi zhe me jian jian kang kang de" --prjDir mixedZhuyin
# 司徒氏終於得到了釋放，可以和先生團聚
python3 compute_gop.py --wavPath /home/ubuntu/trained_kaldi_models/ch_006.wav --txt "si tu shi zhong yu de dao le shi fang ke yi he xian sheng tuan ju" --prjDir mixedZhuyin
"""

# DNN
cmds_dnn = """
# 他們都是大學生麻
curl -s "http://localhost:8080/?wav=001.wav&txt=ta+men+dou+shi+da+xue+sheng+ma"
# 在 教授的指導 下 ，我終於完成了 我的博士論文
curl -s "http://localhost:8080/?wav=ch_002.wav&txt=zai+jiao+shou+de+zhi+dao+xia+wo+zhong+yu+wan+cheng+le+wo+de+bo+shi+lun+wen"
# 茶 有益於 減低血脂， 算是 健康的 飲料
curl -s "http://localhost:8080/?wav=ch_003.wav&txt=cha+you+yi+yu+jian+di+xue+zhi+suan+shi+jian+kang+de+yin+liao"
# 後天 跟 明天 一樣 冷
curl -s "http://localhost:8080/?wav=ch_004.wav&txt=hou+tian+gen+ming+tian+yi+yang+leng"
# 可以使一直這麼健健康康的
curl -s "http://localhost:8080/?wav=ch_005.wav&txt=ke+yi+shi+yi+zhi+zhe+me+jian+jian+kang+kang+de"
# 司徒氏終於得到了釋放，可以和先生團聚
curl -s "http://localhost:8080/?wav=ch_006.wav&txt=si+tu+shi+zhong+yu+de+dao+le+shi+fang+ke+yi+he+xian+sheng+tuan+ju"
"""

def ps(fn,fv=''):
    print(fn, fv)


## random 取 rtblst 中的項目
## bookmax => 取幾個資料
## bookmax = 0  => 取1個
## bookmax = 3  => 取4個
def getrandom_fromlst(rtblst, bookmax=3):

    import random
    bookmax = bookmax
    dbrklst = []

    avoidunlimitcircle = 0
    ## avoidunlimitcircle => 避免無圖檔資料時. 成為無窮迴圈
    while(len(dbrklst) < (bookmax+1) and avoidunlimitcircle < 100):
        r = rtblst[random.randint(0, len(rtblst)-1)]
        if r not in dbrklst: dbrklst.append(r)
        avoidunlimitcircle += 1

    return dbrklst
        

def main():
    pruning = 'no'
    (forknums, modeltype) = sys.argv[1:]
    forknums = int(forknums)

    countdir = './pressures_counts'
    if not os.path.isdir(countdir):
        os.mkdir(countdir)

    # 先移除之前統計紀錄 in ./pressures_counts
    dlst = glob.glob('%s/*.log' % countdir)
    for f in dlst:
        if os.path.isfile(f):
            os.remove(f)

    allcmds = []
    if modeltype == 'tri2':
        cmds = cmds_tri2
    elif modeltype == 'dnn':
        cmds = cmds_dnn
        
    for line in cmds.split('\n'):
        line = line.rstrip()
        if line and line[0] != '#':            
            allcmds.append(line)

            
    # 5, 10, 20, 30, 50
    for x in range(0, forknums):
        cmd = getrandom_fromlst(allcmds, bookmax=1)
        thiscmd = '/usr/bin/time -f ==time:%%e %s > %s/%04d.log' % (cmd[0], countdir, x)
        ps('cmd', thiscmd)
        os.system('nohup %s 2>&1 &' % thiscmd)


if __name__ == '__main__': main()
