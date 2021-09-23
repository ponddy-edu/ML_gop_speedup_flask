# GOP (english / chinese) docker images build

### GOP score
![](https://github.com/ponddy-edu/ML_gop_speedup_flask/blob/develop/media/Gop_score.png?raw=true)


### install docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce

### copy NAS GOP model to Taipei server

```bash=
# 先切到台北 ML_gop_speedup 路徑
cd /opt/deeplearning/ML_gop_speedup
# 將舊 ponddyEng 資料夾先改名
mv ponddyEng ponddEng6.7

# 將 nas 的 GOP model 複製到/opt/deeplearning/ML_gop_speedup，並改名為ponddyEng
sudo rsync -avzh ponddy@68.255.152.146:/media/nas/GOP_models/ponddyEng6.8_api /opt/deeplearning/ML_gop_speedup/
mv ponddyEng6.8_api/ ponddyEng/
```

### build docker image

```
bash
English:
sudo docker build --no-cache -t "ponddy/gop_eng_flask" -f Dockerfile.flask_eng .
Chinese:
sudo docker build --no-cache -t "ponddy/gop_chi_flask" -f Dockerfile.flask_chi .
```

### Run docker container


#### English

```
bash
# 目前使用這種
# 沒有共用路徑 -- 直接跑可以看到訊息
sudo docker run --name ponddy_gop_eng_web_flask -p 9899:8506 ponddy/gop_eng_flask:latest
# 沒有共用路徑 背景跑
sudo docker run --name ponddy_gop_eng_web_flask -d -p 9899:8506 ponddy/gop_eng_flask:latest

# 有共用路徑
sudo docker run --name ponddy_gop_eng_web_flask -p 9899:8506 -v /models/kaldi_wavs:/models/kaldi_wavs ponddy/gop_eng_flask:latest
# 背景跑
sudo docker run --name ponddy_gop_eng_web_flask -d -p 9899:8506 -v  /models/kaldi_wavs:/models/kaldi_wavs ponddy/gop_eng_flask:latest

# 執行API checker，確認有無異常
bash API_checker_local_docker_web_gop-english-post.bat

# 將docker images version 推送到dockerhub上
sudo docker tag {IMAGE ID} ponddy/gop_eng_flask:v1.0
sudo docker tag {IMAGE ID} ponddy/gop_eng_flask:latest
sudo docker push ponddy/gop_eng_flask:v1.0
sudo docker push ponddy/gop_eng_flask:latest
```

#### 更新ML版號紀錄

[ML模型版號紀錄](https://docs.google.com/spreadsheets/d/142uhBXcScwvFSOqFkwgPvBijZPJGB4RZXilQ3_ARadc/edit#gid=194926086)


#### Chinese

```
bash
# 目前使用這種
# 沒有共用路徑 -- 直接跑可以看到訊息
sudo docker run --name ponddy_gop_chi_web_flask -p 9898:8505 ponddy/gop_chi_flask:latest
# 沒有共用路徑 背景跑
sudo docker run --name ponddy_gop_chi_web_flask -d -p 9898:8505 ponddy/gop_chi_flask:latest

# 有共用路徑
sudo docker run --name ponddy_gop_chi_web_flask -p 9898:8505 -v /models/kaldi_wavs:/models/kaldi_wavs ponddy/gop_chi_flask:latest
# 背景跑
sudo docker run --name ponddy_gop_chi_web_flask -d -p 9898:8505 -v  /models/kaldi_wavs:/models/kaldi_wavs ponddy/gop_chi_flask:latest

# 將docker images version 推送到dockerhub上
sudo docker tag {xxx} ponddy/gop_chi_flask:v1.0
sudo docker tag {xxx} ponddy/gop_chi_flask:latest
sudo docker push ponddy/gop_chi_flask:v1.0
sudo docker push ponddy/gop_chi_flask:latest
```

#### 更新ML版號紀錄

[ML模型版號紀錄](https://docs.google.com/spreadsheets/d/142uhBXcScwvFSOqFkwgPvBijZPJGB4RZXilQ3_ARadc/edit#gid=194926086)