# GOP (english / chinese) docker images build

### GOP score
![](https://imgur.com/a/f6iM8he)


### install docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce

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

# 將docker images version 推送到dockerhub上
sudo docker tag {xxx} ponddy/gop_eng_flask:v1.0
sudo docker tag {xxx} ponddy/gop_eng_flask:latest
sudo docker push ponddy/gop_eng_flask:v1.0
sudo docker push ponddy/gop_eng_flask:latest
```

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