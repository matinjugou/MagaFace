# MagaFace 校园安保软件

## 项目结构
* Attribute/ 人体属性SDK
* extraction/ 人脸检测SDK
* frontend/ 校园安保软件
* recognition/ 人脸检测SDK
* Reid/ 行人检测SDK
* sdk/ 人脸识别检测SDK封装

## 安装
提供`Linux`与`Windows`平台的版本，主分支下为`Linux`系统可用的版本，`Windows`分支下为`Windows`系统上可用的版本 

本程序推荐在Python 3.7版本下运行。目前已知的是更高的Python版本，其包之间可能会有冲突。

在clone代码，并将工作目录切换到仓库根目录后，执行下面的命令以配置环境：

```bash
python3 -m venv venv
source venv/bin/activate
DISABLE_BCOLZ_AVX2=true pip install -r requirements.txt
cd Reid/DCNv2_latest
./make.sh
cd ../..
```

而后，执行下面的命令启动应用：

```bash
python3 frontend/main.py
```

## 使用
请参考[frontend/README.md](frontend/README.md)中的使用说明使用