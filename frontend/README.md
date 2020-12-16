# 行人检索应用前端部分

## 1 配置环境

本程序推荐在Python 3.7版本下运行。目前已知的是更高的Python版本，其包之间可能会有冲突。

在clone代码，并将工作目录切换到仓库根目录后，执行下面的命令以配置环境：

```bash
python3 -m venv venv
source venv/bin/activate
DISABLE_BCOLZ_AVX2=true pip install -r requirements.txt
```

而后，执行下面的命令启动应用：

```bash
python3 frontend/main.py
```

## 2 使用指南

### 2.1 菜单栏

1. File：文件的打开等。
   1. Open Video：打开本地视频，如果存在同名pickle文件，则将该文件作为分析（可能只是部分的）结果载入。
   2. Open RTSP：打开RTSP流视频，输入的URL一般以`rtsp://`开头。
   3. Save Results：保存当前分析的结果，到同名的pickle文件，仅对本地视频有用。
   4. Load Blacklist：加载黑名单pickle文件，多个视频可以共享一个黑名单文件。
   5. Save Blacklist：保存黑名单文件。
   6. Exit：退出。
2. View：切换试图。
   1. Show Sidebar (checkable)：折叠或展开侧栏。其实拖拽侧栏和主界面之间的handle也可以实现折叠侧栏，甚至可以折叠主界面。
3. Analyze：
   1. Detector (checkable)：是否开始分析，如果不勾选，会按照正常速率播放视频。

### 2.2 侧栏

#### 2.2.1 Video Info标签页

1. File：文件名。
2. Frame：（随视频播放改变）当前第几帧，变化范围为[1, total frames]。
3. Time：（随视频播放改变）当前视频时间。
4. Progress：（随视频播放改变）当前播放进度，这部分直接获取的opencv，返回值似乎不太正确，建议忽略。
5. Width：视频宽度，像素。
6. Height：视频高度，像素。
7. Frame Rate：FPS。
8. Total Frames：总帧数。


#### 2.2.2 Pedestrians标签页

此标签页下有3个子标签页。再继续之前我们需要区分下面4个对象（命名可能不太考究，多多包含）：

- Pedestrian：每个帧检测出的行人，它一定属于某个Instance，可能有一个可选的Face，可能是一个可选的Blacklist。
- Instance：不同的帧被判定为是相同的行人组成的一个组，可以被命名，会有随即分配的颜色。
- Face：不同的帧被判定为是相同的脸组成的一个组，可以被命名，会有随即分配的颜色。
- Blacklist：被特殊关注的某个Face，当视频中出现与之配对的Pedestrian时，会被显著标记。

标签页的下方有个Save Detection Results按钮，功能同菜单栏Save Results。

##### Pedestrians子标签页

这个标签页存放了所有检测到的行人，不同帧的同一行人也被分开列出。各列内容如下：

1. Frame：对象所属帧号，变化范围为[1, total frames]。
2. Instance ID：对应的Instance的UUID前8个字符，如果该Instance已被命名，就显示它的名字。Ctrl+单击该列的单元格会跳转到Instances子标签页对应的行。双击单元格则可以为该Instance分配名字。
3. Instance Color：对应的Instance的颜色，
4. Face ID：对应的Face的UUID前8个字符，如果该Face已被命名，就显示它的名字。如果没有找到行人的脸，可能为空。Ctrl+单击该列的单元格会跳转到Faces子标签页对应的行。双击单元格则可以为该Face分配名字。
5. Face Color：对应的Face的颜色，如果没有找到行人的脸，可能为空。
6. Blacklist：对应的Blacklist的UUID前8个字符，如果该Blacklist已被命名，就显示它的名字。如果没有找到行人的脸或者这个脸并不和某个Blacklist配对，则为空。Ctrl+单击该列的单元格会跳转到Blacklist标签页对应的行。双击单元格则可以为该Blacklist分配名字。
7. Attributes：抽取出的Pedestrian特征。

##### Instances子标页

这个标签页存放所有被认为是相同的行人组成的组：

1. ID：Instance的UUID。
2. Name：Instance的名字，双击可以为Instance分配名字。
3. Color：Instance随即分配得到的颜色。
4. Frame：Instance被检测到的帧号，范围为[1, total frames]，为了节省空间会采用若干闭区间表示被识别出的范围。

##### Faces子标页

这个标签页存放所有被认为是相同的脸组成的组：

1. ID：Face的UUID。
2. Name：Face的名字，双击可以为Face分配名字。
3. Action：点击Add按钮将当前该组的特征添加到Blacklist里（由于组的特征是会随着分析变化的，所以我们允许Blacklist里有多个同样的Face）。注意，Blacklist的UUID与Face的UUID无关系，但默认的名字是Face的名字。
4. Color：Face随即分配得到的颜色。
5. Frame：Face被检测到的帧号，范围为[1, total frames]，为了节省空间会采用若干闭区间表示被识别出的范围。

#### 2.2.2 Blacklist标签页

这个标签页存放的是黑名单信息，由于黑名单不属于某个视频。所以它是个独立的标签页而非子标签页。

1. ID：Blacklist的UUID。
2. Name：Blacklist的名字，双击可以为Blacklist分配名字。
3. Action：点击Remove按钮将移除此行。
4. Frame：Blacklist被检测到的帧号，范围为[1, total frames]，为了节省空间会采用若干闭区间表示被识别出的范围。

标签页的下方有Load Blacklist和Save Blacklist按钮，功能同菜单栏同名选项。

### 2.3 主界面

#### 2.3.1 播放窗口

播放窗口会用较粗的线标记Instance,用较细的线标记Face。在Instance上方会有行人属性和Instance ID；在Face下方会有Face ID和可能的Blacklist ID。如果有Blacklist ID，也就是脸被认为是在黑名单中，原本半透明的白色文本背景会变成红色文本背景，以提示用户。

#### 2.3.2 播放控制区

最左侧是播放、暂停按钮。当视频还没有放完的时候，它能暂停或继续播放视频，当视频播放完，点击会重新播放上一个视频。

中间是进度条。进度条可以拖拽（建议是暂停后再拖拽）。在暂停播放时，你可以将键盘焦点集中到进度条上后，用键盘的左右方向键精细地调整帧。非常不建议将进度条拖拽至视频尚未分析的部分，这会强迫后端开始分析后面的视频，而丢失时间连贯的信息。

最右侧是进度提示，其格式为`currentTime/totalTime currentFrame/TotalFrame`。

## 3 实现细节

### 3.1 分组的特征选取

Instance和Face对应的其实都是一系列的特征。我们会将新的特征和之前总结的特征做加权平均构造出心的总结的特征。这样程序能够应对转头、转身等特征发生连续变化的情况。但问题是这种策略会造成旧的特征指数衰减，从而新的特征化错类别这种错误会被不断地放大。
