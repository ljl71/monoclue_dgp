## 一、首选：tmux（最稳定、好用）

### 1. 安装（AutoDL 一般已装，没有就执行）

```
apt-get update && apt-get install -y tmux
```

### 2. 启动会话 + 运行训练（关键）

在 **PyCharm 远程终端 / AutoDL Web 终端** 执行：

```
# 新建一个叫 train 的会话
tmux new -s train

# 进入后：激活环境、cd 到代码、启动训练
conda activate your_env
cd /root/autodl-tmp/your_project
python train.py  # 正常启动训练
```

### 3. 分离会话（后台运行）

训练开始后，按快捷键：

```
Ctrl + B  → 松开 → 再按 D
```

看到 `detached` 表示成功后台。**现在可以关 PyCharm、断网、关机**。

### 4. 重新连接查看进度

```
# 列出所有会话
tmux ls

# 回到 train 会话
tmux attach -t train
```

### 5. 退出 / 关闭会话（训练结束后）

```
# 在会话内：
exit

# 或外部关闭：
tmux kill-session -t train
```

## 一、必补常用操作（训练时高频用到）

### 1. 不小心退出会话，想回到后台

```
tmux ls
tmux a -t train  # 等价于 tmux attach -t train，更短
```

### 2. 会话太多，想删除不用的

```
# 删除指定会话
tmux kill-session -t train

# 删除所有会话
tmux kill-server
```

### 3. 滚动查看历史输出（非常重要）

直接滚轮无法滚动，需要先进入**复制模式**：

1. 按 `Ctrl + B`，松开
2. 按 `[`
3. 用方向键 / 滚轮上下翻页
4. 按 `q` 退出

------

## 二、AutoDL 场景下的关键提醒

1. **绝对不要在 PyCharm 里直接运行训练代码**

   一定要在 **tmux 内部** 执行 `python train.py`，否则断网依然会停。

   

2. **重启机器后 tmux 会话会消失**

   所以一定要在代码里写 **checkpoint 保存权重**，方便断点续训。

   

3. 如果你用的是 AutoDL 托管容器，**关机后实例停止**，训练也会停

   tmux 只能保证**不断开 SSH 时后台运行**，不能阻止实例关机。

   

------

## 三、防止误操作：不小心按到 Ctrl+C 停训练

如果你想避免在 tmux 里误触终止程序，可以用

```
python train.py &
```

让程序在会话内**再后台一层**，这样按 Ctrl+C 不会停。

想查看输出可以用：

```
tail -f nohup.out
```

------

## 四、极简总结：你真正需要记住的

- 新建：`tmux new -s train`
- 后台：`Ctrl+B` → `D`
- 查看会话：`tmux ls`
- 重连：`tmux a -t train`
- 删除：`tmux kill-session -t train`
- 翻日志：`Ctrl+B` → `[`，`q` 退出