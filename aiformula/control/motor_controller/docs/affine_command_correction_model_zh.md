# Affine Command-Correction Motor Controller 说明

本文档解释当前 `motor_controller` 里的 affine command-correction model 是怎么工作的。

重点先说清楚：

- 神经网络不直接输出 CAN。
- 神经网络不直接输出左右轮 RPM。
- 神经网络输出的是当前车辆的动态 `ax+b` 响应关系。
- `motor_controller` 用这个 `ax+b` 关系反解修正后的 `cmd_vel`。
- 修正后的 `cmd_vel` 再走传统 differential-drive 轮速换算，最后编码成 8-byte CAN frame。

## 1. 整体结构

运行链路是：

```text
base cmd_vel
  + recent vehicle response history
  -> neural network estimates dynamic [a_v, a_omega, b_v, b_omega]
  -> inverse affine correction
  -> corrected cmd_vel
  -> left/right wheel RPM
  -> 8-byte CAN payload
```

也就是说，学习模型不是替代传统 motor controller，而是在传统控制命令后面加一层补偿。

## 2. `ax+b` 是什么意思

理想情况下，如果发送：

```text
cmd_v
cmd_omega
```

车辆应该产生同样的响应：

```text
meas_v     ~= cmd_v
meas_omega ~= cmd_omega
```

这就是 ideal differential-drive baseline：

```text
y = x
```

写成 affine form：

```text
y = a x + b
```

理想 baseline 对应：

```text
a = 1
b = 0
```

但真实车辆会有响应不足、响应过强、摩擦、延迟、偏置等问题。所以模型学习当前车辆更真实的关系：

```text
meas_v     = a_v     * cmd_v     + b_v
meas_omega = a_omega * cmd_omega + b_omega
```

这里的 `a` 和 `b` 不是固定常数，而是每个 runtime step 根据最近历史动态估计出来的：

```text
[a_v, a_omega, b_v, b_omega] = f(history, base_cmd)
```

直观理解：

- `a_v < 1`：前向速度响应偏弱，同样 command 下车跑得不够快。
- `a_v > 1`：前向速度响应偏强。
- `b_v != 0`：前向速度有固定偏置。
- `a_omega < 1`：yaw rate 响应偏弱。
- `b_omega != 0`：转向存在固定偏置或 drift。

## 3. 神经网络训练输入

模型训练时使用固定长度历史窗口：

```text
H = [z_{t-20}, z_{t-19}, ..., z_{t-1}]
```

每个历史点 `z` 的特征顺序必须严格是：

```text
[cmd_v, cmd_omega, meas_v, meas_omega]
```

含义：

- `cmd_v`：当时实际发送给车的线速度命令。
- `cmd_omega`：当时实际发送给车的角速度命令。
- `meas_v`：车辆观测到的车体前向速度。
- `meas_omega`：车辆观测到的 yaw rate。

训练数据里：

- `meas_v` 来自 `vn_body_vx`
- `meas_omega` 来自 `odom_omega_z`

训练采样周期是：

```text
dt = 0.05 s
```

所以 20 个历史样本大约表示 1 秒历史。

除了历史窗口，模型还输入当前 base command：

```text
u_base = [base_v, base_omega]
```

所以训练输入可以写成：

```text
(H, u_base)
```

## 4. 神经网络训练输出

神经网络输出 4 个数：

```text
[a_v, a_omega, b_v, b_omega]
```

然后用这 4 个数构造当前响应预测：

```text
pred_meas_v     = a_v     * cmd_v     + b_v
pred_meas_omega = a_omega * cmd_omega + b_omega
```

训练监督目标是观测到的车辆响应：

```text
[meas_v, meas_omega]
```

所以训练的核心目标是：

```text
predicted observed response ~= actual observed response
```

换句话说，模型学习的不是“直接应该发什么 command”，而是学习：

```text
当前车辆在最近状态下，command 到 observed response 的局部 affine 映射是什么。
```

## 5. Runtime 输入

运行时，`motor_controller` 需要三类输入。

### 5.1 Base command

订阅：

```text
sub_speed_command
```

消息类型：

```text
geometry_msgs/msg/Twist
```

使用字段：

```text
linear.x  -> base_v
angular.z -> base_omega
```

这个 command 是传统控制器或 gamepad 原本想发给车的命令。

### 5.2 Forward velocity measurement

订阅：

```text
/aiformula_sensing/vectornav/velocity_body
```

消息类型：

```text
nav_msgs/msg/Odometry
```

使用字段：

```text
twist.twist.linear.x -> meas_v
```

### 5.3 Yaw-rate measurement

订阅：

```text
/aiformula_sensing/gyro_odometry_publisher/odom
```

消息类型：

```text
nav_msgs/msg/Odometry
```

使用字段：

```text
twist.twist.angular.z -> meas_omega
```

## 6. Runtime 历史窗口

模型训练时看到的是：

```text
20 samples, 0.05 s per sample
```

所以 runtime 也必须尽量保持同样的历史结构。

历史 buffer 持续保存真实采样点：

```text
[last_sent_cmd_v, last_sent_cmd_omega, latest_meas_v, latest_meas_omega]
```

注意这里用的是 `last_sent_cmd`，不是当前刚收到的 `base_cmd`。原因是历史要记录“上一次真正发给车的 command”和车辆随后观测到的响应。

### 为什么不能 fake padding

刚启动时，history 还没有 20 个真实样本。

这时候不能用重复样本补齐，例如：

```text
[first_sample, first_sample, ..., first_sample]
```

因为模型训练时没有见过这种假的时间历史。GRU 会把它当成真实 1 秒车辆动态，这会导致错误 correction。

所以 runtime 逻辑是：

```text
if history is not ready:
    u_send = u_base
```

也就是历史没准备好时直接 bypass 模型。

### History ready 条件

只有同时满足以下条件时才使用模型：

```text
len(history) >= history_steps
latest_meas_v exists
latest_meas_omega exists
history time span is close to training span
```

训练期望跨度是：

```text
expected_span = (history_steps - 1) * 0.05
```

当 `history_steps = 20` 时：

```text
expected_span = 19 * 0.05 = 0.95 s
```

如果 runtime 历史跨度和这个值差太多，模型也会被 bypass。

## 7. Runtime 输出

当 history ready 时，模型输出：

```text
a_v, a_omega, b_v, b_omega
```

然后 motor controller 反解 corrected command：

```text
corrected_v     = (base_v     - b_v)     / a_v
corrected_omega = (base_omega - b_omega) / a_omega
```

这个反解的意思是：

```text
base command 是我们希望车辆实际产生的响应。
模型估计当前车辆满足 y = ax + b。
所以要反过来求 x，也就是应该发送给车的 corrected command。
```

如果 history 没准备好，输出就是：

```text
corrected_v     = base_v
corrected_omega = base_omega
```

## 8. Motor Controller 到 RPM

得到 corrected command 后：

```text
[corrected_v, corrected_omega]
```

motor controller 沿用传统 differential-drive 逻辑换算左右轮速度。

概念上：

```text
right_wheel_velocity = corrected_v + tread / 2 * corrected_omega
left_wheel_velocity  = corrected_v - tread / 2 * corrected_omega
```

然后根据轮径、齿比和已有 motor compensation 参数换算为：

```text
right_rpm
left_rpm
```

学习模型只修正 `cmd_vel`，不直接改 RPM 公式，也不直接改 CAN 编码。

## 9. CAN 输出

输出 topic：

```text
pub_can
```

消息类型：

```text
can_msgs/msg/Frame
```

CAN frame 保持传统格式不变：

```text
id  = 0x210
dlc = 8
```

payload 是 8 bytes：

```text
bytes 0-3: right wheel RPM, signed int32 little-endian
bytes 4-7: left wheel RPM,  signed int32 little-endian
```

所以这里是 8-byte CAN payload，不是模型输出 8-bit signal。

## 10. Debug CSV

debug CSV 用来确认 runtime 是否真的在合理条件下启用了模型。

关键字段包括：

```text
history_len
history_ready
history_span_sec
used_model
latest_meas_v
latest_meas_omega
a_v
a_omega
b_v
b_omega
base_v
base_omega
corrected_v
corrected_omega
delta_v
delta_omega
left_rpm
right_rpm
```

检查时优先看：

```text
history_ready
used_model
history_span_sec
```

如果 `used_model = False`，说明当前 command 是直接通过传统逻辑发送的，没有经过 affine correction。

## 11. 一句话总结

这个 motor controller 的核心是：

```text
神经网络估计当前车辆 command -> response 的动态 ax+b 关系；
runtime 用这个关系反解 corrected cmd_vel；
然后沿用传统 diff-drive RPM 和 CAN 编码输出。
```

神经网络负责估计车辆当前响应偏差，motor controller 负责把这个偏差变成实际可发送的电机命令。
