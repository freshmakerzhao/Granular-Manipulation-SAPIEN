# Granular-Manipulation-SAPIEN 项目交接笔记（2026-04-22）

## 1. 总体目标

本项目目标是基于 **SAPIEN 3.0** 构建可复现的“挖掘机/机械臂 + 颗粒土壤”仿真环境，为后续：

1. 轨迹生成与闭环控制（先 scripted，再 ACT 风格策略）
2. 土壤表面/颗粒状态建图（RGB-D + 粒子状态）
3. 基于图神经网络（L-GBND 思路）学习“动作轨迹 -> 土壤形变”关系

提供稳定数据与评测基础。

---

## 2. 当前核心结论

1. **SAPIEN 可跑通挖掘机+颗粒场景**，并已支持 CPU/GPU 两条路径。
2. **GPU 能用**（日志可见 `Using PhysX GPU system (cuda)`），但 GPU 下关节控制 API 约束更严格，需要特定写法。
3. 已完成 **统一配置化**：模型路径与初始位姿集中在 `configs/config.json`。
4. 已有三种控制模式：`none` / `scripted` / `ee-scripted`。
5. 已支持 **RGB-D 数据采集**（`rgb/depth/edge/intrinsic/extrinsic`）。
6. 粒子接触参数已向“高摩擦、低弹性、强耗散”调优，显著减少“一碰就弹飞”。

---

## 3. 当前文件结构与职责

### 3.1 主要脚本

- `excavator_pool_env.py`
  - 主场景：右侧挖掘机平台 + 左侧颗粒池
  - 粒子生成与物理参数
  - 控制模式切换（none/scripted/ee-scripted）
  - RGB-D 相机与保存
  - GPU/CPU 后端切换

- `scripted_policy.py`
  - 关节关键帧插值策略 `LinearJointKeyframePolicy`
  - 末端位姿关键帧插值策略 `LinearEEKeyframePolicy`
  - JSON 加载器（joint/ee）
  - 内置默认 keyframe 轨迹生成

- `robot_only_env.py`
  - 单机体（机器人）加载和检查脚本，用于模型验证与姿态调试

- `sandbox_env.py`
  - 纯沙盘/颗粒环境基础脚本

### 3.2 配置与数据

- `configs/config.json`
  - `urdf_candidates`：模型名 -> URDF 候选路径
  - `initial_pose`：模型名 -> 场景名 -> 初始 qpos
  - 当前已含 `excavator_simple` / `excavator_full` / `excavator_s010` / `fairino5_single`

- `configs/excavator_s010_scripted_keyframes.json`
  - 关节空间关键帧（`t` + `q`）

- `configs/excavator_s010_ee_keyframes.json`
  - 末端位姿关键帧（`t` + `xyz` + `rpy`）

### 3.3 模型资产

- `assets/excavator_s010/excavator.urdf`（当前主推）
- `assets/excavator_simple/excavator.urdf`
- `assets/excavator_full/excavator.urdf`

关节命名已统一为：
- `j1_swing`
- `j2_boom`
- `j3_stick`
- `j4_bucket`

---

## 4. 关键运行命令（常用）

### 4.1 基础运行

```bash
python excavator_pool_env.py
```

### 4.2 强制 CPU（便于稳定调试与可重复控制）

```bash
python excavator_pool_env.py --cpu
```

### 4.3 Scripted 关节轨迹

```bash
python excavator_pool_env.py \
  --control-mode scripted \
  --scripted-keyframes-json configs/excavator_s010_scripted_keyframes.json \
  --start-unpaused
```

### 4.4 EE 轨迹 + IK

```bash
python excavator_pool_env.py \
  --control-mode ee-scripted \
  --cpu \
  --start-unpaused
```

如需指定 EE 关键帧：

```bash
python excavator_pool_env.py \
  --control-mode ee-scripted \
  --ee-keyframes-json configs/excavator_s010_ee_keyframes.json \
  --cpu \
  --start-unpaused
```

### 4.5 采集模式（挖掘机去重力）

```bash
python excavator_pool_env.py --data-collection-mode --cpu
```

### 4.6 启用 RGB-D

```bash
python excavator_pool_env.py \
  --enable-rgbd \
  --rgbd-save-dir outputs/rgbd \
  --rgbd-capture-interval-steps 24
```

---

## 5. 物理与性能调优抓手

`excavator_pool_env.py` 顶部常量是主要调参入口：

1. 粒子规模
- `--particle-count`
- `--particle-radius`

2. “沙土感”参数
- `SAND_STATIC_FRICTION`
- `SAND_DYNAMIC_FRICTION`
- `SAND_RESTITUTION`
- `SAND_LINEAR_DAMPING`
- `SAND_ANGULAR_DAMPING`
- `SAND_MAX_DEPENETRATION_VEL`
- `SAND_MAX_CONTACT_IMPULSE`

3. 工具接触稳定性
- `TOOL_STATIC_FRICTION`
- `TOOL_DYNAMIC_FRICTION`
- `TOOL_RESTITUTION`
- `TOOL_MAX_DEPENETRATION_VEL`

4. GPU 大接触场景内存
- `GPU_MAX_RIGID_CONTACT_COUNT`
- `GPU_MAX_RIGID_PATCH_COUNT`

说明：这两个参数用于避免 GPU PhysX 在高密度粒子接触时出现 patch/contact buffer overflow。

---

## 6. 已处理问题与经验

1. **URDF 网格路径错误**
- 现象：`Error opening file ... No such file or directory`
- 原因：URDF 中 mesh 路径与实际目录不一致
- 处理：统一相对路径与软链接，最终可正常加载

2. **MuJoCo 可见但 SAPIEN 拼接异常**
- 现象：SAPIEN 里模型看起来错位
- 处理后：用户确认 `robot_only_env.py` 已恢复为正确版本并显示正常

3. **GPU 初始化顺序问题**
- 现象：`enable_gpu() must be called before creating a PhysX GPU system`
- 处理：已在创建 scene 前显式 `sapien.physx.enable_gpu()`

4. **GPU API 限制**
- 现象：GPU direct 模式下部分关节接口不可直接调用（如 drive target）
- 处理：代码中按 CPU/GPU 分支采用不同控制路径

5. **Patch buffer overflow**
- 现象：`Patch buffer overflow detected`
- 处理：增大 `GPU_MAX_RIGID_*` 参数

6. **粒子“弹飞”**
- 处理：将 restitution 压低为 0，增大摩擦与阻尼，限制去穿透/冲量速度

---

## 7. 当前遗留问题（非常重要）

最近一次反馈：

```bash
python excavator_pool_env.py --control-mode ee-scripted --cpu --start-unpaused
```

日志已无崩溃，但“挖掘机没有明显运动”。

当前状态判断：

1. 先前 `TypeError: only length-1 arrays can be converted` 已修复（IK 返回值解析做了兼容）。
2. 仍需确认 EE 轨迹是否产生了“可见幅度”的目标变化。
3. `configs/excavator_s010_ee_keyframes.json` 当前内容是绝对世界坐标风格模板，若直接使用，可能与当前初始 EE 位姿不匹配。
4. 建议优先用 `scripted` 关节模式验证动作闭环，再回到 `ee-scripted` 做 IK 轨迹。

---

## 8. 推荐的下一阶段执行顺序

1. **先打通稳定动作回放（关节空间）**
- 使用 `--control-mode scripted --cpu --start-unpaused`
- 调 `configs/excavator_s010_scripted_keyframes.json` 得到可靠挖掘轨迹

2. **再打通数据采集链路**
- 同步记录：`qpos/qvel`、粒子状态（尤其表层）、RGB-D 帧
- 形成统一 episode 数据目录结构

3. **再攻 EE-IK 工作流**
- 让 EE keyframe 采用“相对初始位姿增量”或录制式生成
- 避免直接使用不匹配的绝对世界坐标模板

4. **最后对接学习**
- 先做可监督单步/短时预测（动作 -> 高度图/表层粒子变化）
- 再扩展到多步 rollout 与闭环策略

---

## 9. 建议的数据格式（面向后续 GNN）

每个 episode 至少保存：

1. 控制信号
- `u_t`：关节目标/实际（至少 4 DOF）

2. 机器人状态
- `q_t`, `dq_t`, EE pose

3. 土壤状态（两套可并行）
- 视觉侧：RGB-D / heightmap / edge
- 物理侧：表层粒子采样点（位置、速度、局部法向/密度）

4. 监督目标
- `s_{t+1}` 或未来 `k` 步地形变化

---

## 10. 开新 Chat 可直接复制的上下文摘要

可以把下面这段原样发给新 AI：

```text
项目：Granular-Manipulation-SAPIEN（SAPIEN 3.0）
目标：做挖掘机+颗粒土壤仿真，采集轨迹和土壤形变数据，后续做 L-GBND/GNN 表征与预测。
主脚本：excavator_pool_env.py
控制模式：none / scripted / ee-scripted
配置中心：configs/config.json（urdf_candidates + initial_pose[model][scene/default]）
当前主模型：assets/excavator_s010/excavator.urdf（4DOF: j1_swing,j2_boom,j3_stick,j4_bucket）
已支持：GPU/CPU、粒子物理调参、RGB-D采集、scripted关键帧加载。
当前待解：ee-scripted 虽无报错，但动作不明显；优先先用 scripted 模式确保挖掘动作与数据采集链路稳定。
```

---

## 11. 当前工作区变更快照（写本笔记时）

`git status --short`：

- `M excavator_pool_env.py`
- `?? scripted_policy.py`
- `?? configs/excavator_s010_scripted_keyframes.json`
- `?? configs/excavator_s010_ee_keyframes.json`

建议下一步在确认 scripted 轨迹稳定后，做一次小步提交（先不混入大规模重构）。

