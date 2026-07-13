# RL 攻防对抗 — 自主攻击链智能体

训练强化学习智能体在模拟内网中**自主完成完整攻击链**（侦察→扫漏→利用→提权→横向移动→窃取），并在动态、不可见、利用会失效的真实模拟条件下稳定学深。

- **算法**：PPO（on-policy，治本 DQN 遗忘）+ Action Masking
- **环境**：YAML 外部化拓扑/检测率/漏洞/奖励，支持动态检测（IDS 告警升级）+ 概率化漏洞利用 + POMDP 不可见检测
- **结果**：4/6 节点 × 静态/动态组合 PPO 全部稳定收敛，final 演示拿 flag；敏感度分析证明策略对手编参数 ±20% 波动 ROBUST

> **快速了解全貌看 [`SUMMARY.md`](SUMMARY.md)**（高层演进总结），**技术细节看 [`GUIDE.md`](GUIDE.md)**（13 章完整解读）。

## 项目结构

```
rl-attack-defense/
├── src/
│   ├── env.py          # v1: 网络拓扑环境 (路径选择)
│   ├── agent.py        # v1: DQN 智能体
│   ├── test_local.py   # v1: 冒烟测试
│   ├── env_v2.py       # v2: 攻击链环境 (侦察→利用→横向→窃取)
│   ├── agent_v2.py     # v2: Double Dueling DQN (训练 + best快照 + 演示)
│   ├── demo_v2.py      # v2: 加载已保存模型直接演示 (无需重训)
│   └── test_local_v2.py# v2: 冒烟测试
├── train.sbatch        # v1 Slurm 脚本
├── train_v2.sbatch     # v2 Slurm 脚本
├── GUIDE.md            # 完整技术解读 (专业名词附中文解释)
├── configs/            # 环境配置 (拓扑/检测率/漏洞/奖励, YAML)
│   └── env_default.yaml# 默认 4 节点拓扑
├── logs/               # Slurm 日志 (.out / .err)
└── models/             # v2 训练产出的 best_v2.pt
```

> v2 环境从 `configs/env_default.yaml` 加载。换拓扑/检测率/奖励只需改 YAML，不用动代码：`AttackChainEnv(config_path="configs/your_scenario.yaml")`。HPC 需 `pip install pyyaml`。

## v2: 自主攻击链智能体

v1 只做路径选择；v2 让智能体自主决策完整攻击链：何时侦察、用什么漏洞、如何提权、何时横向移动、何时窃取数据。

**两套算法**（共享同一环境）：
- **DQN** (`src/agent_v2.py`)：第一代，实测"学不深"（经验回放遗忘，后期策略退化），演示靠 best 快照
- **PPO** (`src/agent_ppo.py`)：换轨后的版本，on-policy 不遗忘，稳定学到贪心策略，final 即 best。**推荐**

**运行**：
```bash
cd ~/projects/rl-attack-defense

# PPO 4 节点 (推荐, ~90min)
ALGO=ppo sbatch --time=01:30:00 train_v2.sbatch

# PPO 6 节点双层拓扑
ALGO=ppo SCENARIO=6node sbatch --time=01:30:00 train_v2.sbatch

# DQN 4 节点 (对比基线)
sbatch train_v2.sbatch

# 用已保存模型重跑演示 (不重训)
.venv/bin/python src/demo_ppo.py                 # PPO 4 节点
.venv/bin/python src/demo_ppo.py --model 6node   # PPO 6 节点
.venv/bin/python src/demo_v2.py                  # DQN 4 节点
```

环境配置在 `configs/`：
- `env_default.yaml` (4节点静态) / `env_6node.yaml` (6节点静态)
- `env_default_dynamic.yaml` (4节点 + 动态检测) / `env_6node_dynamic.yaml` (6节点 + 动态)
- `env_default_dynamic_prob.yaml` (4节点 + 动态 + 概率利用)
- `env_fitted.yaml` (4节点，检测率由 CIC-IDS2017 数据拟合，见 `src/fit_detection_from_cicids.py`)

换拓扑/检测行为只改 YAML。HPC venv 需 `pip install PyYAML`（USTC 镜像无，用清华镜像）。

DQN→PPO 的换轨过程与对比见 `GUIDE.md` 第 10 章。详细技术解读见全文。

## 环境拓扑

```
          [Internet]  检测=0.1
           /      \
     [DMZ-Web]  [DMZ-Mail]  检测=0.3
           \      /
        [Internal-App]  检测=0.5
           /      \
    [File-Svr]  [DB-Server]  检测=0.7   ← ★ flag
           \      /
        [Admin-WS]  检测=0.9             ← ★ flag
```

## 奖励设计

| 事件 | 奖励 |
|---|---|
| 到达目标节点 | +10 |
| 每移动一步 | -1 |
| 被 IDS 检测到 | -5 |

agent 要在路径最短、避开高检测节点之间找平衡。

---

## 操作步骤

### 1. 上传到平台

在平台 GUI「文件管理」中，把整个 `rl-attack-defense/` 文件夹上传到

```
~/projects/rl-attack-defense/
```

> 提示：可以先在本地打包成 `rl-attack-defense.zip`，上传后在 Web Shell 里 `unzip rl-attack-defense.zip`。

### 2. 创建 logs 目录

在平台 Shell 中：

```bash
cd ~/projects/rl-attack-defense
mkdir -p logs
```

### 3. 跑冒烟测试（验证环境）

```bash
cd ~/projects/rl-attack-defense
python src/test_local.py
```

预期看到 agent 随机走动直到到达目标或超时。确认没有 import 报错即可。

### 4. 检查 PyTorch 是否可用

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

如果报 `ModuleNotFoundError: No module named 'torch'`，看下面「环境安装」。

如果不需要 GPU 训练，CPU 也可以（800 轮大概 5-10 分钟）。

### 5. 提交 Slurm 作业

> ⚠️ 把下面 `-p Students` 和 `--qos=qos_stu_default` 换成你上次确认过、能用的那个分区名和 qos。

用你之前确认可用的参数提交 `train.sbatch`。脚本里的 `#SBATCH` 指令已经就位，直接 `sbatch train.sbatch` 即可。

看状态：
```bash
squeue -u "$USER"
```

### 6. 看训练日志

```bash
tail -f logs/rl-attack-*.out
```

训练过程中会每 50 轮输出一行：

```
[Ep   50/800] reward=   5.00  avg100=   3.20  epsilon=0.xxx  buffer=xxxx
```

avg100 应该从负数/正小值慢慢增长到 4~7 左右，说明智能体在学会避开高检测路径。

训练结束后会自动打印完整渗透路径演示。

---

## 环境安装（如果平台没有 torch）

### 选项 A：pip 装（最快）

```bash
pip install --user torch numpy
```

### 选项 B：conda 隔离环境（推荐长期用）

```bash
# 创建环境（一次性）
conda create -n rl python=3.12 -y
conda activate rl
pip install torch numpy

# 之后每次用
conda activate rl
```

如果在 sbatch 脚本里用 conda 环境，需要在脚本里加：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl
```

---

## 扩展思路（你下一步可以改的方向）

1. **换拓扑** — 改 `ADJ` 和 `DETECTION`，模仿某真实内网的简化版
2. **加 IDS 响应机制** — 被检测次数超过阈值后节点被"封堵"
3. **换算法** — 用 PPO (stable-baselines3) 替代 DQN
4. **多 agent** — 红队攻击 + 蓝队防御，两个 RL agent 对抗
5. **真实流量成本建模** — 基于 CIC-IDS2017 数据拟合真实的各节点检测概率

---

# v2: 自主攻击链决策

从"路径选择"升级为完整的攻击链决策：智能体自主决定 **何时侦察、用什么漏洞利用、如何横向移动、何时窃取数据**。

## 环境拓扑

```
               [0] Gateway (检测=0.05)       ← 入口, 已公开
              /    \
             /      \
   [1] WebServer   [2] VPN-Gateway
   (HTTP,PHP)      (SSH,OpenVPN)
   检测=0.60        检测=0.10
   漏洞: CVE,SQLI   漏洞: weak-creds
        |                |
   [3] DB-Server    [4] JumpBox
   (MySQL)          (SSH)
   检测=0.30        检测=0.15
   漏洞: SQLI        漏洞: weak-creds
        |                |
        └──── [5] DC ★ ───┘
        (LDAP,SMB,RDP)
        检测=0.85
        漏洞: CVE, weak-creds
        FLAG: classified_data.tar
```

## 攻击链阶段

一条完整的攻击链要经历：

```
入口发掘 → 侦察(RECON) → 漏洞扫描(VULN_SCAN) → 漏洞利用(EXPLOIT)
                                                    ↓
                                              获得节点访问权
                                                    ↓
                                          权限提升(PRIVESC) → root
                                                    ↓
                                         横向移动(MOVE) → 下一节点
                                                    ↓
                                               ... 重复 ...
                                                    ↓
                                        数据窃取(EXFIL) → 🏁 flag
```

## 动作空间 (13 种)

| 动作 | 说明 | 前置条件 |
|------|------|---------|
| RECON | 扫描当前节点服务 | — |
| VULN_SCAN | 扫描已知服务的漏洞 | RECON 完成 |
| EXPLOIT_CVE | CVE 漏洞利用 | VULN_SCAN 完成 |
| EXPLOIT_CREDS | 弱口令攻击 | VULN_SCAN 完成 |
| EXPLOIT_SQLI | SQL 注入 | VULN_SCAN 完成 |
| PRIVESC | user → root | 已攻陷 |
| EXFIL | 窃取 flag | root + 有 flag |
| MOVE_0~5 | 横向移动 | 邻接节点 |

## 算法升级

| | v1 | **v2** |
|---|---|---|
| 网络结构 | MLP (128→64) | **Dueling (V+A分离)** |
| Q-target | Standard DQN | **Double DQN** |
| 动作选择 | ε-greedy | ε-greedy + **Action Mask** |
| 状态 | 6D (位置+检测) | **32D (6节点×4状态+服务+漏洞)** |
| 动作 | 5 (纯移动) | **13 (完整攻击链)** |

## 使用

```bash
# 冒烟测试
python src/test_local_v2.py

# 训练 (Slurm)
sbatch train_v2.sbatch
```

## 两条路线的安全对比

| | 快路径 (Web) | 安全路径 (VPN) |
|---|---|---|
| 路径 | Gateway→Web→DB→DC | Gateway→VPN→Jump→DC |
| 步骤 | 14 步 | 14 步 |
| 平均检测 | ~3.8 次 | ~1.5 次 |

智能体需要学会**从开始就选安全路径**——在 Gateway 处选 VPN (MOVE_2) 而非 Web (MOVE_1)。