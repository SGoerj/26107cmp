# 项目总结：从路径选择 DQN 到自主攻击链 PPO 智能体

> 一份高层演进总结，适合快速了解项目全貌。技术细节见 `GUIDE.md`。

## 一句话

训练一个能在模拟内网里**自主完成完整攻击链**（侦察→扫漏→利用→提权→横向移动→窃取）的强化学习智能体，并在动态、不可见、利用会失效的真实模拟条件下**稳定学深**，且参数不精确也能稳健。

## 演进脉络

项目经历了六个阶段，每一阶段都解决了上一阶段暴露的真实问题：

### 阶段 1：v1 路径选择（DQN）
智能体只在网络拓扑里选移动路径，在"到达目标"和"避开检测"间平衡。**问题**：只会"走路"，不会"攻击"——离真实渗透还有距离。

### 阶段 2：v2 攻击链决策（Double Dueling DQN）
扩展为完整攻击链：自主决定何时侦察、用什么漏洞、如何提权、何时移动、何时窃取。**问题**：DQN 训练后期策略持续退化（avg 从 +14 跌到 -52），演示靠 best 快照在高探索期碰运气，不是真学深。

### 阶段 3：换轨 PPO（关键转折）
用实证发现 DQN 学不深的根因——**经验回放的灾难性遗忘**：好策略被后期坏经验稀释。换 PPO（on-policy）治本：4 节点稳定收敛 avg +32、flag% 100%，**final 即 best**，演示不再是碰运气。

### 阶段 4：环境工程化
- **YAML 外部化**：拓扑/检测率/漏洞/奖励写配置文件，真实部署可从 Nessus/IDS 报告导入
- **POMDP A 步**：隐藏"被检测次数"，智能体只从惩罚奖励间接推断暴露，贴近真实渗透
- **维度动态化**：状态/动作维度从拓扑自动计算，换规模只改配置

### 阶段 5：让环境真正"动"起来（动态检测 + 概率利用）
- **动态检测**：检测率随被攻击行为升降（模拟 IDS 告警升级），智能体不可见。PPO 学会"打了就跑"——攻陷后立即离开被警觉节点
- **概率化漏洞利用**：EXPLOIT 按成功率随机成败，模拟真实利用会失败。PPO 学会容错
- 4/6 节点 × 静态/动态组合全部成功，6 节点动态（最难）加量训练后 avg +36、flag% 100%

### 阶段 6：参数稳健性（敏感度分析）
回答"真实参数会变化怎么办"：9 组参数扰动 × 3 种子，27 次训练全部 100% 拿 flag。**策略对手编参数 ±20% 波动 ROBUST**，参数不精确问题在分析范围内可接受。

### 阶段 7：真实数据拟合（CIC-IDS2017）
把检测率从经验估计升级为真实数据拟合。用 CIC-IDS2017 数据集算每类攻击的流量可识别度（逻辑回归 AUC），按相对排序映射 detection。**数据纠正了经验直觉**：CIC-IDS 显示 SSH 爆破（JumpBox）流量最易检测（AUC 0.996），而非我们手编的"最安全"；内网渗透（DC）最隐蔽（仅 36 样本）。拟合后 PPO 学到与经验版完全相反的路径（选 WebServer 入口而非 JumpBox）——这是数据驱动的最优策略。

### 阶段 8：真实 IDS 验证（代理指标的局限）
用 Suricata + 社区规则回放 CIC-IDS2017 原始 PCAP，验证 AUC 代理是否等于真实 IDS 检测。**负面但重要的发现**：三个节点 AUC 都 0.96-0.99，但真实 IDS 对 SSH 爆破、Web 攻击（72 个 SQLi 请求）**全部 0 检出**。证明 AUC 代理系统性高估真实检测能力——AUC 是"离线 78 特征可分性"，真实 IDS 靠"规则匹配"非 ML 分类，加密协议尤其失真。这定义了"数据驱动"的诚实边界：项目做的是"数据驱动代理 detection"，真实 IDS detection 需订阅规则或主机日志，超出当前条件。敏感度分析（策略对 detection robust）的意义反而被强化——策略本就不该依赖 detection 精确值。

### 阶段 9：蓝队对抗（self-play 边界探索）
尝试双智能体——引入蓝队防御智能体，通过 self-play（交替训练 + 对手快照池）让红蓝双方同时学习。设计四层兜底验证：随机蓝队→规则蓝队→self-play（调奖励）→self-play（加预测观测）。三轮 self-play 实验均收敛到蓝队不防御、红队 100%。**根因不是奖励或观测**（调过没用），而是 self-play 的非平稳性——蓝队在追移动靶，学不到一致策略。这是诚实的负面发现，定义了当前方法论的边界：self-play 在不完美信息 + 非零和下，蓝队学不到主动防御。解决需虚拟博弈/种群训练等专门博弈算法，超出当前条件。代码保留在 `src/multiagent/`，不影响单智能体基线。

## 核心能力

最终智能体在 6 节点 + 动态检测 + 概率利用环境下，演示走通完整攻击链：

```
Gateway → JumpDMZ (侦察/扫漏/利用creds/提权) → 立即走
       → PivotSrv (侦察/扫漏/利用creds/提权) → 立即走
       → DC (侦察/扫漏/利用cve/提权/窃取) 🏁
```

它自主学会了三件真实渗透的关键行为：
1. **选安全路径**：走低检测节点（JumpDMZ 0.20）而非高风险节点（WebDMZ 0.55）
2. **打了就跑**：攻陷提权后立即移动，不在被警觉的节点停留（动态检测下反复攻击会推高检测率）
3. **容错**：利用会失败时仍能完成攻击链（概率利用下学会重试/换路）

## 关键工程教训

1. **算法选择比调参重要**：DQN 调了四轮（PER、固定 ε、best 逻辑）都没解决遗忘，换 PPO 一次到位。先确认算法适合问题，再调参。

2. **评估 RL 要看低探索期真实表现**：高 ε 下的高 avg 是探索噪声，不代表策略质量。DQN 之前所有"成功演示"都是 best 在高探索期碰巧抓到能走通的快照。

3. **换算法时环境不动**：积累的环境外部化、POMDP 设计都复用——只换 agent，不重做环境。

4. **敏感度分析要多种子**：单种子短训练可能因运气不收敛（实验中 α=0.08 单种子 0%，3 种子取中位数后 100%）。

5. **小环境别过度工程**：PER、RNN 在 4 节点用不上，硬上反而更糟。让需求先出现，再上工具。

## 文件结构

```
configs/        环境配置 (YAML): env_default / env_6node / env_default_dynamic / env_6node_dynamic / env_default_dynamic_prob
src/
  env_v2.py     攻击链环境 (动态检测 + 概率利用 + POMDP)
  agent_v2.py   DQN 智能体 (对比基线)
  agent_ppo.py  PPO 智能体 (主力)
  demo_v2.py / demo_ppo.py   加载模型演示
  sensitivity.py   敏感度分析
  test_local_v2.py  冒烟测试
GUIDE.md        完整技术解读 (13 章)
```

## 运行

提交前先 `cd ~/projects/rl-attack-defense`（日志走相对路径 `logs/`）。

**关键实验提交命令**（对应下面"已验证的实验"表）：

```bash
# 1. PPO 4节点静态 (基础验证)
ALGO=ppo SCENARIO=default sbatch --time=01:30:00 train_v2.sbatch

# 2. PPO 6节点静态 (长路径)
ALGO=ppo SCENARIO=6node sbatch --time=01:30:00 train_v2.sbatch

# 3. PPO 4节点动态检测 (打了就跑)
ALGO=ppo SCENARIO=default_dynamic sbatch --time=01:30:00 train_v2.sbatch

# 4. PPO 6节点动态检测 (规模收尾, 需加量)
ALGO=ppo SCENARIO=6node_dynamic ROLLOUTS=120 sbatch --time=01:30:00 train_v2.sbatch

# 5. PPO 4节点动态+概率利用 (三维度叠加)
ALGO=ppo SCENARIO=default_dynamic_prob ROLLOUTS=90 sbatch --time=01:30:00 train_v2.sbatch

# 6. PPO 数据驱动检测率 (CIC-IDS 拟合)
ALGO=ppo SCENARIO=fitted ROLLOUTS=90 sbatch --time=01:30:00 train_v2.sbatch

# 7. 敏感度分析 (9组参数 × 3种子, 约20分钟)
RUN=sensitivity sbatch --time=01:30:00 train_v2.sbatch
```

**不训练, 直接加载已保存模型演示**：

```bash
.venv/bin/python src/demo_ppo.py --model ppo                     # 4节点
.venv/bin/python src/demo_ppo.py --model 6node                   # 6节点
.venv/bin/python src/demo_ppo.py --model default_dynamic_prob    # 动态+概率
.venv/bin/python src/demo_ppo.py --model fitted                  # 数据驱动
```

**数据驱动配置生成**（本地跑, 非HPC, 产出 `configs/env_fitted.yaml`）：

```bash
python3 src/fit_detection_from_cicids.py --data-dir archive --out configs/env_fitted.yaml
```

**环境变量说明**：
- `ALGO=ppo|dqn` 选算法（默认 dqn, 推荐 ppo）
- `SCENARIO=<name>` 选 `configs/env_<name>.yaml`
- `ROLLOUTS=N` PPO 训练量（默认 60, 难场景加量）
- `RUN=sensitivity` 跑敏感度分析（覆盖 SCENARIO/ALGO）

## 已验证的实验

| 场景 | 算法 | 日志 | avg | flag% | 关键行为 |
|------|------|------|-----|-------|---------|
| 4 节点静态 | PPO | 21109 | +32 | 100% | 选安全路径 |
| 6 节点静态 | PPO | 21115 | +37 | 100% | 长路径选路 |
| 4 节点动态 | PPO | 21220 | +30 | 100% | 打了就跑 |
| 6 节点动态 | PPO | 21258 | +36 | 100% | 长路径+动态 |
| 4 节点动态+概率 | PPO | 21267 | +26 | 100% | 三维度叠加容错 |
| 敏感度分析 | PPO | 21295 | — | 100%×27 | 参数波动 ROBUST |

## 下一步可能方向

1. **蓝队对抗（防御智能体）**：引入会学习的防御方，形成非平稳博弈。真实攻防对抗的终极形态，但论文级难度（自博弈 + 非平稳收敛）。
2. **真实数据拟合**：用 CIC-IDS2017 等数据集拟合检测率，把参数从经验估计升级为数据驱动。
3. POMDP B 步（RNN 策略）：当前规模未暴露记忆需求，暂缓。

详细技术解读、局限分析、术语表见 `GUIDE.md`。
