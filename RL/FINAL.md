# 项目最终总结：RL 自主攻防智能体

## 一句话

从 DQN 路径选择到 PPO 攻击链决策，再到数据驱动检测率和多智能体博弈——每一阶段都推进到明确边界，正面发现与负面发现均诚实记录。

## 成果全览（按下行顺序）

### 第一部分：算法演进（rl-attack-defense 主项目）

| 阶段 | 内容 | 关键发现 | 日志 |
|------|------|---------|------|
| v1 | DQN 路径选择 | 能在拓扑中选路但不能攻击 | — |
| v2 | Double Dueling DQN 攻击链 | 能攻击但学不深（avg 从+14跌到-52） | 21024 |
| **PPO换轨** | on-policy治本遗忘 | **avg +32稳定不退化，final即best** | 21109 |
| 环境工程 | YAML外部化+POMDP+动态维度 | 可配置、部分可观测 | 21050/21066/21082 |
| 动态检测 | IDS告警升级，不可见 | **学会"打了就跑"** | 21220 |
| 概率利用 | EXPLOIT会失败 | 容错 | 21267 |
| 敏感度分析 | 9组×3种子 | **策略ROBUST（全100%）** | 21295 |
| 6节点扩规模 | 双层拓扑长路径 | PPO仍稳（+36, 100%） | 21258 |
| CIC-IDS拟合 | AUC代理拟合detection | **纠正直觉**：SSH爆破最易检测，DC最隐蔽 | 21562 |
| B2更细映射 | 乘数+利用率 | 乘数拟合失败（诚实退回经验），利用率部分拟合 | 21733 |
| **B1真实IDS** | Suricata回放PCAP | **AUC系统性高估**：真实IDS对SSH/Web全0检出 | — |
| 蓝队对抗 | self-play×3轮 | **蓝队学不动**：奖励/观测/预测信号全调无效 | — |
| 6节点动态 | 规模收尾 | 加量120 rollout后学会（+36, 100%） | 21258 |

### 第二部分：博弈边界（ctf-selfplay 子项目）

| 轮次 | 设置 | 结论 |
|------|------|------|
| 1 | 8节点双网 | 无人赢（路径太长） |
| 2 | DC直连 | 无人赢 |
| 3 | 27D从零 | 无人赢 |
| 4 | 27D预训练 | 无人赢（对称锁死，熵0.75） |
| 5 | 双网络分化 | B 30-0碾压（分化但无竞争） |

**核心结论**：对称+分化都无法产生有意义的竞争博弈。先到者赢、落后方无反制。

## 最重要的三项发现

1. **DQN学不深** → 4次实验（PER/固定ε/严格best/宽松best）均后期退化。换PPO一次解决。**算法选择 > 调参。**

2. **AUC代理高估真实检测** → Suricata+社区规则回放PCAP，SSH爆破和Web攻击全0检出。AUC是"离线78特征可分性"，真实IDS靠规则匹配。**定义了"数据驱动"的诚实边界。**

3. **self-play在当前框架下无法形成博弈** → 9轮实验（红蓝×3 + CTF×5 + 红蓝调参×1）。非对称→蓝队追移动靶崩溃；对称→锁死或碾压。**定义了multi-agent RL攻防的方法论边界。**

## 复现命令（全部可跑）

```bash
# === 主项目 (rl-attack-defense) ===
cd ~/projects/rl-attack-defense

# PPO 4节点静态
ALGO=ppo SCENARIO=default sbatch --time=01:30:00 train_v2.sbatch

# PPO 6节点静态
ALGO=ppo SCENARIO=6node sbatch --time=01:30:00 train_v2.sbatch

# PPO 4节点动态检测
ALGO=ppo SCENARIO=default_dynamic sbatch --time=01:30:00 train_v2.sbatch

# PPO 6节点动态检测（规模收尾，需加量）
ALGO=ppo SCENARIO=6node_dynamic ROLLOUTS=120 sbatch --time=01:30:00 train_v2.sbatch

# PPO 4节点动态+概率利用（三维度叠加）
ALGO=ppo SCENARIO=default_dynamic_prob ROLLOUTS=90 sbatch --time=01:30:00 train_v2.sbatch

# PPO 数据驱动检测率（CIC-IDS拟合）
ALGO=ppo SCENARIO=fitted ROLLOUTS=90 sbatch --time=01:30:00 train_v2.sbatch

# 敏感度分析（9组×3种子，20分钟）
RUN=sensitivity sbatch --time=01:30:00 train_v2.sbatch

# === CTF子项目 (ctf-selfplay) ===
cd ~/projects/ctf-selfplay
cp ~/projects/rl-attack-defense/models/best_ppo.pt models/
.venv/bin/python -B src/train_ctf.py --model models/best_ppo.pt --rounds 100 --seed 0
```

## 文件结构

```
rl-attack-defense/          # 主项目
├── src/
│   ├── env_v2.py           # 攻击链环境（动态/概率/POMDP）
│   ├── agent_v2.py         # DQN（对比基线）
│   ├── agent_ppo.py        # PPO（主力）
│   ├── fit_detection_from_cicids.py  # CIC-IDS拟合
│   ├── sensitivity.py      # 敏感度分析
│   ├── demo_ppo.py / demo_v2.py
│   └── multiagent/         # 蓝队对抗（隔离）
│       ├── env_ma.py
│       ├── train_red_vs_random.py
│       ├── train_red_vs_rule.py
│       └── train_selfplay.py
├── configs/                # 全部YAML配置
│   ├── env_default.yaml / env_6node.yaml
│   ├── env_default_dynamic.yaml / env_6node_dynamic.yaml
│   ├── env_default_dynamic_prob.yaml
│   └── env_fitted.yaml
├── GUIDE.md                # 13章技术解读
├── SUMMARY.md              # 9阶段全貌
├── FINAL.md                # 本文件
└── models/

ctf-selfplay/               # CTF子项目
├── src/
│   ├── ctf_env.py
│   └── train_ctf.py
└── SUMMARY.md
```

## 边界与未来

| 能做 | 不能做（当前框架下） |
|------|---------------------|
| 单智能体PPO稳定学到攻击链 | self-play产生有意义的攻防博弈 |
| 动态检测+概率利用真实模拟 | 真实IDS检测率拟合（AUC是高估代理） |
| CIC-IDS数据驱动代理detection | 蓝队从不完美信息学到主动防御 |
| 敏感度分析证明策略ROBUST | 多智能体博弈收敛 |

跨越边界的方向（需要论文级投入）：虚拟博弈/NFSP/种群训练、真实IDS规则覆盖、蓝队预测性观测、差异化竞争奖励设计。