r"""
攻防对抗环境 v2 —— 自主攻击链决策

智能体在异构内网中自主决策完整的攻击链:
  RECON → VULN_SCAN → EXPLOIT → PRIVESC → MOVE → ... → EXFIL

拓扑 (4 节点, 2 条到 flag 的路线):

               [0] Gateway (检测=0.05)       ← 入口, 已 root
              /    \
             /      \
   [1] WebServer   [2] JumpBox
   (HTTP)          (SSH)
   检测=0.70        检测=0.15
   漏洞: CVE        漏洞: weak-creds
        |                |
        └──── [3] DC ★ ───┘
        (DB)
        检测=0.90
        漏洞: CVE
        FLAG: classified_data.tar

两条路线 (各 10 步):
  🔴 快路径: Gateway→Web→RECON→VULN→EXPLOIT_CVE→MOVE_DC→RECON→VULN→EXPLOIT_CVE→PRIVESC→EXFIL
  🟢 安全路径: Gateway→Jump→RECON→VULN→EXPLOIT_CREDS→MOVE_DC→RECON→VULN→EXPLOIT_CVE→PRIVESC→EXFIL

核心决策: 在 Gateway 选 Web(0.7) 还是 JumpBox(0.15)?
  → Web 直达 DC 但检测率高
  → JumpBox 同样直达 DC 但检测率低
  → DQN 必须学会优先选安全入口

动作空间 (11 种):
  RECON(0):      扫描当前节点服务          前置: -
  VULN_SCAN(1):  扫描已知服务的漏洞        前置: RECON完成
  EXPLOIT_CVE(2):      CVE 利用            前置: VULN_SCAN完成 + 漏洞存在
  EXPLOIT_CREDS(3):    弱口令攻击           前置: VULN_SCAN完成 + 漏洞存在
  EXPLOIT_SQLI(4):     SQL 注入             前置: VULN_SCAN完成 + 漏洞存在
  PRIVESC(5):     user → root              前置: 已攻陷, 未root
  EXFIL(6):       窃取 flag 数据            前置: root + 节点有flag (唯一动作)
  MOVE_0~3(7-10): 横向移动到目标节点        前置: 邻接 + 已攻陷当前节点

状态表示 (动态维度, POMDP 部分可观测):
  设 N=节点数, S=服务数, V=漏洞数。state_dim = 4N + S + V + 1 + N
    [0:N]        current_node one-hot
    [N:2N]       compromised per node (自己做过的, 可见)
    [2N:3N]      recon_done per node
    [3N:4N]      vuln_scan_done per node
    [4N:4N+S]    services_known on current (须先 RECON, 否则全 0)
    [4N+S:4N+S+V] vulns_known on current (须先 VULN_SCAN, 否则全 0)
    [4N+S+V]     steps_norm
    [4N+S+V+1:5N+S+V+1] visited per node
  默认 4 节点/3 服务/3 漏洞 → 27D。
  注: 已移除 detection_norm —— 智能体不可见"被检测次数",
        只能从检测惩罚奖励 (r 里的 -5) 间接推断暴露, 贴近真实渗透
"""

import os
from pathlib import Path

import numpy as np
import yaml

# 默认配置路径: <repo>/configs/env_default.yaml
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "env_default.yaml"


def load_config(config_path: str | os.PathLike | None = None) -> dict:
    """从 YAML 加载环境配置。config_path=None 用默认 4 节点配置。"""
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ═══════════════════════════════════════════════════════════════════════
# 动作常量
# ═══════════════════════════════════════════════════════════════════════
RECON = 0
VULN_SCAN = 1
EXPLOIT_CVE = 2
EXPLOIT_CREDS = 3
EXPLOIT_SQLI = 4
PRIVESC = 5
EXFIL = 6
MOVE_BASE = 7       # MOVE to node N = MOVE_BASE + N

# 固定动作名 (不随拓扑变化); 移动动作 MOVE_0..MOVE_{N-1} 按节点数动态生成
_FIXED_ACTION_NAMES = [
    "RECON", "VULN_SCAN",
    "EXPLOIT_CVE", "EXPLOIT_CREDS", "EXPLOIT_SQLI",
    "PRIVESC", "EXFIL",
]


def _make_action_names(n_nodes: int) -> list[str]:
    """动作名 = 7 个固定动作 + n_nodes 个 MOVE_x。"""
    return _FIXED_ACTION_NAMES + [f"MOVE_{i}" for i in range(n_nodes)]


def _compute_state_dim(n_nodes: int, n_services: int, n_vulns: int) -> int:
    """state 维度 = 4·N (current/compromised/recon/vuln_scan 各 N 位 one-hot)
    + n_services + n_vulns + 1 (steps) + N (visited)。
    注: POMDP 已移除 detection 位。
    """
    return 4 * n_nodes + n_services + n_vulns + 1 + n_nodes


# 漏洞类型 -> EXPLOIT 动作的映射 (硬编码, 因为固定动作的编号是常量)
_VULN_TO_EXPLOIT = {"cve": EXPLOIT_CVE, "creds": EXPLOIT_CREDS, "sqli": EXPLOIT_SQLI}


# ═══════════════════════════════════════════════════════════════════════
# 从配置构建环境数据
# ═══════════════════════════════════════════════════════════════════════
def _build_from_config(cfg: dict) -> dict:
    """把 YAML 配置转成环境运行时用的内部结构。

    节点用 id 关联邻接, 这里把 id 解析成数组下标, 供环境内部用整数索引访问。
    返回 dict: nodes, node_names, service_map, vuln_map, rewards, entry_idx, n_nodes。
    """
    # 服务/漏洞 -> 槽位索引 (按配置列表顺序)
    service_map = {svc: i for i, svc in enumerate(cfg["services"])}
    vuln_map = {v: i for i, v in enumerate(cfg["vulns"])}

    # id -> 数组下标
    raw_nodes = cfg["nodes"]
    id_to_idx = {n["id"]: i for i, n in enumerate(raw_nodes)}

    nodes = []
    entry_idx = 0
    for i, n in enumerate(raw_nodes):
        nodes.append({
            "id": n["id"],
            "name": n["name"],
            "services": set(n["services"]),
            "vulns": set(n["vulns"]),
            "detection": float(n["detection"]),
            "has_flag": bool(n["has_flag"]),
            "adj": [id_to_idx[a] for a in n["adj"]],
        })
        if n.get("entry", False):
            entry_idx = i

    # 校验: 必须恰好一个入口
    entries = [n for n in raw_nodes if n.get("entry", False)]
    if len(entries) != 1:
        raise ValueError(f"配置错误: 需要恰好一个 entry 节点, 实际 {len(entries)} 个")

    n_services = len(cfg["services"])
    n_vulns = len(cfg["vulns"])
    n_nodes = len(nodes)

    # 校验: 节点引用的服务/漏洞必须在配置表里
    for n in nodes:
        for svc in n["services"]:
            if svc not in service_map:
                raise ValueError(f"配置错误: 节点 {n['id']} 的服务 '{svc}' 不在 services 表中")
        for v in n["vulns"]:
            if v not in vuln_map:
                raise ValueError(f"配置错误: 节点 {n['id']} 的漏洞 '{v}' 不在 vulns 表中")
            if v not in _VULN_TO_EXPLOIT:
                raise ValueError(
                    f"配置错误: 漏洞 '{v}' 没有对应的 EXPLOIT 动作 "
                    f"(目前只支持 {sorted(_VULN_TO_EXPLOIT)})"
                )

    state_dim = _compute_state_dim(n_nodes, n_services, n_vulns)
    action_names = _make_action_names(n_nodes)

    # 动态检测配置 (可选)。启用后节点检测率随被攻击行为升降, 模拟 IDS 告警升级。
    dyn = cfg.get("dynamic_detection", {})
    dynamic_cfg = {
        "enabled": bool(dyn.get("enabled", False)),
        "alpha_attack": float(dyn.get("alpha_attack", 0.08)),   # 攻击性动作 +α
        "alpha_detected": float(dyn.get("alpha_detected", 0.15)),  # 被检测 +α
        "alpha_decay": float(dyn.get("alpha_decay", 0.01)),    # 每步 -α (衰减)
        "cap_high": float(dyn.get("cap_high", 0.95)),          # 检测率上限
    }

    # 概率化漏洞利用 (可选)。启用后 EXPLOIT 按成功率随机成功/失败, 模拟真实利用
    # 可能失败 (补丁、ASLR、监控干扰)。失败不攻陷节点, 给 fail 惩罚, 但仍触发检测
    # (攻击行为本身暴露)。智能体要学会重试或换路。
    prob = cfg.get("exploit_success_prob", {})
    prob_cfg = {
        "enabled": bool(prob.get("enabled", False)),
        "cve": float(prob.get("cve", 1.0)),
        "creds": float(prob.get("creds", 1.0)),
        "sqli": float(prob.get("sqli", 1.0)),
    }

    # 动作检测乘数 (可选, B2 数据拟合)。启用后 _roll_detection 用配置乘数,
    # 否则用硬编码默认 (RECON/VULN=0.1, EXPLOIT=0.5, PRIVESC/EXFIL=1.0, MOVE=0.3)。
    am = cfg.get("action_detection_multiplier", {})
    am_cfg = {
        "enabled": bool(am.get("enabled", False)),
        "recon": float(am.get("recon", 0.1)),
        "vuln_scan": float(am.get("vuln_scan", 0.1)),
        "exploit": float(am.get("exploit", 0.5)),
        "privesc": float(am.get("privesc", 1.0)),
        "exfil": float(am.get("exfil", 1.0)),
        "move": float(am.get("move", 0.3)),
    }

    return {
        "nodes": nodes,
        "node_names": [n["name"] for n in nodes],
        "service_map": service_map,
        "vuln_map": vuln_map,
        "rewards": cfg["rewards"],
        "entry_idx": entry_idx,
        "n_nodes": n_nodes,
        "n_services": n_services,
        "n_vulns": n_vulns,
        "state_dim": state_dim,
        "action_names": action_names,
        "n_actions": len(action_names),
        "max_steps": cfg.get("max_steps", 40),
        "dynamic": dynamic_cfg,
        "exploit_prob": prob_cfg,
        "action_mult": am_cfg,
    }


# 默认加载 (兼容旧代码的模块级常量; 也可传自定义配置给 AttackChainEnv)
_CFG = _build_from_config(load_config())

NODES = _CFG["nodes"]
NODE_NAMES = _CFG["node_names"]
SERVICE_MAP = _CFG["service_map"]
VULN_MAP = _CFG["vuln_map"]
N_NODES = _CFG["n_nodes"]
ENTRY_IDX = _CFG["entry_idx"]
STATE_DIM = _CFG["state_dim"]          # 动态: 4N + n_services + n_vulns + 1 + N
N_ACTIONS = _CFG["n_actions"]          # 动态: 7 固定 + N 移动
ACTION_NAMES = _CFG["action_names"]


# ═══════════════════════════════════════════════════════════════════════
# 奖赏常量 (从默认配置加载; AttackChainEnv 用自定义配置时会覆盖)
# ═══════════════════════════════════════════════════════════════════════
_RW = _CFG["rewards"]
STEP_COST = _RW["step_cost"]
RECON_REWARD = _RW["recon"]
VULN_SCAN_REWARD = _RW["vuln_scan"]
EXPLOIT_SUCCESS_REWARD = _RW["exploit_success"]
EXPLOIT_FAIL_PENALTY = _RW["exploit_fail"]
PRIVESC_REWARD = _RW["privesc"]
EXFIL_REWARD = _RW["exfil"]
DETECTION_PENALTY = _RW["detection"]
INVALID_PENALTY = _RW["invalid"]
EXPLORE_BONUS = _RW["explore_bonus"]
REVISIT_PENALTY = _RW["revisit_penalty"]


# ═══════════════════════════════════════════════════════════════════════
# 环境
# ═══════════════════════════════════════════════════════════════════════
class AttackChainEnv:
    """自主攻击链决策环境 (拓扑/奖励由 YAML 配置驱动)。

    智能体从入口节点 (entry, 已 root) 出发, 选择跳板攻击 flag 节点。
    传 config_path 可换一套完全不同的拓扑; 不传则用 configs/env_default.yaml。
    """

    def __init__(self, max_steps: int | None = None, config_path: str | os.PathLike | None = None):
        if config_path is not None:
            cfg = _build_from_config(load_config(config_path))
        else:
            cfg = _CFG   # 默认配置 (已加载为模块级常量)
        self.nodes = cfg["nodes"]
        self.node_names = cfg["node_names"]
        self.service_map = cfg["service_map"]
        self.vuln_map = cfg["vuln_map"]
        self.rewards = cfg["rewards"]
        self.entry_idx = cfg["entry_idx"]
        self.n_nodes = cfg["n_nodes"]
        self.n_services = cfg["n_services"]
        self.n_vulns = cfg["n_vulns"]
        self.state_dim = cfg["state_dim"]
        self.action_names = cfg["action_names"]
        self.n_actions = cfg["n_actions"]
        self._cfg_max_steps = cfg["max_steps"]
        self.dynamic = cfg["dynamic"]
        self.exploit_prob = cfg["exploit_prob"]
        self.action_mult = cfg["action_mult"]

        # 显式 max_steps 覆盖配置里的值
        self.max_steps = max_steps if max_steps is not None else self._cfg_max_steps
        self.reset()

    # ── 重置 ──────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self.current = self.entry_idx
        self.steps = 0
        self.detections = 0

        self.compromised = [False] * self.n_nodes
        self.root = [False] * self.n_nodes
        self.recon_done = [False] * self.n_nodes
        self.vuln_scan_done = [False] * self.n_nodes

        # 动态检测率: 每节点当前检测水平, reset 时回到 base。启用动态检测时
        # 随被攻击行为升降 (见 step); 不启用时恒等于 base, 行为与之前一致。
        self.det_level = [self.nodes[i]["detection"] for i in range(self.n_nodes)]

        # 入口节点已 root——防止 farming
        self.compromised[self.entry_idx] = True
        self.root[self.entry_idx] = True
        self.recon_done[self.entry_idx] = True
        self.vuln_scan_done[self.entry_idx] = True

        self._flag_captured = False
        self._visited = {0}
        return self._get_state()

    # ── 单步执行 ──────────────────────────────────────────────────────
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        rw = self.rewards
        info = {
            "action": action,
            "action_name": self.action_names[action],
            "node": self.current,
            "node_name": self.node_names[self.current],
            "invalid": False,
            "detected": False,
            "compromised": False,
            "flag_captured": False,
        }

        if not self._is_legal(action):
            info["invalid"] = True
            return self._get_state(), rw["invalid"], False, info

        reward = rw["step_cost"]

        detected = self._roll_detection(action)
        if detected:
            self.detections += 1
            reward += rw["detection"]
            info["detected"] = True

        # 动态检测率更新 (攻击性动作 +α, 被检测 +α, 每步 -α_decay)。
        # 放在动作执行前更新当前节点, 让本步的攻击后果反映到检测水平。
        self._update_dynamic_detection(action, detected)

        if action == RECON:
            self.recon_done[self.current] = True
            reward += rw["recon"]
            info["services_found"] = sorted(self.nodes[self.current]["services"])

        elif action == VULN_SCAN:
            self.vuln_scan_done[self.current] = True
            reward += rw["vuln_scan"]
            info["vulns_found"] = sorted(self.nodes[self.current]["vulns"])

        elif action in (EXPLOIT_CVE, EXPLOIT_CREDS, EXPLOIT_SQLI):
            vuln_type = {
                EXPLOIT_CVE: "cve",
                EXPLOIT_CREDS: "creds",
                EXPLOIT_SQLI: "sqli",
            }[action]
            if vuln_type in self.nodes[self.current]["vulns"]:
                # 概率化利用: 启用后按成功率随机判定。真实世界利用会失败
                # (补丁、ASLR、监控干扰)。失败不攻陷, 给 fail 惩罚, 但本步
                # 的检测判定已在上面执行 (攻击行为本身暴露)。智能体可重试或换路。
                if self.exploit_prob["enabled"]:
                    success = np.random.random() < self.exploit_prob[vuln_type]
                else:
                    success = True
                if success:
                    self.compromised[self.current] = True
                    reward += rw["exploit_success"]
                    info["compromised"] = True
                    info["vuln_used"] = vuln_type
                else:
                    reward += rw["exploit_fail"]
                    info["exploit_failed"] = True
            else:
                reward += rw["exploit_fail"]

        elif action == PRIVESC:
            self.root[self.current] = True
            reward += rw["privesc"]

        elif action == EXFIL:
            self._flag_captured = True
            reward += rw["exfil"]
            info["flag_captured"] = True
            self.steps += 1
            return self._get_state(), reward, True, info

        elif action >= MOVE_BASE:
            target = action - MOVE_BASE
            self.current = target
            if target not in self._visited:
                reward += rw["explore_bonus"]
                self._visited.add(target)
            else:
                reward += rw["revisit_penalty"]

        self.steps += 1
        done = self.steps >= self.max_steps
        return self._get_state(), reward, done, info

    # ── 合法动作集 ────────────────────────────────────────────────────
    def legal_actions(self) -> list[int]:
        legal = []

        node = self.nodes[self.current]
        at_recon = self.recon_done[self.current]
        at_vuln = self.vuln_scan_done[self.current]
        at_comp = self.compromised[self.current]
        at_root = self.root[self.current]

        if not at_recon:
            legal.append(RECON)

        if at_recon and not at_vuln:
            legal.append(VULN_SCAN)

        if at_vuln and not at_comp:
            for vuln_type in self.nodes[self.current]["vulns"]:
                if vuln_type == "cve":
                    legal.append(EXPLOIT_CVE)
                elif vuln_type == "creds":
                    legal.append(EXPLOIT_CREDS)
                elif vuln_type == "sqli":
                    legal.append(EXPLOIT_SQLI)

        if at_comp and not at_root:
            legal.append(PRIVESC)

        if at_root and node["has_flag"]:
            return [EXFIL]

        # MOVE: 必须已攻陷当前节点才能以它为跳板
        if at_comp:
            for target in node["adj"]:
                legal.append(MOVE_BASE + target)

        return legal

    # ── 内部方法 ──────────────────────────────────────────────────────
    def _is_legal(self, action: int) -> bool:
        return action in self.legal_actions()

    def _roll_detection(self, action: int) -> bool:
        base = self.det_level[self.current]   # 动态检测率 (启用时随行为升降)
        if self.action_mult["enabled"]:
            # B2: 乘数从配置读 (CIC-IDS 流量偏离度拟合)
            am = self.action_mult
            if action == RECON:
                multiplier = am["recon"]
            elif action == VULN_SCAN:
                multiplier = am["vuln_scan"]
            elif action in (EXPLOIT_CVE, EXPLOIT_CREDS, EXPLOIT_SQLI):
                multiplier = am["exploit"]
            elif action == PRIVESC:
                multiplier = am["privesc"]
            elif action == EXFIL:
                multiplier = am["exfil"]
            elif action >= MOVE_BASE:
                multiplier = am["move"]
            else:
                multiplier = 0.3
        else:
            multiplier = {
                RECON: 0.1, VULN_SCAN: 0.1,
                EXPLOIT_CVE: 0.5, EXPLOIT_CREDS: 0.5, EXPLOIT_SQLI: 0.5,
                PRIVESC: 1.0, EXFIL: 1.0,
            }.get(action, 0.3)
        return np.random.random() < (base * multiplier)

    def _update_dynamic_detection(self, action: int, detected: bool):
        """动态检测率更新: 攻击性动作 +α_attack, 被检测 +α_detected, 每步 -α_decay。
        clip 到 [base, cap_high]。不启用动态检测时直接返回 (det_level 恒为 base)。
        """
        if not self.dynamic["enabled"]:
            return
        n = self.current
        base = self.nodes[n]["detection"]
        cap = self.dynamic["cap_high"]
        # 攻击性动作触发告警升级
        if action in (EXPLOIT_CVE, EXPLOIT_CREDS, EXPLOIT_SQLI, PRIVESC, EXFIL):
            self.det_level[n] += self.dynamic["alpha_attack"]
        # 被检测确认入侵, 升级更快
        if detected:
            self.det_level[n] += self.dynamic["alpha_detected"]
        # 每步自然衰减 (没动静警觉度回落)
        self.det_level[n] -= self.dynamic["alpha_decay"]
        self.det_level[n] = float(np.clip(self.det_level[n], base, cap))

    def _get_state(self) -> np.ndarray:
        N = self.n_nodes
        s = np.zeros(self.state_dim, dtype=np.float32)
        off = 0

        # [off:off+N] current_node one-hot
        s[off + self.current] = 1.0
        off += N

        # [off:off+N] compromised per node
        for i in range(N):
            s[off + i] = 1.0 if self.compromised[i] else 0.0
        off += N

        # [off:off+N] recon_done per node
        for i in range(N):
            s[off + i] = 1.0 if self.recon_done[i] else 0.0
        off += N

        # [off:off+N] vuln_scan_done per node
        for i in range(N):
            s[off + i] = 1.0 if self.vuln_scan_done[i] else 0.0
        off += N

        # [off:off+n_services] services_known on current (须先 RECON)
        if self.recon_done[self.current]:
            for svc in self.nodes[self.current]["services"]:
                idx = self.service_map.get(svc)
                if idx is not None:
                    s[off + idx] = 1.0
        off += self.n_services

        # [off:off+n_vulns] vulns_known on current (须先 VULN_SCAN)
        if self.vuln_scan_done[self.current]:
            for vuln in self.nodes[self.current]["vulns"]:
                idx = self.vuln_map.get(vuln)
                if idx is not None:
                    s[off + idx] = 1.0
        off += self.n_vulns

        # [off] steps_norm (注: detection_norm 已移除——POMDP 隐藏检测状态,
        #     智能体不可见"被检测次数", 只能从检测惩罚奖励间接推断暴露)
        s[off] = min(self.steps, self.max_steps) / self.max_steps
        off += 1

        # [off:off+N] visited per node
        for i in range(N):
            s[off + i] = 1.0 if i in self._visited else 0.0
        off += N

        assert off == self.state_dim, f"state 布局错位: {off} != {self.state_dim}"
        return s

    # ── 调试信息 ──────────────────────────────────────────────────────
    def state_info(self) -> dict:
        node = self.nodes[self.current]
        return {
            "node": self.current,
            "node_name": node["name"],
            "compromised": self.compromised[self.current],
            "root": self.root[self.current],
            "recon_done": self.recon_done[self.current],
            "vuln_scan_done": self.vuln_scan_done[self.current],
            "services": sorted(node["services"]),
            "vulns": sorted(node["vulns"]),
            "has_flag": node["has_flag"],
            "detection_base": node["detection"],
            "legal_actions": [self.action_names[a] for a in self.legal_actions()],
            "detections_so_far": self.detections,
            "steps": self.steps,
        }


# ═══════════════════════════════════════════════════════════════════════
# 攻击链参考 (10 步)
# ═══════════════════════════════════════════════════════════════════════
# 🟢 安全路径 (JumpBox):
#   0. Gateway 已 root → 直接选入口
#   1. MOVE to JumpBox (node 2)
#   2. RECON JumpBox    → ssh
#   3. VULN_SCAN JumpBox → creds
#   4. EXPLOIT_CREDS JumpBox → 攻陷
#   5. MOVE to DC (node 3)
#   6. RECON DC         → db
#   7. VULN_SCAN DC     → cve
#   8. EXPLOIT_CVE DC   → 攻陷
#   9. PRIVESC DC       → root
#  10. EXFIL DC ★
#
# 🔴 快路径 (WebServer):
#   1. MOVE to WebServer (node 1) — 检测 0.70 > 0.15!
#   2-10 同上, 但 EXPLOIT_CVE 替代 EXPLOIT_CREDS
#
# 期望检测差: 快路径多 ~0.5 次检测 ≈ -2.5 奖励差 → DQN 应该能学会