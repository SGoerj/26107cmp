"""
双智能体攻防环境 (multiagent) —— 红队 vs 蓝队

隔离说明: 本文件在 src/multiagent/ 下, 不改动 src/env_v2.py (单智能体基线)。
复用单智能体的拓扑/检测机制, 但支持红蓝双方交替动作 + 蓝队防御动作。

设计 (见对话中的四个设计决策):
  红队动作: 沿用单智能体 (RECON/VULN_SCAN/EXPLOIT/PRIVESC/EXFIL/MOVE)
  蓝队动作: MONITOR / HARDEN_<node> / ISOLATE_<node>  (3类, 9个动作 for N=4)
  蓝队观测: 告警计数 + 自身防御状态 + 检测率 (不看红队位置, 不开上帝视角)
  双方奖励: 非零和, 红队拿flag / 蓝队阻止flag + 精准防御奖励 - 误隔离重罚

兜底: 当前版本蓝队动作由外部传入 (可随机/规则/学习), 环境本身不实现蓝队学习。
      先用随机/规则蓝队验证红队能在对抗下学, 再上 self-play。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from env_v2 import (
    load_config, _build_from_config, RECON, VULN_SCAN,
    EXPLOIT_CVE, EXPLOIT_CREDS, EXPLOIT_SQLI, PRIVESC, EXFIL, MOVE_BASE,
    _VULN_TO_EXPLOIT,
)

# ──────────────────────────────────────────────────────────────────────
# 蓝队动作常量
# ──────────────────────────────────────────────────────────────────────
BLUE_MONITOR = 0
BLUE_HARDEN_BASE = 1      # HARDEN_<node> = BLUE_HARDEN_BASE + node
BLUE_ISOLATE_BASE = None  # 初始化时按 N 设 (ISOLATE_<node> = BLUE_ISOLATE_BASE + node)


def blue_action_names(n_nodes: int) -> list[str]:
    names = ["MONITOR"]
    for i in range(n_nodes):
        names.append(f"HARDEN_{i}")
    for i in range(n_nodes):
        names.append(f"ISOLATE_{i}")
    return names


def n_blue_actions(n_nodes: int) -> int:
    return 1 + 2 * n_nodes


# ──────────────────────────────────────────────────────────────────────
# 奖励常量 (蓝队)
# ──────────────────────────────────────────────────────────────────────
# 红队奖励沿用单智能体配置 rewards。蓝队奖励单独定义。
BLUE_REWARD = {
    "block_flag": 30.0,       # episode 结束红队没拿到 flag
    "flag_stolen": -30.0,     # flag 被偷
    "isolate_alert": 8.0,     # 隔离了有告警的节点 (正确, 提高让防御有收益)
    "isolate_noalert": -3.0,  # 隔离了无告警节点 (误隔离, 降惩罚让蓝队敢做)
    "harden_alert": 5.0,      # 加固有告警节点 (提高)
    "harden_noalert": -2.0,   # 加固无告警节点
    "action_cost": -0.5,      # 每个防御动作成本 (别乱动)
    "alert_detected": 1.0,    # 红队被检测 (IDS 有效, 间接收益)
}


class MultiAgentAttackEnv:
    """双智能体攻防环境。红队进攻, 蓝队防御, 同一 step 双方各出一个动作。"""

    def __init__(self, config_path: str | os.PathLike | None = None,
                 blue_budget: int = 3):
        """blue_budget: 蓝 episode 内 HARDEN+ISOLATE 总次数上限 (防无限隔离)。"""
        if config_path is not None:
            cfg = _build_from_config(load_config(config_path))
        else:
            from env_v2 import _CFG
            cfg = _CFG
        self.nodes = cfg["nodes"]
        self.node_names = cfg["node_names"]
        self.service_map = cfg["service_map"]
        self.vuln_map = cfg["vuln_map"]
        self.rewards = cfg["rewards"]          # 红队奖励
        self.entry_idx = cfg["entry_idx"]
        self.n_nodes = cfg["n_nodes"]
        self.n_services = cfg["n_services"]
        self.n_vulns = cfg["n_vulns"]
        self.dynamic = cfg["dynamic"]
        self.exploit_prob = cfg["exploit_prob"]
        self.action_mult = cfg["action_mult"]
        self._cfg_max_steps = cfg["max_steps"]
        self.max_steps = self._cfg_max_steps

        # 蓝队
        self.blue_budget_max = blue_budget
        self.n_red_actions = 7 + self.n_nodes      # 红队动作数 (同单智能体)
        self.n_blue_actions = n_blue_actions(self.n_nodes)
        self.blue_action_names = blue_action_names(self.n_nodes)
        global BLUE_ISOLATE_BASE
        BLUE_ISOLATE_BASE = 1 + self.n_nodes       # ISOLATE 起始编号

        # 状态维度
        self.red_state_dim = self._compute_red_state_dim()
        self.blue_state_dim = self._compute_blue_state_dim()

        self.reset()

    def _compute_red_state_dim(self):
        return 4 * self.n_nodes + self.n_services + self.n_vulns + 1 + self.n_nodes

    def _compute_blue_state_dim(self):
        # 告警 N + 探测痕迹 N + 加固 N + 隔离 N + 检测率 N + 步数 1 + flag被偷 1
        return 5 * self.n_nodes + 2

    # ── 重置 ──────────────────────────────────────────────────────────
    def reset(self):
        self.current = self.entry_idx
        self.steps = 0
        self.detections = 0

        self.compromised = [False] * self.n_nodes
        self.root = [False] * self.n_nodes
        self.recon_done = [False] * self.n_nodes
        self.vuln_scan_done = [False] * self.n_nodes
        self.compromised[self.entry_idx] = True
        self.root[self.entry_idx] = True
        self.recon_done[self.entry_idx] = True
        self.vuln_scan_done[self.entry_idx] = True

        self._flag_captured = False
        self._visited = {self.entry_idx}

        # 动态检测
        self.det_level = [self.nodes[i]["detection"] for i in range(self.n_nodes)]

        # 蓝队状态
        self.alerts = [0] * self.n_nodes          # 各节点累积告警计数
        self.probe_traces = [0] * self.n_nodes    # 各节点被探测痕迹 (红队RECON/VULN_SCAN累积)
        self.hardened = [False] * self.n_nodes    # 是否被加固
        self.isolated = [False] * self.n_nodes    # 是否被隔离
        self.blue_budget = self.blue_budget_max   # 剩余防御预算

        return self._red_state(), self._blue_state()

    # ── 单步: 红蓝同时出动作 ──────────────────────────────────────────
    def step(self, red_action: int, blue_action: int):
        info = {
            "red_action": red_action, "blue_action": blue_action,
            "red_invalid": False, "blue_invalid": False,
            "detected": False, "flag_captured": False,
            "compromised": False, "node": self.current,
        }
        red_reward = self.rewards["step_cost"]
        blue_reward = 0.0

        # ── 背景误报: 每步以小概率给随机节点 +1 告警 (IDS 误报噪声)。
        # 让蓝队"看到告警"≠"红队在那", 规则蓝队的"隔离告警节点"会误中正常节点,
        # 避免蓝队靠告警准确定位红队 (破坏不开上帝视角的设计)。
        if np.random.random() < 0.08:
            self.alerts[np.random.randint(self.n_nodes)] += 1

        # ── 蓝队动作先执行 (防御部署, 影响本步红队) ──
        blue_reward += self._exec_blue(blue_action, info)

        # ── 红队动作 ──
        if not self._red_is_legal(red_action):
            info["red_invalid"] = True
            red_reward += self.rewards["invalid"]
        else:
            red_reward, r_done, r_info = self._exec_red(red_action, info)
            if info.get("detected"):
                blue_reward += BLUE_REWARD["alert_detected"]

        self.steps += 1
        done = self.steps >= self.max_steps or self._flag_captured

        # 终局蓝队奖励
        if done:
            if self._flag_captured:
                blue_reward += BLUE_REWARD["flag_stolen"]
            else:
                blue_reward += BLUE_REWARD["block_flag"]

        return (self._red_state(), self._blue_state(),
                red_reward, blue_reward, done, info)

    # ── 蓝队动作执行 ──────────────────────────────────────────────────
    def _exec_blue(self, action: int, info: dict) -> float:
        rw = 0.0
        if action == BLUE_MONITOR:
            return rw   # 不动, 不扣成本 (MONITOR 免费鼓励观察)
        if self.blue_budget <= 0:
            info["blue_invalid"] = True
            return self.rewards.get("invalid", -3.0) * 0.5   # 预算用完还动作, 轻罚

        node = None
        is_isolate = False
        if BLUE_HARDEN_BASE <= action < BLUE_ISOLATE_BASE:
            node = action - BLUE_HARDEN_BASE
            is_isolate = False
        elif action >= BLUE_ISOLATE_BASE:
            node = action - BLUE_ISOLATE_BASE
            is_isolate = True

        if node is None or node >= self.n_nodes:
            info["blue_invalid"] = True
            return 0.0

        rw += BLUE_REWARD["action_cost"]
        has_alert = self.alerts[node] > 0
        self.blue_budget -= 1

        if is_isolate:
            # 隔离: 不切断路径 (那会让红队无路可走, 博弈失衡), 而是把该节点
            # 检测率拉到 cap (重点监控)。红队还能走, 但每步高概率被检测。
            if self.isolated[node]:
                info["blue_invalid"] = True
                return rw   # 重复隔离, 无效但不额外罚
            self.isolated[node] = True
            self.det_level[node] = self.dynamic["cap_high"]   # 重点监控
            rw += BLUE_REWARD["isolate_alert"] if has_alert else BLUE_REWARD["isolate_noalert"]
        else:
            # 加固: 提升该节点检测率
            if self.hardened[node]:
                info["blue_invalid"] = True
                return rw
            self.hardened[node] = True
            self.det_level[node] = min(self.dynamic["cap_high"],
                                       self.det_level[node] + 0.2)
            rw += BLUE_REWARD["harden_alert"] if has_alert else BLUE_REWARD["harden_noalert"]
        return rw

    # ── 红队动作执行 (基本复用单智能体逻辑 + 隔离检查) ────────────────
    def _red_is_legal(self, action: int) -> bool:
        return action in self._red_legal_actions()

    def _red_legal_actions(self) -> list[int]:
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
            for v in self.nodes[self.current]["vulns"]:
                if v == "cve":
                    legal.append(EXPLOIT_CVE)
                elif v == "creds":
                    legal.append(EXPLOIT_CREDS)
                elif v == "sqli":
                    legal.append(EXPLOIT_SQLI)
        if at_comp and not at_root:
            legal.append(PRIVESC)
        if at_root and node["has_flag"]:
            return [EXFIL]
        if at_comp:
            for target in node["adj"]:
                legal.append(MOVE_BASE + target)   # 隔离节点可走 (只是高检测)
        return legal

    def red_legal_actions(self) -> list[int]:
        return self._red_legal_actions()

    def blue_legal_actions(self) -> list[int]:
        """蓝队合法动作: MONITOR + 未加固节点HARDEN + 未隔离节点ISOLATE (预算>0)。"""
        legal = [BLUE_MONITOR]
        if self.blue_budget > 0:
            for i in range(self.n_nodes):
                if not self.hardened[i]:
                    legal.append(BLUE_HARDEN_BASE + i)
            for i in range(self.n_nodes):
                if not self.isolated[i]:
                    legal.append(BLUE_ISOLATE_BASE + i)
        return legal

    def _exec_red(self, action: int, info: dict):
        rw = self.rewards
        reward = 0.0
        detected = self._roll_detection(action)
        if detected:
            self.detections += 1
            reward += rw["detection"]
            info["detected"] = True
            self.alerts[self.current] += 1     # 告警计入当前节点
        self._update_dynamic_detection(action, detected)

        done = False
        if action == RECON:
            self.recon_done[self.current] = True
            reward += rw["recon"]
            self.probe_traces[self.current] += 1   # 留下探测痕迹 (蓝队可观测)
        elif action == VULN_SCAN:
            self.vuln_scan_done[self.current] = True
            reward += rw["vuln_scan"]
            self.probe_traces[self.current] += 1   # 留下探测痕迹
        elif action in (EXPLOIT_CVE, EXPLOIT_CREDS, EXPLOIT_SQLI):
            vt = {EXPLOIT_CVE: "cve", EXPLOIT_CREDS: "creds", EXPLOIT_SQLI: "sqli"}[action]
            if vt in self.nodes[self.current]["vulns"]:
                if self.exploit_prob["enabled"]:
                    success = np.random.random() < self.exploit_prob[vt]
                else:
                    success = True
                if success:
                    self.compromised[self.current] = True
                    reward += rw["exploit_success"]
                    info["compromised"] = True
                else:
                    reward += rw["exploit_fail"]
            else:
                reward += rw["exploit_fail"]
        elif action == PRIVESC:
            self.root[self.current] = True
            reward += rw["privesc"]
        elif action == EXFIL:
            self._flag_captured = True
            reward += rw["exfil"]
            info["flag_captured"] = True
            done = True
        elif action >= MOVE_BASE:
            target = action - MOVE_BASE
            self.current = target
            if target not in self._visited:
                reward += rw["explore_bonus"]
                self._visited.add(target)
            else:
                reward += rw["revisit_penalty"]
        return reward, done, info

    # ── 检测 (复用单智能体) ───────────────────────────────────────────
    def _roll_detection(self, action: int) -> bool:
        base = self.det_level[self.current]
        if self.action_mult["enabled"]:
            am = self.action_mult
            if action == RECON:
                m = am["recon"]
            elif action == VULN_SCAN:
                m = am["vuln_scan"]
            elif action in (EXPLOIT_CVE, EXPLOIT_CREDS, EXPLOIT_SQLI):
                m = am["exploit"]
            elif action == PRIVESC:
                m = am["privesc"]
            elif action == EXFIL:
                m = am["exfil"]
            elif action >= MOVE_BASE:
                m = am["move"]
            else:
                m = 0.3
        else:
            m = {RECON: 0.1, VULN_SCAN: 0.1, EXPLOIT_CVE: 0.5, EXPLOIT_CREDS: 0.5,
                 EXPLOIT_SQLI: 0.5, PRIVESC: 1.0, EXFIL: 1.0}.get(action, 0.3)
        return np.random.random() < (base * m)

    def _update_dynamic_detection(self, action: int, detected: bool):
        if not self.dynamic["enabled"]:
            return
        n = self.current
        base = self.nodes[n]["detection"]
        cap = self.dynamic["cap_high"]
        if action in (EXPLOIT_CVE, EXPLOIT_CREDS, EXPLOIT_SQLI, PRIVESC, EXFIL):
            self.det_level[n] += self.dynamic["alpha_attack"]
        if detected:
            self.det_level[n] += self.dynamic["alpha_detected"]
        self.det_level[n] -= self.dynamic["alpha_decay"]
        self.det_level[n] = float(np.clip(self.det_level[n], base, cap))

    # ── 红队观测 (同单智能体 state, 加蓝队隔离标记) ───────────────────
    def _red_state(self) -> np.ndarray:
        N = self.n_nodes
        s = np.zeros(self.red_state_dim, dtype=np.float32)
        off = 0
        s[off + self.current] = 1.0; off += N
        for i in range(N):
            s[off + i] = 1.0 if self.compromised[i] else 0.0
        off += N
        for i in range(N):
            s[off + i] = 1.0 if self.recon_done[i] else 0.0
        off += N
        for i in range(N):
            s[off + i] = 1.0 if self.vuln_scan_done[i] else 0.0
        off += N
        if self.recon_done[self.current]:
            for svc in self.nodes[self.current]["services"]:
                idx = self.service_map.get(svc)
                if idx is not None:
                    s[off + idx] = 1.0
        off += self.n_services
        if self.vuln_scan_done[self.current]:
            for v in self.nodes[self.current]["vulns"]:
                idx = self.vuln_map.get(v)
                if idx is not None:
                    s[off + idx] = 1.0
        off += self.n_vulns
        s[off] = min(self.steps, self.max_steps) / self.max_steps; off += 1
        for i in range(N):
            s[off + i] = 1.0 if i in self._visited else 0.0
        off += N
        return s

    # ── 蓝队观测: 告警 + 防御状态 + 检测率 + 步数 + flag被偷 ──────────
    def _blue_state(self) -> np.ndarray:
        N = self.n_nodes
        s = np.zeros(self.blue_state_dim, dtype=np.float32)
        off = 0
        # 告警计数归一化
        max_alert = max(max(self.alerts), 1)
        for i in range(N):
            s[off + i] = self.alerts[i] / max_alert
        off += N
        # 探测痕迹归一化 (红队 RECON/VULN_SCAN 留下, 比告警更前瞻)
        max_probe = max(max(self.probe_traces), 1)
        for i in range(N):
            s[off + i] = self.probe_traces[i] / max_probe
        off += N
        for i in range(N):
            s[off + i] = 1.0 if self.hardened[i] else 0.0
        off += N
        for i in range(N):
            s[off + i] = 1.0 if self.isolated[i] else 0.0
        off += N
        for i in range(N):
            s[off + i] = self.det_level[i]
        off += N
        s[off] = min(self.steps, self.max_steps) / self.max_steps; off += 1
        s[off] = 1.0 if self._flag_captured else 0.0; off += 1
        return s


if __name__ == "__main__":
    # 快速自测: 随机蓝队 + 随机红队跑一局
    env = MultiAgentAttackEnv()
    print(f"红队: state_dim={env.red_state_dim} n_actions={env.n_red_actions}")
    print(f"蓝队: state_dim={env.blue_state_dim} n_actions={env.n_blue_actions}")
    print(f"蓝队动作: {env.blue_action_names}")
    rs, bs = env.reset()
    print(f"reset: red_state={rs.shape} blue_state={bs.shape}")
    np.random.seed(0)
    total_r, total_b = 0.0, 0.0
    for step in range(env.max_steps):
        red_legal = env.red_legal_actions()
        if not red_legal:
            break   # 红队无合法动作 (被隔离等), 提前结束
        ra = np.random.choice(red_legal)
        ba = np.random.choice(env.blue_legal_actions())
        rs, bs, rr, br, done, info = env.step(ra, ba)
        total_r += rr; total_b += br
        if done:
            break
    print(f"一局结束: red_reward={total_r:+.2f} blue_reward={total_b:+.2f} "
          f"steps={env.steps} flag={'✅' if env._flag_captured else '❌'}")
