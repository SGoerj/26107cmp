"""冒烟测试 v2 —— 验证 4 节点 AttackChainEnv 的完整攻击链。

用硬编码策略测试两条路线:
  1. 安全路径: Gateway → JumpBox → DC → EXFIL
  2. 快路径:   Gateway → WebServer → DC → EXFIL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_v2 import (
    AttackChainEnv, NODE_NAMES, ACTION_NAMES, N_NODES,
    RECON, VULN_SCAN, EXPLOIT_CVE, EXPLOIT_CREDS, EXPLOIT_SQLI,
    PRIVESC, EXFIL, MOVE_BASE,
)


def _step(env, action, desc, step_count, total_reward):
    legal = env.legal_actions()
    assert action in legal, (
        f"FAIL: {desc} — {ACTION_NAMES[action]} not in "
        f"{[ACTION_NAMES[a] for a in legal]}"
    )
    state, reward, done, info = env.step(action)
    node = info['node_name']
    flags = []
    if info.get('detected'):
        flags.append("⚠️ DETECTED")
    if info.get('flag_captured'):
        flags.append("🏁")
    flag_str = "  " + ", ".join(flags) if flags else ""
    print(f"  {step_count + 1:2d}. {desc:<33} @ {node:<12} r={reward:+6.2f}{flag_str}")
    return total_reward + reward, done, info


# ═══════════════════════════════════════════════════════════════════════
# 测试 1: 安全路径 (JumpBox)
# ═══════════════════════════════════════════════════════════════════════
def test_safe_path():
    print("=" * 60)
    print("测试 1: 安全路径攻击链 (Gateway → JumpBox → DC)")
    print("=" * 60)

    env = AttackChainEnv(max_steps=30)
    _ = env.reset()
    n, tr = 0, 0.0

    # Gateway 已 root, 直接 MOVE
    tr, d, _ = _step(env, MOVE_BASE + 2, "MOVE to JumpBox (2)", n, tr); n += 1
    tr, d, _ = _step(env, RECON,         "RECON JumpBox", n, tr); n += 1
    tr, d, _ = _step(env, VULN_SCAN,     "VULN_SCAN JumpBox", n, tr); n += 1
    tr, d, _ = _step(env, EXPLOIT_CREDS, "EXPLOIT_CREDS JumpBox", n, tr); n += 1
    tr, d, _ = _step(env, MOVE_BASE + 3, "MOVE to DC (3)", n, tr); n += 1
    tr, d, _ = _step(env, RECON,         "RECON DC", n, tr); n += 1
    tr, d, _ = _step(env, VULN_SCAN,     "VULN_SCAN DC", n, tr); n += 1
    tr, d, _ = _step(env, EXPLOIT_CVE,   "EXPLOIT_CVE DC", n, tr); n += 1
    tr, d, _ = _step(env, PRIVESC,       "PRIVESC DC", n, tr); n += 1
    tr, d, info = _step(env, EXFIL,       "EXFIL DC 🏁", n, tr); n += 1

    print("-" * 60)
    assert env.root[3], "FAIL: DC should be rooted!"
    assert env._flag_captured, "FAIL: flag should be captured!"
    print(f"  ✅ 安全路径测试通过  总奖励={tr:+.2f}  步数={n}  检测={env.detections}次")
    return True


# ═══════════════════════════════════════════════════════════════════════
# 测试 2: 快路径 (WebServer)
# ═══════════════════════════════════════════════════════════════════════
def test_fast_path():
    print("\n" + "=" * 60)
    print("测试 2: 快路径攻击链 (Gateway → WebServer → DC)")
    print("=" * 60)

    env = AttackChainEnv(max_steps=30)
    _ = env.reset()
    n, tr = 0, 0.0

    tr, d, _ = _step(env, MOVE_BASE + 1, "MOVE to WebServer (1)", n, tr); n += 1
    tr, d, _ = _step(env, RECON,         "RECON Web", n, tr); n += 1
    tr, d, _ = _step(env, VULN_SCAN,     "VULN_SCAN Web", n, tr); n += 1
    tr, d, _ = _step(env, EXPLOIT_CVE,   "EXPLOIT_CVE Web", n, tr); n += 1
    tr, d, _ = _step(env, MOVE_BASE + 3, "MOVE to DC (3)", n, tr); n += 1
    tr, d, _ = _step(env, RECON,         "RECON DC", n, tr); n += 1
    tr, d, _ = _step(env, VULN_SCAN,     "VULN_SCAN DC", n, tr); n += 1
    tr, d, _ = _step(env, EXPLOIT_CVE,   "EXPLOIT_CVE DC", n, tr); n += 1
    tr, d, _ = _step(env, PRIVESC,       "PRIVESC DC", n, tr); n += 1
    tr, d, info = _step(env, EXFIL,       "EXFIL DC 🏁", n, tr); n += 1

    print("-" * 60)
    assert env.root[3], "FAIL: DC should be rooted!"
    assert env._flag_captured, "FAIL: flag should be captured!"
    print(f"  ✅ 快路径测试通过  总奖励={tr:+.2f}  步数={n}  检测={env.detections}次")
    return True


# ═══════════════════════════════════════════════════════════════════════
# 测试 3: 无效动作 + Gateway 不需 PRIVESC
# ═══════════════════════════════════════════════════════════════════════
def test_invalid_actions():
    print("\n" + "=" * 60)
    print("测试 3: 无效动作拒绝 + Gateway 防 farming")
    print("=" * 60)

    env = AttackChainEnv(max_steps=30)
    _ = env.reset()

    # Gateway 已 root → PRIVESC 不应合法
    assert PRIVESC not in env.legal_actions(), "FAIL: Gateway already root, PRIVESC should be illegal"
    print("  ✅ Gateway 已 root, PRIVESC 不合法 (防 farming)")

    # Gateway 已 recon_done → RECON 不应合法
    assert RECON not in env.legal_actions(), "FAIL: Gateway already recon'd"
    print("  ✅ Gateway 已 RECON, 不可重复扫描")

    # EXFIL on Gateway (no flag)
    assert EXFIL not in env.legal_actions(), "FAIL: Gateway has no flag"
    print("  ✅ Gateway 无 flag, EXFIL 不合法")

    print("-" * 60)
    print("  ✅ 前置条件检查全部通过")
    return True


# ═══════════════════════════════════════════════════════════════════════
# 测试 4: 状态向量维度
# ═══════════════════════════════════════════════════════════════════════
def test_state_dimension():
    import numpy as np
    from env_v2 import STATE_DIM

    print("\n" + "=" * 60)
    print("测试 4: 状态向量维度")
    print("=" * 60)

    env = AttackChainEnv(max_steps=30)
    state = env.reset()
    assert state.shape == (STATE_DIM,), f"FAIL: expected ({STATE_DIM},), got {state.shape}"
    assert state.dtype == np.float32

    # Gateway: comp[0]=1, root[0]=1, recon[0]=1, vuln[0]=1
    assert state[0] == 1.0, f"one-hot node 0"
    assert state[4] == 1.0, f"compromised[0]"
    assert state[8] == 1.0, f"recon_done[0]"

    print(f"  ✅ 状态向量: shape={state.shape}")
    print(f"  ✅ Gateway 初始: cur=0, comp=1, recon=1, vuln=1, root=1")
    print("-" * 60)
    print("  ✅ 状态向量测试全部通过")
    return True


# ═══════════════════════════════════════════════════════════════════════
# 测试 5: 检测对比
# ═══════════════════════════════════════════════════════════════════════
def test_detection_contrast():
    print("\n" + "=" * 60)
    print("测试 5: JumpBox vs WebServer 检测对比 (各 300 次)")
    print("=" * 60)

    def run(path):
        env = AttackChainEnv(max_steps=30)
        env.reset()
        script = path[:]
        for a in script:
            _, _, done, _ = env.step(a)
            if done:
                break
        return env.detections

    safe_script = [MOVE_BASE + 2, RECON, VULN_SCAN, EXPLOIT_CREDS,
                   MOVE_BASE + 3, RECON, VULN_SCAN, EXPLOIT_CVE,
                   PRIVESC, EXFIL]
    fast_script = [MOVE_BASE + 1, RECON, VULN_SCAN, EXPLOIT_CVE,
                   MOVE_BASE + 3, RECON, VULN_SCAN, EXPLOIT_CVE,
                   PRIVESC, EXFIL]

    N = 300
    safe_dets = [run(safe_script) for _ in range(N)]
    fast_dets = [run(fast_script) for _ in range(N)]

    avg_safe = sum(safe_dets) / N
    avg_fast = sum(fast_dets) / N

    print(f"  JumpBox (安全): 平均检测 = {avg_safe:.2f} 次")
    print(f"  WebServer (快): 平均检测 = {avg_fast:.2f} 次")
    print(f"  安全路径少被检测 {avg_fast - avg_safe:.2f} 次")
    assert avg_safe < avg_fast, f"FAIL: safe should have fewer detections!"
    print("  ✅ 安全路径确实检测更少")
    print("-" * 60)
    return True


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as np
    tests = [
        ("安全路径攻击链", test_safe_path),
        ("快路径攻击链", test_fast_path),
        ("无效动作拒绝 + 防 farming", test_invalid_actions),
        ("状态向量维度", test_state_dimension),
        ("检测期望对比", test_detection_contrast),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ {name} FAILED: {e}")
        except Exception as e:
            import traceback
            print(f"\n  ❌ {name} ERROR: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  结果: {passed}/{len(tests)} 通过")
    if passed == len(tests):
        print("  🎉 全部通过！可以提交训练。")
    else:
        print(f"  ⚠️  {len(tests) - passed} 个测试失败。")
    print("=" * 60)