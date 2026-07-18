"""
本地快速测试：在环境里走一局随机策略，验证 env.py 没有 bug。

用法（在本地或平台 Shell 里跑）:
    python src/test_local.py
"""
from env import NetworkEnv, NODE_NAMES, TARGET_NODES

env = NetworkEnv(max_steps=30)
node = env.reset()
done = False

print("=" * 60)
print(f"起始节点: {NODE_NAMES[node]}")
print(f"目标节点: {', '.join(NODE_NAMES[t] for t in TARGET_NODES)}")
print("=" * 60)

while not done:
    legal = env.legal_actions()
    action = legal[0]  # 永远选第一个合法动作（非随机，仅测试）
    next_node, reward, done, info = env.step(action)
    status = "⚠️ DETECTED" if info.get("detected") else "✓ clean"
    print(
        f"  {NODE_NAMES[node]:20s} → {NODE_NAMES[next_node]:20s}"
        f"  reward={reward:+6.2f}   {status}"
    )
    node = next_node

print("-" * 60)
print(f"完成 → 节点: {NODE_NAMES[env.current]}  |  步数: {env.steps}  |  检测: {env.detections} 次")