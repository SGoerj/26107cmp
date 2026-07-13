"""
加载已保存的 best model, 直接演示攻击链 (无需重新训练)。

用法:
    python src/demo_v2.py                     # 4 节点默认模型
    python src/demo_v2.py --model best_6node  # 6 节点模型

依赖文件: models/best_*.pt (由 agent_v2.py 训练时生成)
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_v2 import AttackChainEnv, STATE_DIM, N_ACTIONS
from agent_v2 import DuelingNetwork, demonstrate


def load_model(model_path: str, device: torch.device) -> tuple[DuelingNetwork, int, int]:
    """返回 (网络, state_dim, n_actions) —— 维度从 checkpoint 读, 与训练时一致。"""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dim = ckpt.get("state_dim", STATE_DIM)
    n_actions = ckpt.get("n_actions", N_ACTIONS)
    net = DuelingNetwork(state_dim, n_actions).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, state_dim, n_actions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2",
                    help="模型标签: v2 (4节点默认) 或 6node (6节点)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", f"best_{args.model}.pt")

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {os.path.abspath(model_path)}")
        print(f"   请先运行 `python src/agent_v2.py --config configs/env_{args.model.replace('v2','default')}.yaml` 训练。")
        sys.exit(1)

    print(f"[加载] {os.path.abspath(model_path)}  (device={device})")
    q_net, state_dim, n_actions = load_model(model_path, device)

    # 按模型标签选对应配置, 保证 env 维度与训练时一致
    config_path = None
    if args.model == "6node":
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "env_6node.yaml")
    env = AttackChainEnv(config_path=config_path)
    print(f"[环境] state_dim={env.state_dim} (模型期望 {state_dim}), "
          f"n_actions={env.n_actions} (模型期望 {n_actions})")
    assert env.state_dim == state_dim and env.n_actions == n_actions, \
        "模型维度与环境不匹配! 检查 --model 与配置是否对应。"

    demonstrate(env, q_net)

