"""
加载已保存的 PPO 模型, 直接演示攻击链 (无需重训)。

用法:
    python src/demo_ppo.py                 # 4 节点
    python src/demo_ppo.py --model 6node   # 6 节点

依赖文件: models/best_ppo.pt 或 models/best_6node.pt
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_v2 import AttackChainEnv
from agent_ppo import ActorCritic, demonstrate


def load_model(model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dim = ckpt["state_dim"]
    n_actions = ckpt["n_actions"]
    ac = ActorCritic(state_dim, n_actions).to(device)
    ac.load_state_dict(ckpt["state_dict"])
    ac.eval()
    return ac, state_dim, n_actions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo",
                    help="模型标签: ppo (4节点) 或 6node")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", f"best_{args.model}.pt")

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {os.path.abspath(model_path)}")
        sys.exit(1)

    print(f"[加载] {os.path.abspath(model_path)}  (device={device})")
    ac, state_dim, n_actions = load_model(model_path, device)

    config_path = None
    if args.model == "6node":
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "env_6node.yaml")
    env = AttackChainEnv(config_path=config_path)
    print(f"[环境] state_dim={env.state_dim} (模型期望 {state_dim}), "
          f"n_actions={env.n_actions} (模型期望 {n_actions})")
    assert env.state_dim == state_dim and env.n_actions == n_actions, "维度不匹配"

    demonstrate(env, ac)
