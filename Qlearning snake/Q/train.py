import os
import numpy as np
import torch
import torch.nn.functional as F
from env_snake import SnakeEnv
from model import DQN
from replay_buffer import ReplayBuffer
from utils import get_device, linear_epsilon

def train():
    device = get_device()
    print("Using device:", device)

    env = SnakeEnv(render=True, fps=10)  # 训练时也可以 render=False 更快
    policy = DQN().to(device)
    target = DQN().to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    buf = ReplayBuffer(100_000)

    gamma = 0.95
    batch_size = 64
    warmup = 2_000
    sync_every = 1_000
    total_steps = 80_000

    s = env.reset()
    ep_return = 0.0
    ep = 0

    os.makedirs("checkpoints", exist_ok=True)

    for t in range(1, total_steps + 1):
        eps = linear_epsilon(t, 1.0, 0.05, decay_steps=40_000)

        # epsilon-greedy
        if np.random.rand() < eps:
            a = np.random.randint(0, 3)
        else:
            with torch.no_grad():
                qs = policy(torch.tensor(s, device=device).unsqueeze(0))
                a = int(torch.argmax(qs, dim=1).item())

        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)

        ep_return += r
        env.render(last_reward=r, epsilon=eps)

        s = s2

        # learn
        if len(buf) >= max(warmup, batch_size):
            bs, ba, br, bs2, bdone = buf.sample(batch_size)

            bs = torch.tensor(bs, device=device)
            ba = torch.tensor(ba, device=device).unsqueeze(1)
            br = torch.tensor(br, device=device).unsqueeze(1)
            bs2 = torch.tensor(bs2, device=device)
            bdone = torch.tensor(bdone, device=device).unsqueeze(1)

            q = policy(bs).gather(1, ba)

            with torch.no_grad():
                q2 = target(bs2).max(dim=1, keepdim=True)[0]
                y = br + gamma * q2 * (1.0 - bdone)

            loss = F.mse_loss(q, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # sync target
        if t % sync_every == 0:
            target.load_state_dict(policy.state_dict())

        # episode end
        if done:
            ep += 1
            print(f"Episode {ep:4d} | score={info.score:2d} steps={info.steps:4d} ep_return={ep_return:.1f} t={t}")
            s = env.reset()
            ep_return = 0.0

        # save occasionally
        if t % 10_000 == 0:
            torch.save(policy.state_dict(), "checkpoints/snake_dqn.pt")
            print("Saved: checkpoints/snake_dqn.pt")

    torch.save(policy.state_dict(), "checkpoints/snake_dqn.pt")
    env.close()

if __name__ == "__main__":
    train()