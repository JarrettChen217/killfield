"""RECONSTRUCTED schema24 training entry point; never exact v17b lineage.

Actor equations/shapes: author's pinned viewer/src/hybrid.js and hybrid.json.
Critic/init: declared legacy 256->1 critic, orthogonal fresh init, gates start zero.
PPO/Adam/schedule/reward/serialization: existing author duel_ppo/duel engine.
Opt-in only: install() is called explicitly; default schema20 entry stays intact.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import duel_ppo as original
from duel_env import DuelVec

LegacyActorCritic = getattr(original, 'LegacyActorCritic', original.ActorCritic)
ARCH = 'hybrid_schema24_reconstructed_v1'


class HybridActorCritic(LegacyActorCritic):
    def __init__(self):
        super().__init__()
        if self.scalars[0].in_features != 88:
            self.scalars = nn.Sequential(nn.Linear(88, 128), nn.Tanh())
            nn.init.orthogonal_(self.scalars[0].weight, gain=2**.5)
            nn.init.zeros_(self.scalars[0].bias)
        self.dodge_alpha_old = nn.Parameter(torch.zeros(()))
        self.ammo_alpha_old = nn.Parameter(torch.zeros(4))
        self.idle_logit_penalty = nn.Parameter(torch.zeros(()))
        self.dodge_delta = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.ammo_delta = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 4))
        for gate in (self.dodge_delta, self.ammo_delta):
            nn.init.orthogonal_(gate[0].weight, gain=2**.5)
            nn.init.zeros_(gate[0].bias)
            nn.init.zeros_(gate[2].weight)
            nn.init.zeros_(gate[2].bias)

    def forward(self, obs, mask, dodge=None):
        if obs.shape[-1] != 1028 or mask.shape[-1] != 10:
            raise ValueError('reconstructed actor requires schema24/1028+10mask')
        features = self.features(obs, mask)
        logits = self.actor(features)
        dodge = obs[:, 1018:1027] if dodge is None else dodge
        dodge_scale = self.dodge_alpha_old + self.dodge_delta(features).squeeze(-1)
        scales = self.ammo_alpha_old + self.ammo_delta(features)
        ammo, hit, suicide = obs[:, 863], obs[:, 890], obs[:, 891]
        hit_soon = hit * (1 - (obs[:, 893] * 3).clamp(0, 1))
        fire_bias = (scales[:, 0] * ammo + scales[:, 1] * hit_soon
                     - scales[:, 2] * (1-ammo).square() * (1-hit_soon)
                     - scales[:, 3] * suicide)
        logits = logits + dodge_scale[:, None] * dodge.repeat_interleave(2, dim=1)
        fire = torch.arange(18, device=obs.device) % 2
        idle = ((obs[:, 1027] * 25 - 8) / 17).clamp(0, 1)
        idle_actions = ((torch.arange(18, device=obs.device) == 8)
                        | (torch.arange(18, device=obs.device) == 9))
        logits = (logits + fire_bias[:, None] * fire
                  - idle[:, None] * self.idle_logit_penalty * idle_actions)
        return logits, self.critic(features).squeeze(-1)


class HybridDuelVec(DuelVec):
    obs_dim = 1028
    schema_version = 24


def install():
    original.LegacyActorCritic = LegacyActorCritic
    original.ActorCritic = HybridActorCritic
    original.DuelVec = HybridDuelVec
    original.OBS_DIM = 1028
    original.SCALAR_DIM = 88
    original.OBS_SCHEMA_VERSION = 24
    original.ARCH = ARCH


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


@torch.inference_mode()
def evaluate_panel(model, output, deadline, seeds=range(19000, 19100)):
    """Same declared Laika seeds, argmax; rows survive a partial panel."""
    path = Path(output)
    if path.exists():
        raise ValueError('panel already exists; preserve rather than rerun')
    rows = []
    device = next(model.parameters()).device
    model.eval()
    started = time.monotonic()
    with path.open('x') as stream:
        for seed in seeds:
            if time.time() >= deadline:
                break
            env = HybridDuelVec(1, seed, (1, 0, 0), 1)
            row = {'seed': seed, 'result': 'partial', 'frames': 0}
            try:
                for frame in range(env.episode_frames + env.grace_frames + 1):
                    if time.time() >= deadline:
                        break
                    obs = torch.as_tensor(env.obs.copy(), device=device)
                    mask = torch.as_tensor(env.masks.astype(bool), device=device)
                    logits, _ = model(obs, mask)
                    action = logits.argmax(-1).cpu().numpy()
                    env.step(action)
                    row['frames'] = frame + 1
                    if env.dones[0]:
                        row['result'] = {1:'win', 2:'loss', 3:'double', 4:'draw'}[int(env.outcomes[0])]
                        row['reward'] = float(env.rewards[0])
                        break
            finally:
                env.close()
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            if row['result'] == 'partial':
                break
    counts = {key:sum(r['result'] == key for r in rows)
              for key in ('win','loss','double','draw','partial')}
    result = {'status':'complete' if len(rows)==100 and not counts['partial'] else 'partial',
              'cases':len(rows), 'counts':counts, 'seconds':time.monotonic()-started,
              'argmax':True, 'opponent':'Laika', 'seeds':list(seeds),
              'cases_sha256':digest(path)}
    atomic_json(path.with_suffix('.summary.json'), result)
    return result


class ResearchHook:
    """Thin schema24 checkpoint/warmup wiring for the existing author loop.

    Only initial + latest full checkpoint, not a per-update archive/platform.
    The outside bounded process ledger owns cumulative wall accounting.
    """
    def __init__(self, context_path):
        self.context = json.loads(Path(context_path).read_text())
        self.output = Path(self.context['output'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.deadline = self.context['deadline_unix']

    def restore(self, payload):
        if payload is None:
            return 0, 0
        state = payload['research']
        if state['protocol_sha256'] != self.context['protocol_sha256']:
            raise ValueError('foreign reconstruction protocol')
        if not 0 <= state['warmup_completed'] <= self.context['warmup_updates']:
            raise ValueError('invalid warmup progress')
        torch.set_rng_state(state['torch_rng'].cpu())
        np.random.set_state(state['numpy_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        return state['update'], state['warmup_completed']

    def before_rollout(self, batch):
        if time.time() >= self.deadline - self.context.get('checkpoint_reserve_seconds', 10):
            raise SystemExit(75)
        if shutil.disk_usage(self.output).free < 10*1024**3:
            raise RuntimeError('disk reserve below10GiB')

    def publish(self, model, optimiser, config, model_steps, update, warmup_completed,
                save_live, save_resume, started, schedule_steps):
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise FloatingPointError('nonfinite reconstructed model')
        metadata = {'protocol_sha256':self.context['protocol_sha256'],
                    'update':update, 'warmup_completed':warmup_completed,
                    'torch_rng':torch.get_rng_state(), 'numpy_rng':np.random.get_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        save_live(self.output, model, config, model_steps, update, started, model_steps, schedule_steps)
        save_resume(self.output, model, optimiser, model_steps, schedule_steps, research=metadata)
        marker = {'complete':True, 'schema':24, 'arch':ARCH, 'update':update,
                  'model_steps':model_steps, 'warmup_completed':warmup_completed,
                  'files':{f:digest(self.output/f) for f in ('live.pt','live.json','resume.pt')}}
        atomic_json(self.output/'checkpoint.json', marker)
        if update == 0:
            initial = self.output/'initial'
            if initial.exists():
                raise ValueError('initial checkpoint already exists')
            initial.mkdir()
            for f in ('live.pt','live.json','resume.pt','checkpoint.json'):
                shutil.copy2(self.output/f, initial/f)
            if self.context.get('evaluate_initial'):
                result = evaluate_panel(model, self.output/'initial-cases.jsonl',
                                        min(self.deadline, time.time()+240))
                print('INITIAL_PANEL '+json.dumps(result), flush=True)
                if result['status'] != 'complete':
                    raise SystemExit(75)
        print('FULL_CHECKPOINT '+json.dumps({k:v for k,v in marker.items() if k!='files'}), flush=True)
        if self.context.get('pause_after') == update:
            raise SystemExit(75)


if __name__ == '__main__':
    install()
    original.main()
