"""Schema24-specific reference telemetry; original game/PPO are unchanged."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path


class EpisodeTelemetry:
    """Own public positions (float32 arena fractions) -> displacement in cells.

    Fire requests/edges and actual engine Fire(0) events are separate counters.
    Derived latch bookkeeping is diagnostic only, never an actor input.
    """
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.counts = dict.fromkeys(('frames','neutral_requests','fire_on_requests',
            'fire_off_requests','fire_press_edges','fire_release_edges','actual_shots',
            'actual_hits','stationary_frames','moving_request_stationary_frames',
            'idle_pressure_active_frames','alive_frames','alive_fire_on_requests'), 0)
        self.actions = [0] * 18
        self.distance = 0.
        self.streak = self.longest = 0
        self.previous_fire = False
        self.trigger_armed = True

    def record(self, action, before, after, shots, hits):
        if not 0 <= action < 18 or shots not in (0, 1) or hits not in (0, 1):
            raise ValueError('invalid action/native event telemetry')
        c = self.counts
        fire, neutral = action % 2 == 1, action in (8, 9)
        dx = (float(after[856])-float(before[856])) * self.width
        dy = (float(after[857])-float(before[857])) * self.height
        distance = math.hypot(dx, dy)
        if not math.isfinite(distance):
            raise FloatingPointError('nonfinite public movement')
        pressure = min(1., max(0., (float(before[1027])*25-8)/17))
        self.streak = self.streak + 1 if neutral else 0
        self.longest = max(self.longest, self.streak)
        press, release = fire and not self.previous_fire, not fire and self.previous_fire
        armed_before = self.trigger_armed
        if not fire: self.trigger_armed = True
        if shots: self.trigger_armed = False
        alive = float(before[865]) > .5
        for key, value in {'frames':1,'neutral_requests':neutral,'fire_on_requests':fire,
            'fire_off_requests':not fire,'fire_press_edges':press,'fire_release_edges':release,
            'actual_shots':shots,'actual_hits':hits,'stationary_frames':distance<=1e-5,
            'moving_request_stationary_frames':action//6!=1 and distance<=1e-5,
            'idle_pressure_active_frames':pressure>1e-6,'alive_frames':alive,
            'alive_fire_on_requests':alive and fire}.items():
            c[key] += int(value)
        self.actions[action] += 1; self.distance += distance; self.previous_fire = fire
        return {'frame':c['frames'],'action':action,'displacement_cells':distance,
                'x_cells':float(after[856])*self.width,'y_cells':float(after[857])*self.height,
                'idle_streak_before_clipped':float(before[1027])*25,
                'idle_pressure_before':pressure,'fire_on':fire,'press_edge':press,
                'release_edge':release,'ready_before':float(before[864])>.5,
                'derived_trigger_armed_before':armed_before,'actual_shots':shots,
                'actual_hits':hits,'alive_before':alive}

    def summary(self):
        return {**self.counts,'action_counts':self.actions.copy(),
                'displacement_cells':self.distance,'longest_neutral_streak':self.longest,
                'position_units':'cells derived from public float32 arena fractions',
                'stationary_threshold_cells':1e-5}


def arena_cells(obs):
    cells = [(x, y) for x in range(12) for y in range(10)
             if float(obs[(x*10+y)*7]) > .5]
    if not cells: raise ValueError('missing public validity grid')
    return max(x for x,y in cells)+1, max(y for x,y in cells)+1


def evaluate_telemetry(model, output, deadline, seeds=range(19000,19100), trace=False):
    """Fixed Laika cases; aggregate per episode, optionally ONE short trace.

    Read-only evaluation preserves tensor/NumPy RNG and the model training flag.
    No actor/critic/optimizer math or game-state implementation lives here.
    """
    import numpy as np
    import torch
    from hybrid_rebuild import HybridDuelVec, atomic_json, digest
    path = Path(output)
    if path.exists(): raise ValueError('existing cases are immutable; do not rerun')
    seeds = list(seeds); rows=[]; started=time.monotonic()
    device=next(model.parameters()).device
    cpu_rng=torch.get_rng_state(); np_rng=np.random.get_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    prior_training=model.training
    model.eval()
    trajectory = None
    if trace:
        if len(seeds)!=1: raise ValueError('only one short diagnostic trajectory allowed')
        trajectory=path.with_suffix('.trace.jsonl').open('x')
    try:
        with path.open('x') as stream, torch.inference_mode():
            for seed in seeds:
                if time.time()>=deadline: break
                env=HybridDuelVec(1,seed,(1,0,0),1)
                row={'seed':seed,'result':'partial','frames':0}
                tally=EpisodeTelemetry(*arena_cells(env.obs[0]))
                try:
                    frame_cap=750 if trace else env.episode_frames+env.grace_frames+1
                    for frame in range(frame_cap):
                        if time.time()>=deadline: break
                        before=env.obs[0].copy()
                        if not np.isfinite(before).all(): raise FloatingPointError('nonfinite observation')
                        obs=torch.as_tensor(env.obs.copy(),device=device)
                        mask=torch.as_tensor(env.masks.astype(bool),device=device)
                        logits,value=model(obs,mask)
                        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                            raise FloatingPointError('nonfinite policy/value')
                        action=int(logits.argmax(-1).item())
                        env.step([action]); row['frames']=frame+1
                        item=tally.record(action,before,env.obs[0],int(env.shots[0]),int(env.hits[0]))
                        if trajectory:
                            if frame==0:
                                trajectory.write(json.dumps({'columns':list(item)})+'\n')
                            trajectory.write(json.dumps(list(item.values()),separators=(',',':'))+'\n')
                        if env.dones[0]:
                            row['result']={1:'win',2:'loss',3:'double',4:'draw'}[int(env.outcomes[0])]
                            row['reward']=float(env.rewards[0]);break
                finally:env.close()
                row['telemetry']=tally.summary()
                rows.append(row)
                # Fixed numeric field order avoids repeated schema text in1000case rows.
                compact={k:v for k,v in row.items() if k!='telemetry'}
                order=[k for k in row['telemetry'] if k not in ('position_units','stationary_threshold_cells')]
                compact['telemetry_values']=[row['telemetry'][k] for k in order]
                stream.write(json.dumps(compact,separators=(',',':'))+'\n');stream.flush()
                if row['result']=='partial': break
    finally:
        if trajectory:trajectory.close()
        torch.set_rng_state(cpu_rng);np.random.set_state(np_rng)
        if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
        model.train(prior_training)
    counts={key:sum(r['result']==key for r in rows) for key in ('win','loss','double','draw','partial')}
    numeric=tuple(EpisodeTelemetry(1,1).counts)
    aggregate={key:sum(r['telemetry'][key] for r in rows) for key in numeric}
    aggregate['action_counts']=[sum(r['telemetry']['action_counts'][i] for r in rows) for i in range(18)]
    aggregate['displacement_cells']=sum(r['telemetry']['displacement_cells'] for r in rows)
    aggregate['longest_neutral_streak']=max((r['telemetry']['longest_neutral_streak'] for r in rows),default=0)
    result={'status':'complete' if len(rows)==len(seeds) and not counts['partial'] else 'partial',
        'classification':'diagnostic replay, not independent win-rate evidence' if trace else 'same development-panel longitudinal evaluation',
        'cases':len(rows),'counts':counts,'frames_mean':sum(r['frames'] for r in rows)/len(rows) if rows else None,
        'timeout750':sum(r['result']=='draw' and r['frames']==750 for r in rows),
        'telemetry':aggregate,'seconds':time.monotonic()-started,'seeds':seeds,'argmax':True,
        'opponent':'Laika','cases_sha256':digest(path),'rng_restored':True,
        'case_telemetry_order':order if rows else [],
        'position_units':'cells derived from public float32 arena fractions',
        'stationary_threshold_cells':1e-5}
    atomic_json(path.with_suffix('.summary.json'),result)
    return result
