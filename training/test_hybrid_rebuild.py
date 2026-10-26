"""Focused checks for the new reconstruction (not historical baselines)."""
import importlib.util
import unittest


class ReconstructionContract(unittest.TestCase):
    def test_schema24_trainable_actor_is_available(self):
        self.assertIsNotNone(importlib.util.find_spec('hybrid_rebuild'),
                             'schema24 gated actor reconstruction is missing')

    def test_gates_and_critic_receive_finite_gradients(self):
        import torch
        from hybrid_rebuild import HybridActorCritic, install
        install()
        torch.manual_seed(7)
        model = HybridActorCritic()
        obs = torch.rand(8, 1028) * 2 - 1
        mask = torch.zeros(8, 10, dtype=torch.bool)
        logits, value = model(obs, mask)
        self.assertEqual(tuple(logits.shape), (8, 18))
        self.assertEqual(tuple(value.shape), (8,))
        loss = -logits.log_softmax(-1)[:, 5].mean() + (value - 1).square().mean()
        loss.backward()
        for key in ('actor.weight', 'critic.weight', 'dodge_alpha_old',
                    'ammo_alpha_old', 'dodge_delta.2.weight', 'ammo_delta.2.weight'):
            grad = dict(model.named_parameters())[key].grad
            self.assertTrue(torch.isfinite(grad).all(), key)
            self.assertGreater(float(grad.abs().sum()), 0, key)

    def test_terminal_bootstrap_is_cut_and_rollout_tail_bootstraps(self):
        import numpy as np
        from duel_ppo import compute_advantages
        reward = np.array([[1., 0.]], dtype=np.float32)
        value = np.array([[5., 2.]], dtype=np.float32)
        done = np.array([[True, False]])
        advantage = compute_advantages(reward, value, done, np.array([999., 3.]), .9, .95)
        np.testing.assert_allclose(advantage + value, [[1., 2.7]], atol=1e-6)

    def test_original_js_actor_reference_on_identical_inputs(self):
        import json, os, subprocess, tempfile
        from pathlib import Path
        import numpy as np
        import torch
        from hybrid_rebuild import HybridActorCritic, install
        assets = os.environ.get('KF_MATURE_ASSETS')
        if not assets:
            self.skipTest('read-only mature reference assets not supplied')
        assets = Path(assets)
        install()
        model = HybridActorCritic().eval()
        manifest = json.loads((assets/'assets/hybrid.json').read_text())
        weights = np.fromfile(assets/'assets/hybrid.bin', dtype='<f4')
        actor = {key:torch.from_numpy(weights[spec['offset']:spec['offset']+spec['length']]
                                      .copy().reshape(spec['shape']))
                 for key,spec in manifest['tensors'].items()}
        result = model.load_state_dict(actor, strict=False)
        self.assertEqual(set(result.missing_keys), {'critic.weight','critic.bias'})
        self.assertEqual(result.unexpected_keys, [])
        rng = np.random.default_rng(240918)
        obs = rng.uniform(-1,1,(3,1028)).astype(np.float32)
        masks = np.array([[0]*10,[1,0,1,1,0,0,1,0,0,1],[1]*10], bool)
        dodge = rng.uniform(-1,1,(3,9)).astype(np.float32)
        with torch.inference_mode():
            expected = model(torch.from_numpy(obs), torch.from_numpy(masks),
                             torch.from_numpy(dodge))[0].numpy()
        fixture = {'obs':obs.tolist(),'mask':masks.astype(int).tolist(),'dodge':dodge.tolist()}
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as temp:
            inp = Path(temp)/'fixture.json'; inp.write_text(json.dumps(fixture))
            output = subprocess.check_output(['node',str(Path(__file__).with_name('check_hybrid_js.mjs')),
                                              str(assets),str(inp)],text=True)
        actual = np.array(json.loads(output))
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=1e-4)
        self.assertTrue(np.array_equal(actual.argmax(1), expected.argmax(1)))
        print('ACTOR_REFERENCE '+json.dumps({'cases':3,'max_abs_error':float(abs(actual-expected).max()),
                                           'actions':actual.argmax(1).tolist(),
                                           'not_bitwise_or_environment_parity':True}), flush=True)


if __name__ == '__main__':
    unittest.main()
