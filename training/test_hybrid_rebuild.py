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


if __name__ == '__main__':
    unittest.main()
