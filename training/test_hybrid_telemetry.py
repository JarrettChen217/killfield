"""New telemetry checks only; no training or historical qualification reruns."""
import importlib.util
import unittest


class TelemetryContract(unittest.TestCase):
    def test_telemetry_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('hybrid_telemetry'))

    def test_fire_requests_releases_and_shots_stay_distinct(self):
        from hybrid_telemetry import EpisodeTelemetry
        obs = [0.] * 1028
        obs[864] = obs[865] = 1
        t = EpisodeTelemetry(4, 4)
        t.record(9, obs, obs, 1, 0)
        t.record(9, obs, obs, 0, 0)
        t.record(8, obs, obs, 0, 0)
        t.record(9, obs, obs, 1, 0)
        s = t.summary()
        self.assertEqual(s['fire_on_requests'], 3)
        self.assertEqual(s['fire_release_edges'], 1)
        self.assertEqual(s['actual_shots'], 2)
        self.assertEqual(s['longest_neutral_streak'], 4)
        self.assertEqual(s['stationary_frames'], 4)

    def test_movement_and_neutral_request_are_different(self):
        from hybrid_telemetry import EpisodeTelemetry
        before = [0.] * 1028
        after = before.copy(); after[856] = .25
        t = EpisodeTelemetry(4, 4)
        t.record(16, before, after, 0, 0)
        t.record(16, after, after, 0, 0)
        s = t.summary()
        self.assertAlmostEqual(s['displacement_cells'], 1.)
        self.assertEqual(s['neutral_requests'], 0)
        self.assertEqual(s['moving_request_stationary_frames'], 1)

    def test_absolute_optimizer_deadline_refuses_a_step(self):
        # Optional target-only check: no model/env/optimizer is constructed.
        import importlib.util, json, tempfile, time
        from pathlib import Path
        if importlib.util.find_spec('torch') is None:
            self.skipTest('uses existing target-only Torch dependency')
        from hybrid_rebuild import ResearchHook
        with tempfile.TemporaryDirectory() as temp:
            context=Path(temp)/'context.json'
            context.write_text(json.dumps({'output':temp,'deadline_unix':time.time()+30,
                                'optimization_deadline_unix':time.time()-1}))
            hook=ResearchHook(context)
            with self.assertRaises(SystemExit) as raised:
                hook.before_optimizer_step()
            self.assertEqual(raised.exception.code,75)


if __name__ == '__main__':
    unittest.main()
