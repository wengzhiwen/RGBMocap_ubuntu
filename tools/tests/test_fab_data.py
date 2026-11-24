"""FAB numeric record and SMPL-X parameter checks: CPU only, synthetic data only.

No FAB install, no Torch, no weights, no project data. Run with:
    python3 -m unittest discover -s tools/tests -v
"""
import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from mocap import fab_data  # noqa: E402

FRAMES = 7

# A plain Y-up, T-pose-like 22-joint skeleton standing near the floor.
NEUTRAL = np.array([
    (0., .95, 0.), (-.06, .93, 0.), (.06, .93, 0.), (0., 1.05, 0.), (-.06, .55, 0.), (.06, .55, 0.),
    (0., 1.2, 0.), (-.06, .05, .01), (.06, .05, .01), (0., 1.35, 0.), (-.06, .02, .13), (.06, .02, .13),
    (0., 1.45, 0.), (-.03, 1.42, 0.), (.03, 1.42, 0.), (0., 1.6, 0.), (-.18, 1.42, 0.), (-.45, 1.42, 0.),
    (-.7, 1.42, 0.), (.18, 1.42, 0.), (.45, 1.42, 0.), (.7, 1.42, 0.)], float)


def anatomy(active=True):
    """Synthetic target body arrays: each common joint is one template vertex,
    and every shape direction moves it linearly, so the shape fit is invertible."""
    template = np.zeros((fab_data.SMPLX_VERTEX_COUNT, 3))
    directions = np.zeros((fab_data.SMPLX_VERTEX_COUNT, 3, fab_data.SMPL_BETAS))
    regressor = np.zeros((55, fab_data.SMPLX_VERTEX_COUNT))
    rng = np.random.default_rng(7)
    for joint in range(fab_data.JOINT_COUNT):
        vertex = 4000 + joint
        template[vertex] = NEUTRAL[joint]
        regressor[joint, vertex] = 1.
        if active:
            directions[vertex] = rng.normal(scale=.02, size=(3, fab_data.SMPL_BETAS))
    return dict(v_template=template, J_regressor=regressor, shapedirs=directions)


def record(betas_true, native=None, frames=FRAMES, floor=-1.6, seed=3):
    """One valid numeric record built from the synthetic anatomy."""
    body = anatomy()
    shapes = np.einsum('jv,vcd->jcd', body['J_regressor'][:fab_data.JOINT_COUNT],
                       body['shapedirs'][..., :fab_data.SMPL_BETAS])
    rest = body['v_template'][4000:4000 + fab_data.JOINT_COUNT] + np.einsum('jcd,d->jc', shapes, betas_true)
    rng = np.random.default_rng(seed)
    pose = rng.uniform(-1., 1., (frames, fab_data.SMPL_POSE_WIDTH)).astype(np.float32)
    pelvis = np.stack([(-.2 + .04 * t, 1.62, 3.4 + .05 * t) for t in range(frames)]).astype(np.float32)
    native = np.tile(np.asarray(native if native is not None else [.5, -.4, .3, .2, -.1, .1, -.2, .3, -.3, .1],
                                np.float32), (frames, 1))
    intrinsic = np.array([[1400., 0., 960.], [0., 1400., 540.], [0., 0., 1.]], np.float64)
    return dict(smpl_pose=pose, smpl_betas=native,
                smpl_transl=(pelvis - rest[0]).astype(np.float32), camera_pelvis=pelvis,
                contact=rng.uniform(0., 1., (frames, fab_data.CONTACT_CHANNELS)).astype(np.float32),
                boxes=np.tile(np.array([300., 60., 700., 980.], np.float32), (frames, 1)),
                K=intrinsic, world_from_camera=fab_data.world_transform(floor),
                floor_estimate=np.array(floor, np.float64), frames=np.array(frames, np.int32),
                fps=np.array(30., np.float64), median_betas=np.median(native, axis=0).astype(np.float32),
                rest_joints=rest.astype(np.float32))


class NumericRecord(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def save(self, values, name='fab-numeric.npz'):
        path = self.root / name
        np.savez(path, **values)
        return path

    def test_round_trip_keeps_frame_order_and_dimensions(self):
        values = record(np.zeros(fab_data.SMPL_BETAS))
        loaded = fab_data.load(self.save(values), FRAMES)
        self.assertEqual(loaded['frames'], FRAMES)
        self.assertEqual(loaded['fps'], 30.)
        for name, count in (('smpl_pose', fab_data.SMPL_POSE_WIDTH), ('smpl_betas', fab_data.SMPL_BETAS),
                            ('smpl_transl', 3), ('camera_pelvis', 3), ('contact', fab_data.CONTACT_CHANNELS),
                            ('boxes', 4)):
            self.assertEqual(loaded[name].shape, (FRAMES, count))
        self.assertEqual(loaded['rest_joints'].shape, (fab_data.JOINT_COUNT, 3))
        self.assertEqual(loaded['world_from_camera'].shape, (4, 4))
        self.assertTrue(np.array_equal(loaded['smpl_pose'], values['smpl_pose']))
        self.assertEqual(fab_data.load(self.save(values))['frames'], FRAMES)

    def test_invalid_numeric_input_is_rejected(self):
        good = record(np.zeros(fab_data.SMPL_BETAS))
        for name, values in (
                ('smpl_pose', np.full_like(good['smpl_pose'], np.nan)),
                ('smpl_pose', np.full_like(good['smpl_pose'], 40.)),
                ('smpl_betas', np.full_like(good['smpl_betas'], 40.)),
                ('camera_pelvis', np.where(np.arange(FRAMES)[:, None] == 2, -1., good['camera_pelvis'])),
                ('contact', np.full_like(good['contact'], 1.4)),
                ('boxes', np.zeros_like(good['boxes'])),
                ('K', np.zeros_like(good['K'])),
                ('rest_joints', np.zeros_like(good['rest_joints'])),
                ('floor_estimate', np.array(np.nan)),
                ('fps', np.array(25.)),
                ('frames', np.array(FRAMES + 3))):
            broken = dict(good)
            broken[name] = values
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    fab_data.validate(broken)
        for name in ('smpl_pose', 'smpl_betas', 'K', 'contact', 'frames', 'fps', 'rest_joints'):
            broken = {key: value for key, value in good.items() if key != name}
            with self.subTest(missing=name), self.assertRaises(ValueError):
                fab_data.validate(broken)
        with self.assertRaises(ValueError):
            fab_data.validate(dict(good, unexpected=np.zeros(1)))
        with self.assertRaises(ValueError):
            fab_data.validate(dict(good, K=good['K'].astype(np.int64)))
        with self.assertRaises(ValueError):
            fab_data.validate(good, frames=FRAMES + 1)

    def test_transform_must_be_the_recorded_fixed_basis(self):
        good = record(np.zeros(fab_data.SMPL_BETAS))
        for transform in (np.diag([1., 1., -1., 1.]),                    # improper / wrong basis
                          np.diag([1., 1., 1., 1.]),                    # reflection
                          np.diag([1., -1., -1., 1.])):                 # proper but not the fixed basis
            with self.subTest(transform=transform[1, 1]), self.assertRaises(ValueError):
                fab_data.validate(dict(good, world_from_camera=transform))
        rotated = fab_data.world_transform(-1.6)
        rotated[:3, 3] = (.2, 1.6, 0.)
        with self.assertRaises(ValueError):
            fab_data.validate(dict(good, world_from_camera=rotated))

    def test_pickle_and_oversized_records_are_refused(self):
        path = self.root / 'pickle.npz'
        np.savez(path, payload=np.array([object()], dtype=object))
        with self.assertRaises(ValueError):
            np.load(path, allow_pickle=False)['payload']
        large = self.root / 'large.npz'
        large.write_bytes(bytes(fab_data.RECORD_LIMIT_BYTES + 1))
        with self.assertRaises(ValueError):
            fab_data.load(large)


class ResultParameters(unittest.TestCase):
    def setUp(self):
        self.betas_true = np.array([1.2, -.9, .4, .1, -.6, .3, -.2, .5, -.7, .2], np.float32)
        self.values = record(self.betas_true)
        self.body = anatomy()
        self.result = fab_data.result_parameters(self.values, self.body)

    def test_dimensions_and_constant_shape(self):
        for space in ('smpl_params_incam', 'smpl_params_global'):
            params = self.result[space]
            self.assertEqual(set(params), {'global_orient', 'body_pose', 'betas', 'transl',
                                           'left_hand_pose', 'right_hand_pose'})
            self.assertEqual(params['global_orient'].shape, (FRAMES, 3))
            self.assertEqual(params['body_pose'].shape, (FRAMES, fab_data.SMPL_BODY_POSE_WIDTH))
            self.assertEqual(params['betas'].shape, (FRAMES, fab_data.SMPL_BETAS))
            self.assertEqual(params['transl'].shape, (FRAMES, 3))
            for side in ('left', 'right'):
                hand = params[side + '_hand_pose']
                self.assertEqual(hand.shape, (FRAMES, fab_data.HAND_POSE_WIDTH))
                self.assertTrue(np.count_nonzero(hand) == 0)
            self.assertTrue(all(np.isfinite(value).all() for value in params.values()))
        betas = self.result['smpl_params_incam']['betas']
        self.assertTrue(np.array_equal(betas, np.broadcast_to(betas[:1], betas.shape)))
        self.assertEqual(self.result['contact_logits'].shape, (1, FRAMES, fab_data.CORRECTION_CHANNELS))
        self.assertEqual(self.result['net_outputs']['static_conf_logits'].shape, (1, FRAMES, 6))
        self.assertEqual(self.result['K'].shape, (3, 3))
        self.assertEqual(self.result['rest_joints'].shape, (fab_data.JOINT_COUNT, 3))

    def test_shape_is_fitted_not_copied(self):
        fitted = self.result['smpl_params_incam']['betas'][0]
        self.assertTrue(np.abs(fitted - self.betas_true).max() < 1e-3)
        self.assertLess(self.result['shape']['residual_m'], 1e-4)
        # FAB's native SMPL coefficients are a different model's basis.
        self.assertGreater(np.abs(fitted - self.values['smpl_betas'][0]).max(), .1)
        self.assertFalse(np.array_equal(fitted, self.values['median_betas']))
        self.assertTrue(np.array_equal(self.result['native']['smpl_betas'], self.values['smpl_betas']))

    def test_target_pivot_and_camera_translation(self):
        anatomy_values = fab_data.target_anatomy(self.body)
        pivot = (anatomy_values['rest_template']
                 + np.einsum('jcd,d->jc', anatomy_values['rest_shapes'],
                             self.result['smpl_params_incam']['betas'][0]))[0]
        self.assertTrue(np.abs(pivot - self.result['pelvis_pivot']).max() < 1e-6)
        for space in ('smpl_params_incam', 'smpl_params_global'):
            place = self.result[space]['transl'] + pivot
            self.assertTrue(np.isfinite(place).all())
        camera = self.result['smpl_params_incam']['transl'] + pivot
        self.assertTrue(np.abs(camera - self.values['camera_pelvis']).max() < 1e-5)

    def test_world_uses_a_proper_basis_and_one_clip_shift(self):
        world = self.result['world_from_camera']
        inverse = self.result['camera_from_world']
        rotation = world[:3, :3]
        self.assertTrue(np.abs(rotation.T @ rotation - np.eye(3)).max() < 1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1., places=12)
        self.assertTrue(np.abs(rotation - np.diag([1., -1., -1.])).max() == 0.)
        self.assertTrue(np.abs(world[:3, 3] - np.array([0., -self.values['floor_estimate'], 0.])).max() < 1e-12)
        self.assertTrue(np.abs(world @ inverse - np.eye(4)).max() < 1e-12)
        camera_space = self.result['smpl_params_incam']
        world_space = self.result['smpl_params_global']
        # Only the root and the pelvis move between spaces; local rotations stay native.
        self.assertTrue(np.array_equal(camera_space['body_pose'], world_space['body_pose']))
        self.assertTrue(np.abs(world_space['betas'] - camera_space['betas']).max() == 0.)
        from scipy.spatial.transform import Rotation
        for frame in range(FRAMES):
            from_world = Rotation.from_matrix(rotation) * Rotation.from_rotvec(camera_space['global_orient'][frame])
            to_world = Rotation.from_rotvec(world_space['global_orient'][frame])
            self.assertLess(float((from_world.inv() * to_world).magnitude()), 1e-6)
            world_place = world[:3, :3] @ (camera_space['transl'][frame] + self.result['pelvis_pivot']) \
                + world[:3, 3]
            self.assertTrue(np.abs(world_place - (world_space['transl'][frame] + self.result['pelvis_pivot'])).max() < 1e-5)

    def test_contact_channels_are_clipped_logits_in_the_correction_order(self):
        logits = self.result['contact_logits']
        self.assertTrue(np.isfinite(logits).all())
        self.assertTrue(np.abs(logits[0, :, 4:]).max() == 0.)
        probability = 1. / (1. + np.exp(-logits[0, :, :4]))
        expected = np.clip(self.values['contact'][:, :4], fab_data.CONTACT_EPS, 1 - fab_data.CONTACT_EPS)
        self.assertLess(np.abs(probability - expected).max(), 1e-5)

    def test_unmappable_shape_is_refused(self):
        flat = anatomy(active=False)
        with self.assertRaises(ValueError) as caught:
            fab_data.result_parameters(self.values, flat)
        self.assertIn('target body', str(caught.exception))

    def test_rest_joints_and_bones_are_validated(self):
        self.assertEqual(len(fab_data.PARENTS), fab_data.JOINT_COUNT)
        parents = np.asarray(fab_data.PARENTS)
        self.assertEqual(parents[0], -1)
        self.assertTrue((parents[1:] < np.arange(1, fab_data.JOINT_COUNT)).all())
        with self.assertRaises(ValueError):
            fab_data.neutral_shape(np.zeros((fab_data.JOINT_COUNT, 3)), fab_data.target_anatomy(anatomy()))
        with self.assertRaises(ValueError):
            fab_data.target_anatomy(dict(v_template=np.zeros((10, 3)), J_regressor=np.zeros((22, 10)),
                                         shapedirs=np.zeros((10, 3, 10))))


if __name__ == '__main__':
    unittest.main()
