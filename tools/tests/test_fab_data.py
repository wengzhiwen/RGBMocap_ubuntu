"""FAB numeric record and SMPL-X parameter checks: CPU only, synthetic data only.

Covers the optional FAH finger pass (``fab --fah``) as well: FAB's stored 2D
hand points, the hand boxes cut from them, and how the fitted finger rotations
are published.

No FAB install, no Torch, no weights, no project data. Run with:
    python3 -m unittest discover -s tools/tests -v
"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from mocap import fab_data  # noqa: E402

FRAMES = 7

# A plain Y-up, T-pose-like 22-joint skeleton standing near the floor, in the
# native SMPL / SMPL-X joint order: the left and right arm joints alternate.
NEUTRAL = np.array([
    (0., .95, 0.), (-.06, .93, 0.), (.06, .93, 0.), (0., 1.05, 0.), (-.06, .55, 0.), (.06, .55, 0.),
    (0., 1.2, 0.), (-.06, .05, .01), (.06, .05, .01), (0., 1.35, 0.), (-.06, .02, .13), (.06, .02, .13),
    (0., 1.45, 0.), (-.03, 1.42, 0.), (.03, 1.42, 0.), (0., 1.6, 0.), (-.18, 1.42, 0.), (.18, 1.42, 0.),
    (-.45, 1.42, 0.), (.45, 1.42, 0.), (-.7, 1.42, 0.), (.7, 1.42, 0.)], float)

# FAB's 2D hand points of one hand: 21 points whose bounding box is exactly the
# unit square, so a hand of ``extent`` pixels around ``corner`` is easy to place.
HAND_PATTERN = np.stack([np.linspace(0., 1., fab_data.HAND_KEYPOINTS),
                         (np.arange(fab_data.HAND_KEYPOINTS) % 2).astype(float)], -1)


def hand_points(extent, corner):
    return np.asarray(corner, float) + HAND_PATTERN * extent


def hand_array(hands):
    """(frames,2,21,2) points: one (2,21,2) pair, repeated over the frames, or one pair per frame."""
    hands = np.asarray(hands, np.float32)
    return np.tile(hands[None], (FRAMES, 1, 1, 1)) if hands.shape == (2, fab_data.HAND_KEYPOINTS, 2) else hands


def hand_body():
    """The synthetic target body with the two per-hand mean poses of a real SMPL-X body."""
    return SimpleNamespace(**anatomy(), **HAND_MEANS)


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


# Two different per-hand mean poses, as the registered SMPL-X body exposes them.
HAND_MEANS = {'left_hand_mean': np.linspace(-.6, .5, fab_data.HAND_JOINTS * 3).reshape(fab_data.HAND_JOINTS, 3),
              'right_hand_mean': np.linspace(.4, -.3, fab_data.HAND_JOINTS * 3).reshape(fab_data.HAND_JOINTS, 3)}
FITTED = np.zeros((FRAMES, 2), bool)
FITTED[::2, 0] = True            # left hand: frames 0, 2, 4, 6
FITTED[1::3, 1] = True           # right hand: frames 1, 4


def fingers(seed=11):
    """One FAH finger pass: local rotations of every fitted hand, identity elsewhere."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    rotations = np.tile(np.eye(3), (FRAMES, 2, fab_data.HAND_JOINTS, 1, 1))
    for side, name in enumerate(HAND_MEANS):
        rows = np.flatnonzero(FITTED[:, side])
        pose = HAND_MEANS[name] + rng.normal(0., .4, (len(rows), fab_data.HAND_JOINTS, 3))
        rotations[rows, side] = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(len(rows), fab_data.HAND_JOINTS, 3, 3)
    return dict(rotations=rotations, valid=FITTED.copy())


class NumericRecord(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.zeros = np.zeros((FRAMES, 2, fab_data.HAND_KEYPOINTS, 2), np.float32)

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


    def test_hand_points_round_trip_and_rejection(self):
        good = record(np.zeros(fab_data.SMPL_BETAS))
        # Records written before the finger pass have no hand points; only that pass reads them.
        self.assertIsNone(fab_data.validate(dict(good))['hand_keypoints'])
        points = np.stack([hand_points(40., (480., 380.)), hand_points(60., (870., 390.))], 0)
        points = hand_array(points)
        loaded = fab_data.load(self.save(dict(good, hand_keypoints=points)), FRAMES)
        self.assertEqual(loaded['hand_keypoints'].shape, (FRAMES, 2, fab_data.HAND_KEYPOINTS, 2))
        self.assertTrue(np.array_equal(loaded['hand_keypoints'], points))
        nan = np.where(np.arange(FRAMES)[:, None, None, None] == 3, np.nan, points.astype(np.float64))
        for name, broken in (('points of one hand', points[:, :1]), ('20 points', points[:, :, :20]),
                             ('other frame count', points[::2]), ('integer pixels', points.astype(np.int16)),
                             ('not finite', nan)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fab_data.validate(dict(good, hand_keypoints=broken))

    def test_hand_boxes_are_squares_around_fab_hand_points(self):
        good = record(np.zeros(fab_data.SMPL_BETAS))
        empty, usable = fab_data.hand_boxes(good, 1920, 1080)
        self.assertEqual(empty.shape, (FRAMES, 2, 4))
        self.assertEqual(empty.dtype, np.float32)
        self.assertFalse(usable.any())
        self.assertTrue(np.count_nonzero(empty) == 0)
        # slot 0 is the left hand, slot 1 the right one. The body box is 400x920, so a
        # hand box may be at most 0.8 x 920 = 736 pixels wide.
        #                           frame 0            frame 1             frame 2                 frame 3
        left = [hand_points(40., (480., 380.)), hand_points(10., (500., 400.)), hand_points(16., (500., 400.)),
                hand_points(100., (-135., 400.)),   # 7.5 % of its box inside the frame
                #              frame 4                        frame 5                                   frame 6
                hand_points(40., (480., 380.)), hand_points(360., (100., 100.)), np.zeros((fab_data.HAND_KEYPOINTS, 2))]
        # Frame 1: 10 pixels is too small to carry finger pixels, and 12 pixels is below the
        # same floor although its box would pass. Frame 2: 16 and 20 pixels get the 32-pixel floor.
        # Frame 5: 720 pixels is still within the body share, 800 is larger than it.
        right = [hand_points(60., (870., 390.)),     # 2 x 60 pixels
                 hand_points(12., (900., 420.)),     # below the floor although its box would pass
                 hand_points(20., (900., 420.)),     # the 32-pixel floor
                 hand_points(100., (1850., 400.)),   # clipped by the frame edge, 60 % visible
                 hand_points(40., (870., 390.)),
                 hand_points(400., (-40., 60.)),     # 800 pixels: larger than the body share
                 hand_points(60., (870., 390.))]
        points = hand_array(list(zip(left, right)))
        boxes, usable = fab_data.hand_boxes(dict(good, hand_keypoints=points), 1920, 1080)
        self.assertEqual(usable.dtype, bool)
        self.assertEqual(usable.tolist(), [[True, True], [False, False], [True, True], [False, True], [True, True],
                                           [True, False], [False, True]])
        self.assertTrue(np.count_nonzero(boxes[~usable]) == 0)
        for frame, side, extent, corner in ((0, 0, 40., (480., 380.)), (0, 1, 60., (870., 390.)),
                                            (2, 0, 16., (500., 400.)), (2, 1, 20., (900., 420.)),
                                            (3, 1, 100., (1850., 400.)), (4, 0, 40., (480., 380.)),
                                            (4, 1, 40., (870., 390.)), (5, 0, 360., (100., 100.)),
                                            (6, 1, 60., (870., 390.))):
            box = boxes[frame, side]
            length = box[2:] - box[:2]
            self.assertTrue(np.allclose(length, max(fab_data.HAND_BOX_SCALE * extent, fab_data.HAND_BOX_MIN_SIDE)),
                            msg=f'frame {frame} hand {side}: {box}')
            np.testing.assert_allclose((box[:2] + box[2:]) / 2, np.asarray(corner) + extent / 2, atol=1e-4)
        self.assertTrue(usable[3, 1])                   # the box keeps its full size at the frame edge
        np.testing.assert_allclose(boxes[3, 1], [1800., 350., 2000., 550.])
        with self.assertRaises(ValueError):
            fab_data.hand_boxes(dict(good, hand_keypoints=points[:, :, :20]), 1920, 1080)


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

    def test_parent_table_is_the_body_models_tree(self):
        from mocap.mixamo_rig import BODY_PARENTS
        self.assertEqual(len(fab_data.PARENTS), fab_data.JOINT_COUNT)
        parents = np.asarray(fab_data.PARENTS)
        self.assertEqual(parents[0], -1)
        self.assertTrue((parents[1:] < np.arange(1, fab_data.JOINT_COUNT)).all())
        # The same tree the export rig and both body models use, for the same joint order.
        self.assertEqual(tuple(fab_data.PARENTS), tuple(BODY_PARENTS))
        self.assertEqual(parents[13:22].tolist(), [9, 9, 12, 13, 14, 16, 17, 18, 19])
        # So every arm bone joins neighbouring joints of one arm, and no bone crosses the chest.
        for chain in (NEUTRAL[[13, 16, 18, 20]], NEUTRAL[[14, 17, 19, 21]]):     # collar, shoulder, elbow, wrist
            lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
            self.assertTrue((lengths > .1).all() and (lengths[1:] < .4).all(), msg=str(lengths))
        bones = NEUTRAL[1:] - NEUTRAL[parents[1:]]
        self.assertTrue((np.linalg.norm(bones[15:], axis=1) < .5).all())

    def test_rest_joints_and_bones_are_validated(self):
        body = fab_data.target_anatomy(anatomy())
        for name, rest in (('all zero', np.zeros((fab_data.JOINT_COUNT, 3))),
                           ('one joint missing', NEUTRAL[1:]),
                           ('not finite', np.where(np.arange(22)[:, None] == 5, np.nan, NEUTRAL)),
                           ('zero-length bone', NEUTRAL - np.where(np.arange(22)[:, None] == 18, NEUTRAL[18] - NEUTRAL[16], 0.))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fab_data.neutral_shape(rest, body)
        with self.assertRaises(ValueError):
            fab_data.target_anatomy(dict(v_template=np.zeros((10, 3)), J_regressor=np.zeros((22, 10)),
                                         shapedirs=np.zeros((10, 3, 10))))

    def test_fah_fingers_are_published_and_nothing_else_changes(self):
        body = hand_body()
        plain = fab_data.result_parameters(self.values, body)
        result = fab_data.result_parameters(self.values, body, fingers())
        self.assertEqual(plain['hand_pose_policy'], fab_data.HAND_POSE_POLICY)
        self.assertEqual(plain['fingers'], dict(source=None, frames=FRAMES))
        self.assertEqual(result['hand_pose_policy'], fab_data.FAH_HAND_POSE_POLICY)
        valid = fingers()['valid']
        fitted = dict(source='fah', frames=FRAMES, left=int(valid[:, 0].sum()), right=int(valid[:, 1].sum()))
        self.assertEqual(result['fingers'], fitted)
        self.assertTrue(all(type(value) is int for value in (fitted['frames'], fitted['left'], fitted['right'])))
        self.assertTrue(0 < fitted['left'] < FRAMES and 0 < fitted['right'] < FRAMES)
        self.assertEqual(json.loads(json.dumps(result['fingers'])), result['fingers'])   # plain ints for the receipt
        poses = fab_data.finger_poses(fingers(), body)
        self.assertTrue(np.count_nonzero(poses) > 0)
        for space in ('smpl_params_incam', 'smpl_params_global'):
            for side, hand in enumerate(('left_hand_pose', 'right_hand_pose')):
                # (frames,45): the shape the SMPL-X hand pose path reads slot by slot.
                self.assertEqual(result[space][hand].shape, (FRAMES, fab_data.HAND_POSE_WIDTH))
                self.assertTrue(np.array_equal(result[space][hand], poses[side]))
                self.assertTrue(np.array_equal(plain[space][hand], np.zeros_like(poses[side])))
            # The finger pass never touches the FAB body, shape, root or placement.
            for key in ('global_orient', 'body_pose', 'betas', 'transl'):
                self.assertTrue(np.array_equal(result[space][key], plain[space][key]), msg=key)
        for key in ('rest_joints', 'pelvis_pivot', 'contact_logits', 'world_from_camera', 'camera_from_world',
                    'K', 'boxes', 'floor_estimate_m', 'frames', 'fps'):
            self.assertTrue(np.array_equal(result[key], plain[key]), msg=key)
        for key in plain['shape']:
            self.assertTrue(np.array_equal(result['shape'][key], plain['shape'][key]), msg='shape.' + key)
        for key in plain['native']:
            self.assertTrue(np.array_equal(result['native'][key], plain['native'][key]), msg=key)

    def test_a_pass_that_fitted_no_hand_is_not_published_as_fah(self):
        body = hand_body()
        plain = fab_data.result_parameters(self.values, body)
        none = fingers()
        none['valid'][:] = False
        result = fab_data.result_parameters(self.values, body, none)
        self.assertEqual(result['hand_pose_policy'], fab_data.HAND_POSE_POLICY)
        self.assertEqual(result['fingers'], dict(source=None, frames=FRAMES, left=0, right=0))
        for space in ('smpl_params_incam', 'smpl_params_global'):
            for hand in ('left_hand_pose', 'right_hand_pose'):
                self.assertEqual(result[space][hand].shape, (FRAMES, fab_data.HAND_POSE_WIDTH))
                self.assertTrue(np.count_nonzero(result[space][hand]) == 0)
            for key in ('global_orient', 'body_pose', 'betas', 'transl'):
                self.assertTrue(np.array_equal(result[space][key], plain[space][key]), msg=key)
        # One fitted hand is a FAH result; the receipt counts the other hand as unfitted.
        one = fingers()
        one['valid'][:, 1] = False
        single = fab_data.result_parameters(self.values, body, one)
        self.assertEqual(single['hand_pose_policy'], fab_data.FAH_HAND_POSE_POLICY)
        self.assertEqual(single['fingers'], dict(source='fah', frames=FRAMES, left=int(one['valid'][:, 0].sum()), right=0))


class HandPoints(unittest.TestCase):
    """The FAB entry's reader for the predictor's own 2D hand points, the finger pass's only input."""

    def test_slots_follow_the_mhr_order_and_one_bad_point_stores_no_hand(self):
        from mocap import fab_infer_entry
        rng = np.random.default_rng(3)
        person = dict(keypoints2d=rng.uniform(0., 1000., (70, 2)).astype(np.float32))
        points = fab_infer_entry.hand_keypoints(person, np)
        self.assertEqual(points.shape, (2, fab_data.HAND_KEYPOINTS, 2))
        # MHR order: right hand first, then left; each block ends with its wrist.
        np.testing.assert_array_equal(points[0], person['keypoints2d'][42:63])
        np.testing.assert_array_equal(points[1], person['keypoints2d'][21:42])
        person['keypoints2d'][50, 1] = np.nan
        points = fab_infer_entry.hand_keypoints(person, np)
        self.assertTrue(np.count_nonzero(points[0]) == 0)              # the left hand is dropped as a whole
        np.testing.assert_array_equal(points[1], person['keypoints2d'][21:42])
        for broken in (None, np.zeros((0, 2)), np.zeros((62, 2)), np.zeros((70, 3))):
            with self.subTest(shape=getattr(broken, 'shape', None)):
                self.assertTrue(np.count_nonzero(fab_infer_entry.hand_keypoints(dict(keypoints2d=broken), np)) == 0)
                self.assertTrue(np.count_nonzero(fab_infer_entry.hand_keypoints(object(), np)) == 0)


class FingerPoses(unittest.TestCase):
    def test_fitted_slots_are_offsets_from_each_hand_mean(self):
        from scipy.spatial.transform import Rotation
        poses = fab_data.finger_poses(fingers(), hand_body())
        self.assertEqual(poses.shape, (2, FRAMES, fab_data.HAND_POSE_WIDTH))
        self.assertEqual(poses.dtype, np.float32)
        rotations = fingers()['rotations']
        for side, name in enumerate(HAND_MEANS):
            rows = np.flatnonzero(FITTED[:, side])
            # An SMPL-X hand pose is added to the hand's mean pose before it becomes a rotation.
            rebuilt = HAND_MEANS[name] + poses[side, rows].reshape(len(rows), fab_data.HAND_JOINTS, 3)
            back = Rotation.from_rotvec(rebuilt.reshape(-1, 3)).as_matrix().reshape(len(rows), fab_data.HAND_JOINTS, 3, 3)
            np.testing.assert_allclose(back, rotations[rows, side], atol=1e-5)
            # A hand FAH did not fit keeps the relaxed mean pose; nothing is carried over between frames.
            self.assertTrue(np.count_nonzero(np.delete(poses[side], rows, axis=0)) == 0)
        # Each hand is offset from its own mean pose.
        self.assertFalse(np.allclose(poses[0, 0], poses[1, 1]))

    def test_finger_rotations_are_checked(self):
        body = hand_body()
        scaled = fingers()
        scaled['rotations'][0, 0, 3] *= 1.2
        reflected = fingers()
        reflected['rotations'][1, 1, 7] = np.diag([1., -1., 1.])
        loose = fingers()
        loose['rotations'][2, 0, 9] = loose['rotations'][2, 0, 9] + .01       # no longer a rotation
        for name, broken in (('scaled', scaled), ('reflected', reflected), ('loose', loose)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fab_data.finger_poses(broken, body)
        good = fingers()
        for name, changed in (('valid as integers', dict(good, valid=good['valid'].astype(np.int64))),
                              ('frames differ', dict(good, rotations=good['rotations'][::2])),
                              ('other joint count', dict(good, rotations=good['rotations'][:, :, :14])),
                              ('not finite', dict(good, rotations=np.where(np.arange(FRAMES)[:, None, None, None, None] == 3,
                                                                             np.nan, good['rotations'])))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fab_data.finger_poses(changed, body)
        with self.assertRaisesRegex(ValueError, 'hand mean'):
            fab_data.finger_poses(good, anatomy())


class FahOption(unittest.TestCase):
    """What a project was saved with against what its committed FAB result actually used."""

    def setUp(self):
        from mocap import fab_pipeline
        self.pipeline = fab_pipeline
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.project = Path(folder.name) / 'fab'

    def test_saved_choice_must_be_on_or_off(self):
        fah_enabled = self.pipeline.fah_enabled
        self.assertIs(fah_enabled({}), False)
        self.assertIs(fah_enabled({'fab_options': None}), False)
        self.assertIs(fah_enabled({'fab_options': {}}), False)
        self.assertIs(fah_enabled({'fab_options': {'fah': True}}), True)
        self.assertIs(fah_enabled({'fab_options': {'fah': False}}), False)
        for value in (1, 0, '1', 'true', None, [True]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fah_enabled({'fab_options': {'fah': value}})

    def write(self, text):
        self.project.mkdir(exist_ok=True)
        (self.project / self.pipeline.RECEIPT).write_text(text, encoding='utf-8')

    def test_only_the_receipt_says_what_a_result_used(self):
        fah_fingers = self.pipeline.fah_fingers
        self.assertFalse(fah_fingers(self.project))                     # no result yet
        self.write(json.dumps(dict(hand_pose_policy=fab_data.FAH_HAND_POSE_POLICY,
                                   fingers=dict(source='fah', frames=3, left=3, right=2))))
        self.assertTrue(fah_fingers(self.project))
        # Saved with the choice on, but committed before the finger pass ran.
        self.write(json.dumps(dict(hand_pose_policy=fab_data.HAND_POSE_POLICY, fab_options={'fah': True})))
        self.assertFalse(fah_fingers(self.project))
        self.write(json.dumps(dict(fingers=dict(source='fah', frames=3, left=3, right=3))))
        self.assertFalse(fah_fingers(self.project))
        for text in ('{not json', '[]', '"fah-fingers-on-fab-body-v1"', ''):
            self.write(text)
            with self.subTest(text=text):
                self.assertFalse(fah_fingers(self.project))


if __name__ == '__main__':
    unittest.main()
