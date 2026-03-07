import unittest
from unittest.mock import patch, MagicMock, call

from utils.metrics import compute_fid


def _make_image_paths(n):
    """Return n MagicMock path objects that look like .jpg files."""
    paths = []
    for _ in range(n):
        m = MagicMock()
        m.suffix.lower.return_value = '.jpg'
        paths.append(m)
    return paths


class TestComputeFID(unittest.TestCase):

    @patch('utils.metrics.Path')
    def test_raises_below_2048(self, mock_path_cls):
        """compute_fid raises ValueError when fewer than 2048 real images are found."""
        real_mock = MagicMock()
        real_mock.glob.return_value = iter(_make_image_paths(100))
        gen_mock = MagicMock()
        gen_mock.glob.return_value = iter(_make_image_paths(100))
        mock_path_cls.side_effect = [real_mock, gen_mock]

        with self.assertRaises(ValueError) as ctx:
            compute_fid('/real/dir', '/gen/dir')

        self.assertIn('2048', str(ctx.exception))

    @patch('utils.metrics.calculate_fid_given_paths')
    @patch('utils.metrics.Path')
    def test_num_workers_zero(self, mock_path_cls, mock_fid_fn):
        """compute_fid calls calculate_fid_given_paths with num_workers=0."""
        real_mock = MagicMock()
        real_mock.glob.return_value = iter(_make_image_paths(3000))
        gen_mock = MagicMock()
        gen_mock.glob.return_value = iter(_make_image_paths(500))
        mock_path_cls.side_effect = [real_mock, gen_mock]

        mock_fid_fn.return_value = 10.0
        compute_fid('/real/dir', '/gen/dir')

        _, kwargs = mock_fid_fn.call_args
        self.assertEqual(kwargs.get('num_workers', None), 0,
                         f"Expected num_workers=0, got call_args={mock_fid_fn.call_args}")

    @patch('utils.metrics.calculate_fid_given_paths')
    @patch('utils.metrics.Path')
    def test_returns_float(self, mock_path_cls, mock_fid_fn):
        """compute_fid returns a float for valid dirs."""
        real_mock = MagicMock()
        real_mock.glob.return_value = iter(_make_image_paths(3000))
        gen_mock = MagicMock()
        gen_mock.glob.return_value = iter(_make_image_paths(500))
        mock_path_cls.side_effect = [real_mock, gen_mock]

        mock_fid_fn.return_value = 42.5
        result = compute_fid('/real/dir', '/gen/dir')

        self.assertIsInstance(result, float)

    @patch('utils.metrics.calculate_fid_given_paths')
    @patch('utils.metrics.Path')
    def test_raises_empty_gen_dir(self, mock_path_cls, mock_fid_fn):
        """compute_fid raises when gen_dir contains no images."""
        real_mock = MagicMock()
        real_mock.glob.return_value = iter(_make_image_paths(3000))
        gen_mock = MagicMock()
        gen_mock.glob.return_value = iter([])  # no generated images
        mock_path_cls.side_effect = [real_mock, gen_mock]

        with self.assertRaises((ValueError, FileNotFoundError)):
            compute_fid('/real/dir', '/gen/dir')


if __name__ == '__main__':
    unittest.main()
