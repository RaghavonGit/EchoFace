import io
import sys
import unittest
from unittest.mock import patch, MagicMock

from utils.dataset_loader import load_dataset


def _make_path_mock(exists=True, files=None):
    """Return a MagicMock that mimics a Path instance.

    files: list of MagicMock path objects returned by iterdir().
    If files is None, iterdir() returns no files.
    """
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = exists
    if files is None:
        files = []
    mock_path_instance.iterdir.return_value = iter(files)
    return mock_path_instance


def _make_file_mock(suffix='.jpg'):
    """Return a MagicMock file path with the given suffix."""
    m = MagicMock()
    m.suffix.lower.return_value = suffix
    return m


def _make_image_open_side_effect(fail_on_index=None):
    """Return a side_effect function for Image.open that fails on a specific call index."""
    call_count = [0]

    def side_effect(path):
        idx = call_count[0]
        call_count[0] += 1
        if fail_on_index is not None and idx == fail_on_index:
            raise Exception("Unreadable file")
        mock_img = MagicMock()
        mock_img.convert.return_value.copy.return_value = MagicMock()
        return mock_img

    return side_effect


class TestLoadDataset(unittest.TestCase):

    @patch('utils.dataset_loader.Path')
    def test_missing_dir_raises(self, mock_path_cls):
        """load_dataset raises FileNotFoundError when directory does not exist."""
        mock_path_cls.return_value = _make_path_mock(exists=False)
        with self.assertRaises(FileNotFoundError):
            load_dataset('/nonexistent/path')

    @patch('utils.dataset_loader.Image')
    @patch('utils.dataset_loader.Path')
    def test_empty_dir_raises(self, mock_path_cls, mock_image):
        """load_dataset raises FileNotFoundError when directory exists but has no supported images."""
        # Return files with unsupported extension only
        unsupported_file = _make_file_mock(suffix='.txt')
        mock_path_cls.return_value = _make_path_mock(exists=True, files=[unsupported_file])
        with self.assertRaises(FileNotFoundError):
            load_dataset('/empty/path')

    @patch('utils.dataset_loader.Image')
    @patch('utils.dataset_loader.Path')
    def test_skips_unreadable(self, mock_path_cls, mock_image):
        """load_dataset skips unreadable files and prints skip count."""
        file1 = _make_file_mock('.jpg')
        file2 = _make_file_mock('.jpg')
        file3 = _make_file_mock('.jpg')
        mock_path_cls.return_value = _make_path_mock(exists=True, files=[file1, file2, file3])

        # Image.open raises on the second file (index 1)
        mock_image.open.side_effect = _make_image_open_side_effect(fail_on_index=1)

        captured = io.StringIO()
        sys.stdout = captured
        try:
            result = load_dataset('/some/path')
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertEqual(len(result), 2)
        self.assertIn('1', output)  # skip count of 1 mentioned in output

    @patch('utils.dataset_loader.Image')
    @patch('utils.dataset_loader.Path')
    def test_max_images(self, mock_path_cls, mock_image):
        """load_dataset caps result to max_images when argument is provided."""
        files = [_make_file_mock('.jpg') for _ in range(5)]
        mock_path_cls.return_value = _make_path_mock(exists=True, files=files)

        mock_image.open.side_effect = _make_image_open_side_effect()

        result = load_dataset('/some/path', max_images=3)
        self.assertEqual(len(result), 3)

    @patch('utils.dataset_loader.Image')
    @patch('utils.dataset_loader.Path')
    def test_returns_pil_list(self, mock_path_cls, mock_image):
        """load_dataset returns a list of PIL Images for a valid populated directory."""
        files = [_make_file_mock('.png'), _make_file_mock('.jpg')]
        mock_path_cls.return_value = _make_path_mock(exists=True, files=files)

        mock_image.open.side_effect = _make_image_open_side_effect()

        result = load_dataset('/some/path')
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
