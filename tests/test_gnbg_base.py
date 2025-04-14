import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from scipy.io import savemat
from iohgnbg import get_problem, get_problems
import numpy as np




class TestGNBGBase(unittest.TestCase):

    @patch("iohgnbg.gnbg_base.loadmat")
    def test_get_problem_file_not_found(self, mock_loadmat):
        # Simulate file not found error
        mock_loadmat.side_effect = FileNotFoundError("File not found")

        # Call the function and assert exception
        instances_folder = "/path/to/instances"
        problem_index = 1
        with self.assertRaises(FileNotFoundError):
            get_problem(instances_folder, problem_index)

    @patch("iohgnbg.gnbg_base.loadmat")
    def test_get_problem_invalid_data(self, mock_loadmat):
        # Simulate invalid data in the .mat file
        mock_loadmat.return_value = {'GNBG': None}

        # Call the function and assert exception
        instances_folder = "/path/to/instances"
        problem_index = 1
        with self.assertRaises(TypeError):
            get_problem(instances_folder, problem_index)

    @patch("iohgnbg.gnbg_base.get_problem")
    def test_get_problems_with_range(self, mock_get_problem):
        # Mock get_problem to return a dummy problem
        mock_problem = MagicMock()
        mock_get_problem.return_value = mock_problem

        # Call the function
        instances_folder = "/path/to/instances"
        problem_indices = 3
        result = get_problems(instances_folder, problem_indices)

        # Assertions
        self.assertEqual(len(result), 3)
        mock_get_problem.assert_any_call(instances_folder, 1)
        mock_get_problem.assert_any_call(instances_folder, 2)
        mock_get_problem.assert_any_call(instances_folder, 3)

    @patch("iohgnbg.gnbg_base.get_problem")
    def test_get_problems_with_list(self, mock_get_problem):
        # Mock get_problem to return a dummy problem
        mock_problem = MagicMock()
        mock_get_problem.return_value = mock_problem

        # Call the function
        instances_folder = "/path/to/instances"
        problem_indices = [1, 3, 5]
        result = get_problems(instances_folder, problem_indices)

        # Assertions
        self.assertEqual(len(result), 3)
        mock_get_problem.assert_any_call(instances_folder, 1)
        mock_get_problem.assert_any_call(instances_folder, 3)
        mock_get_problem.assert_any_call(instances_folder, 5)


if __name__ == "__main__":
    unittest.main()