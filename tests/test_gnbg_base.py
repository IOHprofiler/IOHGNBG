import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from scipy.io import savemat
from iohgnbg import get_problem, get_problems, create_problem, GNBG_COMPETITION_INSTANCES
import numpy as np

class TestGNBGBase(unittest.TestCase):
    
    def test_get_problem_valid(self):

        # Test get_problem with valid data
        problem_index = 1

        problem = get_problem(problem_index)

        self.assertIsNotNone(problem)
        self.assertEqual(problem.meta_data.name, f"GNBG_{GNBG_COMPETITION_INSTANCES}_f{problem_index}")
    
    def test_get_problem_invalid(self):

        # Test get_problem with valid data
        problem_index = 25

        with self.assertRaises(FileNotFoundError):
            problem = get_problem(problem_index)

        

    def test_get_problems_with_list(self):
        # Mock get_problem to return dummy problems
        
        problem_indices = [1, 2, 3]
        problems = get_problems(problem_indices)
        self.assertEqual(len(problems), 3)
        self.assertEqual(problems[0].meta_data.name, f"GNBG_{GNBG_COMPETITION_INSTANCES}_f{1}")
        self.assertEqual(problems[1].meta_data.name, f"GNBG_{GNBG_COMPETITION_INSTANCES}_f{2}")
        self.assertEqual(problems[2].meta_data.name, f"GNBG_{GNBG_COMPETITION_INSTANCES}_f{3}")

    def test_get_problems_with_int(self):
        # Mock get_problem to return dummy problems
        
        problem_indices = 24
        problems = get_problems(problem_indices)
        self.assertEqual(len(problems), 24)
        self.assertEqual(problems[0].meta_data.name, f"GNBG_{GNBG_COMPETITION_INSTANCES}_f{1}")
        self.assertEqual(problems[1].meta_data.name, f"GNBG_{GNBG_COMPETITION_INSTANCES}_f{2}")
        self.assertEqual(problems[22].meta_data.name, f"GNBG_{GNBG_COMPETITION_INSTANCES}_f{23}")
        self.assertEqual(problems[23].meta_data.name, f"GNBG_{GNBG_COMPETITION_INSTANCES}_f{24}")

    def test_create_problem_valid(self):
        # Test create_problem with valid parameters
        problem = create_problem(
            MaxEvals=10000,
            AcceptanceThreshold=1e-8,
            Dimension=2,
            CompNum=1,
            MinCoordinate=-1,
            MaxCoordinate=1,
            CompMinPos=np.array([[0.0, 0.0]]),
            CompSigma=np.array([[0.1]]),
            CompH=np.array([[0.0, 1.0]]),
            Mu=np.array([[0.0, 0.0]]),
            Omega=np.array([[1.0, 0.0, 0.0, 0.0]]),
            Lambda=np.array([[1.0]]),
            RotationMatrix=np.array([[[1.0], [0.0]], [[0.0], [1.0]]]),
            OptimumValue=0.1,
            OptimumPosition=np.array([[0.0, 0.0]])
        )

        self.assertIsNotNone(problem)
        self.assertEqual(problem.meta_data.name, "GNBG_Custom")
        self.assertEqual(problem.meta_data.n_variables, 2)

    def test_create_problem_invalid_shapes(self):
        # Test create_problem with invalid parameter shapes
        with self.assertRaises(AssertionError):
            create_problem(
                CompMinPos=np.array([[0.0]]),  # Invalid shape
                CompSigma=np.array([[0.1]]),
                CompH=np.array([[0.0, 1.0]]),
                Mu=np.array([[0.0, 0.0]]),
                Omega=np.array([[1.0, 0.0, 0.0, 0.0]]),
                Lambda=np.array([[1.0]]),
                RotationMatrix=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
                OptimumValue=0.1,
                OptimumPosition=np.array([[0.0, 0.0]])
            )


if __name__ == "__main__":
    unittest.main()
    