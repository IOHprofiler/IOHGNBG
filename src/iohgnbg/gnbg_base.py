import numpy as np
from .gnbg_problem import GNBG
import ioh
from scipy.io import loadmat
import os 

def get_problem(instances_folder : str, problem_index : int) -> ioh.ProblemClass.REAL:
    """
    Load a GNBG problem instance from a specified folder and wrap it as an IOH problem.
    Args:
        instances_folder (str): The path to the folder containing the GNBG problem instance files.
        problem_index (int): The index of the problem to load (used to determine the filename).
    Returns:
        ioh.ProblemClass.REAL: An IOH problem object representing the GNBG problem instance.
    Notes:
        - The function expects the problem instance files to be in MATLAB `.mat` format.
        - The filename is constructed as `f{problem_index}.mat`.
        - The GNBG problem instance is wrapped into an IOH problem with the objective function
            adjusted by subtracting the `OptimumValue`.
        - The problem's bounds, dimension, and other parameters are extracted from the `.mat` file.
    """
    filename = f'f{problem_index}.mat'
    GNBG_tmp = loadmat(os.path.join(instances_folder, filename))['GNBG']
    MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
    AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
    Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
    CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
    MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
    MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
    CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
    CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
    CompH = np.array(GNBG_tmp['Component_H'][0, 0])
    Mu = np.array(GNBG_tmp['Mu'][0, 0])
    Omega = np.array(GNBG_tmp['Omega'][0, 0])
    Lambda = np.array(GNBG_tmp['lambda'][0, 0])
    RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
    OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
    OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
    gnbg = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
    
    # def transform_objectives(y: float, instance_id:int) -> float:
    #     return y + OptimumValue
    f = ioh.wrap_problem(lambda x: gnbg.fitness(x) - OptimumValue, f"GNBG_{instances_folder}_f{problem_index}", ioh.ProblemClass.REAL, Dimension, 0, lb=MinCoordinate, ub=MaxCoordinate,
                 calculate_objective=lambda x,y : ioh.RealSolution(OptimumPosition[0], 0),  optimization_type=ioh.OptimizationType.MIN)#, transform_objectives=transform_objectives)
    f.set_id(problem_index)
    return f

def get_problems(instances_folder :str, problem_indices : int | list[int]) -> list[ioh.ProblemClass.REAL]:
    """
    Retrieves a list of problems based on the specified indices.

    Args:
        instances_folder (str): The folder path where problem instances are stored.
        problem_indices (int | list[int]): An integer specifying the range of problem indices 
            (0 to problem_indices - 1) or a list of specific problem indices to retrieve.

    Returns:
        list[ioh.ProblemClass.REAL]: A list of problem instances corresponding to the specified indices.
        """
    problems = []
    if(isinstance(problem_indices, int)):
        problem_indices = list(range(1,problem_indices+1))

    for problem_index in problem_indices:
        try:
            problems.append(get_problem(instances_folder, problem_index))
        except Exception as e:
            print(f"Error loading problem instance {problem_index}: {e}")
            continue
    return problems