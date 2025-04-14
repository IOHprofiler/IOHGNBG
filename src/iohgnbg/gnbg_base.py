import numpy as np
from .gnbg_problem import GNBG
import ioh
from scipy.io import loadmat
import os 
from .utils import get_static_package_path

GNBG_COMPETITION_INSTANCES = "GECCO_2025"


def get_problem(problem_index : int, instances_folder : str | None = None) -> ioh.ProblemClass.REAL:
    if(instances_folder is None):
        instances_folder = os.path.join(get_static_package_path(), GNBG_COMPETITION_INSTANCES)
        problem_name = f"GNBG_{GNBG_COMPETITION_INSTANCES}_f{problem_index}"
    else:
        problem_name =  f"GNBG_{instances_folder}_f{problem_index}"

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

    gnbg = GNBG(MaxEvals, 
                AcceptanceThreshold, 
                Dimension, 
                CompNum, 
                MinCoordinate, 
                MaxCoordinate, 
                CompMinPos, 
                CompSigma, 
                CompH, 
                Mu, 
                Omega, 
                Lambda, 
                RotationMatrix, 
                OptimumValue, 
                OptimumPosition
                )
    
    # def transform_objectives(y: float, instance_id:int) -> float:
    #     return y + OptimumValue
    f = ioh.wrap_problem(
        lambda x: gnbg.fitness(x) - OptimumValue, 
        name = problem_name, 
        problem_class=ioh.ProblemClass.REAL,
        optimization_type=ioh.OptimizationType.MIN,
        dimension = Dimension, 
        instance = 0,
        lb=MinCoordinate, 
        ub=MaxCoordinate,
        calculate_objective=lambda x,y : ioh.RealSolution(OptimumPosition[0], 0),  
        )
    f.set_id(problem_index)
    return f

def get_problems(problem_indices : int | list[int], instances_folder :str | None = None) -> list[ioh.ProblemClass.REAL]:
    problems = []
    if(isinstance(problem_indices, int)):
        problem_indices = list(range(1,problem_indices+1))

    for problem_index in problem_indices:
        try:
            problems.append(get_problem(problem_index, instances_folder))
        except Exception as e:
            print(f"Error loading problem instance {problem_index}: {e}")
            continue
    return problems

def create_problem(MaxEvals: np.int32 = 10000,
                    AcceptanceThreshold: np.float64 = 1e-8,
                    Dimension : np.int32 = 2,
                    CompNum: np.int32 = 1,
                    MinCoordinate: np.float64 = -1,
                    MaxCoordinate: np.float64 = 1,
                    CompMinPos: np.ndarray = np.array([[0.0, 0.0]]),
                    CompSigma: np.ndarray = np.array([[0.1]]),
                    CompH: np.ndarray = np.array([ [0.0, 1.0]]),
                    Mu: np.ndarray = np.array([[0.0, 0.0]]),
                    Omega: np.ndarray = np.array([[1.0, 0.0, 0.0, 0.0]]),
                    Lambda: np.ndarray = np.array([[1.0]]),
                    RotationMatrix: np.ndarray = np.array([[[1.0], [0.0]], [[0.0], [1.0]]]),
                    OptimumValue: np.float64 = 0.1,
                    OptimumPosition: np.ndarray = np.array([[0.0, 0.0]]),
                    ) -> ioh.ProblemClass.REAL:
    """
    Creates a problem instance for optimization based on the given parameters.
    Parameters:
        MaxEvals (np.int32): The maximum number of evaluations allowed for the problem.
        AcceptanceThreshold (np.float64): The threshold for accepting solutions.
        Dimension (np.int32): The dimensionality of the problem.
        CompNum (np.int32): The number of components in the problem.
        MinCoordinate (np.float64): The minimum coordinate value for the search space.
        MaxCoordinate (np.float64): The maximum coordinate value for the search space.
        CompMinPos (np.ndarray): The minimum positions for each component.
                                 Shape should be (CompNum, Dimension).
        CompSigma (np.ndarray): The standard deviations for each component.
                                Shape should be (CompNum, 1).
        CompH (np.ndarray): The heights of each component.
                            Shape should be (CompNum, Dimension).
        Mu (np.ndarray): The mean values for the components.
                         Shape should be (CompNum, 2).
        Omega (np.ndarray): The weights for the components.
                            Shape should be (CompNum, 4).
        Lambda (np.ndarray): The scaling factors for the components.
                             Shape should be (CompNum, 1).
        RotationMatrix (np.ndarray): The rotation matrix applied to the components.
                                     Shape should be (Dimension, Dimension, CompNum). 
        OptimumValue (np.float64): The optimal value of the objective function.
        OptimumPosition (np.ndarray): The position corresponding to the optimal value.
                                      Shape should be (1, Dimension).
    Returns:
        ioh.ProblemClass.REAL: An instance of a real-valued optimization problem.

    Raises:
        AssertionError: If any of the input arrays do not have the expected shape.
    """
    assert CompMinPos.shape == (CompNum, Dimension), f"CompMinPos must have shape ({CompNum}, {Dimension})"
    assert CompSigma.shape == (CompNum, 1), f"CompSigma must have shape ({CompNum}, 1)"
    assert CompH.shape == (CompNum, Dimension), f"CompH must have shape ({CompNum}, {Dimension})"
    assert Mu.shape == (CompNum, 2), f"Mu must have shape ({CompNum}, 2)"
    assert Omega.shape == (CompNum, 4), f"Omega must have shape ({CompNum}, 4)"
    assert Lambda.shape == (CompNum, 1), f"Lambda must have shape ({CompNum}, 1)"
    assert RotationMatrix.shape == (Dimension, Dimension, CompNum), f"RotationMatrix must have shape ({Dimension}, {Dimension}, {CompNum})"
    assert OptimumPosition.shape == (1, Dimension), f"OptimumPosition must have shape (1, {Dimension})"
    
    gnbg = GNBG(MaxEvals, 
                AcceptanceThreshold, 
                Dimension, 
                CompNum, 
                MinCoordinate, 
                MaxCoordinate, 
                CompMinPos, 
                CompSigma, 
                CompH, 
                Mu, 
                Omega, 
                Lambda, 
                RotationMatrix, 
                OptimumValue, 
                OptimumPosition
                )
    
    problem_name = "GNBG_Custom"
    f = ioh.wrap_problem(
        lambda x: gnbg.fitness(x) - OptimumValue, 
        name=problem_name, 
        problem_class=ioh.ProblemClass.REAL,
        optimization_type=ioh.OptimizationType.MIN,
        dimension=Dimension, 
        instance=0,
        lb=MinCoordinate, 
        ub=MaxCoordinate,
        calculate_objective=lambda x, y: ioh.RealSolution(OptimumPosition[0], 0),  
    )
    f.set_id(0)
    return f

