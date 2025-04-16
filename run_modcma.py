import os
import argparse

# import iohgnbg
import ioh
import numpy as np
import modcma.c_maes as modcma
import gnbg
import iohgnbg

BASE = os.path.realpath(os.path.dirname(__file__))


def get_problem(fid, use_cpp_instances = False):
    if use_cpp_instances:
        gnbg.set_root(f"{BASE}/gnbg/data/")
    else:
        gnbg.set_root(f"{BASE}/src/iohgnbg/static/GECCO_2025/")
        
    obj = gnbg.GNBG(fid)
    return ioh.wrap_problem(lambda x: obj(x) - obj.OptimumValue, f"GNBG_{fid}",
       dimension=30, 
       lb= -100,
       ub=100,
       calculate_objective=lambda i, j: ioh.RealSolution(obj.OptimumPosition, 0.0),     
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=25)
    parser.add_argument("--use_cpp_instances", action="store_true")
    args = parser.parse_args()
    
    problem = get_problem(args.fid, args.use_cpp_instances)
    pyproblem = iohgnbg.get_problem(problem_index=args.fid)
    
    if not args.use_cpp_instances:
        assert np.all(problem.optimum.x == pyproblem.optimum.x)
        assert problem.optimum.y == pyproblem.optimum.y
        assert np.all(problem.bounds.lb == pyproblem.bounds.lb)
        assert np.all(problem.bounds.ub == pyproblem.bounds.ub)
    
    mods = modcma.parameters.Modules()

    total_time = 0
    total_sucs = 0
    total_dfs = 0
    
    for _ in range(args.n_runs):
        settings = modcma.Settings(
            dim=problem.meta_data.n_variables,
            modules=mods,
            target=problem.optimum.y + 1e-8,
            budget=problem.meta_data.n_variables * 10_000,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
            x0=np.random.uniform(problem.bounds.lb, problem.bounds.ub),
            verbose=False
        )
        # print(problem.optimum)
        cma = modcma.ModularCMAES(settings)
        # cma.p.repelling.coverage = 2
        while cma.step(problem):
            # dx = problem.state.current_best.x - problem.optimum.x
            # print(dx, problem.state.current_best.y)
            # if np.isclose(np.abs(dx).sum(), 0.0):
            #     breakpoint()
            pass
        print(
            problem.state.evaluations,
            problem.state.final_target_found,
            problem.state.current_best.y,
            problem.optimum.y,
        )
        total_time += problem.state.evaluations
        total_sucs += problem.state.final_target_found
        total_dfs += problem.state.current_best_internal.y
        problem.reset()
        
    total_time += settings.budget * ((args.n_runs) - total_sucs)
    print()
    print(
        "\t",
        problem,
        f"ert: {total_time / total_sucs if total_sucs > 0 else float('inf'):.2f}",
        f"average delta f: {total_dfs / (args.n_runs): .2e}",
        
    )
    