import numpy as np
import cvxpy as cp
import networkx as nx

from collections import defaultdict

from mage.utils.gnn_helpers import subgraph_connected_components, get_neighbors


def _find_motifs(indices, num_motifs, beta=1., omega=0.5, cnvt_constr={}, verbose=False):
    P = indices
    n = indices.shape[0]
    beta = beta * n if beta <= 1 else beta
    omega = np.clip(omega, 0.01, 0.99)
    P_diag = np.diag(P)

    # solver
    Zt = [cp.Variable((n, n), symmetric=True) for _ in range(num_motifs)]
    x = cp.Variable((num_motifs, n), boolean=True)
    # Linearizing absolute objective function by BigM trick
    u = cp.Variable(num_motifs)
    y = cp.Variable(num_motifs, boolean=True)

    constraints = [Z >= 0 for Z in Zt]
    constraints += [
        cp.diag(Zt[k]) == 0 for k in range(num_motifs)
    ]

    for k in range(num_motifs):
        for i in range(n):
            for j in range(i+1, n):
                constraints += [
                    Zt[k][i, j] <= x[k, i],
                    Zt[k][i, j] <= x[k, j],
                    x[k, i] + x[k, j] - 1 <= Zt[k][i, j],
                ]

    constraints += [
        cp.sum(x, axis=0) <= 1
    ]
    constraints += [
        cp.sum(x) <= beta
    ]
    ### connectivity constraints
    for C, v_lst in cnvt_constr.items():
        constraints += [
            cp.sum(x[k, list(C)]) >= x[k, v] for v in v_lst for k in range(num_motifs)
        ]

    big_M = 1e3 + 7
    for k in range(num_motifs):
        obj_k = cp.trace(P.T @ Zt[k]) + 2 * P_diag.T @ x[k]
        constraints += [
            obj_k + big_M * y[k] >= 2 * (1 - omega) * u[k],
            - obj_k + big_M * (1 - y[k]) >= 2 * omega * u[k],
            obj_k <= u[k],
            -obj_k <= u[k],
        ]
    
    prob = cp.Problem(cp.Maximize(cp.sum(u)), constraints)
    prob.solve(solver=cp.MOSEK,
               mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME':  30.0},
               verbose=verbose)
    # env = gurobipy.Env()
    # env.setParam('TimeLimit', 30)
    # prob.solve(solver=cp.GUROBI,
    #            env=env,
    #            verbose=verbose)
    # prob.solve(solver=cp.CPLEX,
    #            cplex_params={"timelimit": 30},
    #            verbose=verbose)

    x_opt = x.value
    motifs = []
    for k in range(num_motifs):
        motif = list(np.nonzero(x_opt[k])[0])
        if len(motif) > 0:
            motifs.append(motif)
    return motifs


def _find_motifs_viky(
        indices,
        adj,
        num_motifs,
        beta=1.,
        omega=0.5,
        verbose=False,
        cnvt_constr={}):
    """
        viky formulation requires more axillary variables (+n^2)
    """
    P = indices
    n = indices.shape[0]
    beta = beta * n if beta <= 1 else beta
    omega = np.clip(omega, 0.01, 0.99)
    P_diag = np.diag(P)

    # solver
    Zt = [cp.Variable((n, n), symmetric=True) for _ in range(num_motifs)]
    x = cp.Variable((num_motifs, n), boolean=True)
    # Linearizing absolute objective function by BigM trick
    u = cp.Variable(num_motifs)
    y = cp.Variable(num_motifs, boolean=True)
    # No need to use integer variables here
    Gamma = [cp.Variable((n, n), symmetric=True) for _ in range(num_motifs)]
    # Gamma = [cp.Variable((n, n), boolean=True) for _ in range(num_motifs)]

    constraints = [Z >= 0 for Z in Zt]
    constraints += [
        cp.diag(Zt[k]) == 0 for k in range(num_motifs)
    ]

    for k in range(num_motifs):
        for i in range(n):
            for j in range(i+1, n):
                constraints += [
                    Zt[k][i, j] <= x[k, i],
                    Zt[k][i, j] <= x[k, j],
                    x[k, i] + x[k, j] - 1 <= Zt[k][i, j],
                ]

    # \sum x <= 1
    constraints += [
        cp.sum(x, axis=0) <= 1
    ]

    # cardinality function
    constraints += [
        cp.sum(x) <= beta
    ]

    # connectivity constraints
    # # gamma is symmetric
    # constraints += [
    #     gamma == gamma.T for gamma in Gamma
    # ]
    # constrained by the adjacency matrix
    constraints += [
        gamma >= 0 for gamma in Gamma
    ]
    # constrained by the adjacency matrix
    constraints += [
        gamma <= adj for gamma in Gamma
    ]
    constraints += [
        Gamma[k][i, :] <= x[k] for k in range(num_motifs) for i in range(n)
    ]
    # number of edges in each cluster should be greater than the number of nodes
    constraints += [
        2 * cp.sum(x[k]) - 2 <= cp.sum(Gamma[k]) for k in range(num_motifs)
    ]
    # # each vertex should connect to at least one vertex from the same cluster
    constraints += [
        x[k, i] <= cp.sum(Gamma[k][i, :]) for k in range(num_motifs) for i in range(n)
    ]

    ### for repeated connectivity constraints
    for C, v_lst in cnvt_constr.items():
        constraints += [
            cp.sum(x[k, list(C)]) >= x[k, v] for v in v_lst for k in range(num_motifs)
        ]

    # For objective function
    big_M = 1e3 + 7
    for k in range(num_motifs):
        obj_k = cp.trace(P.T @ Zt[k]) + 2 * P_diag.T @ x[k]
        constraints += [
            obj_k + big_M * y[k] >= 2 * (1 - omega) * u[k],
            - obj_k + big_M * (1 - y[k]) >= 2 * omega * u[k],
            obj_k <= u[k],
            -obj_k <= u[k],
        ]
    
    prob = cp.Problem(cp.Maximize(cp.sum(u)), constraints)
    prob.solve(solver=cp.MOSEK,
               mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME':  20.0},
               verbose=verbose)
    # env = gurobipy.Env()
    # env.setParam('TimeLimit', 30)
    # prob.solve(solver=cp.GUROBI,
    #            env=env,
    #            verbose=verbose)
    # prob.solve(solver=cp.CPLEX,
    #            cplex_params={"timelimit": 30},
    #            verbose=verbose)

    x_opt = x.value
    motifs = []
    for k in range(num_motifs):
        motif = list(np.nonzero(x_opt[k])[0])
        if len(motif) > 0:
            motifs.append(motif)
    return motifs


def create_tensor(n):
    ret = [[[cp.Variable() for i in range(n)] for j in range(n)] for k in range(n)]
    return ret


def _find_motifs_viky_third_order(
        indices,
        adj,
        num_motifs,
        beta=1.,
        omega=0.5,
        verbose=False,
        cnvt_constr={}):
    """
        viky formulation requires more axillary variables (+n^2)
    """
    P = indices
    n = indices.shape[0]
    beta = beta * n if beta <= 1 else beta
    omega = np.clip(omega, 0.01, 0.99)

    # solver
    Ut = [create_tensor(n) for _ in range(num_motifs)]
    x = cp.Variable((num_motifs, n), boolean=True)
    # Linearizing absolute objective function by BigM trick
    u = cp.Variable(num_motifs)
    y = cp.Variable(num_motifs, boolean=True)

    # Gamma for connectivity constraints. No need to use integer variables here
    Gamma = [cp.Variable((n, n), symmetric=True) for _ in range(num_motifs)]
    # Gamma = [cp.Variable((n, n), boolean=True) for _ in range(num_motifs)]

    constraints = []
    # constraints = [U >= 0 for U in Ut]
    for t in range(num_motifs):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    constraints += [Ut[t][i][j][k] >= 0]

    # constraints += [
    #     cp.diag(Ut[k]) == 0 for k in range(num_motifs)
    # ]

    for t in range(num_motifs):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    constraints += [
                        Ut[t][i][j][k] <= x[t, i],
                        Ut[t][i][j][k] <= x[t, j],
                        Ut[t][i][j][k] <= x[t, k],
                        x[t, i] + x[t, j] + x[t, k] - 2 <= Ut[t][i][j][k],
                    ]

    # \sum x <= 1
    constraints += [
        cp.sum(x, axis=0) <= 1
    ]

    # cardinality function
    constraints += [
        cp.sum(x) <= beta
    ]

    # connectivity constraints
    # # gamma is symmetric
    # constraints += [
    #     gamma == gamma.T for gamma in Gamma
    # ]
    # constrained by the adjacency matrix
    constraints += [
        gamma >= 0 for gamma in Gamma
    ]
    # constrained by the adjacency matrix
    constraints += [
        gamma <= adj for gamma in Gamma
    ]
    constraints += [
        Gamma[k][i, :] <= x[k] for k in range(num_motifs) for i in range(n)
    ]
    # number of edges in each cluster should be greater than the number of nodes
    constraints += [
        2 * cp.sum(x[k]) - 2 <= cp.sum(Gamma[k]) for k in range(num_motifs)
    ]
    # # each vertex should connect to at least one vertex from the same cluster
    constraints += [
        x[k, i] <= cp.sum(Gamma[k][i, :]) for k in range(num_motifs) for i in range(n)
    ]

    ### for repeated connectivity constraints
    for C, v_lst in cnvt_constr.items():
        constraints += [
            cp.sum(x[k, list(C)]) >= x[k, v] for v in v_lst for k in range(num_motifs)
        ]

    # For objective function
    big_M = 1e3 + 7
    for t in range(num_motifs):
        # obj_t = cp.trace(P.T @ Zt[k]) + 2 * P_diag.T @ x[k]
        obj_t = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    obj_t += P[i, j, k] * Ut[t][i][j][k]

        for i in range(n):
            for j in range(i+1, n):
                obj_t += P[i, j, i] * Ut[t][i][j][i]

        for i in range(n):
            obj_t += P[i, i, i] * Ut[t][i][i][i]

        constraints += [
            obj_t + big_M * y[t] >= 2 * (1 - omega) * u[t],
            - obj_t + big_M * (1 - y[t]) >= 2 * omega * u[t],
            obj_t <= u[t],
            -obj_t <= u[t],
        ]

    prob = cp.Problem(cp.Maximize(cp.sum(u)), constraints)
    prob.solve(solver=cp.MOSEK,
               mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME':  100.0},
               verbose=verbose)
    # env = gurobipy.Env()
    # env.setParam('TimeLimit', 100)
    # prob.solve(solver=cp.GUROBI,
    #            env=env,
    #            verbose=verbose)
    # prob.solve(solver=cp.CPLEX,
    #            cplex_params={"timelimit": 30},
    #            verbose=verbose)

    x_opt = x.value
    motifs = []
    for k in range(num_motifs):
        motif = list(np.nonzero(x_opt[k])[0])
        if len(motif) > 0:
            motifs.append(motif)
    # print(x.value)
    # print(num_motifs)
    # print(motifs)
    return motifs


def rearrange_motifs(motifs, num_motifs, G):
    new_motifs = []
    cpns = []
    for motif in motifs:
        cpns.extend(list(subgraph_connected_components(G.adj, motif)))

    cpns = sorted(cpns, key=lambda x: -len(x))

    for cpn in cpns:
        if len(new_motifs) < num_motifs:
            new_motifs.append(cpn)
        else:
            new_motifs[-1].extend(cpn)

    return new_motifs


def _find_motifs_with_connectivity(indices, G: nx.Graph, beta, num_motifs, omega=0.5,
                                   max_iterations=5, connectivity='viky', ord=2, verbose=False):
    cnvt_constr = defaultdict(set)
    n = G.number_of_nodes()
    adj = nx.adjacency_matrix(G).todense()
    beta = beta * n if beta <= 1 else beta

    motifs = []
    for it in range(int(max_iterations)):
        if connectivity == 'viky':
            if ord == 2:
                motifs = _find_motifs_viky(
                    indices=indices,
                    adj=adj,
                    beta=beta,
                    num_motifs=num_motifs,
                    omega=omega,
                    cnvt_constr=cnvt_constr,
                    verbose=verbose,
                )
            else:
                motifs = _find_motifs_viky_third_order(
                    indices=indices,
                    adj=adj,
                    beta=beta,
                    num_motifs=num_motifs,
                    omega=omega,
                    cnvt_constr=cnvt_constr,
                    verbose=verbose,
                )
        else:
            motifs = _find_motifs(
                indices=indices,
                beta=beta,
                omega=omega,
                num_motifs=num_motifs,
                cnvt_constr=cnvt_constr,
                verbose=verbose,
            )

        if len(motifs) == 0 or connectivity == 'none' or it == max_iterations - 1:
            return motifs

        if verbose:
            print("clusters are not connected, reoptimizing")
            print(motifs)

        motifs = rearrange_motifs(motifs, num_motifs, G)
        cpns = subgraph_connected_components(G.adj, motifs[-1])
        if len(cpns) == 1:
            return motifs
        else:
            for cpn in cpns:
                if len(cpn) < beta:
                    cpn_neighbors = get_neighbors(G, cpn)
                    cnvt_constr[tuple(cpn_neighbors)].update(cpn)

    return motifs
