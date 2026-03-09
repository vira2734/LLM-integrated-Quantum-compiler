import argparse
import ast
import datetime
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
import time
from asyncio.subprocess import PIPE

import numpy as np
import qiskit
import qiskit.circuit
import qiskit.dagcircuit
import scipy.sparse.csgraph
from pysat.solvers import Solver

import architectures
from common import compose_swaps, extract2qubit, extract_qubits, extract_gates, get_multi_qubit_indices, get_chunk_ranges

try:
    import hardware_spec as hw_spec
except ImportError:
    hw_spec = None

# Controls whether debug is output (overwritten by Local if True)
DEBUG_GLOBAL = True


def _wbo_cmd(iterations, cnf_path):
    """Build Open-WBO command. On Linux, prepend stdbuf -o0 so stdout is unbuffered and the model (v line) is flushed when optimum is found, reducing risk of missing assignment on timeout."""
    base = ["lib/Open-WBO-Inc/open-wbo-inc_release", "-iterations=" + str(iterations), cnf_path]
    if sys.platform == "linux" and shutil.which("stdbuf"):
        return ["stdbuf", "-o0"] + base
    return base


## Topological layering ##

def getLayers(cnots):
    layers = [0]
    for i in range(len(cnots)):
        if inconsistent(cnots[:i], cnots[i]) and len(layers) == 1:
            layers.append(i)
        elif len(layers) >1 and  inconsistent(cnots[layers[-1]:i], cnots[i]):
            layers.append(i)
    return layers


def inconsistent(cnots, cnot):
    relevantQubits = [c for (c, _) in cnots] + [t for (_, t) in cnots]
    return (cnot[0] in relevantQubits or cnot[1] in relevantQubits)

def sortCnots(logNum, cnots):
    qc = qiskit.QuantumCircuit(logNum,0)
    for (c, t) in cnots:
        qc.cx(c,t)
    dag = qiskit.converters.circuit_to_dag(qc)
    sorted_cnots = []
    for layer in dag.layers():
       pairs = layer["partition"]
       sorted_cnots = sorted_cnots + list(map(lambda p: tuple(map(lambda q: q.index, p)), pairs))
    return sorted_cnots

## Constraint Generation ##


def generateAndWriteClauses(logNum, liveCnots, cnots, cm, swapNum, ffClauses, path, routing=True, weighted=False, boundedAbove=False, layering=False, calibrationData=None, spec=None, multi_qubit_gates=None):
    '''
        Writes the constraints corresponding to a particular MaxSat Instance to the given path as a wcnf file.
        When spec and multi_qubit_gates are provided, uses writeMultiQubitGateConstraint and match_counts
        for all clause writing; otherwise uses writeCnotConstraint (legacy 2-qubit only).
        Returns (solver, match_counts or None).
    '''
    physNum = len(cm)
    if multi_qubit_gates is not None and spec is not None:
        numCnots = len(multi_qubit_gates)
        layers = list(range(numCnots))
    else:
        numCnots = len(cnots)
        if layering:
            layers = getLayers(cnots)
        else:
            layers = list(range(len(cnots)))
    liveLog = range(logNum)
    numP = physNum * physNum * numCnots
    numR = numP
    numS = numCnots * physNum * physNum * swapNum
    top = numS + numR + 1
    s = Solver(name='cd')
    match_counts = None
    with open(path, "w") as f:
        f.write("p wcnf " + str(42) + " " + str(42) + " " + str(top) + "\n")
        if multi_qubit_gates is not None and spec is not None:
            match_counts = writeMultiQubitGateConstraint(multi_qubit_gates, spec, physNum, logNum, swapNum, top, f, satSolver=s)
        else:
            writeCnotConstraint(cnots, cm, physNum, logNum, swapNum, top, f, satSolver=s)
        writeFunConConstraint(numCnots, liveLog, physNum, logNum, swapNum, top, f, satSolver=s, match_counts=match_counts)
        writeInjectivityConstraint(numCnots, liveLog, physNum, logNum, swapNum, top, f, satSolver=s, match_counts=match_counts)
        if routing:
            writeSwapChoiceConstraint(swapNum, layers, cm, physNum, logNum, numCnots, top, f, satSolver=s, match_counts=match_counts)
            writeSwapEffectConstraint(swapNum, layers, liveLog, physNum, cm, logNum, numCnots, top, f, satSolver=s, match_counts=match_counts)
        elif weighted:
            writeDistanceConstraint(swapNum, physNum, logNum, numCnots, top, f, satSolver=s, match_counts=match_counts)
        elif boundedAbove:
            writeMaxDisplacedConstraint(5, physNum, logNum, swapNum, numCnots, top, f, satSolver=s, match_counts=match_counts)
        for clause in ffClauses:
            writeHardClause(f, top, clause, physNum, logNum, numCnots, swapNum, match_counts=match_counts)
        writeOptimizationConstraints(swapNum, physNum, numCnots, cm, logNum, routing, weighted, calibrationData, f, match_counts=match_counts)
    return (s, match_counts)
# Mapping Constraints #


# Every logical qubit is mapped to exactly one physical qubit
def writeFunConConstraint(numCnots, liveLog, physNum, logNum, swapNum, top, path, satSolver=None, match_counts=None):
    for k in range(numCnots):
        for j in liveLog:
            atLeastOneJ = []
            for i in range(physNum):
                atLeastOneJ.append((False,"x", i,j,k))
                for i2 in range(i):
                    clause=[(True, "x", i2, j, k), (True,"x", i,j,k)]
                    writeHardClause(path, top, clause, physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)
            writeHardClause(path, top, atLeastOneJ, physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)

# No two logical qubits are mapped to the same physical qubit
def writeInjectivityConstraint(numCnots, liveLog, physNum, logNum,  swapNum, top, path,  satSolver=None, match_counts=None):
    for i in range(physNum):
        for j in liveLog:
            for k in range(numCnots):
                for j2 in range(j):
                   writeHardClause(path, top, [(True, "x", i, j2, k), (True,"x",i,j,k)], physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)


# Control and target are mapped to adjacent physical qubits
def writeCnotConstraint(cnots, cm, physNum, logNum, swapNum, top, path, satSolver=None):
    numCnots = len(cnots)
    for k in range(len(cnots)):
        (c,t) = cnots[k]
        edgeUsed = []
        nonzeroIndices = np.argwhere(cm>0)
        for edge in nonzeroIndices:
            [u,v] = edge
            edgeUsed.append((False, "p", u, v, k))
            edgeUsed.append((False, "r",  u, v, k))
            clauses =  [[(False, "x", u, c, k), (True, "p", u, v, k)],
                        [(False, "x", v, t, k), (True, "p", u, v, k)],
                        [(False, "x", u, t, k), (True, "r", u, v, k)],
                        [(False, "x", v, c, k), (True, "r", u, v, k)]]
            for clause in clauses:
                writeHardClause(path, top, clause, physNum, logNum, numCnots, swapNum, satSolver=satSolver)
        writeHardClause(path, top, edgeUsed, physNum, logNum, numCnots, swapNum, satSolver=satSolver)


# Multi-qubit gate constraint: at least one allowed n-tuple (SubgraphMatches) per gate (Option A: match variables)
def writeMultiQubitGateConstraint(multi_qubit_gates, spec, physNum, logNum, swapNum, top, path, satSolver=None):
    """
    For each multi-qubit gate at step k, add: OR_j match(j,k) and for each j: match(j,k) -> AND_i map(q_i, p_ji, k).
    Uses hardware_spec.get_subgraph_matches(spec, arity, type). Raises if subgraph_matches missing for an arity.
    Returns match_counts (list of length numCnots) for use in flattenedIndex/unravel.
    """
    if hw_spec is None:
        raise ImportError("hardware_spec required for writeMultiQubitGateConstraint")
    numCnots = len(multi_qubit_gates)
    match_counts = []
    for k, gate in enumerate(multi_qubit_gates):
        arity = gate.get("arity", 2)
        type_name = gate.get("type")
        qubits = gate.get("qubits", ())
        if len(qubits) != arity:
            raise ValueError(f"Gate at k={k} has arity {arity} but {len(qubits)} qubits")
        matches = hw_spec.get_subgraph_matches(spec, arity, type_name)
        if not matches:
            raise ValueError(
                f"No subgraph_matches for arity {arity}" + (f" and type {type_name!r}" if type_name else "")
                + "; add subgraph_matches to hardware spec or check gate type."
            )
        match_counts.append(len(matches))
        # At least one match: OR_j match(j,k)
        at_least_one = [(False, "m", j, k) for j in range(len(matches))]
        writeHardClause(path, top, at_least_one, physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)
        # For each j: match(j,k) -> map(q_i, p_ji, k) for all i  =>  ¬match(j,k) ∨ map(q_i, p_ji, k)
        for j, tuple_p in enumerate(matches):
            for i, p in enumerate(tuple_p):
                q = qubits[i]
                clause = [(True, "m", j, k), (False, "x", p, q, k)]
                writeHardClause(path, top, clause, physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)
    return match_counts


# Routing Constraints #


def writeDistanceConstraint(swapNum, physNum, logNum, numCnots, top, path, satSolver=None, match_counts=None):
    for k in range(1, numCnots):
        for i in range(physNum):
            for i2 in range(physNum):
                if i2 != i:
                    for j in range(logNum):
                        writeHardClause(path, top, [(True, "x", i, j, k-1),(True, "x", i2, j, k), (False, "w", i, i2, k)], physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)

# Exactly one swap sequence is chosen
def writeSwapChoiceConstraint(swapNum, layers, cm, physNum, logNum, numCnots,top, path, satSolver=None, match_counts=None):
    allowedSwaps = np.append(np.argwhere(cm>0), [[0,0]], axis=0)
    for k in layers:
        for t in range(swapNum):
            atLeastOne = []
            for (u,v) in allowedSwaps:
                i = 0
                atLeastOne.append((False, "s", u, v, t, k))
                if i != 0:
                    writeHardClause(path, top, [(True, "s", u, v, t, k), (False,"b", i-1, t, k)], physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)
                    writeHardClause(path, top, [(True, "s", u, v, t, k), (True, "b", i, t, k)], physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)
                    writeHardClause(path, top, [(True,"b", i-1, t, k), (False, "b", i, t, k), (False, "s", u, v, t, k)], physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)
                writeHardClause(path, top, [(False, "b", i, t, k), (True,"b", i+1, t, k)], physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)
                i += 1
            writeHardClause(path, top, atLeastOne, physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)   


# The chosen swap sequence determines the next mapping
def writeSwapEffectConstraint(swapNum, layers, liveLog, physNum, cm, logNum, numCnots, top, path, satSolver=None, match_counts=None):
    allowedSwaps = np.append(np.argwhere(cm>0), [[0,0]], axis=0) 
    swapSeqs = itertools.product(allowedSwaps, repeat=swapNum)
    for swapSeq in swapSeqs:
        indexed_swaps = list(enumerate(swapSeq))
        perm = compose_swaps(swapSeq, range(physNum))
        for k in range(1, len(layers)):
            swapLits = [(True, "s", u, v, t, layers[k]) for (t, [u,v]) in indexed_swaps]
            for i in range(physNum):
                for j in liveLog:
                    for prev in range(layers[k-1], layers[k]):
                        if k == len(layers)-1: currentRange = [layers[k]]
                        else: currentRange = range(layers[k], layers[ k+1])
                        for current in currentRange:
                            writeHardClause(path, top, swapLits + [(False, "x", i, j, prev), (True, "x", perm[i], j, current)], physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)
                            writeHardClause(path, top, swapLits + [(True, "x", i, j, prev), (False, "x",perm[i], j, current)], physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)

def writeMaxDisplacedConstraint(maxDisplaced, physNum, logNum, swapNum, numCnots, top, path, satSolver=None, match_counts=None):
    for k in range(1,numCnots):
        for i in range(physNum):
            for j in range(logNum):
             writeHardClause(path, top, [(True, "x", i, j, k-1), (False, "x", i, j, k), (False, "d", j, k)],  physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)
    displacementSets = itertools.combinations([(True, "d", j, k) for j in range(logNum) for k in range(numCnots)], maxDisplaced)
    for displacementSet in displacementSets:
            writeHardClause(path, top, displacementSet, physNum, logNum, numCnots, swapNum, satSolver=satSolver, match_counts=match_counts)

# Soft Constraints #

def writeOptimizationConstraints(swapNum, physNum, numCnots, cm, logNum, routing, weighted, calibrationData, path, match_counts=None):
    if routing:
        if calibrationData:
            edges = np.argwhere(cm>0)
            for k in range(numCnots):
                    for i in range(len(edges)):
                        [u, v] = edges[i]
                        success_rate = 1-calibrationData[i]

                        writeSoftClause(path, (-1000*math.log(success_rate), [(True, "p", u, v, k)]), physNum, logNum, numCnots, swapNum, match_counts=match_counts)
                        writeSoftClause(path, (-1000*math.log(success_rate), [(True, "r", u, v, k)]), physNum, logNum, numCnots, swapNum, match_counts=match_counts)
                        for t in range(swapNum):
                            writeSoftClause(path, (-3000*math.log(success_rate), [(True, "s", u, v, t, k)]), physNum, logNum, numCnots, swapNum, match_counts=match_counts)

        else:
            for k in range(numCnots):
                for t in range(swapNum):
                    for (u,v) in itertools.product(range(physNum), repeat=2):
                        if u != v:
                            writeSoftClause(path, (1, [(True, "s", u, v, t, k)]), physNum, logNum, numCnots, swapNum, match_counts=match_counts)
    elif weighted:
            for k in range(1, numCnots):
                for i in range(physNum):
                    for i2 in range(physNum):
                        if i != i2:
                            writeSoftClause(path, (scipy.sparse.csgraph.shortest_path(cm)[i][i2], [(True, "w", i, i2, k)]), physNum, logNum, numCnots, swapNum, match_counts=match_counts)
    else:
        for k in range(1, numCnots):
                for i in range(physNum):
                    for j in range(logNum):
                        writeSoftClause(path, (1, [(True, "x", i, j, k-1), (False, "x", i, j, k)]), physNum, logNum, numCnots, swapNum, match_counts=match_counts)


## Conversion to MaxSat solver input format ##

def _match_offset(k, match_counts):
    """Cumulative offset for match variables: gate k has match indices in [offset[k], offset[k]+match_counts[k])."""
    return sum(match_counts[:k]) if match_counts else 0


def flattenedIndex(lit, physNum, logNum, numCnots, swapNum, match_counts=None):
    '''
        Converts the tuple representation of literals into integers.
        match_counts: optional list of length numCnots (number of allowed matches per multi-qubit gate);
        when provided, "m" literals (match(j,k)) are supported.
    '''
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numS = numCnots * physNum * physNum * swapNum
    numB = numS
    numW = physNum * physNum * numCnots
    numD = logNum * numCnots
    numM = sum(match_counts) if match_counts else 0
    indices = lit[2:]
    if lit[1] == "p":
        pos = np.ravel_multi_index(indices, (physNum, physNum, numCnots))
    elif lit[1] == "r":
        pos = (np.ravel_multi_index(indices, (physNum, physNum, numCnots)) + numP)
    elif lit[1] == "x":
        pos = (np.ravel_multi_index(
            indices, (physNum, logNum, numCnots)) + numP + numR)
    elif lit[1] == "s":
        pos = (np.ravel_multi_index(
            indices, (physNum, physNum, swapNum, numCnots)) + numP + numR + numX)
    elif lit[1] == "b":
        pos = (np.ravel_multi_index(indices, (physNum * physNum,
               swapNum, numCnots)) + numP + numR + numX+numS)
    elif lit[1] == "w":
        pos = (np.ravel_multi_index(indices, (physNum, physNum, numCnots)) + numP + numR + numX+numS + numB)
    elif lit[1] == "d":
        pos = (np.ravel_multi_index(indices, (logNum, numCnots)) + numP + numR + numX+numS + numB + numW)
    elif lit[1] == "m":
        if not match_counts:
            raise ValueError("match_counts required for 'm' literals")
        j, k = indices[0], indices[1]
        pos = _match_offset(k, match_counts) + j + numP + numR + numX + numS + numB + numW + numD
    else:
        raise ValueError(f"Unknown literal kind: {lit[1]}")
    pos = pos + 1
    if lit[0]:
        pos = -pos
    return pos

def flattenedWeightedClause(clause, physNum, logNum, numCnots, swapNum, match_counts=None):
    return (clause[0], [flattenedIndex(lit, physNum, logNum, numCnots, swapNum, match_counts) for lit in clause[1]])
def flattenedClause(clause, physNum, logNum, numCnots, swapNum, match_counts=None):
    return [flattenedIndex(lit, physNum, logNum, numCnots, swapNum, match_counts) for lit in clause]

def writeHardClause(f, top, clause, physNum, logNum, numCnots, swapNum, satSolver=None, match_counts=None):
        flatClause = flattenedClause(clause, physNum, logNum, numCnots, swapNum, match_counts)
        if satSolver:
            satSolver.add_clause([int(lit) for lit in flatClause])
        f.write(str(top))
        f.write(" ")
        for lit in flatClause:
            f.write(str(lit))
            f.write(" ")
        f.write("0\n")

def writeSoftClause(f, clause, physNum, logNum, numCnots, swapNum, match_counts=None):
    flattenedClause = flattenedWeightedClause(clause, physNum, logNum, numCnots, swapNum, match_counts)
    f.write(str(int(clause[0])))
    f.write(" ")
    for lit in flattenedClause[1]:
        f.write(str(lit))
        f.write(" ")
    f.write("0\n")

## Reading MaxSat solver output ##

def unravel(flatLit, physNum, logNum, numCnots, swapNum, match_counts=None):
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numS = numCnots * swapNum * physNum * physNum
    numB = numS
    numW = physNum * physNum * numCnots
    numD = logNum * numCnots
    numM = sum(match_counts) if match_counts else 0
    flipped = flatLit < 0
    shifted = abs(flatLit) - 1
    if shifted < numP:
        return (flipped, "p", np.unravel_index(shifted, (physNum, physNum, numCnots)))
    elif shifted < (numP + numR):
        return (flipped, "r", np.unravel_index(shifted-numP, (physNum, physNum, numCnots)))
    elif shifted < (numX + numP + numR):
        return (flipped, "x", np.unravel_index(shifted-(numP+numR), (physNum, logNum, numCnots)))
    elif shifted < (numP+numR+numX+numS):
        return (flipped, "s", np.unravel_index(shifted-(numP+numR+numX), (physNum, physNum, swapNum, numCnots)))
    elif shifted < (numP+numR+numX+numS+numB):
        return (flipped, "b", np.unravel_index(shifted-(numP+numR+numX+numS), (physNum*physNum, swapNum, numCnots)))
    elif shifted < (numP+numR+numX+numS+numB+numW):
        return (flipped, "w", np.unravel_index(shifted-(numP+numR+numX+numS+numB), (physNum, physNum,numCnots)))
    elif shifted < (numP+numR+numX+numS+numB+numW+numD):
        return (flipped, "d", np.unravel_index(shifted-(numP+numR+numX+numS+numB+numW), (logNum, numCnots)))
    elif numM > 0 and shifted < (numP+numR+numX+numS+numB+numW+numD+numM):
        m_shifted = shifted - (numP+numR+numX+numS+numB+numW+numD)
        # Decode (j, k) from flat index: m_shifted = offset[k] + j
        for k in range(numCnots):
            off = _match_offset(k, match_counts)
            if m_shifted < off + match_counts[k]:
                j = m_shifted - off
                return (flipped, "m", (j, k))
        raise AssertionError("match decode failed")
    else:
        return (flipped, "d", np.unravel_index(shifted-(numP+numR+numX+numS+numB+numW), (logNum, numCnots)))


def readMaxSatOutput(physNum, logNum, numCnots, swapNum, fname, match_counts=None):
    with open(fname) as f:
        for line in f:
            if line.startswith("v"):
                lits = line.split()[1:]
                return [unravel(int(lit), physNum, logNum, numCnots, swapNum, match_counts) for lit in lits]
    return []

def readPySatOutput(physNum, logNum, numCnots, swapNum, solver, match_counts=None):
    lits = solver.get_model()
    return [unravel(int(lit), physNum, logNum, numCnots, swapNum, match_counts) for lit in lits] 
    



def readCost(fname):
    best = math.inf
    with open(fname) as f:
        for line in f:
            if line.startswith("o") and int(line.split()[1]) < best:
                best = int(line.split()[1])
    return best


def mappingVars(parseFun, physNum, logNum, numCnots, permNum, source, match_counts=None):
    lits = parseFun(physNum, logNum, numCnots, permNum, source, match_counts=match_counts)
    return map(lambda x: x[2], filter(lambda x: not x[0] and x[1] == "x", lits))


## Interface with Haskell for no route version

def writeForRouting(initial, final, cm, fname="toHaskell.txt"):
    init_no_k = [(y,x) for (x,y, _) in initial]
    final_no_k = [(y,x) for (x,y, _) in final]
    a = scipy.sparse.csgraph.shortest_path(cm)
    d = [((i,j), int(a[i][j])) for i in range(len(cm)) for j in range(len(cm))]
    with open(fname, "w") as f:
        print( init_no_k, file=f)
        print( final_no_k, file=f )
        print( d, file=f)

def swapsFromMaps(initial, final, mapStr):
    init_no_k = [(y,x) for (x,y, _) in initial]
    final_no_k = [(y,x) for (x,y, _) in final]
    mapList = [init_no_k] + ast.literal_eval(mapStr.replace("fromList", "")) + [final_no_k]
    swaps = []
    for t in range(len(mapList)-1):
        t_done = False
        for (q, p) in mapList[t]:
            for (q1, p1) in mapList[t+1]:
                if q  == q1 and p != p1 and not t_done:
                    swaps.append((p, p1, t))
                    t_done = True
    return swaps 

## Solving ##

def extractMappingCore(solver, initialMapping, logNum, physNum, numCnots, swapNum, match_counts=None):
    i = 1
    for i in range(1,len(initialMapping)):
        submaps = list(itertools.combinations(initialMapping, i))
        for submap in submaps:
            assump = []
            for clause in submap:
                flatClause = flattenedClause(clause, physNum, logNum, numCnots, swapNum, match_counts)
                assump.append(int(flatClause[0]))
            if not solver.solve(assumptions = assump): 
                return submap



def solve_bounded_above(progName, cm, swapNum, chunks, pname="test", sname="out"):
    return_results = {}
    cost = 0  # <-- number of SWAPs added
    time_elapsed_wbo = 0
    physNum = len(cm)
    logNum = max(extract_qubits(cnots)) + 1
    return_results = {}
    hack = qiskit.QuantumCircuit.from_qasm_file(progName)
    (head, tail) = os.path.split(progName)
    with open(os.path.join(head, "qiskit-" + tail), "w") as f:
        f.write(hack.qasm())
    cnots = extract2qubit(os.path.join(head, "qiskit-" + tail))
    numCnots = len(cnots)

    layers= range(len(cnots))
    chunkSize = len(layers)//chunks
    currentChunk = 0
    addedSwaps = [0 for _ in range(chunks)]
    negatedModels = [[] for i in range(chunks)]
    solvers = [None for i in range(chunks)] 
    while currentChunk < chunks:
        print("current chunk is", currentChunk)
        print("negated", len(negatedModels[currentChunk]), "models")
        if currentChunk == chunks - 1: end = numCnots
        else: end = layers[chunkSize*(currentChunk+1)]
        currentSize = end - layers[chunkSize*(currentChunk)]
        print("current size:", currentSize)
        if currentChunk == 0:
            swapBack = []
            gen_write_s = time.process_time()
            s_out, _ = generateAndWriteClauses(logNum, cnots[:end], cnots[:end], cm, swapNum+addedSwaps[0], negatedModels[0] + swapBack, pname+"-chnk"+str(currentChunk)+".cnf", boundedAbove=True, routing=False)
            solvers[currentChunk] = s_out
            gen_write_f = time.process_time()
            print("generation and write time:", gen_write_f - gen_write_s)
            t_s = time.process_time()
            solvers[currentChunk].solve()
            t_f = time.process_time()
        else:
            prevSize = layers[chunkSize*currentChunk] - layers[chunkSize*(currentChunk-1)]
            prevAssignments = filter(lambda x : x[2] == prevSize-1, mappingVars(readPySatOutput, physNum, logNum, prevSize, swapNum+addedSwaps[currentChunk-1], solvers[currentChunk-1]))
            consistencyClauses = [[(False, "x", phys, log, 0)] for (phys, log, _) in prevAssignments]
            swapBack = []
            gen_write_s = time.process_time()
            print("start:", layers[chunkSize*(currentChunk)])
            print("end:", end)
            s_out, _ = generateAndWriteClauses(logNum, cnots[:end], cnots[layers[chunkSize*(currentChunk)]:end], cm, swapNum+addedSwaps[currentChunk], consistencyClauses+negatedModels[currentChunk]+swapBack,  pname+"-chnk"+str(currentChunk)+".cnf", boundedAbove=True, routing=False)
            solvers[currentChunk] = s_out
            gen_write_f = time.process_time()
            print("generation and write time:", gen_write_f - gen_write_s)
            t_s = time.process_time()
            solvers[currentChunk].solve()
            t_f = time.process_time()
        if solvers[currentChunk].solve(): 
            print("chunk", currentChunk, "solved")
            currentChunk = currentChunk+1
        else:
                if len(negatedModels[currentChunk-1]) < 50*(addedSwaps[currentChunk]+1): 
                    print("got stuck on chunk", currentChunk, "backtracking to chunk", currentChunk-1)
                    prevAssignments = filter(lambda x : x[2] == prevSize-1, mappingVars(readPySatOutput, physNum, logNum, prevSize, swapNum+addedSwaps[currentChunk-1], solvers[currentChunk-1]))
                    negatedModel =  [(True, "x", phys, log, lastGate) for (phys, log, lastGate) in prevAssignments]
                    print(negatedModel)
                    core = extractMappingCore(solvers[currentChunk], consistencyClauses,  logNum, len(cm), currentSize, swapNum+addedSwaps[currentChunk])
                    negatedSubmap = [(True, x, phys, log, prevSize-1) for [(_, x, phys, log, _)] in  core]
                    print(negatedSubmap)
                    negatedModels[currentChunk-1].append(negatedSubmap)
                    currentChunk = currentChunk-1

                else:
                    print("got stuck on chunk", currentChunk, "repeatedly, increasing swap count")
                    addedSwaps[currentChunk] += 1 
    a_star_time = 0
    cost = 0
    swaps = [[] for _ in range(chunks)]
    for i in range(chunks):
        if i == chunks - 1: end = numCnots
        else: end = layers[chunkSize*(i+1)]
        size = end - layers[chunkSize*(i)]
        for k in range(1,size):
            initial = list(filter(lambda x : x[2] == k-1, mappingVars(readPySatOutput, physNum, logNum, size, swapNum+addedSwaps[i], solvers[i])))
            final = list(filter(lambda x : x[2] == k, mappingVars(readPySatOutput, physNum, logNum, size, swapNum+addedSwaps[i], solvers[i])))
            writeForRouting(initial, final, cm)
            a_star_start = time.process_time()
            p = subprocess.run(["./route","toHaskell.txt"], stdout=PIPE )
            a_star_end = time.process_time()
            a_star_time += a_star_end - a_star_start
            out = p.stdout.decode()
            swaps[i].extend([(u, v, t, k) for  (u,v,t) in swapsFromMaps(initial, final, out.splitlines()[1].replace("mappings: ", ""))])
            cost += float(out.splitlines()[0].split()[1])
    return_results["cost"] = cost
    return_results['a_star_time'] = a_star_time
    return_results["swaps"] = swaps
    return_results['solvers'] = solvers
    return_results['algorithm'] = 'satmap_bounded_above'
    return return_results

def solve(progName, cm, swapNum, chunks, iterations=100, time_wbo_max = 3600, initial_map=None, qaoa=False, _routing=True, _weighted=False, _calibrationData=None,  pname="test", sname="out", spec=None):
    ''' The SAT-solving loop. When spec is provided, uses gate list + multi-qubit constraint; else legacy 2-qubit only. '''
    DEBUG_LOCAL = False
    (head, tail) = os.path.split(progName)
    os.makedirs("aux_files", exist_ok=True)
    progPath = os.path.join("aux_files", "qiskit-" + tail)

    if spec is not None:
        if hw_spec is None:
            raise ImportError("hardware_spec required when spec is provided")
        spec = hw_spec.load_spec(spec)
        cm = hw_spec.build_cm_from_spec(spec)
        hack = qiskit.QuantumCircuit.from_qasm_file(progName)
        with open(progPath, "w") as f:
            f.write(hack.qasm())
        gate_list = extract_gates(progPath)
        chunk_ranges = get_chunk_ranges(gate_list, chunks)
        if not chunk_ranges:
            return_results = {'qubits': max(extract_qubits(gate_list)) + 1 if gate_list else 0, 'chunks': 0, 'circ': tail, 'cnots': 0, 'swaps': 0, 'solve_time': 0, 'algorithm': 'satmap_classic'}
            return return_results
        logNum = max(extract_qubits(gate_list)) + 1
        physNum = len(cm)
        numCnots = sum(r[2] for r in chunk_ranges)
        return_results = {'qubits': logNum, 'chunks': chunks, 'circ': tail, 'cnots': numCnots, 'gate_list': gate_list, 'chunk_ranges': chunk_ranges, 'spec': spec}
        match_counts_per_chunk = [None] * chunks
        currentChunk = 0
        addedSwaps = [0 for _ in range(chunks)]
        negatedModels = [[] for _ in range(chunks)]
        time_elapsed_wbo = 0
        while currentChunk < chunks:
            start, end_excl, currentSize = chunk_ranges[currentChunk]
            multi_qubit_chunk = [g for g in gate_list[start:end_excl] if g.get("arity", 0) >= 2]
            dummy_cnots = [(0, 0)] * currentSize
            ff_clauses = negatedModels[currentChunk]
            if currentChunk > 0:
                prev_size = chunk_ranges[currentChunk - 1][2]
                prev_assignments_all = list(mappingVars(readMaxSatOutput, physNum, logNum, prev_size, swapNum + addedSwaps[currentChunk - 1], sname + "-chnk" + str(currentChunk - 1) + ".txt", match_counts=match_counts_per_chunk[currentChunk - 1]))
                # Pin step 0 of current chunk to *last* step of previous chunk (same as legacy).
                prev_assignments = [(phys, log, k) for (phys, log, k) in prev_assignments_all if k == prev_size - 1]
                ff_clauses = [[(False, "x", phys, log, 0)] for (phys, log, _) in prev_assignments] + negatedModels[currentChunk]
            s, match_counts = generateAndWriteClauses(logNum, dummy_cnots, dummy_cnots, cm, swapNum + addedSwaps[currentChunk], ff_clauses, pname + "-chnk" + str(currentChunk) + ".cnf", routing=_routing, weighted=_weighted, calibrationData=_calibrationData, spec=spec, multi_qubit_gates=multi_qubit_chunk)
            match_counts_per_chunk[currentChunk] = match_counts
            t_s = time.time()
            if time_wbo_max:
                solve_time_rem = time_wbo_max - time_elapsed_wbo
            try:
                with open(sname + "-chnk" + str(currentChunk) + ".txt", "w") as out_f:
                    p = subprocess.Popen(_wbo_cmd(iterations, pname + "-chnk" + str(currentChunk) + ".cnf"), stdout=out_f)
                    p.wait(timeout=solve_time_rem / max(1, chunks - currentChunk) if time_wbo_max else None)
            except subprocess.TimeoutExpired:
                p.terminate()
                time.sleep(1)
            time_elapsed_wbo += time.time() - t_s
            assignments_all = list(mappingVars(readMaxSatOutput, physNum, logNum, currentSize, swapNum + addedSwaps[currentChunk], sname + "-chnk" + str(currentChunk) + ".txt", match_counts=match_counts))
            # Advance only if we have a valid mapping at the *last* step (same as legacy).
            assignments = [(phys, log, k) for (phys, log, k) in assignments_all if k == currentSize - 1]
            if assignments:
                currentChunk += 1
            else:
                if currentChunk > 0 and len(negatedModels[currentChunk - 1]) < 50 * (addedSwaps[currentChunk] + 1):
                    prev_size = chunk_ranges[currentChunk - 1][2]
                    prev_assignments_all = list(mappingVars(readMaxSatOutput, physNum, logNum, prev_size, swapNum + addedSwaps[currentChunk - 1], sname + "-chnk" + str(currentChunk - 1) + ".txt", match_counts=match_counts_per_chunk[currentChunk - 1]))
                    prev_assignments = [(phys, log, k) for (phys, log, k) in prev_assignments_all if k == prev_size - 1]
                    consistency_clauses = [[(False, "x", phys, log, 0)] for (phys, log, _) in prev_assignments]
                    core = extractMappingCore(s, consistency_clauses, logNum, len(cm), currentSize, swapNum + addedSwaps[currentChunk], match_counts=match_counts)
                    if core is not None:
                        negatedSubmap = [(True, "x", phys, log, prev_size - 1) for [(_, _, phys, log, _)] in core]
                    else:
                        negatedSubmap = [(True, "x", phys, log, prev_size - 1) for (phys, log, _) in prev_assignments]
                    negatedModels[currentChunk - 1].append(negatedSubmap)
                    currentChunk -= 1
                else:
                    addedSwaps[currentChunk] += 1
        cost = 0
        for i in range(chunks):
            try:
                with open(sname + "-chnk" + str(i) + ".txt") as f:
                    for line in f:
                        if line.startswith("o"):
                            cost += int(line.split()[1])
                            break
            except (FileNotFoundError, ValueError):
                pass
        return_results['swaps'] = cost
        return_results['solve_time'] = time_elapsed_wbo
        return_results['algorithm'] = 'satmap_classic'
        return_results['match_counts_per_chunk'] = match_counts_per_chunk
        return return_results

    hack = qiskit.QuantumCircuit.from_qasm_file(progName)
    with open(progPath, "w") as f:
        f.write(hack.qasm())
    cnots = extract2qubit(progPath)
    numCnots = len(cnots)
    cost = 0
    time_elapsed_wbo = 0
    physNum = len(cm)
    logNum = max(extract_qubits(cnots)) + 1
    return_results = {}
    return_results['qubits'] = logNum
    return_results["chunks"] = chunks 
    return_results['circ'] = tail
    return_results['cnots'] = numCnots
    layers = range(len(cnots))
    chunkSize = len(layers)//chunks

    if(DEBUG_LOCAL and DEBUG_GLOBAL):
        print(f'logNum={logNum}, physNum={physNum}, cnots={cnots}, numCnots{numCnots}, layers={layers}, chunkSize={chunkSize}')

    currentChunk = 0
    addedSwaps = [0 for _ in range(chunks)]
    negatedModels = [[] for i in range(chunks)]
    time_elapsed_wbo = 0
    while currentChunk < chunks:
        print("current chunk is", currentChunk)
        print("negated", len(negatedModels[currentChunk]), "models")
        if currentChunk == chunks - 1: end = numCnots
        else: end = layers[chunkSize*(currentChunk+1)]
        currentSize = end - layers[chunkSize*(currentChunk)]
        print("current size:", currentSize)
        if negatedModels[currentChunk]:
            if(DEBUG_LOCAL and DEBUG_GLOBAL):
                print(set.intersection(*[set(l)
                      for l in negatedModels[currentChunk]]))
        if currentChunk == 0:
            swapBack = []
            if qaoa and currentChunk == chunks-1:
                swapBack = [[(False, "x", phys, log, currentSize-1), (True, "x", phys, log, 0) ] for phys in range(physNum) for log in range(logNum)] +  [[(True, "x", phys, log, currentSize-1), (False, "x", phys, log, 0) ] for phys in range(physNum) for log in range(logNum)]
            gen_write_s = time.time()
            s, _ = generateAndWriteClauses(logNum, cnots[:end], cnots[:end], cm, swapNum+addedSwaps[0], negatedModels[0] + swapBack, pname+"-chnk"+str(currentChunk)+".cnf", routing=_routing, weighted =_weighted, calibrationData=_calibrationData)
            gen_write_f = time.time()
            print("generation and write time:", gen_write_f - gen_write_s)
            t_s = time.time()
            if time_wbo_max:
                solve_time_rem = time_wbo_max-time_elapsed_wbo 
            try:
               p = subprocess.Popen(_wbo_cmd(iterations, pname+"-chnk"+str(currentChunk)+".cnf"),  stdout=open( sname + "-chnk0" + ".txt", "w"))
               p.wait(timeout=solve_time_rem/(chunks-currentChunk))
            except subprocess.TimeoutExpired:
                print("exiting open-wbo because of solve time alloted...")
                p.terminate()
                time.sleep(10)
            t_f = time.time()
            time_elapsed_wbo += t_f - t_s
        else:
            prevSize = layers[chunkSize*currentChunk] - layers[chunkSize*(currentChunk-1)]
            prevAssignments = filter(lambda x : x[2] == prevSize-1, mappingVars(readMaxSatOutput, physNum, logNum, prevSize, swapNum+addedSwaps[currentChunk-1], sname + "-chnk" + str(currentChunk-1) + ".txt"))
            consistencyClauses = [[(False, "x", phys, log, 0)] for (phys, log, _) in prevAssignments]
            swapBack = []
            if qaoa and currentChunk == chunks-1:
                initialSize = layers[chunkSize] - layers[0]
                initialMapping =  filter(lambda x : x[2] == 0, mappingVars(readMaxSatOutput, physNum, logNum, initialSize, swapNum+addedSwaps[0], sname + "-chnk" + str(0) + ".txt")) 
                swapBack = [[(False, "x", phys, log, currentSize-1)] for (phys, log, _) in initialMapping]
            gen_write_s = time.process_time()
            print("start:", layers[chunkSize*(currentChunk)])
            print("end:", end)
            s, _ = generateAndWriteClauses(logNum, cnots[:end], cnots[layers[chunkSize*(currentChunk)]:end], cm, swapNum+addedSwaps[currentChunk], consistencyClauses+negatedModels[currentChunk]+swapBack,  pname+"-chnk"+str(currentChunk)+".cnf", routing=_routing, weighted=_weighted, calibrationData=_calibrationData)
            gen_write_f = time.process_time()
            print("generation and write time:", gen_write_f - gen_write_s)
            t_s = time.process_time()
            if time_wbo_max:
                solve_time_rem = time_wbo_max- time_elapsed_wbo 
            try:
                p = subprocess.Popen(_wbo_cmd(iterations, pname+"-chnk"+str(currentChunk)+".cnf"), stdout=open(sname + "-chnk" + str(currentChunk) + ".txt", "w"))
                p.wait(timeout=solve_time_rem/(chunks-currentChunk))
            except subprocess.TimeoutExpired:
                print("exiting open-wbo because of solve time alloted...")
                p.terminate()
                time.sleep(10)
            t_f = time.process_time()
            time_elapsed_wbo += t_f - t_s
        assignments = filter(lambda x : x[2] == currentSize-1, mappingVars(readMaxSatOutput, physNum, logNum, currentSize, swapNum+addedSwaps[currentChunk], sname + "-chnk" + str(currentChunk) + ".txt"))
        if list(assignments): 
            print("chunk", currentChunk, "solved")
            currentChunk = currentChunk+1
        else:
                if len(negatedModels[currentChunk-1]) < 50*(addedSwaps[currentChunk]+1): 
                    print("got stuck on chunk", currentChunk, "backtracking to chunk", currentChunk-1)
                    prevAssignments = filter(lambda x : x[2] == prevSize-1, mappingVars(readMaxSatOutput, physNum, logNum, prevSize, swapNum+addedSwaps[currentChunk-1], sname + "-chnk" + str(currentChunk-1) + ".txt"))
                    negatedModel =  [(True, "x", phys, log, lastGate) for (phys, log, lastGate) in prevAssignments]
                    print(negatedModel)
                    core = extractMappingCore(s, consistencyClauses,  logNum, len(cm), currentSize, swapNum+addedSwaps[currentChunk])
                    negatedSubmap = [(True, x, phys, log, prevSize-1) for [(_, x, phys, log, _)] in  core]
                    print(negatedSubmap)
                    negatedModels[currentChunk-1].append(negatedSubmap)
                    currentChunk = currentChunk-1

                else:
                    print("got stuck on chunk", currentChunk, "repeatedly, increasing swap count")
                    addedSwaps[currentChunk] += 1 
    cost=0
    for i in range(chunks):
        with open(sname + "-chnk" + str(i) + ".txt") as f:
            for line in f:
                if line.startswith("o"):
                    count = int(line.split()[1])
        cost += count
    return_results['swaps'] = cost
    return_results['solve_time'] = time_elapsed_wbo
    return_results['algorithm'] = 'satmap_classic'
    if not _routing:
        a_star_time = 0
        cost = 0
        swaps = [[] for _ in range(chunks)]
        for i in range(chunks):
            if i == chunks - 1: end = numCnots
            else: end = layers[chunkSize*(i+1)]
            size = end - layers[chunkSize*(i)]
            for k in range(1,size):
                initial = list(filter(lambda x : x[2] == k-1, mappingVars(readMaxSatOutput, physNum, logNum, size, swapNum+addedSwaps[i], sname + "-chnk" + str(i) + ".txt")))
                final = list(filter(lambda x : x[2] == k, mappingVars(readMaxSatOutput, physNum, logNum, size, swapNum+addedSwaps[i], sname + "-chnk" + str(i) + ".txt")))
                writeForRouting(initial, final, cm)
                a_star_start = time.process_time()
                p = subprocess.run(["./route","toHaskell.txt"], stdout=PIPE )
                a_star_end = time.process_time()
                a_star_time += a_star_end - a_star_start
                out = p.stdout.decode()
                swaps[i].extend([(u, v, t, k) for  (u,v,t) in swapsFromMaps(initial, final, out.splitlines()[1].replace("mappings: ", ""))])
                cost += float(out.splitlines()[0].split()[1])
        return_results["cost"] = cost
        return_results['a_star_time'] = a_star_time
        return_results["swaps"] = swaps
        return_results['algorithm'] = 'satmap_classic'
        print(return_results)
        return return_results
    return return_results


## Converting solutions to circuits, verifying correctness ##

def toQasm(physNum, logNum, numCnots, swapNum, solSource, progPath, cm, prevMap, start=0, append_rest=False, swapList=None, match_counts=None, spec=None):
    circ = qiskit.QuantumCircuit(physNum, physNum)
    prog = qiskit.QuantumCircuit.from_qasm_file(progPath)
    temp = qiskit.QuantumCircuit(physNum, physNum)
    temp.compose(prog, inplace=True)
    edges = np.argwhere(cm > 0)
    i = start
    if append_rest:
        while len(circ) + start < len(temp):
            circ.append(*temp[i])
            i += 1
    else:
        multi_count = 0
        while multi_count < numCnots and i < len(temp):
            circ.append(*temp[i])
            if len(temp[i][1]) >= 2:
                multi_count += 1
            i += 1
    if type(solSource) is str:
        lits = readMaxSatOutput(physNum, logNum, numCnots, swapNum, solSource, match_counts=match_counts)
    else:
        lits = readPySatOutput(physNum, logNum, numCnots, swapNum, solSource, match_counts=match_counts)
    if swapList is not None:
        swaps = swapList
    else:
        swaps = [s[2] for s in filter(lambda x : not x[0] and x[1] == "s" and x[2][0] != x[2][1], lits)]
    mappingVars =  [x[2] for x in filter(lambda x : not x[0] and x[1] == "x", lits)]
    logToPhys = { (j,k) : i for (i,j,k) in mappingVars}
    physToLog = { (i,k) : j for (i,j,k) in mappingVars}
    swapIndices = [s[3] for s in swaps]
    for k in range(numCnots):
        mapKLog = list(filter(lambda x: x[0][1] == k, logToPhys.items()))
        assert(len(list(mapKLog)) == len(set(mapKLog))), "Invalid solution: non-injective"
        if k == 0 and prevMap:
            # Compare (log, phys) mapping; prevMap has keys (log, k_prev), mapKLog has (log, 0).
            assert set((x[0][0], x[1]) for x in mapKLog) == set((x[0][0], x[1]) for x in prevMap), "Invalid solution: slices aren't consistent"
        mapKPhys = list(filter(lambda x: x[0][1] == k, physToLog.items()))
        assert(len(list(mapKPhys)) == len(set(mapKPhys))), "Invalid solution: non-function"
        swapsK = filter(lambda s: s[3] == k, swaps)
        
        justPhys = [s[:2] for s in swapsK]
        for (phys1,phys2) in justPhys:
            assert([phys1, phys2] in edges.tolist()), "Invalid solution: bad swap"
        if k>0:
            for l in range(logNum):
                if (l,k) in logToPhys.keys():
                    physPrev = logToPhys[(l,k-1)]
                    assert(logToPhys[(l,k)] == compose_swaps(justPhys,range(physNum))[physPrev]), "Invalid solution: unexpected SWAP"      
    mappedCirc = qiskit.QuantumCircuit(circ.num_qubits)
    multiCount = 0
    for j in range(len(circ)):
        qubits = list(map(lambda q: qiskit.circuit.Qubit(q.register, logToPhys[(q.index, min(multiCount, numCnots - 1))]), circ[j][1]))
        if len(circ[j][1]) >= 2:
            if multiCount in swapIndices:
                swapsK = filter(lambda s: s[3] == multiCount, swaps)
                for s in swapsK:
                    mappedCirc.swap(s[0], s[1])
            phys_qubits = tuple(logToPhys[(q.index, multiCount)] for q in circ[j][1])
            if spec is not None and hw_spec is not None:
                n = len(phys_qubits)
                allowed = hw_spec.get_subgraph_matches(spec, n, None)
                assert list(phys_qubits) in allowed, f"Invalid solution: gate qubits {phys_qubits} not in SubgraphMatches for arity {n}"
            else:
                if len(circ[j][1]) == 2:
                    assert [phys_qubits[0], phys_qubits[1]] in edges.tolist(), "Invalid solution: unsatisfied 2q gate"
            multiCount += 1
        mappedCirc.append(circ[j][0], qubits)
    # Final mapping is at last step (k = numCnots - 1); pass to next chunk for consistency assert.
    finalMap = list(filter(lambda x: x[0][1] == numCnots - 1, logToPhys.items()))
    return (mappedCirc, i, finalMap)
          
def toQasmFF(progName, cm, swapNum, chunks, solSource, swaps=None, gate_list=None, chunk_ranges=None, spec=None, match_counts_per_chunk=None):
    physNum = len(cm)
    if gate_list is not None and chunk_ranges is not None:
        logNum = max(extract_qubits(gate_list)) + 1
        prevMap = None
        circ = qiskit.QuantumCircuit(len(cm), len(cm))
        for i in range(chunks):
            start, end_excl, currentSize = chunk_ranges[i]
            is_last = (i == chunks - 1)
            match_counts = match_counts_per_chunk[i] if match_counts_per_chunk else None
            if type(solSource) is str:
                (mapped_circ, gates, finalMap) = toQasm(physNum, logNum, currentSize, swapNum, solSource + "-chnk" + str(i) + ".txt", progName, cm, prevMap, append_rest=is_last, start=start, swapList=swaps[i] if swaps else None, match_counts=match_counts, spec=spec)
            else:
                (mapped_circ, gates, finalMap) = toQasm(physNum, logNum, currentSize, swapNum, solSource[i], progName, cm, prevMap, append_rest=is_last, start=start, swapList=swaps[i] if swaps else None, match_counts=match_counts, spec=spec)
            prevMap = finalMap
            circ.compose(mapped_circ, inplace=True)
        return circ.qasm()
    cnots = extract2qubit(progName)
    logNum = max(extract_qubits(cnots)) + 1
    numCnots = len(cnots)
    layers = range(len(cnots))
    chunkSize = len(layers) // chunks
    pointer = 0
    prevMap = None
    circ = qiskit.QuantumCircuit(len(cm), len(cm))
    for i in range(chunks):
        is_last = (i == chunks - 1)
        if i == chunks - 1:
            end = numCnots
        else:
            end = layers[chunkSize * (i + 1)]
        currentSize = end - layers[chunkSize * (i)]
        if type(solSource) is str:
            (mapped_circ, gates, finalMap) = toQasm(physNum, logNum, currentSize, swapNum, solSource + "-chnk" + str(i) + ".txt", progName, cm, prevMap, append_rest=is_last, start=pointer, swapList=swaps[i] if swaps else None)
        else:
            (mapped_circ, gates, finalMap) = toQasm(physNum, logNum, currentSize, swapNum, solSource[i], progName, cm, prevMap, append_rest=is_last, start=pointer, swapList=swaps[i] if swaps else None)
        pointer = gates
        prevMap = finalMap
        circ.compose(mapped_circ, inplace=True)
    return circ.qasm()

def computeFidelity(circ, calibrationData):
    fid=1
    for i in range(len(circ)):
        if circ[i][0].name == 'cx':
            [c, t] = circ[i][1]
            fid = fid*(1-calibrationData[(c.index, t.index)])
        elif circ[i][0].name == 'swap':
            [q, q1] = circ[i][1]
            fid = fid*((1-calibrationData[(q.index, q1.index)])**3)
    return fid
