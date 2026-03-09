import argparse
import ast
import datetime
import os
import numpy as np
import architectures
from common import extract2qubit
import satmap_core
import satmap_hybrid

def transpile(progname, cm, swapNum=1, cnfname='test', sname='out', slice_size=25, max_sat_time=600, routing=True, weighted=False, calibrationData = None, bounded_above=True, hybrid=None, hardware_spec=None):
    """
    When hardware_spec is provided (path to JSON or dict), use multi-qubit pipeline:
    load spec, build cm from spec, extract_gates, chunk by multi-qubit gates, solve with writeMultiQubitGateConstraint, output with toQasmFF(gate_list=..., spec=...).
    Otherwise use legacy 2-qubit path with cm and extract2qubit.
    """
    if hardware_spec is not None:
        from common import extract_gates
        import hardware_spec as hw_spec
        gate_list = extract_gates(progname)
        num_multi = len([g for g in gate_list if g.get("arity", 0) >= 2])
        if num_multi == 0:
            print('Exiting... circuit contains no multi-qubit gates')
            with open(progname) as f:
                return (None, f.read())
        chunks = -(num_multi // -slice_size)
        if chunks <= 0:
            chunks = 1
        stats = satmap_core.solve(progname, cm, swapNum, chunks, pname=cnfname, sname=sname, time_wbo_max=max_sat_time, _calibrationData=calibrationData, spec=hardware_spec)
        qasm_path = os.path.join("aux_files", "qiskit-" + os.path.basename(progname))
        spec = stats.get('spec')
        cm_from_spec = hw_spec.build_cm_from_spec(spec) if spec else cm
        qasm = satmap_core.toQasmFF(qasm_path, cm_from_spec, swapNum, chunks, sname, gate_list=stats.get('gate_list'), chunk_ranges=stats.get('chunk_ranges'), spec=spec, match_counts_per_chunk=stats.get('match_counts_per_chunk'))
        return (stats, qasm)
    chunks = -(len(extract2qubit(progname)) // -slice_size)
    if len(extract2qubit(progname)) == 0:
        print('Exiting... circuit contains no two qubit gates')
        with open(progname) as f:
            return (None, f.read())
    elif hybrid:
        return satmap_hybrid.solve_with_sabre(progname, coupling_map=cm, swap_num=swapNum, explore=hybrid, timeout=max_sat_time)
    elif routing:
        stats = satmap_core.solve(progname, cm, swapNum, chunks, pname=cnfname, sname=sname, time_wbo_max=max_sat_time, _calibrationData=calibrationData)
        return (stats, satmap_core.toQasmFF(os.path.join("aux_files", "qiskit-"+os.path.split(progname)[1]),  cm, swapNum, chunks, sname))
    elif bounded_above:
        results = satmap_core.solve_bounded_above(progname, cm, swapNum, chunks, pname=cnfname, sname=sname)
        return ((results['cost'], results['a_star_time']), satmap_core.toQasmFF(os.path.join("aux_files", "qiskit-"+os.path.split(progname)[1]),  cm, swapNum, chunks, results['solvers'], swaps=results['swaps']))
    else: 
      results = satmap_core.solve(progname, cm, swapNum, chunks, pname=cnfname, sname=sname, _routing=False, _weighted=weighted)
      return ((results['cost'], results['time_wbo'], results['a_star_time']), satmap_core.toQasmFF(os.path.join("aux_files", "qiskit-"+os.path.split(progname)[1]),  cm, swapNum, chunks, sname, swaps=results['swaps']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prog", help="path to input program file")
    parser.add_argument("-o_p", "--output_path", help="where to write the resulting qasm")
    parser.add_argument("-a", "--arch", help="name of qc architecture")
    parser.add_argument("-to", "--timeout", type=int, default=1800,help="maximum run time for a mapper in seconds")
    parser.add_argument("--k", type=int, default=25, help="SolveSwapsFF: k-value")
    parser.add_argument("--cyclic", choices=["on", "off"], default="off", help="cyclic mapping")
    parser.add_argument("--no_route",  action="store_true", help="SolveSwapsFF routing")
    parser.add_argument("--weighted",  action="store_true", help="SolveSwapsFF weighting on dist")
    parser.add_argument("--err", choices=['fake_tokyo', 'fake_linear'], help="olsq: 2 qubit gate error rates")
    parser.add_argument('--hybrid', choices=['vertically', 'horizontally', 'horizontal_sliding_window'])
    parser.add_argument("-H", "--hardware_spec", help="Path to JSON hardware spec (edges + subgraph_matches). Uses multi-qubit pipeline.")
    archs =  {
        "tokyo" : architectures.ibmTokyo,
        "toronto" : architectures.ibmToronto,
        "4x4_mesh" : architectures.meshArch(4,4),
        'small_linear' : architectures.linearArch(6),
        "16_linear" : architectures.linearArch(16),
        "tokyo_full_diags" : architectures.tokyo_all_diags(),
        "tokyo_no_diags" : architectures.tokyo_no_diags(),
        'tokyo_drop_2' : architectures.tokyo_drop_worst_n(2, architectures.tokyo_error_map()),
        'tokyo_drop_6' : architectures.tokyo_drop_worst_n(6, architectures.tokyo_error_map()),
        'tokyo_drop_10' : architectures.tokyo_drop_worst_n(10, architectures.tokyo_error_map()),
        'tokyo_drop_14' : architectures.tokyo_drop_worst_n(14, architectures.tokyo_error_map()),
    }
    error_rates = {
        'fake_tokyo' : architectures.tokyo_error_list(),
        'fake_linear' : architectures.fake_linear_error_list()
    }
    args = parser.parse_args()
    hardware_spec_path = getattr(args, 'hardware_spec', None)
    if hardware_spec_path:
        import hardware_spec as hw_spec
        spec = hw_spec.load_spec(args.hardware_spec)
        arch = hw_spec.build_cm_from_spec(spec)
    else:
        if not args.arch:
            parser.error("either -a/--arch or -H/--hardware_spec is required")
        if args.arch in archs:
            arch = archs[args.arch]
        else:
            with open(args.arch) as f:
                arch = np.array(ast.literal_eval(f.read()))
    hybrid = args.hybrid
    base, _ = os.path.splitext(os.path.basename(args.prog))
    os.makedirs(f"aux_files", exist_ok =True)
    (stats, qasm) = transpile(args.prog, arch, 1, os.path.join("aux_files", "prob_"+base), os.path.join("aux_files", "sol_"+base), slice_size=args.k, max_sat_time=args.timeout, routing=not args.no_route, weighted=args.weighted, calibrationData=error_rates[args.err] if args.err else None, bounded_above=True, hybrid=hybrid, hardware_spec=hardware_spec_path)
    print(stats)
    if args.arch in archs:
        stats["arch"] = args.arch
    else:
        stats['arch'] = f"custom arch with {len(arch)} qubits"
    with open(f"stats_{base}.txt", "w") as f:
        f.write(str(stats))
    out_file = args.output_path if args.output_path else "mapped_"+os.path.basename(args.prog)
    if out_file != "no_qasm":
        with open(out_file, "w") as f:
            f.write(qasm) 
    