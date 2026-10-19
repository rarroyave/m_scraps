"""Installed entry points; the native optimizer is built against local MPI."""
import sys
from . import _TOOLS


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare-atat":
        from compare_atat import main as compare
        return compare(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        from structure_analysis import main as analyze
        return analyze(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='benchmark':
        from benchmark_submission import main as benchmark
        return benchmark(sys.argv[2:])
    from scraps_wizard import main as wizard
    return wizard()


def run():
    from run_sqs import main
    return main()


def verify():
    from verify_sro import main
    return main()


def sweep():
    from scraps_htp import main
    return main()
