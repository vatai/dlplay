#!/usr/bin/env python
import subprocess
from pathlib import Path


def error_msg(result: subprocess.CompletedProcess):
    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")
    return f"\nstdout:\n{stdout}\nstderr:\n{stderr}"


class PolybenchPpcg:
    def __init__(
        self,
        benchmark_dir="datamining/correlation",
        polybench_dir="~/code/polybench-c-4.2.1-beta/",
        out_dir="/tmp/",
    ):
        self.polybench_dir = Path(polybench_dir).expanduser()
        self.ppcg = Path("~/code/ppcg/ppcg").expanduser()
        self.out_dir = out_dir

        self.benchmark_dir = f"{self.polybench_dir}/{benchmark_dir}"
        self.benchmark_name = benchmark_dir.split("/")[-1]

        self.common_args = [
            "-DMINI_DATASET",
            # "-DPOLYBENCH_DUMP_ARRAYS",
            "-DPOLYBENCH_TIME",
            "-DPOLYBENCH_USE_C99_PROTO",
            f"-I{self.polybench_dir}/utilities",
            f"-I{self.benchmark_dir}",
        ]

        self.compiler_args = [
            f"{self.polybench_dir}/utilities/polybench.c",
            "-lm",
            # "--std=gnu99",
        ]

        self.orig_c_file = f"{self.benchmark_dir}/{self.benchmark_name}.c"
        self.ppcg_c_file = f"{self.out_dir}/{self.benchmark_name}.ppcg.c"
        self.orig_bin_file = f"{self.out_dir}/{self.benchmark_name}.orig"
        self.ppcg_bin_file = f"{self.out_dir}/{self.benchmark_name}.ppcg"

    def create_ppcg_c_file(self, target: str, sizes: str):
        args = (
            [self.ppcg, self.orig_c_file]
            + [f"--target={target}", "--tile"]
            + ["-o", self.ppcg_c_file]
            + self.common_args
        )
        result = subprocess.run(args, capture_output=True)
        assert result.returncode == 0, error_msg(result)

    def compile(self, c_file: str, bin_file: str, compiler: str):
        args = (
            [compiler, c_file]
            + ["-o", bin_file]
            + self.compiler_args
            + self.common_args
        )
        result = subprocess.run(args, capture_output=True)
        assert result.returncode == 0, error_msg(result)

    def run_bin(self, bin_file: str):
        result = subprocess.run([bin_file], capture_output=True)
        time = float(result.stdout.decode("utf-8"))
        return time

    def run_orig(self, compiler: str = "gcc"):
        self.compile(self.orig_c_file, self.orig_bin_file, compiler)
        time = self.run_bin(self.orig_bin_file)
        print(time)

    def run_ppcg(self, sizes: str, target: str = "c", compiler: str = "gcc"):
        self.create_ppcg_c_file(target, sizes)
        self.compile(self.ppcg_c_file, self.ppcg_bin_file, compiler)
        time = self.run_bin(self.ppcg_bin_file)
        print(time)


def main():
    polybench_dir = Path("~/code/polybench-c-4.2.1-beta/").expanduser()
    for benchmark in list(polybench_dir.glob("**/*.c"))[:3]:
        if benchmark.name == benchmark.parent.name + ".c":
            benchmark_dir = str(benchmark.parent).replace(str(polybench_dir), "")[1:]
            print(benchmark_dir)
            pp = PolybenchPpcg(benchmark_dir)
            pp.run_orig("nvcc")
            pp.run_ppcg(None, "cuda", "nvcc")


main()
