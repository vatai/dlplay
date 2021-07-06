#!/usr/bin/bash
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
    ):
        self.polybench_dir = Path(polybench_dir).expanduser()
        self.ppcg_dir = Path("~/code/ppcg").expanduser()
        self.out_dir = "/tmp/"

        self.benchmark_dir = benchmark_dir
        self.benchmark_name = self.benchmark_dir.split("/")[-1]

        self.common_args = [
            "-DMEDIUM_DATASET",
            # "-DPOLYBENCH_DUMP_ARRAYS",
            "-DPOLYBENCH_TIME",
            "-DPOLYBENCH_USE_C99_PROTO",
            f"-I{self.polybench_dir}/utilities",
            f"-I{self.polybench_dir}/{self.benchmark_dir}",
        ]

        self.orig_c_file = (
            f"{self.polybench_dir}/{self.benchmark_dir}/{self.benchmark_name}.c"
        )
        self.ppcg_c_file = f"{self.out_dir}/{self.benchmark_name}.ppcg.c"
        self.orig_bin_file = f"{self.out_dir}/{self.benchmark_name}.orig"
        self.ppcg_bin_file = f"{self.out_dir}/{self.benchmark_name}.ppcg"

    def create_ppcg_c_file(self):
        args = [
            f"{self.ppcg_dir}/ppcg",
            self.orig_c_file,
            "--target=c",
            "--tile",
            "-o",
            self.ppcg_c_file,
        ] + self.common_args
        result = subprocess.run(args, capture_output=True)
        assert result.returncode == 0

    def gcc(self, c_file, bin_file):
        args = [
            "gcc",
            c_file,
            f"{self.polybench_dir}/utilities/polybench.c",
            "-lm",
            "--std=gnu99",
            "-o",
            bin_file,
        ] + self.common_args
        result = subprocess.run(args, capture_output=True)
        assert result.returncode == 0

    def run_bin(self, bin_file):
        result = subprocess.run([bin_file], capture_output=True)
        time = float(result.stdout.decode("utf-8"))
        return time

    def run_orig(self):
        self.gcc(self.orig_c_file, self.orig_bin_file)
        time = self.run_bin(self.orig_bin_file)
        print(time)

    def run_ppcg(self):
        self.create_ppcg_c_file()
        self.gcc(self.ppcg_c_file, self.ppcg_bin_file)
        time = self.run_bin(self.ppcg_bin_file)
        print(time)


def main():
    polybench_dir = Path("~/code/polybench-c-4.2.1-beta/").expanduser()
    for i in polybench_dir.glob("**/*.c"):
        if i.name == i.parent.name + ".c":
            benchmark_dir = str(i.parent).replace(str(polybench_dir), "")[1:]
            print(benchmark_dir)
            pp = PolybenchPpcg(benchmark_dir)
            pp.run_orig()
            pp.run_ppcg()


main()
