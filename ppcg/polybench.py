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
        out_dir="/tmp",
    ):
        self.polybench_dir = Path(polybench_dir).expanduser()
        self.ppcg = Path("~/code/ppcg/ppcg").expanduser()
        self.out_dir = out_dir

        self.benchmark_dir = f"{self.polybench_dir}/{benchmark_dir}"
        self.benchmark_name = benchmark_dir.split("/")[-1]

        self.common_args = [
            "-DLARGE_DATASET",
            # "-DPOLYBENCH_DUMP_ARRAYS",
            "-DPOLYBENCH_TIME",
            f"-I{self.polybench_dir}/utilities",
            f"-I{self.benchmark_dir}",
        ]

        self.compiler_args = [
            f"{self.polybench_dir}/utilities/polybench.c",
            "-lm",
            # "--std=gnu99",
        ]

        self.orig_c_file = f"{self.benchmark_dir}/{self.benchmark_name}.c"
        self.orig_bin_file = f"{self.out_dir}/{self.benchmark_name}.orig"
        self.ppcg_c_file = f"{self.out_dir}/{self.benchmark_name}.ppcg.c"

    def ppcg_bin_file(self, target: str):
        return f"{self.out_dir}/{self.benchmark_name}.{target}.ppcg"

    def create_ppcg_c_file(self, target: str, sizes: str):
        args = [self.ppcg, self.orig_c_file, f"--target={target}"]
        args += [
            "--tile",
            # "--dump-sizes",
        ]
        if sizes:
            args.append(f"--sizes={sizes}")
        args += self.common_args
        if target != "cuda":
            args += ["-o", self.ppcg_c_file]  # ignored for `target=="cuda"`
        result = subprocess.run(args, capture_output=True)
        assert result.returncode == 0, error_msg(result)
        if target == "cuda":
            self.add_extern_c()

    def add_extern_c(self):
        file_path = f"{self.benchmark_name}_host.cu"
        with open(file_path, "r") as fp:
            lines = fp.readlines()
            for i, line in enumerate(lines):
                if "#include" in line and "polybench.h" in line:
                    pos = i
            lines.insert(pos + 1, "}\n")
            lines.insert(pos, 'extern "C" {\n')

        with open(file_path, "w") as fp:
            fp.writelines(lines)

    def compile(self, files: list[str], bin_file: str, compiler: str):
        args = [compiler] + files
        if compiler != "nvcc":
            args.append("-DPOLYBENCH_USE_C99_PROTO")
        args += self.common_args + self.compiler_args + ["-o", bin_file]
        # print(" ".join(args))
        result = subprocess.run(args, capture_output=True)
        assert result.returncode == 0, error_msg(result)

    def run_bin(self, bin_file: str):
        result = subprocess.run([bin_file], capture_output=True)
        time = float(result.stdout.decode("utf-8"))
        return time

    def run_orig(self, compiler: str = "gcc"):
        self.compile([self.orig_c_file], self.orig_bin_file, compiler)
        time = self.run_bin(self.orig_bin_file)
        return time

    def run_ppcg(self, sizes: str, target: str = "c", compiler: str = "gcc"):
        self.create_ppcg_c_file(target, sizes)
        if target == "cuda":
            suffixes = ["host.cu", "kernel.cu"]
            files = [f"{self.benchmark_name}_{suffix}" for suffix in suffixes]
        else:
            files = [self.ppcg_c_file]
        bin_file = self.ppcg_bin_file(target)
        self.compile(files, bin_file, compiler)
        time = self.run_bin(bin_file)
        return time


def main():
    polybench_dir = Path("~/code/polybench-c-4.2.1-beta/").expanduser()
    for benchmark in list(polybench_dir.glob("**/*.c"))[:5]:
        if benchmark.name == benchmark.parent.name + ".c":
            benchmark_dir = str(benchmark.parent).replace(str(polybench_dir), "")[1:]
            pp = PolybenchPpcg(benchmark_dir)
            orig_time = pp.run_orig("gcc")
            gcc_time = pp.run_ppcg(
                "{ kernel[0] -> tile[2048,1024]; kernel[i] -> block[16] : i != 4 }",
                # "",
                "c",
                "gcc",
            )
            cuda_time = pp.run_ppcg(
                # "{ kernel[0] -> tile[2048,1024]; kernel[i] -> block[16] : i != 4 }",
                "",
                "cuda",
                "nvcc",
            )

            print(f"{benchmark_dir}: {orig_time:10}{gcc_time:10}{cuda_time:10}")


main()
