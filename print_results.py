import glob
import os

def print_latest_results(index_code: str):
    """
    查找并打印指定指数的最新买入公式和性能指标文件内容
    index_code: 例如 '000905'
    """
    # 查找最新的公式文件
    formula_files = sorted(glob.glob(f"{index_code}_best_formula_*.txt"))
    metrics_files = sorted(glob.glob(f"{index_code}_metrics_*.txt"))

    if formula_files:
        latest_formula_file = formula_files[-1]
        with open(latest_formula_file, "r", encoding="utf-8") as f:
            print("最终买入公式:\n", f.read())
    else:
        print("未找到买入公式文件")

    if metrics_files:
        latest_metrics_file = metrics_files[-1]
        with open(latest_metrics_file, "r", encoding="utf-8") as f:
            print("性能指标内容:\n", f.read())
    else:
        print("未找到性能指标文件")


if __name__ == "__main__":
    # 这里可以改成你要查看的指数代码，例如 '000905'
    print_latest_results("000905")
