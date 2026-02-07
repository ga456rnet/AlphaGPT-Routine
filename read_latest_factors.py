#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 AlphaGPT-Routine 最新回测结果中的优选因子
放置在 main/ 目录下运行
"""

import os
import glob
import pandas as pd

def get_latest_parquet_file(data_dir="margin_balance"):
    """
    在指定目录下查找最新日期的 parquet 文件
    """
    files = glob.glob(os.path.join(data_dir, "*_margin_data.parquet"))
    if not files:
        raise FileNotFoundError("未找到任何 parquet 文件，请确认 workflow 已生成结果。")
    # 按文件名排序，取最新的
    files.sort()
    return files[-1]

def read_selected_factors(parquet_file):
    """
    从 parquet 文件中读取优选因子
    假设文件中有 'selected_factors' 或类似列
    """
    df = pd.read_parquet(parquet_file)
    # 尝试读取可能的因子列
    candidate_cols = [c for c in df.columns if "factor" in c.lower()]
    if not candidate_cols:
        raise ValueError(f"文件 {parquet_file} 中未找到因子相关列，请检查数据结构。")
    
    # 输出优选因子列表
    factors = set()
    for col in candidate_cols:
        factors.update(df[col].dropna().unique())
    
    return list(factors)

if __name__ == "__main__":
    latest_file = get_latest_parquet_file()
    print(f"最新回测文件: {latest_file}")
    factors = read_selected_factors(latest_file)
    print("优选因子列表:")
    for f in factors:
        print("-", f)
