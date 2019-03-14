"""Explore the experience profiling results after a small experiment set

Needs a parameter ('json' or 'csv') in order to save the timer results in the
accurate type of output

See https://docs.python.org/3.6/library/profile.html for additional details

See also https://github.com/python/cpython/blob/master/Lib/pstats.py for diving
into 'pstats' module code

"""

import argparse
import json
from pathlib import Path
import pstats
import sys

import pandas as pd

PROJECT_NAME = "geo3dfeatures"


def export_timer_to_json(xp_name):
    profiling_in_folder = Path("data", "output", xp_name, "profiling")
    profiling_out_folder = Path("data", "output", xp_name, "timers")
    for profiling_path in profiling_in_folder.iterdir():
        stats = {}
        profiling_file = profiling_path.name
        print(profiling_file)
        _, nb_points, nb_neighbors, feature_set = profiling_file.split('-')
        p = pstats.Stats(str(profiling_path))
        _, function_list = p.get_print_list([PROJECT_NAME])
        for f in function_list:
            full_printable_function = pstats.func_std_string(f)
            printable_function = full_printable_function.split("/")[-1]
            nb_cum_calls, nb_calls, total_time, cum_time, _ = p.stats[f]
            stats[printable_function] = {
                "nb_points": nb_points,
                "nb_neighbors": nb_neighbors,
                "feature_set": feature_set,
                "nb_calls": nb_calls,
                "total_time": total_time,  # the function without sub-calls
                "total_time_per_call": total_time/nb_calls,
                "cum_time": cum_time,  # with calls to sub-functions
                "cum_time_per_call": cum_time/nb_cum_calls
            }
        with open(profiling_out_folder / (profiling_file + ".json"), 'w') as f:
            json.dump(stats, f)


def export_timer_to_csv(xp_name):
    full_stats = []
    profiling_in_folder = Path("data", "output", xp_name, "profiling")
    profiling_out_folder = Path("data", "output", xp_name, "timers")
    columns = [
        "function", "nb_points", "nb_neighbors", "feature_set",
        "nb_calls", "total_time", "total_time_per_call",
        "cum_time", "cum_time_per_call"
    ]
    for profiling_path in profiling_in_folder.iterdir():
        stats = []
        profiling_file = profiling_path.name
        print(profiling_file)
        _, nb_points, nb_neighbors, feature_set = profiling_file.split('-')
        p = pstats.Stats(str(profiling_path))
        _, function_list = p.get_print_list([PROJECT_NAME])
        for f in function_list:
            full_printable_function = pstats.func_std_string(f)
            printable_function = full_printable_function.split("/")[-1]
            nb_cum_calls, nb_calls, total_time, cum_time, _ = p.stats[f]
            current_timer = [
                printable_function, nb_points, nb_neighbors, feature_set,
                nb_calls, total_time, total_time/nb_calls,
                cum_time, cum_time/nb_cum_calls
            ]
            stats.append(current_timer)
            full_stats.append(current_timer)
        df = pd.DataFrame(stats, columns=columns)
        df.to_csv(
            str(profiling_out_folder / (profiling_file + ".csv")),
            index=False
        )
    full_df = pd.DataFrame(full_stats, columns=columns)
    full_df.to_csv(
        str(profiling_out_folder / "timers.csv"),
        index=False
    )


def main(opts):
    if opts.file_format == "csv":
        export_timer_to_csv(opts.experiment)
    elif opts.file_format == "json":
        export_timer_to_json(opts.experiment)
    else:
        raise ValueError("Wrong file extension. Choose between csv and json")
