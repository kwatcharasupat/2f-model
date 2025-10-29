from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


def load_df():
    df = pd.read_csv("tests/assets/sassec/reference_mms_est_profile.csv")

    # df["ref_algo"] = df["ref"].apply(lambda x: x.split("/")[0])
    df["test_algo"] = df["test"].apply(lambda x: x.split("/")[0])
    df["signal"] = df["ref"].apply(lambda x: x.split("/")[-1])
    # df["signal_t"] = df["test"].apply(lambda x: x.split("/")[-1])
    df["signal_type"] = df["signal"].apply(lambda x: x.split("_")[0])
    # df["signal_mode"] = df["signal"].apply(lambda x: x.split("_")[1])
    # df["signal_gen"] = df["signal"].apply(lambda x: x.split("_")[2])
    df["signal_num"] = df["signal"].apply(lambda x: x.split("_")[3])

    for varname in ["mms", "amd1b", "adb"]:
        df["error_" + varname] = df["computed_" + varname] - df[varname]

    return df


def plot_comparison(df: pd.DataFrame, varname: str, fullname: str, compare: str):
    f, ax = plt.subplots(figsize=(8, 8), dpi=300)
    sns.scatterplot(
        data=df,
        x=varname,
        y=f"{compare}_{varname}",
        hue="test_algo",
        style="signal_type",
        ec="none",
        alpha=0.5,
        ax=ax,
    )

    ax.set(
        xlabel="AudioLabs Reference Value",
        ylabel="Seabass Implementation"
        + (" Error" if compare == "error" else " Value"),
        title=fullname,
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02))

    plt.savefig(
        f"docs/docs/assets/validation/{varname}-{compare}.png", bbox_inches="tight"
    )


def compute_metrics(df: pd.DataFrame):
    corrs = []

    for varname, fullname in [
        ("mms", "Estimated Mean MUSHRA Score"),
        ("amd1b", "Average Modulation Difference 1 Basic"),
        ("adb", "Average Block Distortion Basic"),
    ]:
        for corr_func, corr_name in [
            (lambda x, y: stats.pearsonr(x, y)[0], "Pearson"),
            (lambda x, y: stats.spearmanr(x, y)[0], "Spearman"),
            (mean_absolute_error, "MAE"),
            (mean_squared_error, "MSE"),
            (lambda x, y: mean_absolute_percentage_error(x, y) * 100, "MA%E"),
        ]:
            corr = corr_func(df[varname], df["computed_" + varname])
            corrs.append(
                {
                    "Variable": fullname,
                    "Metric": corr_name,
                    "Value": corr,
                }
            )

    dfc = pd.DataFrame(corrs)
    dfc = dfc.pivot(index="Variable", columns="Metric", values="Value")

    markdown = dfc.to_markdown(
        floatfmt=("", ".3f", ".4f", ".7f", ".6f", ".5f"), tablefmt="github"
    )

    print(markdown)


if __name__ == "__main__":
    sns.set_theme("talk", "whitegrid")

    df = load_df()

    for varname, fullname in [
        ("mms", "Estimated Mean MUSHRA Score"),
        ("amd1b", "Average Modulation Difference 1 Basic"),
        ("adb", "Average Block Distortion Basic"),
    ]:
        plot_comparison(df, varname, fullname, "computed")
        plot_comparison(df, varname, fullname, "error")

    compute_metrics(df)
