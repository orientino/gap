import numpy as np
import pandas as pd

metric2text = {
    "tr_loss_min": "Final train loss (MIN)",
    "tr_loss_avg": "Final train loss (AVG)",
    "tr_loss_ema": "Final train loss",
    "tr_loss": "Final train loss",
    "vl_loss": "Final val loss",
    "vl_loss_min": "Final val loss (MIN)",
    "vl_loss_avg": "Final val loss (AVG)",
    "vl_loss_ema": "Final val loss",
    "vl_acc": "Final val acc",
}


def add_aggregate_loss(df, window=0.01, ema_alpha=0.05):
    df["tr_loss_history"] = df["tr_loss_history"].apply(eval)
    df["tr_loss_history"] = df["tr_loss_history"].map(
        lambda lst: (
            [
                np.nan if (isinstance(v, str) and v.strip().lower() == "nan") else v
                for v in lst
            ]
            if isinstance(lst, list)
            else lst
        )
    )
    df["tr_loss_ema"] = df["tr_loss_history"].apply(lambda x: pd.Series(x).ewm(alpha=ema_alpha, adjust=False).mean().iloc[-1])  # fmt: skip
    df["tr_loss_min"] = df["tr_loss_history"].apply(lambda x: np.min(x[-int(window * len(x)):]))  # fmt: skip
    df["tr_loss_avg"] = df["tr_loss_history"].apply(lambda x: np.mean(x[-int(window * len(x)):]))  # fmt: skip
    return df
