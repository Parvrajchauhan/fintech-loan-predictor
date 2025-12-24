from typing import Dict
from Project.api.deps import fetch_and_merge_features


def get_regression_features(
    table_name: str,
    index_id: int,
    user_features: Dict
) -> Dict:
    """
    Fetch regression features from Postgres and
    override with user-provided inputs.
    """

    features = fetch_and_merge_features(
        table_name=table_name,
        index_id=index_id,
        user_features=user_features,
    )

    # Regression-specific safety defaults
    features.setdefault("avg_prev_amt_requested", 0.0)
    features.setdefault("cnt_fam_members", 1)

    return features
