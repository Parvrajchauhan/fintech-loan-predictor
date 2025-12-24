from typing import Dict
from Project.api.deps import fetch_and_merge_features


def get_classification_features(
    table_name: str,
    index_id: int,
    user_features: Dict
) -> Dict:
    """
    Fetch classification features from Postgres and
    override with user-provided inputs.
    """

    features = fetch_and_merge_features(
        table_name=table_name,
        index_id=index_id,
        user_features=user_features,
    )

    # Classification-specific defaults / safety
    features.setdefault("pos_num_loans", 0)
    features.setdefault("avg_cc_max_limit_used", 0.0)

    return features
