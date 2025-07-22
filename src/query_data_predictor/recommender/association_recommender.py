import pandas as pd
from typing import List, Dict, Any
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import logging

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)

class AssociationRecommender(BaseRecommender):
    """
    Recommender that uses association rules to predict the next query.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assoc_config = self.config.get('association_rules', {})
        self.min_support = assoc_config.get('min_support', 0.1)
        self.min_confidence = assoc_config.get('min_threshold', 0.5)
        self.metric = assoc_config.get('metric', 'confidence')
        self.max_unique_values = assoc_config.get('max_unique_values', 15)

    def recommend_tuples(self, current_results: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Recommend next queries based on the user's query history.
        """
        self._validate_input(current_results)
        if current_results.empty or len(current_results) < 2:
            return pd.DataFrame(columns=current_results.columns)

        # Limit unique values to prevent combinatorial explosion
        df_limited = self._limit_unique_values(current_results)

        # Mine association rules from the query history
        rules = self._mine_rules_from_history(df_limited)

        if rules.empty:
            return pd.DataFrame(columns=current_results.columns)

        # Get the set of items from the last row of the dataframe
        query_sequence = self._df_to_query_sequence(df_limited)
        current_items = set(query_sequence[-1])

        # Find rules where the antecedent is a subset of the current items
        matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(current_items))]

        if matching_rules.empty:
            return pd.DataFrame(columns=current_results.columns)

        # Get the consequents of the matching rules
        recommendations = []
        for _, rule in matching_rules.iterrows():
            consequents = rule['consequents']
            # Ensure we don't recommend items already present
            if not any(item in current_items for item in consequents):
                recommendations.append(consequents)

        if not recommendations:
            return pd.DataFrame(columns=current_results.columns)

        # Convert the recommendations to a dataframe
        recs_df = self._recommendations_to_df(recommendations, list(current_results.columns))
        return self._limit_output(recs_df)

    def _limit_unique_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_limited = df.copy()
        for col in df_limited.columns:
            if df_limited[col].nunique() > self.max_unique_values:
                top_values = df_limited[col].value_counts().nlargest(self.max_unique_values - 1).index
                df_limited[col] = df_limited[col].where(df_limited[col].isin(top_values), 'OTHER')
        return df_limited

    def _mine_rules_from_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mine association rules from a DataFrame.
        """
        query_sequence = self._df_to_query_sequence(df)
        if not query_sequence:
            return pd.DataFrame()

        te = TransactionEncoder()
        te_ary = te.fit(query_sequence).transform(query_sequence)
        encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = fpgrowth(encoded_df, min_support=self.min_support, use_colnames=True)

        if frequent_itemsets.empty:
            return pd.DataFrame()

        rules = association_rules(frequent_itemsets, metric=self.metric, min_threshold=self.min_confidence)
        return rules

    def _df_to_query_sequence(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Convert a DataFrame to a query sequence with column names prepended.
        """
        df_with_names = df.copy()
        for column in df_with_names.columns:
            df_with_names[column] = df_with_names[column].apply(lambda x: f"{column}_{x}")
        return df_with_names.astype(str).values.tolist()

    def _recommendations_to_df(self, recommendations: List[frozenset], columns: list) -> pd.DataFrame:
        """
        Convert a list of frozenset recommendations back to a DataFrame.
        """
        recs_list = []
        for rec in recommendations:
            rec_dict = {}
            for item in rec:
                parts = str(item).split('_', 1)
                if len(parts) == 2:
                    col, val = parts
                    if col in columns:
                        rec_dict[col] = val
            if rec_dict:
                recs_list.append(rec_dict)

        return pd.DataFrame(recs_list, columns=columns)

    def name(self) -> str:
        return "AssociationRecommender"