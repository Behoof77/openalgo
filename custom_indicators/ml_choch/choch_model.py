"""
CHoCH ML Model -- Random Forest classifier for CHoCH probability prediction.

Uses scikit-learn RandomForestClassifier to predict whether a CHoCH event
will lead to a successful favorable move. Maintains a database of historical
events, retrains periodically, and computes take-profit targets from
successful neighbor events.

Why Random Forest over KNN:
- Robust to feature scale differences (volume_delta ~[-1,1], displacement ~[0,10+])
- Handles noisy financial data and outliers without overfitting
- Well-calibrated probability estimates
- Built-in feature importance (which of the 3 features drives prediction)
- Less sensitive to class imbalance in small datasets
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[assignment]

from .feature_engine import ChochEvent


class ChochMLModel:
    """Random Forest model for CHoCH signal prediction.

    Maintains an in-memory database of historical CHoCH events. On prediction,
    trains a Random Forest on matching-direction events and outputs success
    probability. Also computes TP target levels from favorable runs of
    successfully-predicted historical events.

    Attributes:
        n_trees: Number of trees in the forest
        window: Maximum database size (oldest events are evicted)
        min_events: Minimum events needed before model can predict
    """

    def __init__(
        self,
        n_trees: int = 100,
        window: int = 1500,
        min_events: int = 10,
    ) -> None:
        self.n_trees = n_trees
        self.window = window
        self.min_events = min_events
        self._database: List[ChochEvent] = []

    @property
    def size(self) -> int:
        """Current number of events in the database."""
        return len(self._database)

    def add_event(self, event: ChochEvent) -> None:
        """Add a single event to the database, evicting oldest if at capacity."""
        self._database.append(event)
        if len(self._database) > self.window:
            self._database = self._database[-self.window:]

    def add_events(self, events: List[ChochEvent]) -> None:
        """Add multiple events to the database."""
        for event in events:
            self.add_event(event)

    def fit(self, events: List[ChochEvent]) -> None:
        """Replace the database with provided events."""
        self._database = events[-self.window:]

    def _get_matching_events(
        self, direction: bool
    ) -> List[ChochEvent]:
        """Filter events by direction (bullish/bearish)."""
        return [e for e in self._database if e.is_bullish == direction]

    def _build_forest(
        self, direction: bool
    ) -> Optional[Tuple[RandomForestClassifier, List[ChochEvent]]]:
        """Build and fit a Random Forest from events matching the direction.

        Args:
            direction: True for bullish, False for bearish

        Returns:
            Tuple of (fitted_classifier, matched_events) or None if insufficient data
        """
        matched = self._get_matching_events(direction)

        if len(matched) < self.min_events:
            return None

        X = np.array([e.feature_vector() for e in matched])
        y = np.array([1.0 if e.outcome_value > 0 else 0.0 for e in matched])

        n_estimators = min(self.n_trees, max(10, len(matched) // 2))

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max(3, min(8, len(matched) // 3)),
            min_samples_split=max(2, len(matched) // 10),
            min_samples_leaf=max(1, len(matched) // 20),
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X, y)

        return clf, matched

    def predict(
        self, features: np.ndarray, direction: bool
    ) -> Tuple[float, List[ChochEvent]]:
        """Predict success probability for a new CHoCH event.

        Trains a Random Forest on matching-direction events and returns:
        - Probability: P(successful outcome) as percentage 0-100
        - Historical events with similar feature profiles (for target computation)

        Args:
            features: 3D feature vector [volume_delta, displacement_z, price_velocity]
            direction: True for bullish, False for bearish

        Returns:
            Tuple of (probability 0-100, list of relevant historical events)

        If insufficient data, returns (50.0, []) -- neutral probability.
        """
        result = self._build_forest(direction)
        if result is None:
            return 50.0, []

        clf, matched_events = result
        features_arr = np.asarray(features, dtype=np.float64).reshape(1, -1)

        # Get prediction probabilities
        prob = clf.predict_proba(features_arr)[0]
        # prob[1] = probability of positive class (successful outcome)
        success_prob = float(prob[1]) * 100.0 if len(prob) > 1 else 50.0

        # Find most similar events using the forest's tree votes
        # Use proximity: count how many trees agree with each training sample
        leaf_indices = clf.apply(features_arr)[0]  # Which leaf each tree puts this sample in
        train_leaves = clf.apply(np.array([e.feature_vector() for e in matched_events]))

        # Proximity = fraction of trees where test and training sample land in same leaf
        proximity = np.array([
            np.mean(leaf_indices == train_leaves[i])
            for i in range(len(matched_events))
        ])

        # Get top-K most similar events (by proximity)
        top_k = min(self.min_events, len(matched_events))
        top_indices = np.argsort(proximity)[-top_k:][::-1]
        similar_events = [matched_events[i] for i in top_indices]

        return success_prob, similar_events

    def compute_targets(
        self,
        current_price: float,
        neighbors: List[ChochEvent],
        direction: bool,
        scalar: float = 0.5,
    ) -> Tuple[float, float, float]:
        """Compute TP1/TP2/TP3 target levels from neighbor favorable runs.

        Uses only events with positive outcomes (successful moves) for targets:

        - TP1 (Conservative): mean(favorable_runs) * scalar
        - TP2 (Median): median(favorable_runs)
        - TP3 (Aggressive): 75th percentile of favorable_runs

        All targets are offset from current_price in the trade direction.

        Args:
            current_price: Current close price
            neighbors: Matched neighbor events from predict()
            direction: True for bullish (targets above), False for bearish (below)
            scalar: Conservative scaling factor for TP1 (default 0.5)

        Returns:
            Tuple of (tp1, tp2, tp3) price levels
        """
        if not neighbors:
            return 0.0, 0.0, 0.0

        # Only use successful events for target computation
        successful_runs = np.array([
            e.favorable_run for e in neighbors
            if e.outcome_value > 0 and e.favorable_run > 0
        ])

        if len(successful_runs) == 0:
            return 0.0, 0.0, 0.0

        mean_run = float(np.mean(successful_runs))
        median_run = float(np.median(successful_runs))
        p75_idx = int(np.round(len(successful_runs) * 0.75)) - 1
        p75_idx = max(0, min(p75_idx, len(successful_runs) - 1))
        aggr_run = float(np.sort(successful_runs)[p75_idx])

        direction_sign = 1.0 if direction else -1.0

        tp1 = current_price + direction_sign * mean_run * scalar
        tp2 = current_price + direction_sign * median_run
        tp3 = current_price + direction_sign * aggr_run

        return tp1, tp2, tp3

    def feature_importance(self, direction: bool) -> Optional[dict]:
        """Get feature importance from the trained forest.

        Args:
            direction: True for bullish, False for bearish

        Returns:
            Dict mapping feature names to importance scores, or None if not trained
        """
        result = self._build_forest(direction)
        if result is None:
            return None

        clf, _ = result
        names = ["volume_delta", "displacement_z", "price_velocity"]
        importances = clf.feature_importances_
        return dict(zip(names, importances.tolist()))

    def save(self, path: str | Path) -> None:
        """Persist the model database to disk via joblib.

        Args:
            path: File path for the saved model (.joblib or .pkl)
        """
        if joblib is None:
            raise ImportError(
                "joblib is required for model persistence. "
                "Install with: pip install joblib"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "n_trees": self.n_trees,
            "window": self.window,
            "min_events": self.min_events,
            "database": self._database,
        }, path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved model database from disk.

        Args:
            path: File path of the saved model
        """
        if joblib is None:
            raise ImportError(
                "joblib is required for model persistence. "
                "Install with: pip install joblib"
            )
        data = joblib.load(path)
        self.n_trees = data["n_trees"]
        self.window = data["window"]
        self.min_events = data.get("min_events", 10)
        self._database = data["database"]
