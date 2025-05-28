import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plotly import graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. DISMISSAL PROBABILITY ENGINE - Predicting when batsmen are vulnerable
# ==============================================================================


def build_dismissal_predictor(df):
    """Build a model to predict when batsmen are most likely to get out"""

    # Create features for each ball
    prediction_data = []

    for match_id in df["match_id"].unique():
        match_data = df[df["match_id"] == match_id].copy()

        for innings in match_data["match_innings"].unique():
            innings_data = match_data[match_data["match_innings"] == innings].copy()
            innings_data = innings_data.sort_values(
                ["over", "ball_in_over"]
            ).reset_index(drop=True)

            # Track batsman-specific metrics
            batsman_stats = {}

            for idx, ball in innings_data.iterrows():
                striker = ball["striker_id"]

                # Initialize batsman stats if first ball
                if striker not in batsman_stats:
                    batsman_stats[striker] = {
                        "balls_faced": 0,
                        "runs_scored": 0,
                        "dot_balls": 0,
                        "boundaries": 0,
                        "balls_since_boundary": 0,
                    }

                # Get current stats before this ball
                current_stats = batsman_stats[striker].copy()

                # Create features
                features = {
                    "match_id": match_id,
                    "match_innings": innings,
                    "match_format": ball["match_length_id"],
                    "over": ball["over"],
                    "ball_in_over": ball["ball_in_over"],
                    "powerplay": ball["power_play"],
                    "striker_id": striker,
                    "bowler_id": ball["bowler_id"],
                    "batsman_balls_faced": current_stats["balls_faced"],
                    "batsman_runs_scored": current_stats["runs_scored"],
                    "batsman_strike_rate": (
                        (
                            current_stats["runs_scored"]
                            * 100
                            / current_stats["balls_faced"]
                        )
                        if current_stats["balls_faced"] > 0
                        else 0
                    ),
                    "dot_ball_percentage": (
                        (
                            current_stats["dot_balls"]
                            * 100
                            / current_stats["balls_faced"]
                        )
                        if current_stats["balls_faced"] > 0
                        else 0
                    ),
                    "balls_since_boundary": current_stats["balls_since_boundary"],
                    "recent_form": (
                        1
                        if current_stats["balls_faced"] >= 10
                        and current_stats["runs_scored"] / current_stats["balls_faced"]
                        > 1
                        else 0
                    ),
                    "under_pressure": (
                        1 if current_stats["balls_since_boundary"] > 10 else 0
                    ),
                    "new_batsman": 1 if current_stats["balls_faced"] < 5 else 0,
                    "dismissed": 1 if ball["striker_dismissed"] == 1 else 0,
                }

                prediction_data.append(features)

                # Update batsman stats after this ball
                if ball["legal_ball"] == 1:
                    batsman_stats[striker]["balls_faced"] += 1
                    batsman_stats[striker]["runs_scored"] += ball["bat_score"]

                    if ball["bat_score"] == 0:
                        batsman_stats[striker]["dot_balls"] += 1
                        batsman_stats[striker]["balls_since_boundary"] += 1
                    elif ball["bat_score"] >= 4:
                        batsman_stats[striker]["boundaries"] += 1
                        batsman_stats[striker]["balls_since_boundary"] = 0
                    else:
                        batsman_stats[striker]["balls_since_boundary"] += 1

    prediction_df = pd.DataFrame(prediction_data)

    # Build the model (only if we have dismissals)
    if prediction_df["dismissed"].sum() > 0:
        # Prepare features for modeling
        feature_cols = [
            "over",
            "powerplay",
            "batsman_balls_faced",
            "batsman_strike_rate",
            "dot_ball_percentage",
            "balls_since_boundary",
            "recent_form",
            "under_pressure",
            "new_batsman",
        ]

        X = prediction_df[feature_cols].fillna(0)
        y = prediction_df["dismissed"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": rf_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Predict dismissal probability for each ball
        prediction_df["dismissal_probability"] = rf_model.predict_proba(X)[:, 1]

        # Find high-risk moments
        high_risk_balls = prediction_df[
            prediction_df["dismissal_probability"] > 0.7
        ].copy()

        # Visualize dismissal probabilities by match
        plt.figure(figsize=(15, 8))

        for i, match_id in enumerate(prediction_df["match_id"].unique()):
            plt.subplot(2, 2, i + 1)
            match_pred = prediction_df[prediction_df["match_id"] == match_id]

            # Plot probability over time
            match_pred_sorted = match_pred.sort_values(
                ["match_innings", "over", "ball_in_over"]
            )

            for i, inns in enumerate(match_pred_sorted["match_innings"].unique()):
                data = match_pred_sorted.loc[
                    match_pred_sorted["match_innings"] == inns,
                    ["dismissal_probability", "dismissed"],
                ]
                data.reset_index(inplace=True, drop=True)
                plt.plot(
                    data.index,
                    data.dismissal_probability,
                    alpha=0.7,
                    linewidth=1,
                    label=f"Innings {inns}",
                    color=sns.color_palette()[i % 10],
                )
                dismissals = data[data.dismissed == 1]
                plt.scatter(
                    dismissals.index,
                    [0.9] * len(dismissals),
                    s=100,
                    marker="x",
                    label=f"Actual Dismissal (Inns: {inns})",
                    color=sns.color_palette()[i % 10],
                )

            plt.title(f"Match {match_id} - Dismissal Risk Over Time")
            plt.xlabel("Ball Sequence")
            plt.ylabel("Dismissal Probability")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)

            if len(dismissals) > 0:
                plt.legend()

        plt.tight_layout()
        plt.show()

        return prediction_df, rf_model, feature_importance
    else:
        print("No dismissals found in dataset for modeling")
        return prediction_df, None, None


def build_df(df, match_id):
    # Track batsman-specific metrics
    prediction_data = []

    for innings in df["match_innings"].unique():
        batsman_stats = {}
        innings_data = df[df["match_innings"] == innings].copy()
        for idx, ball in innings_data.iterrows():
            striker = ball["striker_id"]

            # Initialize batsman stats if first ball
            if striker not in batsman_stats:
                batsman_stats[striker] = {
                    "balls_faced": 0,
                    "runs_scored": 0,
                    "dot_balls": 0,
                    "boundaries": 0,
                    "balls_since_boundary": 0,
                }

            # Get current stats before this ball
            current_stats = batsman_stats[striker].copy()

            # Create features
            features = {
                "match_id": match_id,
                "match_format": ball["match_length_id"],
                "match_innings": innings,
                "over": ball["over"],
                "ball_in_over": ball["ball_in_over"],
                "powerplay": ball["power_play"],
                "striker_id": striker,
                "bowler_id": ball["bowler_id"],
                "batsman_balls_faced": current_stats["balls_faced"],
                "batsman_runs_scored": current_stats["runs_scored"],
                "batsman_strike_rate": (
                    (current_stats["runs_scored"] * 100 / current_stats["balls_faced"])
                    if current_stats["balls_faced"] > 0
                    else 0
                ),
                "dot_ball_percentage": (
                    (current_stats["dot_balls"] * 100 / current_stats["balls_faced"])
                    if current_stats["balls_faced"] > 0
                    else 0
                ),
                "balls_since_boundary": current_stats["balls_since_boundary"],
                "recent_form": (
                    1
                    if current_stats["balls_faced"] >= 10
                    and current_stats["runs_scored"] / current_stats["balls_faced"] > 1
                    else 0
                ),
                "under_pressure": (
                    1 if current_stats["balls_since_boundary"] > 10 else 0
                ),
                "new_batsman": 1 if current_stats["balls_faced"] < 5 else 0,
                "dismissed": 1 if ball["striker_dismissed"] == 1 else 0,
            }

            prediction_data.append(features)

            # Update batsman stats after this ball
            if ball["legal_ball"] == 1:
                batsman_stats[striker]["balls_faced"] += 1
                batsman_stats[striker]["runs_scored"] += ball["bat_score"]

                if ball["bat_score"] == 0:
                    batsman_stats[striker]["dot_balls"] += 1
                    batsman_stats[striker]["balls_since_boundary"] += 1
                elif ball["bat_score"] >= 4:
                    batsman_stats[striker]["boundaries"] += 1
                    batsman_stats[striker]["balls_since_boundary"] = 0
                else:
                    batsman_stats[striker]["balls_since_boundary"] += 1

    return pd.DataFrame(prediction_data)


def predict_dismissal_probability(model, df):
    """
    Predict dismissal probability for each ball, returning probabilities and a Plotly figure.
    """
    if model is None:
        print("No model provided.")
        return None, None

    df = build_df(df, df["match_id"].iloc[0])

    print(df)

    feature_cols = [
        "over",
        "powerplay",
        "batsman_balls_faced",
        "batsman_strike_rate",
        "dot_ball_percentage",
        "balls_since_boundary",
        "recent_form",
        "under_pressure",
        "new_batsman",
    ]

    # Prepare Plotly figure for progressive prediction
    fig = go.Figure()
    for i, inns in enumerate(df["match_innings"].unique()):
        X = df.loc[df["match_innings"] == inns, feature_cols].fillna(0)
        y = df.loc[df["match_innings"] == inns, "dismissed"]

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        # Predict probabilities
        proba = model.predict_proba(X)[:, 1]

        # Convert seaborn color to hex for Plotly
        color = sns.color_palette("tab10")[i % 10]
        color_hex = '#%02x%02x%02x' % tuple(int(255 * c) for c in color)

        fig.add_trace(
            go.Scatter(
            x=list(range(len(proba))),
            y=proba,
            mode="lines",
            name=f"Dismissal Probability (Inns {inns})",
            line=dict(color=color_hex),
            marker=dict(size=4),
            )
        )

        # add crosses at real wicket balls
        wicket_indices = y[y == 1].index
        if not wicket_indices.empty:
            fig.add_trace(
                go.Scatter(
                    x=wicket_indices,
                    y=[0.9] * len(wicket_indices),  # Place crosses at y=0.9
                    mode="markers",
                    name=f"Actual Dismissal (Inns {inns})",
                    marker=dict(color=color_hex, size=10, symbol="x"),
                )
            )

    fig.update_layout(
        title="Progressive Dismissal Probability per Ball",
        xaxis_title="Ball Sequence",
        yaxis_title="Dismissal Probability",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )

    return proba, fig


def evaluate_dismissal_model(model, X_test, y_test):
    """Evaluate the dismissal prediction model"""
    if model is None:
        print("No model to evaluate.")
        return

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Not Dismissed", "Dismissed"],
        yticklabels=["Not Dismissed", "Dismissed"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/smil0041/Projects/Personal/ca-task/data/enhanced_cricket_data.csv"
    )  # Load your dataset here
    dismissal_df, dismissal_model, feature_importance = build_dismissal_predictor(df)

    # Evaluate the model
    if dismissal_model is not None:
        feature_cols = [
            "over",
            "powerplay",
            "batsman_balls_faced",
            "batsman_strike_rate",
            "dot_ball_percentage",
            "balls_since_boundary",
            "recent_form",
            "under_pressure",
            "new_batsman",
        ]
        X = dismissal_df[feature_cols].fillna(0)
        y = dismissal_df["dismissed"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        evaluate_dismissal_model(dismissal_model, X_test, y_test)
    else:
        print("No model to evaluate.")

    # save model and feature importance
    if dismissal_model is not None:
        import joblib

        joblib.dump(dismissal_model, "dismissal_predictor_model.pkl")
        feature_importance.to_csv("dismissal_feature_importance.csv", index=False)
        print("Model and feature importance saved successfully.")
    else:
        print("No model to save.")
