from pathlib import Path

import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

# Page configuration


DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data
def load_data():
    """Load and cache the cricket dataset"""
    try:
        df = pd.read_csv(DATA_DIR / "enhanced_cricket_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def calculate_match_summary(df, match_id):
    """Calculate match summary statistics"""
    match_data = df[df["match_id"] == match_id].copy()

    # Basic match info
    match_info = {
        "match_date": (
            match_data["match_date"].iloc[0] if not match_data.empty else "N/A"
        ),
        "team_a": (
            match_data["team_name_team_a"].iloc[0] if not match_data.empty else "N/A"
        ),
        "team_b": (
            match_data["team_name_team_b"].iloc[0] if not match_data.empty else "N/A"
        ),
        "venue_id": match_data["venue_id"].iloc[0] if not match_data.empty else "N/A",
        "match_result": parse_match_results(
            match_data[
                [
                    "team_name_team_a",
                    "team_a_match_result_id",
                    "team_name_team_b",
                    "team_b_match_result_id",
                ]
            ].iloc[0],
        ),
        "toss_result": parse_toss_results(
            match_data[["toss_won_by_team_id", "toss_decision_id"]].iloc[0],
        ),
    }

    # Calculate team scores by innings
    team_scores = {}
    for innings in match_data["match_innings"].unique():
        innings_data = match_data[match_data["match_innings"] == innings]
        if not innings_data.empty:
            team_name = innings_data["team_name"].iloc[0]
            total_runs = (
                innings_data["bat_score"].sum()
                + innings_data["wide_runs"].sum()
                + innings_data["noball_runs"].sum()
                + innings_data["bye_runs"].sum()
                + innings_data["legbye_runs"].sum()
                + innings_data["penalty_runs"].sum()
            )

            wickets = (
                innings_data["striker_dismissed"].sum()
                + innings_data["nonstriker_dismissed"].sum()
            )
            overs = (
                innings_data["over"].max() - 1 if "over" in innings_data.columns else 0
            )
            balls = (
                innings_data.loc[
                    innings_data["over"] == overs + 1, "fair_ball_in_over"
                ].max()
                if innings_data["over"].max() == innings_data["over"].iloc[-1]
                else 0
            )

            if balls == 6:
                overs += 1
                balls = 0

            team_scores[f"{team_name}_innings_{innings}"] = {
                "runs": int(total_runs),
                "wickets": int(wickets),
                "overs": f"{overs}.{balls}" if balls > 0 else str(overs),
            }

    return match_info, team_scores


def calculate_player_stats(df, match_id):
    """Calculate player statistics for batting and bowling"""
    match_data = df[df["match_id"] == match_id].copy()

    # Batting statistics
    batting_stats = []
    for striker_id in match_data["striker_id"].unique():
        if pd.notna(striker_id):
            player_balls = match_data[match_data["striker_id"] == striker_id]
            runs = player_balls["bat_score"].sum()
            balls_faced = len(player_balls[player_balls["legal_ball"] == 1])
            dismissed = player_balls["striker_dismissed"].sum() > 0
            fours = len(
                player_balls[
                    (player_balls["bat_score"] == 4)
                    | (player_balls["reached_boundary"] == 1)
                ]
            )
            sixes = len(player_balls[player_balls["bat_score"] == 6])

            batting_stats.append(
                {
                    "player_id": striker_id,
                    "team_name": player_balls["team_name"].iloc[0],
                    "runs": int(runs),
                    "balls": int(balls_faced),
                    "strike_rate": (
                        round((runs / balls_faced) * 100, 2) if balls_faced > 0 else 0
                    ),
                    "fours": int(fours),
                    "sixes": int(sixes),
                    "dismissed": dismissed,
                }
            )

    # Bowling statistics
    bowling_stats = []
    for bowler_id in match_data["bowler_id"].unique():
        if pd.notna(bowler_id):
            bowler_balls = match_data[match_data["bowler_id"] == bowler_id]
            runs_conceded = (
                bowler_balls["bat_score"].sum()
                + bowler_balls["wide_runs"].sum()
                + bowler_balls["noball_runs"].sum()
                + bowler_balls["bye_runs"].sum()
                + bowler_balls["legbye_runs"].sum()
            )
            overs_bowled = len(bowler_balls[bowler_balls["legal_ball"] == 1]) / 6
            wickets = (
                bowler_balls["striker_dismissed"].sum()
                + bowler_balls["nonstriker_dismissed"].sum()
            )

            bowling_stats.append(
                {
                    "player_id": bowler_id,
                    "team_name": bowler_balls["team_name_bowling"].iloc[0],
                    "overs": round(overs_bowled, 1),
                    "runs": int(runs_conceded),
                    "wickets": int(wickets),
                    "economy": (
                        round(runs_conceded / overs_bowled, 2)
                        if overs_bowled > 0
                        else 0
                    ),
                }
            )

    return pd.DataFrame(batting_stats), pd.DataFrame(bowling_stats)


def create_progressive_score_chart(df, match_id):
    """Create progressive score comparison chart with wicket dots"""
    match_data = df[df["match_id"] == match_id].copy()

    # Calculate cumulative scores by over for each team
    progressive_data = []
    wicket_data = []

    for innings in match_data["match_innings"].unique():
        innings_data = match_data[match_data["match_innings"] == innings].sort_values(
            ["match_innings", "over", "ball_in_over", "fair_ball_in_over"]
        )
        team_name = (
            innings_data["team_name"].iloc[0]
            if not innings_data.empty
            else f"Team {innings}"
        )

        cumulative_runs = 0
        for idx, row in innings_data.iterrows():
            ball_runs = (
                row["bat_score"]
                + row["wide_runs"]
                + row["noball_runs"]
                + row["bye_runs"]
                + row["legbye_runs"]
                + row["penalty_runs"]
            )
            cumulative_runs += ball_runs

            over_ball = f"{row['over']}.{row['ball_in_over']}"
            progressive_data.append(
                {
                    "team": team_name,
                    "innings": innings,
                    "over_ball": over_ball,
                    "over": row["over"] + (row["ball_in_over"] / 6),
                    "cumulative_runs": cumulative_runs,
                }
            )

            # Add wicket dot if wicket falls on this ball
            if row.get("striker_dismissed", 0) or row.get("nonstriker_dismissed", 0):
                wicket_data.append(
                    {
                        "team": team_name,
                        "over": row["over"] + (row["ball_in_over"] / 6),
                        "cumulative_runs": cumulative_runs,
                    }
                )

    prog_df = pd.DataFrame(progressive_data)
    wicket_df = pd.DataFrame(wicket_data)

    if not prog_df.empty:
        fig = px.line(
            prog_df,
            x="over",
            y="cumulative_runs",
            color="team",
            title="Progressive Score Comparison",
            labels={"over": "Overs", "cumulative_runs": "Cumulative Runs"},
        )
        # Add dots for wickets
        if not wicket_df.empty:
            for team in wicket_df["team"].unique():
                team_wickets = wicket_df[wicket_df["team"] == team]
                fig.add_scatter(
                    x=team_wickets["over"],
                    y=team_wickets["cumulative_runs"],
                    mode="markers",
                    marker=dict(symbol="circle", size=10, color="red"),
                    name=f"{team} Wickets",
                    showlegend=True,
                )
        fig.update_layout(height=400)
        return fig
    return None


def create_bowling_pitch_map(df, match_id, team):
    """Create a bowling pitch map for the specified team"""
    match_data = df[df["match_id"] == match_id].copy()
    team_data = match_data[match_data["team_name_bowling"] == team]

    pitch_length_min, pitch_length_max = (
        0,
        22000,
    )  # typical pitch length range (from stumps)
    pitch_line_min, pitch_line_max = (
        -1750,
        1750,
    )  # typical pitch line range (across stumps)
    stump_lines = [-250, 0, 250]
    crease_line = 1220

    if team_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw pitch rectangle (20.12m = 2012mm, but use 2200mm for margin)
    ax.add_patch(
        plt.Rectangle(
            (pitch_line_min, pitch_length_min),
            pitch_line_max - pitch_line_min,
            pitch_length_max - pitch_length_min,
            edgecolor="saddlebrown",
            facecolor="#ffffff",
            lw=2,
            zorder=0,
            alpha=1,
        )
    )
    # Draw stumps at 0 mm line
    ax.plot([pitch_line_min, pitch_line_max], [0, 0], color="black", lw=3, zorder=1)
    # Draw crease (assume at 0 and 2200mm)
    ax.plot(
        [pitch_line_min, pitch_line_max],
        [0, 0],
        color="gray",
        lw=1,
        linestyle="--",
        zorder=1,
    )
    ax.plot(
        [pitch_line_min, pitch_line_max],
        [pitch_length_max, pitch_length_max],
        color="gray",
        lw=1,
        linestyle="--",
        zorder=1,
    )
    # Draw stump lines
    for line in stump_lines:
        ax.plot(
            [line, line],
            [pitch_length_min, pitch_length_max],
            color="black",
            lw=1,
            linestyle="--",
            zorder=4,
        )
    # Draw horizontal crease line
    ax.hlines(
        crease_line,
        pitch_line_min,
        pitch_line_max,
        color="black",
        lw=1,
        linestyle="--",
        zorder=4,
    )

    sns.kdeplot(
        x=team_data["pitch_line"],
        y=team_data["pitch_length"],
        label="Bowling Density",
        fill=True,
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        bw_adjust=0.5,
        thresh=0.05,
        levels=100,
        legend=True,
        ax=ax,
    )

    sns.scatterplot(
        x=team_data["pitch_line"],
        y=team_data["pitch_length"],
        c="#090E11",
        alpha=0.2,
        s=50,
        ax=ax,
        legend=False,
        zorder=2,
    )

    wicket_balls = team_data[
        (team_data["striker_dismissed"] == 1) & (team_data["legal_ball"] == 1)
    ]
    ax.scatter(
        wicket_balls["pitch_line"],
        wicket_balls["pitch_length"],
        marker="x",
        color="black",
        s=100,
        label="Wicket",
        zorder=3,
    )

    if not wicket_balls.empty:
        ax.legend(loc="upper right")

    ax.set_xlim(pitch_line_min, pitch_line_max)
    ax.set_ylim(
        pitch_length_max, pitch_length_min - 1850
    )  # invert y-axis for cricket view
    ax.set_title(f"Pitch Heatmap - Match {match_id}, Team {team}")
    ax.set_xlabel("Pitch Line (mm from middle stump)")
    ax.set_ylabel("Pitch Length (mm from stumps)")
    plt.legend()
    # reverse x-axis
    ax.invert_xaxis()
    plt.tight_layout()

    # Set background to match Streamlit container background (#0e1117) and white lines/markers
    # ax.set_facecolor("#0730090D")
    # fig.patch.set_facecolor("#0730090D")
    # for spine in ax.spines.values():
    #     spine.set_color("white")
    # ax.title.set_color("white")
    # ax.xaxis.label.set_color("white")
    # ax.yaxis.label.set_color("white")
    # ax.tick_params(axis="x", colors="white")
    # ax.tick_params(axis="y", colors="white")

    # # Change all lines and markers to white
    # for line in ax.get_lines():
    #     line.set_color("white")
    # for text in ax.texts:
    #     text.set_color("white")

    # Save the pitch map as an image file
    output_path = DATA_DIR / f"{team}_bowling_pitch_map.png"
    plt.savefig(output_path, dpi=750)
    plt.close(fig)
    return fig


def parse_match_results(row: pd.Series) -> str:
    lookup = pd.read_csv(DATA_DIR / "Lookups.csv")
    team_a_result = lookup.loc[
        (lookup["lookup_type_id"] == 2804)
        & (lookup["id"] == row["team_a_match_result_id"]),
        "description",
    ].values[0]
    team_b_result = lookup.loc[
        (lookup["lookup_type_id"] == 2804)
        & (lookup["id"] == row["team_b_match_result_id"]),
        "description",
    ].values[0]
    if team_a_result == "Won T20" and team_b_result == "Lost T20":
        return f"{row['team_name_team_a']} Won"
    elif team_a_result == "Lost T20" and team_b_result == "Won T20":
        return f"{row['team_name_team_b']} Won"
    elif team_a_result == "Won (Outright)" and team_b_result == "Lost (Outright)":
        return f"{row['team_name_team_a']} Won"
    elif team_a_result == "Lost (Outright)" and team_b_result == "Won (Outright)":
        return f"{row['team_name_team_b']} Won"
    else:
        return "Match Tied"


def parse_toss_results(row: pd.Series) -> str:
    lookup = pd.read_csv(DATA_DIR / "Lookups.csv")
    teams = pd.read_csv(DATA_DIR / "Teams.csv")

    toss_winner = row["toss_won_by_team_id"]
    toss_winner = teams.loc[teams["team_id"] == toss_winner, "team_name"].values[0]

    toss_decision = lookup.loc[
        (lookup["lookup_type_id"] == 2802) & (lookup["id"] == row["toss_decision_id"]),
        "description",
    ].values[0]
    return f"{toss_winner} chose {toss_decision.lower()}"


def calculate_momentum_tracking(df, match_id, n_block=3):
    """Calculate run rates in 5-over blocks for momentum analysis with team comparison"""
    match_data = df[df["match_id"] == match_id].copy()

    momentum_data = []

    for innings in match_data["match_innings"].unique():
        innings_data = match_data[match_data["match_innings"] == innings]

        for team in innings_data["team_batting_id"].unique():
            batting_innings_data = innings_data[innings_data["team_batting_id"] == team]

            # Get team names for comparison
            batting_team = batting_innings_data["team_name"].iloc[0]
            bowling_team = batting_innings_data["team_name_bowling"].iloc[0]

            # Create 5-over blocks
            batting_innings_data = batting_innings_data.copy()
            batting_innings_data["over_block"] = (
                (batting_innings_data["over"] - 1) // n_block
            ) + 1

            for block in batting_innings_data["over_block"].unique():
                block_data = batting_innings_data[
                    batting_innings_data["over_block"] == block
                ]

                # Calculate metrics for this block
                total_runs = block_data["bat_score"].sum()
                total_balls = len(block_data[block_data["legal_ball"] == 1])
                wickets = (
                    block_data["striker_dismissed"].sum()
                    + block_data["nonstriker_dismissed"].sum()
                )

                run_rate = (total_runs * 6 / total_balls) if total_balls > 0 else 0

                momentum_data.append(
                    {
                        "match_id": match_id,
                        "match_innings": innings,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "over_block": block,
                        "overs": f"{(block-1)*5 + 1}-{min(block*5, batting_innings_data['over'].max())}",
                        "runs": total_runs,
                        "balls": total_balls,
                        "wickets": wickets,
                        "run_rate": run_rate,
                        "match_name": f"{match_data['team_name_team_a'].iloc[0]} vs {match_data['team_name_team_b'].iloc[0]}",
                        "match_result": parse_match_results(
                            match_data[
                                [
                                    "team_name_team_a",
                                    "team_a_match_result_id",
                                    "team_name_team_b",
                                    "team_b_match_result_id",
                                ]
                            ].iloc[0],
                        ),
                    }
                )

    momentum_df = pd.DataFrame(momentum_data)

    # Visualize momentum comparison between teams
    fig = px.line(
        momentum_df,
        x="over_block",
        y="run_rate",
        color="batting_team",
        title="Team Momentum Comparison",
        labels={
            "over_block": "Over Block (5-over phases)",
            "run_rate": "Run Rate",
            "batting_team": "Batting Team",
        },
    )

    if len(momentum_df["match_innings"].unique()) == 2:
        team1_data = momentum_df[momentum_df["match_innings"] == 1]
        team2_data = momentum_df[momentum_df["match_innings"] == 2]

        # Find common over blocks
        common_blocks = set(team1_data["over_block"]).intersection(
            set(team2_data["over_block"])
        )

        for block in common_blocks:
            team1_rr = team1_data[team1_data["over_block"] == block]["run_rate"].iloc[0]
            team2_rr = team2_data[team2_data["over_block"] == block]["run_rate"].iloc[0]
            diff = team2_rr - team1_rr

            fig.add_annotation(
                x=block,
                y=max(team1_rr, team2_rr) + 0.5,
                text=f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}",
                showarrow=False,
                font=dict(
                    size=12, color="green" if diff > 0 else "red", family="Arial Black"
                ),
                align="center",
            )

    plt.tight_layout()

    return fig if fig is not None else None


def identify_match_defining_moments(df):
    """Identify the specific balls that had the biggest impact on match outcomes"""

    print("\n=== 2. MICRO-MOMENT ANALYSIS ===")

    decisive_moments = []
    match_data = df.copy()
    match_format = (
        "T20"
        if match_data["match_length_id"].iloc[0] == 7
        else "Test Match" if not match_data.empty else "Unknown"
    )
    match_id = match_data["match_id"].iloc[0] if not match_data.empty else "N/A"

    for innings in match_data["match_innings"].unique():
        innings_data = match_data[match_data["match_innings"] == innings].copy()
        innings_data = innings_data.sort_values(
            ["match_innings", "over", "ball_in_over"]
        ).reset_index(drop=True)

        cumulative_runs = 0
        cumulative_wickets = 0
        balls_bowled = 0

        for idx, ball in innings_data.iterrows():
            if ball["legal_ball"] == 1:
                balls_bowled += 1

            cumulative_runs += ball["bat_score"]
            cumulative_wickets += (
                ball["striker_dismissed"] + ball["nonstriker_dismissed"]
            )

            # Calculate match pressure metrics
            current_rr = (cumulative_runs * 6 / balls_bowled) if balls_bowled > 0 else 0
            wickets_remaining = 10 - cumulative_wickets

            # Identify high-impact moments
            impact_factors = []
            impact_score = 0

            # Wicket impact
            if ball["striker_dismissed"] == 1 or ball["nonstriker_dismissed"] == 1:
                impact_factors.append("Wicket")
                impact_score += 3

            # Boundary impact
            if ball["bat_score"] >= 4:
                impact_factors.append("Boundary")
                impact_score += 2

            # Pressure situation impact
            if match_format in ["T20", "ODI"]:  # Limited overs
                if ball["over"] >= 18:  # Death overs
                    impact_factors.append("Death_Overs")
                    impact_score += 2
                elif ball["power_play"] == 1:  # Powerplay
                    impact_factors.append("Powerplay")
                    impact_score += 1

            # Low run rate pressure
            if balls_bowled >= 30 and current_rr < 4:
                impact_factors.append("Low_RR_Pressure")
                impact_score += 1

            # High wicket loss pressure
            if wickets_remaining <= 3:
                impact_factors.append("Tail_Pressure")
                impact_score += 2

            # Store moment if significant
            if impact_score >= 3:
                decisive_moments.append(
                    {
                        "match_id": match_id,
                        "match_name": match_data["match_name"].iloc[0],
                        "match_format": match_format,
                        "innings": innings,
                        "team_batting": ball["team_name"],
                        "team_bowling": ball["team_name_bowling"],
                        "over": ball["over"],
                        "ball": ball["fair_ball_in_over"],
                        "striker_id": ball["striker_id"],
                        "bowler_id": ball["bowler_id"],
                        "runs_scored": ball["bat_score"],
                        "wicket_taken": ball["striker_dismissed"]
                        + ball["nonstriker_dismissed"],
                        "match_situation": f"{cumulative_runs}/{cumulative_wickets} (RR: {current_rr:.1f})",
                        "impact_factors": ", ".join(impact_factors),
                        "impact_score": impact_score,
                        "balls_bowled": balls_bowled,
                        "pressure_index": impact_score
                        * (1 + (10 - wickets_remaining) * 0.1),
                    }
                )

    decisive_df = pd.DataFrame(decisive_moments)

    if len(decisive_df) > 0:
        print(f"\nFound {len(decisive_df)} high-impact moments across all matches")

        # Top 10 most decisive moments
        top_moments = decisive_df.nlargest(10, "pressure_index")
        print("\nTop 10 Most Decisive Moments:")
        for idx, moment in top_moments.iterrows():
            print(
                f"Match {moment['match_id']}, Over {moment['over']}.{moment['ball']}: "
                f"{moment['match_situation']} - {moment['impact_factors']} "
                f"(Impact: {moment['impact_score']})"
            )

        # Impact distribution
        fig1 = plt.figure(figsize=(12, 5))
        plt.hist(decisive_df["impact_score"], bins=5, alpha=0.7, color="lightgreen")
        plt.title("Distribution of Impact Scores")
        plt.xlabel("Impact Score")
        plt.ylabel("Frequency")
        plt.gca().tick_params(axis="x", colors="white")
        plt.gca().tick_params(axis="y", colors="white")
        plt.gca().xaxis.label.set_color("white")
        plt.gca().yaxis.label.set_color("white")
        plt.gca().title.set_color("white")
        plt.gca().spines["bottom"].set_color("white")
        plt.gca().spines["top"].set_color("white")
        plt.gca().spines["left"].set_color("white")
        plt.gca().spines["right"].set_color("white")
        plt.tight_layout()

        # Timing of decisive moments
        fig2 = plt.figure(figsize=(12, 5))
        # plot the pressure index by over per team
        sns.lineplot(
            x="over",
            y="pressure_index",
            hue="team_batting",
            data=decisive_df,
            palette="Set2",
        )

        plt.title("Pressure Index by Over")
        plt.xlabel("Over")
        plt.ylabel("Pressure Index")
        plt.gca().tick_params(axis="x", colors="white")
        plt.gca().tick_params(axis="y", colors="white")
        plt.gca().xaxis.label.set_color("white")
        plt.gca().yaxis.label.set_color("white")
        plt.gca().title.set_color("white")
        plt.gca().spines["bottom"].set_color("white")
        plt.gca().spines["top"].set_color("white")
        plt.gca().spines["left"].set_color("white")
        plt.gca().spines["right"].set_color("white")
        # Set legend text and title color to white
        plt.legend(
            title="Batting Team",
            title_fontsize="13",
            fontsize="11",
            loc="upper left",
            frameon=False,
        )
        plt.setp(plt.gca().get_legend().get_texts(), color="white")
        plt.setp(plt.gca().get_legend().get_title(), color="white")
        plt.tight_layout()

        # Impact factors frequency
        fig3 = plt.figure(figsize=(12, 8))
        # Count frequency of each impact factor per team
        impact_factors = (
            decisive_df["impact_factors"]
            .str.get_dummies(sep=", ")
            .groupby(decisive_df["team_batting"])
            .sum()
        )
        impact_factors = impact_factors.T
        impact_factors.reset_index(inplace=True, drop=False, names="impact_factor")
        print(impact_factors)
        # Plot a barplot for each team (each column except 'impact_factor')
        for idx, team in enumerate(impact_factors.columns[1:]):
            sns.barplot(
                x="impact_factor",
                y=team,
                data=impact_factors,
                color=sns.color_palette("Set2")[idx],
                orient="v",
                label=team,
            )
        plt.xlabel("Impact Factor")
        plt.ylabel("Frequency")
        plt.title("Impact Factors Frequency by Team")
        plt.gca().tick_params(axis="x", colors="white")
        plt.gca().tick_params(axis="y", colors="white")
        plt.gca().xaxis.label.set_color("white")
        plt.gca().yaxis.label.set_color("white")
        plt.gca().title.set_color("white")
        plt.gca().spines["bottom"].set_color("white")
        plt.gca().spines["top"].set_color("white")
        plt.gca().spines["left"].set_color("white")
        plt.gca().spines["right"].set_color("white")
        plt.xticks(rotation=45)
        plt.tight_layout()

        return top_moments, [fig1, fig2, fig3]
    else:
        print("No high-impact moments found")
        return pd.DataFrame()


def analyze_bowling_changes(df, match_id):
    """Analyze the impact of bowling changes on match flow"""

    bowling_changes = []
    match_data = df.copy()

    for innings in match_data["match_innings"].unique():
        innings_data = match_data[match_data["match_innings"] == innings].copy()
        innings_data = innings_data.sort_values(
            ["match_innings", "over", "ball_in_over"]
        )

        prev_bowler = None
        current_spell_stats = {"runs": 0, "balls": 0, "dots": 0, "wickets": 0}

        for idx, ball in innings_data.iterrows():
            current_bowler = ball["bowler_id"]

            # Detect bowling change
            if prev_bowler is not None and current_bowler != prev_bowler:
                print("Change detected at ball index:", idx)
                # Analyze impact of the change
                # Look at next 12 balls (2 overs) after change
                remaining_balls = innings_data.iloc[idx : idx + 12]

                if len(remaining_balls) >= 6:  # At least 1 over of data
                    legal_balls = remaining_balls[remaining_balls["legal_ball"] == 1]

                    post_change_stats = {
                        "runs_conceded": legal_balls["bat_score"].sum(),
                        "balls_bowled": len(legal_balls),
                        "dot_balls": len(legal_balls[legal_balls["bat_score"] == 0]),
                        "wickets_taken": legal_balls["striker_dismissed"].sum()
                        + legal_balls["nonstriker_dismissed"].sum(),
                        "economy": (
                            legal_balls["bat_score"].sum() * 6 / len(legal_balls)
                            if len(legal_balls) > 0
                            else 0
                        ),
                        "bowling_team": ball["team_name_bowling"],
                    }

                    # Compare with previous bowler's performance
                    prev_economy = (
                        current_spell_stats["runs"] * 6 / current_spell_stats["balls"]
                        if current_spell_stats["balls"] > 0
                        else 0
                    )

                    bowling_changes.append(
                        {
                            "match_id": match_id,
                            "innings": innings,
                            "change_over": ball["over"],
                            "prev_bowler": prev_bowler,
                            "new_bowler": current_bowler,
                            "prev_bowler_economy": prev_economy,
                            "prev_bowler_dot_pct": (
                                (
                                    current_spell_stats["dots"]
                                    * 100
                                    / current_spell_stats["balls"]
                                )
                                if current_spell_stats["balls"] > 0
                                else 0
                            ),
                            "post_change_economy": post_change_stats["economy"],
                            "post_change_dot_pct": (
                                (
                                    post_change_stats["dot_balls"]
                                    * 100
                                    / post_change_stats["balls_bowled"]
                                )
                                if post_change_stats["balls_bowled"] > 0
                                else 0
                            ),
                            "economy_improvement": prev_economy
                            - post_change_stats["economy"],
                            "dot_pct_improvement": (
                                current_spell_stats["dots"]
                                * 100
                                / current_spell_stats["balls"]
                                if current_spell_stats["balls"] > 0
                                else 0
                            )
                            - (
                                post_change_stats["dot_balls"]
                                * 100
                                / post_change_stats["balls_bowled"]
                                if post_change_stats["balls_bowled"] > 0
                                else 0
                            ),
                            "wicket_impact": post_change_stats["wickets_taken"],
                            "change_effectiveness": (
                                "Positive"
                                if prev_economy > post_change_stats["economy"]
                                else "Negative"
                            ),
                        }
                    )

                # Reset spell stats for new bowler
                current_spell_stats = {
                    "runs": 0,
                    "balls": 0,
                    "dots": 0,
                    "wickets": 0,
                }

            # Update current spell stats
            if ball["legal_ball"] == 1:
                current_spell_stats["balls"] += 1
                current_spell_stats["runs"] += ball["bat_score"]
                if ball["bat_score"] == 0:
                    current_spell_stats["dots"] += 1
                if ball["striker_dismissed"] == 1 or ball["nonstriker_dismissed"] == 1:
                    current_spell_stats["wickets"] += 1

            prev_bowler = current_bowler

    bowling_changes_df = pd.DataFrame(bowling_changes)

    if len(bowling_changes_df) > 0:
        print(f"\nAnalyzed {len(bowling_changes_df)} bowling changes")

        # Most effective bowling changes
        effective_changes = bowling_changes_df[
            bowling_changes_df["change_effectiveness"] == "Positive"
        ]
        print(
            f"Effective changes: {len(effective_changes)} ({len(effective_changes)/len(bowling_changes_df)*100:.1f}%)"
        )

        # Visualize bowling change impacts
        fig1 = plt.figure(figsize=(12, 6))

        # Economy rate before vs after changes
        plt.scatter(
            bowling_changes_df["prev_bowler_economy"],
            bowling_changes_df["post_change_economy"],
            alpha=0.7,
            c=[
                "green" if x == "Positive" else "red"
                for x in bowling_changes_df["change_effectiveness"]
            ],
        )
        plt.plot([0, 15], [0, 15], "k--", alpha=0.5)
        plt.xlabel("Previous Bowler Economy")
        plt.ylabel("New Bowler Economy")
        plt.title("Bowling Change Impact on Economy")
        plt.grid(True, alpha=0.3)
        plt.gca().tick_params(axis="x", colors="white")
        plt.gca().tick_params(axis="y", colors="white")
        plt.gca().xaxis.label.set_color("white")
        plt.gca().yaxis.label.set_color("white")
        plt.gca().title.set_color("white")
        plt.gca().spines["bottom"].set_color("white")
        plt.gca().spines["top"].set_color("white")
        plt.gca().spines["left"].set_color("white")
        plt.gca().spines["right"].set_color("white")
        plt.tight_layout()

        # Correlation between change timing and effectiveness
        fig2 = plt.figure(figsize=(12, 6))
        effectiveness_by_over = bowling_changes_df.groupby("change_over")[
            "economy_improvement"
        ].mean()
        effectiveness_by_over.plot(kind="line", marker="o", color="purple")
        plt.xlabel("Over of Change")
        plt.ylabel("Average Economy Improvement")
        plt.title("Change Effectiveness by Timing")
        plt.grid(True, alpha=0.3)
        plt.gca().tick_params(axis="x", colors="white")
        plt.gca().tick_params(axis="y", colors="white")
        plt.gca().xaxis.label.set_color("white")
        plt.gca().yaxis.label.set_color("white")
        plt.gca().title.set_color("white")
        plt.gca().spines["bottom"].set_color("white")
        plt.gca().spines["top"].set_color("white")
        plt.gca().spines["left"].set_color("white")
        plt.gca().spines["right"].set_color("white")
        plt.tight_layout()

        # Top insights
        print("\n--- KEY INSIGHTS ---")
        avg_improvement = bowling_changes_df["economy_improvement"].mean()
        print(
            f"Average economy improvement from bowling changes: {avg_improvement:.2f}"
        )

        best_change = bowling_changes_df.loc[
            bowling_changes_df["economy_improvement"].idxmax()
        ]
        print(
            f"Best bowling change: Match {best_change['match_id']}, Over {best_change['change_over']} "
            f"(Improvement: {best_change['economy_improvement']:.2f})"
        )

        return bowling_changes_df, [fig1, fig2]
    else:
        print("No bowling changes detected in the dataset")
        return pd.DataFrame(), None


def main():
    st.title("🏏 Cricket Match Dashboard")
    st.markdown("---")

    # File upload
    # uploaded_file = st.file_uploader("Upload Cricket Dataset (CSV)", type=["csv"])

    # Load data
    with st.spinner("Loading data..."):
        df = pd.read_csv(DATA_DIR / "enhanced_cricket_data.csv")

    st.success(f"Data loaded successfully! {len(df)} rows, {len(df.columns)} columns")

    # Match selection
    available_matches = sorted(df["match_id"].unique())
    selected_match = st.selectbox("Select Match", available_matches)

    if selected_match:
        # Calculate match data
        match_info, team_scores = calculate_match_summary(df, selected_match)
        batting_df, bowling_df = calculate_player_stats(df, selected_match)

        # Display match information
        st.subheader("📋 Match Information")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Match ID", selected_match)
        with col2:
            st.metric("Venue ID", match_info["venue_id"])
        with col3:
            st.metric("Date", match_info["match_date"].split(" ")[0])
        with col4:
            st.metric("Team A", match_info["team_a"])
        with col5:
            st.metric("Team B", match_info["team_b"])

        st.markdown("---")

        # Team Scores
        st.subheader("📊 Team Scores")
        score_cols = st.columns(len(team_scores))

        for idx, (team_innings, score_data) in enumerate(team_scores.items()):
            with score_cols[idx % len(score_cols)]:
                team_name = team_innings.split("_innings_")[0]
                innings_num = team_innings.split("_innings_")[1]
                st.metric(
                    f"{team_name} (Innings {innings_num})",
                    f"{score_data['runs']}/{score_data['wickets']}",
                    f"({score_data['overs']} overs)",
                )

        st.markdown("---")

        # Player Statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Batting Performances")
            if not batting_df.empty:
                batting_display = batting_df.sort_values("runs", ascending=False).head(
                    10
                )
                batting_display["Status"] = batting_display["dismissed"].apply(
                    lambda x: "Out" if x else "Not Out"
                )
                st.dataframe(
                    batting_display[
                        [
                            "player_id",
                            "team_name",
                            "runs",
                            "balls",
                            "strike_rate",
                            "fours",
                            "sixes",
                            "Status",
                        ]
                    ],
                    use_container_width=True,
                )
            else:
                st.info("No batting data available")

        with col2:
            st.subheader("Top Bowling Performances")
            if not bowling_df.empty:
                bowling_display = bowling_df.sort_values(
                    "wickets", ascending=False
                ).head(10)
                st.dataframe(
                    bowling_display[
                        [
                            "player_id",
                            "team_name",
                            "overs",
                            "runs",
                            "wickets",
                            "economy",
                        ]
                    ],
                    use_container_width=True,
                )
            else:
                st.info("No bowling data available")

        st.markdown("---")

        # Progressive Score Chart
        st.subheader("📈 Progressive Score Comparison")
        score_chart = create_progressive_score_chart(df, selected_match)
        if score_chart:
            st.plotly_chart(score_chart, use_container_width=True)
        else:
            st.info("Unable to generate progressive score chart")

        st.markdown("---")
        st.subheader("Bowling Pitch Maps")
        col1, col2 = st.columns(2)
        team = df[df["match_id"] == selected_match]["team_name_bowling"].unique()
        with col1:
            pitch_map = create_bowling_pitch_map(df, selected_match, team[0])
            if pitch_map:
                st.image(
                    DATA_DIR / f"{team[0]}_bowling_pitch_map.png",
                    caption=f"{team[0]} Bowling Pitch Map",
                    use_container_width=True,
                )
            else:
                st.info("Unable to generate bowling pitch map")

        with col2:
            pitch_map = create_bowling_pitch_map(df, selected_match, team[1])
            if pitch_map:
                st.image(
                    DATA_DIR / f"{team[1]}_bowling_pitch_map.png",
                    caption=f"{team[1]} Bowling Pitch Map",
                    use_container_width=True,
                )
            else:
                st.info("Unable to generate bowling pitch map")

        st.markdown("---")
        # Momentum Tracking
        st.subheader("📊 Momentum Tracking")
        momentum_chart = calculate_momentum_tracking(df, selected_match)
        if momentum_chart:
            st.pyplot(momentum_chart, use_container_width=True)
        else:
            st.info("Unable to generate momentum tracking chart")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Cricket Match Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    main()
