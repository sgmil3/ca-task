import re
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

import ca_task.dashboard as dbd
from ca_task.dismissal_proba import predict_dismissal_probability
from ca_task.utility import check_password

# Page configuration
st.set_page_config(
    page_title="Data Science Portfolio Presentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

# Custom CSS for professional presentation styling
st.markdown(
    """
<style>
   /* Force dark theme for presentation */
   .stApp {
       background-color: #0a0e0a;
       color: #ffffff;
   }
   
   /* Sidebar styling */
   .css-1d391kg {
       background-color: #0f1a0f;
       border-right: 2px solid #ffe000;
   }
   
   /* Main content area */
   .main .block-container {
       padding-top: 1rem;
       padding-bottom: 2rem;
       max-width: 1200px;
   }
   
   /* Professional header for presentation */
   .presentation-header {
       text-align: center;
       padding: 2.5rem 2rem;
       background: linear-gradient(135deg, #073009 0%, #0a4012 50%, #0d5016 100%);
       border: 2px solid #ffe000;
       border-radius: 20px;
       margin-bottom: 2rem;
       box-shadow: 0 10px 40px rgba(179, 143, 0, 0.2);
       position: relative;
       overflow: hidden;
   }
   
   .presentation-header::before {
       content: '';
       position: absolute;
       top: 0;
       left: -100%;
       width: 100%;
       height: 100%;
       background: linear-gradient(90deg, transparent, rgba(179, 143, 0, 0.1), transparent);
       animation: shimmer 3s infinite;
   }
   
   @keyframes shimmer {
       0% { left: -100%; }
       100% { left: 100%; }
   }
   
   .presentation-header h1 {
       font-size: 2.5rem;
       margin-bottom: 0.5rem;
       color: #ffffff;
       text-shadow: 0 0 20px rgba(179, 143, 0, 0.5);
       position: relative;
       z-index: 1;
   }
   
   .presentation-header .subtitle {
       font-size: 1.3rem;
       color: #b8c5b8;
       margin-bottom: 0.5rem;
       position: relative;
       z-index: 1;
   }
   
   .presentation-header .context {
       font-size: 1rem;
       color: #9dbf9d;
       font-weight: 500;
       position: relative;
       z-index: 1;
   }
   
   /* Section navigation indicators */
   .section-nav {
       display: flex;
       justify-content: center;
       gap: 2rem;
       margin-bottom: 2rem;
       padding: 1rem;
       background: rgba(7, 48, 9, 0.03);
       border-radius: 15px;
       border: 1px solid rgba(179, 143, 0, 0.2);
   }
   
   .nav-item {
       padding: 0.8rem 1.5rem;
       background: rgba(179, 143, 0, 0.1);
       border-radius: 10px;
       color: #ffe000;
       font-weight: 500;
       border: 1px solid rgba(179, 143, 0, 0.3);
   }
   
   /* Professional section headers */
   .section-header {
       color: #ffe000;
       border-bottom: 3px solid;
       border-image: linear-gradient(90deg, #ffe000, #9dbf9d) 1;
       padding-bottom: 1rem;
       margin-bottom: 2rem;
       font-size: 2.2rem;
       text-shadow: 0 0 15px rgba(179, 143, 0, 0.3);
       display: flex;
       align-items: center;
       gap: 1rem;
   }
   
   /* Content cards for presentation flow */
   .presentation-card {
       background: linear-gradient(135deg, rgba(179, 143, 0, 0.05) 0%, rgba(7, 48, 9, 0.05) 100%);
       backdrop-filter: blur(15px);
       border: 1px solid rgba(179, 143, 0, 0.2);
       padding: 2.5rem;
       border-radius: 20px;
       margin-bottom: 2rem;
       box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
       transition: all 0.4s ease;
       position: relative;
   }
   
   
   .presentation-card h3 {
       color: #9dbf9d;
       margin-bottom: 1.5rem;
       font-size: 1.6rem;
       font-weight: 600;
   }
   
   .presentation-card h4 {
       color: #ffe000;
       margin-bottom: 1rem;
       font-size: 1.3rem;
   }

   /* Content cards for presentation flow */
   .st-key-presentation-card, .st-key-presentation-card-2 {
       background: linear-gradient(135deg, rgba(179, 143, 0, 0.05) 0%, rgba(7, 48, 9, 0.05) 100%);
       backdrop-filter: blur(15px);
       border: 1px solid rgba(179, 143, 0, 0.2);
       padding: 2.5rem;
       border-radius: 20px;
       margin-bottom: 2rem;
       box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
       transition: all 0.4s ease;
       position: relative;
   }
   
   
   .st-key-presentation-card h3, .st-key-presentation-card-2 h3 {
       color: #9dbf9d;
       margin-bottom: 1.5rem;
       font-size: 1.6rem;
       font-weight: 600;
   }
   
   .st-key-presentation-card h4, .st-key-presentation-card-2 h4 {
       color: #ffe000;
       margin-bottom: 1rem;
       font-size: 1.3rem;
   }
   
   /* Key points highlighting */
   .key-point {
       background: rgba(7, 48, 9, 0.08);
       border-left: 4px solid #9dbf9d;
       padding: 1.5rem;
       border-radius: 0 12px 12px 0;
       margin: 1.5rem 0;
       box-shadow: 0 5px 20px rgba(7, 48, 9, 0.1);
   }
   
   .key-point h4 {
       color: #9dbf9d;
       margin-bottom: 0.8rem;
       display: flex;
       align-items: center;
       gap: 0.5rem;
   }
   
   /* Process flow indicators */
   .process-step {
       background: linear-gradient(135deg, rgba(179, 143, 0, 0.1), rgba(7, 48, 9, 0.1));
       border: 2px solid rgba(179, 143, 0, 0.3);
       padding: 2rem;
       border-radius: 15px;
       margin-bottom: 1.5rem;
       position: relative;
       overflow: hidden;
   }
   
   .process-step::before {
       content: attr(data-step);
       position: absolute;
       top: -10px;
       right: -10px;
       background: linear-gradient(135deg, #ffe000, #9dbf9d);
       color: #0a0e0a;
       width: 40px;
       height: 40px;
       border-radius: 50%;
       display: flex;
       align-items: center;
       justify-content: center;
       font-weight: bold;
       font-size: 1.2rem;
   }
   
   /* Professional tabs for methodical presentation */
   .stTabs [data-baseweb="tab-list"] {
       gap: 12px;
       background: rgba(7, 48, 9, 0.02);
       padding: 1rem;
       border-radius: 15px;
       border: 1px solid rgba(179, 143, 0, 0.2);
   }
   
   .stTabs [data-baseweb="tab"] {
       background: rgba(179, 143, 0, 0.1);
       border-radius: 12px;
       color: #ffffff;
       border: 1px solid rgba(179, 143, 0, 0.3);
       padding: 1rem 2rem;
       font-weight: 500;
   }
   
   .stTabs [aria-selected="true"] {
       background: linear-gradient(135deg, #ffe000, #073009);
       color: #ffffff;
       font-weight: 600;
   }
   
   /* Navigation styling */
   .stSelectbox > div > div {
       background: linear-gradient(135deg, rgba(179, 143, 0, 0.1), rgba(7, 48, 9, 0.1));
       border: 2px solid rgba(179, 143, 0, 0.3);
       color: #ffffff;
       border-radius: 12px;
       font-weight: 500;
   }
   
   /* Text styling for readability */
   .stApp p, .stApp li {
       color: #e8f4f8;
       line-height: 1.7;
       font-size: 1.05rem;
   }
   
   /* Dashboard preview section */
   .dashboard-preview {
       background: linear-gradient(135deg, rgba(7, 48, 9, 0.03), rgba(179, 143, 0, 0.03));
       border: 2px solid rgba(7, 48, 9, 0.2);
       border-radius: 20px;
       padding: 3rem;
       text-align: center;
       margin: 2rem 0;
   }
   
   .dashboard-preview h2 {
       color: #9dbf9d;
       font-size: 2rem;
       margin-bottom: 1rem;
   }
   
   /* Footer for presentation context */
   .presentation-footer {
       text-align: center;
       color: #9dbf9d;
       padding: 2rem;
       margin-top: 3rem;
       border-top: 2px solid rgba(179, 143, 0, 0.2);
       background: rgba(179, 143, 0, 0.02);
       border-radius: 15px;
   }
   
   /* Responsive adjustments */
   @media (max-width: 768px) {
       .presentation-header h1 {
           font-size: 2rem;
       }
       .section-nav {
           flex-direction: column;
           gap: 1rem;
       }
       .presentation-card {
           padding: 1.5rem;
       }
   }

    #section-nav-link {
        color: #00d4ff;
        text-decoration: none;
        font-weight: 500;
    }

    /* Fainter, more subtle border for st.metric boxes */
    [data-testid="stMetric"] {
        border: 1px solid rgba(179, 143, 0, 0.10);
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(179, 143, 0, 0.03), rgba(7, 48, 9, 0.03));
        box-shadow: 0 2px 8px rgba(179, 143, 0, 0.03);
        padding: 1.2rem 1rem;
        margin-bottom: 1rem;
    }

    [data-testid="stMetricDelta"] svg {
        display: none;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar navigation with presentation flow
st.sidebar.title("Presentation Navigation")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Select Section:",
    ["Introduction & Context", "Dashboard Presentation"],
    help="Navigate through the presentation sections",
)

# Add presentation context in sidebar
st.sidebar.markdown("---")

if page == "Introduction & Context":
    # Professional presentation header
    st.markdown(
        """
    <div class="presentation-header">
        <h1>Cricket Australia Task Interview</h1>
        <p class="context">Stuart Mills</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Section 1: Introduction - Interview Opening
    st.markdown(
        '<h2 class="section-header" id="s1">Introduction & Opening Discussion</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
        <div class="presentation-card">
            <h3>Welcome & Presentation Overview</h3>
            <p>
                Thank you for this opportunity to present my approach to the technical task. This presentation 
                will walk through my complete data science workflow, from initial data exploration to final 
                dashboard implementation.
            </p>
            <p>
                I'll be demonstrating how I approached the problem systematically, the analytical decisions 
                I made along the way, and how these led to the final dashboard solution and why I chose to
                implement it in this way.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="key-point">
            <h4>Presentation Agenda</h4>
            <ul>
                <li>Problem understanding & approach</li>
                <li>Data exploration methodology</li>
                <li>Technical decisions & rationale</li>
                <li>Interactive dashboard walkthrough</li>
                <li>Key insights & take aways</li>
                <li>Q&A</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Section 2: EDA - Data Exploration Process
    st.markdown(
        '<h2 class="section-header" id="s2">Exploratory Data Analysis Process</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="presentation-card">
        <h3>My Approach to Understanding the Data</h3>
        <p>
            Before diving into the technical implementation, I conducted a thorough exploratory data analysis 
            (EDA) to understand the dataset's structure, identify key variables, and uncover initial insights. 
            This foundational work informed my analytical approach and dashboard design.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="process-step" data-step="1">
            <h4>Data Profiling</h4>
            <p>
                Comprehensive assessment of data structure, types, completeness, and initial quality checks. 
                Identified key variables, missing patterns, and data distribution characteristics that would 
                impact analysis approach. This also included understanding the relationships between different
                features and their relevance, and what data was present in the sample dataset compared
                to the full data dictionary.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="process-step" data-step="2">
            <h4>Feature Engineering & Insights Discovery</h4>
            <p>
                Explored and created new features from the raw data, such as run rates, partnership lengths, and pressure indices. 
                This step uncovered hidden patterns and provided actionable insights, directly shaping the metrics and visualisations 
                used in the dashboard.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="process-step" data-step="3">
            <h4>Visual Exploration</h4>
            <p>
                Created exploratory visualisations to understand data patterns, outliers, and relationships. 
                This meant I could explore what the data actually looked like and represented once 
                visualised and some important features had been included.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Section 3: Methodology & Justification
    st.markdown(
        '<h2 class="section-header id="s3">Methodology & Decision Rationale</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="presentation-card">
        <h3>Strategic Decisions Behind the Implementation</h3>
        <p>
            Every technical choice in this project was made with the end output and data 
            characteristics in mind. I'll walk through the key decisions that shaped the final dashboard 
            and explain the reasoning behind each choice.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Analytical Approach", "Technical Stack", "Visualisation Strategy"]
    )

    with tab1:
        st.markdown(
            """
        <div class="presentation-card">
            <h4>Why This Analytical Framework?</h4>
            <p>
                Based on the data characteristics and task at hand, I chose to demonstrate both my ability to understand key metrics 
                for the game, but also try to branch out and try new things to see new insights. The data available is rich and meaningful,
                I made this choice consciously to reflect the dynamic nature of cricket analytics.
            </p>
            <div class="key-point">
                <h4>Key Considerations</h4>
                <ul>
                    <li><strong>Cricket Context:</strong> Aligned analytics with expected knowledge and convention</li>
                    <li><strong>Data Quality:</strong> Chose specific methods that mesh with the data limitations</li>
                    <li><strong>Interpretability:</strong> Balanced known approaches with new ones to maximise understanding</li>
                    <li><strong>Scalability:</strong> Made choices for scalability to accommodate future data growth but also technical capability enhancements</li>
                </ul>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            """
        <div class="presentation-card">
            <h4>Technology Choices & Implementation</h4>
            <p>
                The technical stack was chosen to maximise development efficiency but 
                also giving me the ability to create a rich, interactive dashboard. 
                Additionally, I can flex some of my Python skills to demonstrate
                my ability to work with data in a way that is meaningful and insightful.
            </p>
            <div class="key-point">
                <h4>Technology Rationale</h4>
                <ul>
                    <li><strong>Python Ecosystem:</strong> Leveraged pandas, numpy for robust data processing, scikit-learn for machine learning, and streamlit for visualisation</li>
                    <li><strong>Visualisation:</strong> Selected relevant tools that balance functionality with user experience as well as the existing use cases.</li>
                    <li><strong>Dashboard Framework:</strong> Chosen for rapid prototyping and deployment flexibility</li>
                    <li><strong>Performance:</strong> Optimised for responsive user interactions</li>
                </ul>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown(
            """
        <div class="presentation-card">
            <h4>Dashboard Design Philosophy</h4>
            <p>
                The visualisation strategy focuses on clear communication of insights while maintaining 
                the ability for users to explore data independently. Design choices support both 
                high-level overview and detailed analysis needs. The visualisations are not all encompassing,
                but rather a starting point for deeper exploration and understanding of the data.
            </p>
            <div class="key-point">
                <h4>Design Principles</h4>
                <ul>
                    <li><strong>User-Centric:</strong> Designed for the target audience's analytical needs</li>
                    <li><strong>Progressive Disclosure:</strong> Created a flow to the data to also highlight my visualisation experience and knowledge</li>
                    <li><strong>Interactive Elements:</strong> Enables self-service exploration</li>
                    <li><strong>Performance:</strong> Optimised loading and rendering for smooth experience</li>
                </ul>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Transition to Dashboard
    st.markdown(
        """
    <div class="dashboard-preview">
        <h2>Ready for Dashboard Demonstration</h2>
        <p style="font-size: 1.2rem; color: #e8f4f8;">
            Now that we've covered the analytical foundation and methodology, 
            let's explore the interactive dashboard that brings these insights to life.
        </p>
        <p style="color: #64ffda; margin-top: 1rem;">
            Navigate to <strong>"Dashboard Presentation"</strong> to continue
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

elif page == "Dashboard Presentation":
    st.markdown(
        """
    <div class="presentation-header">
        <h1>Interactive Dashboard</h1>
        <p class="subtitle">Technical Implementation & Key Insights</p>
        <p class="context">Live Dashboard Demonstration</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.container(key="presentation-card"):
        st.markdown(
            '<h3 class="section-header" id="s4">Game Summaries</h3>',
            unsafe_allow_html=True,
        )
        # Load data
        with st.spinner("Loading data..."):
            df = pd.read_csv(dbd.DATA_DIR / "enhanced_cricket_data.csv")

        # Match selection
        available_matches = sorted(df["match_name"].unique())
        selected_match = st.selectbox("Select Match", available_matches)

        if selected_match:
            # Calculate match data
            match_id = df.loc[df["match_name"] == selected_match, "match_id"].iloc[0]
            match_info, team_scores = dbd.calculate_match_summary(df, match_id)
            batting_df, bowling_df = dbd.calculate_player_stats(df, match_id)  # type: ignore

            # Display match information
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Format", re.match(r"\[(.*?)\]", selected_match).group(1))  # type: ignore
            with col2:
                # Format date as dd-mon-yyyy
                date_obj = datetime.strptime(
                    match_info["match_date"].split(" ")[0], "%Y-%m-%d"
                )
                formatted_date = date_obj.strftime("%a %d %b %Y")
                st.metric("Date", formatted_date)
            with col3:
                st.metric("Team A", match_info["team_a"])
            with col4:
                st.metric("Team B", match_info["team_b"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Match Result", match_info["match_result"])
            with col2:
                st.metric("Toss", match_info["toss_result"])

            st.markdown("---")
            score_cols = st.columns(len(team_scores))

            for idx, (team_innings, score_data) in enumerate(team_scores.items()):
                with score_cols[idx % len(score_cols)]:
                    team_name = team_innings.split("_innings_")[0]
                    innings_num = team_innings.split("_innings_")[1]
                    st.metric(
                        label=f"{team_name} (Innings {innings_num})",
                        value=f"{score_data['runs']}/{score_data['wickets']}",
                        delta=f"({score_data['overs']} overs)",
                        delta_color="off",
                    )

            col1, col2 = st.columns(2)
            teams = batting_df["team_name"].unique()

            with col1:
                team = teams[0]
                st.subheader("Top Batting Performers")
                if not batting_df[batting_df["team_name"] == team].empty:
                    batting_display = (
                        batting_df[batting_df["team_name"] == team]
                        .sort_values(["runs", "balls"], ascending=[False, True])
                        .head(3)
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

                st.subheader("Top Bowling Performances")
                if not bowling_df[bowling_df["team_name"] == team].empty:
                    bowling_display = (
                        bowling_df[bowling_df["team_name"] == team]
                        .sort_values(["wickets", "runs"], ascending=[False, True])
                        .head(3)
                    )

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

            with col2:
                team = teams[1]
                st.subheader("Top Batting Performers")
                if not batting_df[batting_df["team_name"] == team].empty:
                    batting_display = (
                        batting_df[batting_df["team_name"] == team]
                        .sort_values(["runs", "balls"], ascending=[False, True])
                        .head(3)
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
                st.subheader("Top Bowling Performances")
                if not bowling_df[bowling_df["team_name"] == team].empty:
                    bowling_display = (
                        bowling_df[bowling_df["team_name"] == team]
                        .sort_values(["wickets", "runs"], ascending=[False, True])
                        .head(3)
                    )

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
            st.subheader("Worm")
            score_chart = dbd.create_progressive_score_chart(df, match_id)
            if score_chart:
                st.plotly_chart(score_chart, use_container_width=True)
            else:
                st.info("Unable to generate progressive score chart")

            st.markdown("---")
            st.subheader("Bowling Heat Maps")
            col1, col2 = st.columns(2)
            team = df[df["match_id"] == selected_match]["team_name_bowling"].unique()
            with col1:
                pitch_map = dbd.create_bowling_pitch_map(df, match_id, teams[0])
                if pitch_map:
                    st.image(
                        dbd.DATA_DIR / f"{teams[0]}_bowling_pitch_map.png",
                        caption=f"{teams[0]} Bowling Pitch Map",
                        use_container_width=True,
                    )
                else:
                    st.info("Unable to generate bowling pitch map")

            with col2:
                pitch_map = dbd.create_bowling_pitch_map(df, match_id, teams[1])
                if pitch_map:
                    st.image(
                        dbd.DATA_DIR / f"{teams[1]}_bowling_pitch_map.png",
                        caption=f"{teams[1]} Bowling Pitch Map",
                        use_container_width=True,
                    )
                else:
                    st.info("Unable to generate bowling pitch map")

            st.markdown("---")
            st.subheader("Match Metrics")
            metric = st.selectbox(
                "Select Metric to Visualise",
                (
                    [
                        "Momentum",
                        "Turning Points",
                        "All Metrics",
                    ]
                    if selected_match.__contains__("T20")
                    else ["Turning Points", "All Metrics"]
                ),
                help="Select the metric to visualise",
                label_visibility="collapsed",
                key="match_metric_select",
                placeholder="Select Metric",
            )
            if metric == "Momentum" or metric == "All Metrics":
                momentum_teams_df = df[df.match_id.isin(teams)]
                st.markdown("---")
                st.subheader("Match Momentum")
                st.markdown(
                    "This chart shows the momentum of the match over time, highlighting key moments and shifts in team performance."
                    "The comparison is based on the team batting second, therefore the annotations relate to whether the team batting second's run rate in that block was higher or lower than the team batting first in the same block."
                )
                block_size = st.number_input(
                    "Block Size (overs)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    help="Overs per block for momentum tracking",
                    label_visibility="collapsed",
                    key="momentum_block_size",
                    placeholder="Block size",
                )
                momentum_chart = dbd.calculate_momentum_tracking(
                    df, match_id, n_block=block_size
                )
                if momentum_chart:
                    st.plotly_chart(momentum_chart, use_container_width=True)
                else:
                    st.info("Unable to generate match momentum chart")

            if metric == "Turning Points" or metric == "All Metrics":
                top_moments, pressure = dbd.identify_match_defining_moments(
                    df[df["match_id"] == match_id]
                )
                if pressure:
                    st.markdown("---")
                    st.subheader("Match Defining Moments")
                    st.dataframe(
                        top_moments[
                            [
                                "match_name",
                                "over",
                                "ball",
                                "match_situation",
                                "innings",
                                "team_batting",
                                "team_bowling",
                                "impact_score",
                                "pressure_index",
                            ]
                        ],
                        use_container_width=True,
                    )
                    fig1, fig2, fig3 = pressure
                    st.markdown(
                        "This chart highlights key moments in the match that significantly impacted the outcome, such as wickets, partnerships, and scoring bursts."
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    st.markdown(
                        "This chart shows the pressure index over the course of the match, indicating periods of high and low pressure for each team."
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.markdown(
                        "This chart visualises the impact of key moments on the match outcome, showing how each moment contributed to the final result."
                    )
                    st.markdown(
                        "Key Impact Factors: {}".format(
                            ", ".join(
                                list(
                                    set(
                                        (
                                            ", ".join(
                                                top_moments["impact_factors"].to_list()
                                            ).split(", ")
                                        )
                                    )
                                )
                            )
                            .replace("_", " ")
                            .title()
                        )
                    )
                    st.plotly_chart(fig3, use_container_width=True)

    with st.container(key="presentation-card-2"):
        st.markdown(
            '<h3 class="section-header" id="s5">Machine Learning</h3>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            This model predicts the likelihood of a ball being high risk (for the batting team). 
            It uses a simple Random Forest Classifier trained on the match data supplied, and serves as just a
            proof of concept for how machine learning could be applied to the data.
            """
        )
        # Load the model
        dismissal_model = joblib.load("dismissal_predictor_model.pkl")
        # match selector
        available_matches = sorted(df["match_name"].unique())
        selected_match = st.selectbox(
            "Select Match for Dismissal Prediction", available_matches
        )
        if selected_match:
            # Calculate match data
            match_data = df[df["match_name"] == selected_match]
            prob, fig = predict_dismissal_probability(
                model=dismissal_model, df=match_data
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Unable to generate dismissal probability chart")
