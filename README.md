NYC Crash Analysis Dashboard
Overview
This interactive dashboard visualizes and analyzes NYC traffic crash data, helping users explore trends, risk factors, and demographic impacts. It’s built for city planners, safety advocates, and anyone interested in understanding urban traffic safety.

---

Tech Stack
• Python
• Dash (by Plotly)
• Plotly Express & Graph Objects
• Dash Bootstrap Components
• Pandas
• NumPy
• Scikit-learn (for clustering)
• Plotly Figure Factory
• Data Source: NYC crash data (loaded from Kaggle)

---

Dashboard Visualizations
Tab 1: Crash Geography
• Crash Locations Map
Visualizes the geographic distribution of crashes across NYC, highlighting hotspots and borough patterns.
• Crash Trends Over Time
Shows how crash frequency changes year by year, helping identify long-term trends and the impact of interventions.
• Crashes by Borough
Compares crash counts across boroughs, revealing which areas are most affected.
• Injuries by Borough
Displays total injuries per borough, clarifying where the most severe incidents occur.

---

Tab 2: Vehicles & Factors
• Contributing Factors
Ranks the top causes of crashes (e.g., distracted driving, speeding), helping target prevention efforts.
• Vehicle vs Factor Heatmap
Shows which vehicle types are most associated with specific contributing factors, revealing risk patterns.
• Vehicle Type Trends
Tracks how crash involvement by vehicle type changes over time (e.g., rise in bicycle crashes).

---

Tab 3: People & Injuries
• Safety Equipment
Breaks down usage of safety gear (seatbelts, helmets), highlighting gaps in protection.
• Injury Types
Categorizes bodily injuries, helping understand the severity and nature of crash outcomes.
• Emotional State
Visualizes reported emotional status post-crash, offering insight into psychological impacts.
• Ejection Status
Shows how often people are ejected from vehicles, by person type, indicating crash severity.
• Position in Vehicle
Analyzes injury counts by seating position, informing vehicle safety design.
• Person Types Over Time
Tracks injuries by person type (pedestrian, cyclist, motorist) across years.
• Top Complaints
Lists the most common complaints reported after crashes, grouped by person type.

---

Tab 4: Demographics
• Age Distribution
Histogram of ages involved in crashes, revealing vulnerable age groups.
• Gender Distribution
Pie chart showing male/female/other proportions in crash data.
• Real-time Statistics
Displays live counts for total crashes, injuries, fatalities, and average injuries per crash.

---

Tab 5: Advanced Analytics
• Crash Hotspot Clustering
Uses machine learning to identify geographic clusters with high crash frequency.
• Risk Correlation Matrix
Heatmap showing relationships between risk factors and crash outcomes (e.g., age vs. severity).
• Temporal Risk Patterns
Reveals peak crash times by day and hour, guiding targeted interventions.
• Severity Prediction Factors
Analyzes which boroughs and factors lead to the most severe outcomes.
• Spatial Risk Density
Heatmap of severe crash concentrations, highlighting high-risk zones.

---

Why These Graphs?
Each visualization is designed to make complex crash data easy to interpret, supporting data-driven decisions for safer streets.
• Geographic maps pinpoint problem areas.
• Trend charts show changes over time.
• Factor and vehicle analyses reveal root causes.
• Demographic charts highlight who is most at risk.
• Advanced analytics uncover hidden patterns for proactive safety planning