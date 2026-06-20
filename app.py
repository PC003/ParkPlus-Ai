import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px



st.set_page_config(
    page_title="ParkPulse AI",
    page_icon="🚔",
    layout="wide"
)

st.title("🚔 ParkPulse AI")
st.caption("AI-Powered Parking-Induced Congestion Risk Prediction System")



@st.cache_data
def load_data():
    return pd.read_csv("dashboard_data.csv")

df = load_data()



st.sidebar.header("Filters")

station = st.sidebar.selectbox(
    "Police Station",
    ["All"] + sorted(df["dominant_police_station"].dropna().unique())
)

risk = st.sidebar.multiselect(
    "Risk Level",
    ["Critical","High","Medium","Low"],
    default=["Critical","High","Medium","Low"]
)

if station != "All":
    df = df[df["dominant_police_station"] == station]

df = df[df["predicted_risk"].isin(risk)]



latest_df = (
    df.sort_values("hour")
      .groupby("hotspot_id", as_index=False)
      .last()
)



critical = (latest_df.predicted_risk=="Critical").sum()
high = (latest_df.predicted_risk=="High").sum()
medium = (latest_df.predicted_risk=="Medium").sum()
hotspots = latest_df.hotspot_id.nunique()

c1,c2,c3,c4 = st.columns(4)

c1.metric("🔴 Critical", critical)
c2.metric("🟠 High", high)
c3.metric("🟡 Medium", medium)
c4.metric("📍 Hotspots", hotspots)

st.divider()



st.subheader("🗺️ Bangalore Hotspot Map")

colors = {
    "Low":"green",
    "Medium":"gold",
    "High":"orange",
    "Critical":"red"
}

radius = {
    "Low":5,
    "Medium":8,
    "High":11,
    "Critical":15
}

m = folium.Map(
    location=[12.97,77.59],
    zoom_start=11,
    tiles="CartoDB Positron"
)

for _, row in latest_df.iterrows():

    popup = f"""
    <b>📍 Hotspot</b><br>{row.hotspot_id}<br><br>

    <b>Risk:</b> {row.predicted_risk}<br>

    <b>Confidence:</b> {row.confidence:.1%}<br>

    <b>Police Station:</b> {row.dominant_police_station}<br>

    <b>Recommendation:</b><br>{row.recommendation}
    """

    folium.CircleMarker(
        location=[row.latitude,row.longitude],
        radius=radius[row.predicted_risk],
        color=colors[row.predicted_risk],
        fill=True,
        fill_color=colors[row.predicted_risk],
        fill_opacity=0.8,
        popup=popup
    ).add_to(m)

st_folium(
    m,
    width=1400,
    height=650
)

st.divider()



col1,col2 = st.columns(2)

with col1:

    fig = px.pie(
        latest_df,
        names="predicted_risk",
        title="Risk Distribution",
        hole=0.45
    )

    st.plotly_chart(fig,use_container_width=True)

with col2:

    fig = px.bar(
        latest_df["predicted_risk"].value_counts().reset_index(),
        x="predicted_risk",
        y="count",
        title="Hotspots by Risk"
    )

    st.plotly_chart(fig,use_container_width=True)

st.divider()



st.subheader("🚨 Top Critical Hotspots")

critical_df = (
    latest_df[latest_df.predicted_risk=="Critical"]
    .sort_values("confidence",ascending=False)
)

st.dataframe(
    critical_df[
        [
            "hotspot_id",
            "dominant_police_station",
            "confidence",
            "recommendation"
        ]
    ],
    use_container_width=True
)

st.divider()



st.subheader("🔍 Hotspot Details")

selected = st.selectbox(
    "Select Hotspot",
    latest_df.hotspot_id.unique()
)

row = latest_df[latest_df.hotspot_id==selected].iloc[0]

col1,col2 = st.columns(2)

with col1:

    st.metric("Risk",row.predicted_risk)

    st.metric(
        "Confidence",
        f"{row.confidence:.1%}"
    )

with col2:

    st.success(row.recommendation)

    st.info(row.reason)

deployment = (
    latest_df
    .groupby("dominant_police_station")
    .agg(
        Critical=("predicted_risk",
                  lambda x:(x=="Critical").sum()),

        High=("predicted_risk",
              lambda x:(x=="High").sum())
    )
)

deployment["Priority"] = (
    deployment["Critical"]*3
    +
    deployment["High"]*2
)

deployment = deployment.sort_values(
    "Priority",
    ascending=False
)

st.subheader("🚔 Officer Deployment Priority")

st.dataframe(
    deployment,
    use_container_width=True
)
critical_count = (latest_df["predicted_risk"] == "Critical").sum()

if critical_count > 0:
    top = (
        latest_df[latest_df["predicted_risk"] == "Critical"]
        .sort_values("confidence", ascending=False)
        .iloc[0]
    )

    st.error(
        f"""
🚨 ALERT: {critical_count} Critical Hotspots Detected

Highest Priority: {top.hotspot_id}

Recommendation: {top.recommendation}
"""
    )
else:
    st.success("✅ No Critical Hotspots Detected")

    st.subheader("🤖 AI Insights")

top_station = deployment.index[0]

st.info(f"""
### AI Summary

• Total Critical Hotspots: **{critical}**

• Total High Risk Hotspots: **{high}**

• Highest Deployment Priority:
**{top_station} Police Station**

• Junction-based hotspots contribute significantly to congestion risk.

• Immediate enforcement is recommended for Critical hotspots.
""")


trend = (
    latest_df.groupby("hour")["confidence"]
    .mean()
    .reset_index()
)

fig = px.line(
    trend,
    x="hour",
    y="confidence",
    markers=True,
    title="Average Prediction Confidence by Hour"
)

st.plotly_chart(fig, use_container_width=True)


csv = latest_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Deployment Report",
    csv,
    "deployment_report.csv",
    "text/csv"
)


st.markdown("---")

st.markdown(
"""
<center>

### 🚔 Bengaluru Traffic Command Center

Powered by **ParkPulse AI**

Developed for **GridLock Hackathon 2.0**

</center>
""",
unsafe_allow_html=True
)