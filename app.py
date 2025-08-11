import ast
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots

from src.config import PROCESSED_DATA_DIR

# Cấu hình trang
st.set_page_config(
    page_title="HUIM Results Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS tùy chỉnh
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stMetric > div > div > div > div {
        color: #667eea;
        font-weight: bold;
    }
    .filter-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
<div class="main-header">
    <h1>🛒 HUIM Results Dashboard</h1>
    <p>High Utility Itemset Mining Analysis - Phân tích tập mục có độ hữu ích cao</p>
</div>
""",
    unsafe_allow_html=True,
)


def load_and_process_data():
    """Load và xử lý dữ liệu từ CSV"""
    data = pd.read_csv(PROCESSED_DATA_DIR / "hui_results.csv")

    df = pd.DataFrame(data)

    # Xử lý dữ liệu
    df["items_list"] = df["items"].apply(lambda x: ast.literal_eval(x))
    df["itemset_size"] = df["items_list"].apply(len)
    df["items_display"] = df["items_list"].apply(lambda x: " + ".join(x))
    df["utility_formatted"] = df["utility"].apply(lambda x: f"{x:,}")

    return df


# Load dữ liệu
df = load_and_process_data()

# Sidebar cho filters (ẩn theo mặc định)
with st.expander("🔧 Bộ lọc và tùy chọn", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        min_utility = st.slider(
            "Utility tối thiểu",
            min_value=int(df["utility"].min()),
            max_value=int(df["utility"].max()),
            value=int(df["utility"].min()),
            step=100000,
        )

    with col2:
        itemset_sizes = st.multiselect(
            "Kích thước itemset",
            options=sorted(df["itemset_size"].unique()),
            default=sorted(df["itemset_size"].unique()),
        )

    with col3:
        top_n = st.slider(
            "Hiển thị top N", min_value=10, max_value=len(df), value=20, step=5
        )

# Lọc dữ liệu
filtered_df = df[
    (df["utility"] >= min_utility) & (df["itemset_size"].isin(itemset_sizes))
].head(top_n)

# Metrics overview
st.markdown("## 📊 Tổng quan")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Tổng số itemsets",
        value=len(filtered_df),
        delta=f"{len(filtered_df)/len(df)*100:.1f}% của tổng",
    )

with col2:
    st.metric(
        label="Utility cao nhất",
        value=f"{filtered_df['utility'].max():,}",
        delta=filtered_df.iloc[0]["items_display"] if len(filtered_df) > 0 else "N/A",
    )

with col3:
    st.metric(
        label="Utility trung bình",
        value=f"{filtered_df['utility'].mean():,.0f}",
        delta=f"±{filtered_df['utility'].std():,.0f}",
    )

with col4:
    st.metric(
        label="Kích thước itemset phổ biến",
        value=(
            filtered_df["itemset_size"].mode().iloc[0]
            if len(filtered_df) > 0
            else "N/A"
        ),
        delta=f"{(filtered_df['itemset_size'] == filtered_df['itemset_size'].mode().iloc[0]).sum()} itemsets",
    )

with col5:
    total_utility = filtered_df["utility"].sum()
    st.metric(label="Tổng utility", value=f"{total_utility:,}", delta="VNĐ")

# Main visualizations
st.markdown("## 📈 Biểu đồ phân tích")

# Row 1: Top itemsets và phân bố theo kích thước
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🏆 Top Itemsets có Utility cao nhất")

    # Horizontal bar chart
    fig_bar = px.bar(
        filtered_df.head(15),
        x="utility",
        y="items_display",
        orientation="h",
        color="utility",
        color_continuous_scale="viridis",
        title=f"Top 15 High Utility Itemsets",
        labels={"utility": "Utility", "items_display": "Itemsets"},
    )
    fig_bar.update_layout(
        height=600, yaxis={"categoryorder": "total ascending"}, showlegend=False
    )
    fig_bar.update_traces(texttemplate="%{x:,}", textposition="inside")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### 📏 Phân bố theo kích thước Itemset")

    size_dist = filtered_df["itemset_size"].value_counts().reset_index()
    size_dist.columns = ["size", "count"]

    fig_pie = px.pie(
        size_dist,
        values="count",
        names="size",
        title="Phân bố kích thước itemset",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

# Row 2: Heatmap và scatter plot
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔥 Heatmap - Items xuất hiện cùng nhau")

    # Tạo ma trận đồng xuất hiện
    all_items = set()
    for items_list in filtered_df["items_list"]:
        all_items.update(items_list)

    all_items = sorted(list(all_items))
    co_occurrence = np.zeros((len(all_items), len(all_items)))

    for items_list in filtered_df["items_list"]:
        for i, item1 in enumerate(all_items):
            for j, item2 in enumerate(all_items):
                if item1 in items_list and item2 in items_list and i != j:
                    co_occurrence[i][j] += 1

    fig_heatmap = px.imshow(
        co_occurrence,
        x=all_items,
        y=all_items,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="Ma trận đồng xuất hiện các items",
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    st.markdown("### 📊 Scatter: Kích thước vs Utility")

    fig_scatter = px.scatter(
        filtered_df,
        x="itemset_size",
        y="utility",
        size="utility",
        color="itemset_size",
        hover_data=["items_display"],
        title="Mối quan hệ giữa kích thước itemset và utility",
        labels={"itemset_size": "Kích thước Itemset", "utility": "Utility"},
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Row 3: Item frequency và utility distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Tần suất xuất hiện của từng item")

    item_frequency = Counter()
    for items_list in filtered_df["items_list"]:
        item_frequency.update(items_list)

    freq_df = pd.DataFrame(list(item_frequency.items()), columns=["item", "frequency"])
    freq_df = freq_df.sort_values("frequency", ascending=True)

    fig_freq = px.bar(
        freq_df.tail(10),
        x="frequency",
        y="item",
        orientation="h",
        title="Top 10 items xuất hiện nhiều nhất",
        color="frequency",
        color_continuous_scale="blues",
    )
    fig_freq.update_layout(height=400)
    st.plotly_chart(fig_freq, use_container_width=True)

with col2:
    st.markdown("### 📊 Phân bố Utility")

    fig_hist = px.histogram(
        filtered_df,
        x="utility",
        nbins=20,
        title="Phân bố Utility của các itemsets",
        labels={"utility": "Utility", "count": "Số lượng"},
        color_discrete_sequence=["#ff7f0e"],
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# Detailed table
st.markdown("## 📋 Bảng chi tiết")
st.markdown("### Top High Utility Itemsets")

display_df = filtered_df[
    ["items_display", "utility", "itemset_size", "utility_formatted"]
].copy()
display_df.columns = ["Itemsets", "Utility", "Kích thước", "Utility (Formatted)"]

st.dataframe(
    display_df,
    use_container_width=True,
    height=400,
    column_config={
        "Utility": st.column_config.NumberColumn(
            "Utility", help="Giá trị utility của itemset", format="%d"
        ),
        "Kích thước": st.column_config.NumberColumn(
            "Kích thước", help="Số lượng items trong itemset", format="%d"
        ),
    },
)

# Insights section
st.markdown("## 💡 Insights và Khuyến nghị")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    ### 🔍 Phân tích chính:
    """
    )

    top_item = item_frequency.most_common(1)[0] if item_frequency else ("N/A", 0)
    avg_size = filtered_df["itemset_size"].mean()

    st.info(
        f"""
    • **Item phổ biến nhất**: {top_item[0]} (xuất hiện {top_item[1]} lần)
    • **Kích thước itemset trung bình**: {avg_size:.1f} items
    • **Utility cao nhất**: {filtered_df['utility'].max():,} VNĐ
    • **Tỷ lệ itemsets 2 items**: {(filtered_df['itemset_size'] == 2).sum()/len(filtered_df)*100:.1f}%
    """
    )

with col2:
    st.markdown(
        """
    ### 📈 Khuyến nghị kinh doanh:
    """
    )

    st.success(
        """
    • **Tập trung vào combo "Heo + Củ quả"** - utility cao nhất
    • **Phát triển bundle sản phẩm** với items xuất hiện cùng nhau
    • **Tối ưu hóa kho hàng** dựa trên tần suất xuất hiện
    • **Cross-selling** các sản phẩm có utility cao
    """
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 1rem;'>
    📊 HUIM Dashboard | Phân tích High Utility Itemset Mining | 
    <small>Dữ liệu được cập nhật theo thời gian thực</small>
</div>
""",
    unsafe_allow_html=True,
)
