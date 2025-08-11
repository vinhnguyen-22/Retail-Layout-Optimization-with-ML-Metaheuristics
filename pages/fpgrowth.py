import ast
import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ====== CẤU HÌNH ĐƯỜNG DẪN FILE CSV ======
CSV_PATH = "data/processed/association_rules.csv"  # Sửa path đúng của bạn

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Association Rules Dashboard")


def plot_association_network_streamlit(
    rules_df,
    top_n=100,
    min_degree=2,
    color_by="out_degree",
    highlight_quantile=0.9,
    label_degree_type="both",
    layout_type="spring",
):
    top_rules = rules_df.sort_values("lift", ascending=False).head(top_n)
    G = nx.DiGraph()
    # Build graph: mỗi sản phẩm là node riêng
    for _, row in top_rules.iterrows():
        try:
            ant_list = (
                ast.literal_eval(row["antecedent"])
                if isinstance(row["antecedent"], str)
                else list(row["antecedent"])
            )
        except Exception:
            ant_list = [str(row["antecedent"])]
        try:
            cons_list = (
                ast.literal_eval(row["consequent"])
                if isinstance(row["consequent"], str)
                else list(row["consequent"])
            )
        except Exception:
            cons_list = [str(row["consequent"])]
        for a in ant_list:
            for c in cons_list:
                G.add_edge(
                    a.strip(),
                    c.strip(),
                    weight=row["lift"],
                    confidence=row["confidence"],
                    support=row.get("support", None),
                )
    # Layout
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=0.5, seed=42)
    else:
        pos = nx.circular_layout(G)
    out_deg = np.array([G.out_degree(n) for n in G.nodes()])
    in_deg = np.array([G.in_degree(n) for n in G.nodes()])
    deg_for_color = out_deg if color_by == "out_degree" else in_deg
    colorbar_title = (
        "Số luật đi từ node này" if color_by == "out_degree" else "Số luật đến node này"
    )
    threshold = np.quantile(deg_for_color, highlight_quantile)
    node_color = []
    for v in deg_for_color:
        if v >= threshold and v > 0:
            node_color.append("#FF5C5C")
        else:
            node_color.append(v)
    # Scale node size hợp lý
    size_min, size_max = 8, 12
    if deg_for_color.max() > deg_for_color.min():
        node_size = [
            size_min
            + (size_max - size_min)
            * (v - deg_for_color.min())
            / (deg_for_color.max() - deg_for_color.min())
            for v in deg_for_color
        ]
    else:
        node_size = [size_min for _ in deg_for_color]

    # Chọn label
    if label_degree_type == "out_degree":
        node_text = [n if (G.out_degree(n) >= min_degree) else "" for n in G.nodes()]
    elif label_degree_type == "in_degree":
        node_text = [n if (G.in_degree(n) >= min_degree) else "" for n in G.nodes()]
    else:
        node_text = [
            n if (G.out_degree(n) >= min_degree or G.in_degree(n) >= min_degree) else ""
            for n in G.nodes()
        ]
    node_hovertext = [
        f"{n}<br>Luật đi: {G.out_degree(n)}<br>Luật đến: {G.in_degree(n)}"
        for n in G.nodes()
    ]
    # Vẽ cạnh (line)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#BBB"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )
    # Vẽ node
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        text=node_text,
        hoverinfo="text",
        hovertext=node_hovertext,
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            color=node_color,
            size=node_size,
            colorbar=dict(thickness=15, title=colorbar_title, xanchor="left"),
            line_width=2,
        ),
        textposition="top center",
    )

    # ====== VẼ MŨI TÊN (marker tam giác) ======
    arrow_x, arrow_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # Move mũi tên về gần node đích (tránh overlap node)
        shrink_ratio = 0.12
        dx, dy = x1 - x0, y1 - y0
        x1_arrow = x1 - dx * shrink_ratio
        y1_arrow = y1 - dy * shrink_ratio
        arrow_x.append(x1_arrow)
        arrow_y.append(y1_arrow)

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Network Graph các Association Rules (node là từng sản phẩm)",
                font=dict(size=18),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig, G


def plot_scatter_rules(rules_df, top_n=300):
    rules = rules_df.sort_values("lift", ascending=False).head(top_n)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=rules["support"],
                y=rules["confidence"],
                text=[
                    f"{a} → {c}"
                    for a, c in zip(rules["antecedent"], rules["consequent"])
                ],
                mode="markers",
                marker=dict(
                    size=10
                    + 20
                    * (rules["support"] - rules["support"].min())
                    / (rules["support"].max() - rules["support"].min() + 1e-9),
                    color=rules["support"],
                    colorscale="YlGnBu",
                    showscale=True,
                    colorbar=dict(title="Support"),
                ),
                hovertemplate="Antecedent: %{text}<br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<br>Support: %{marker.color:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Scatter Plot: Support vs Confidence (color = Support)",
        xaxis_title="Support",
        yaxis_title="Confidence",
        margin=dict(b=40, l=20, r=20, t=60),
        height=500,
    )
    return fig


def plot_lift_histogram(rules_df, top_n=300):
    rules = rules_df.sort_values("lift", ascending=False).head(top_n)
    fig = go.Figure(
        data=[go.Histogram(x=rules["lift"], nbinsx=30, marker_color="teal")]
    )
    fig.update_layout(
        title="Histogram phân phối Lift của các luật",
        xaxis_title="Lift",
        yaxis_title="Số lượng luật",
        margin=dict(b=40, l=20, r=20, t=60),
        height=400,
    )
    return fig


def plot_metrics_correlation_heatmap(rules_df):
    """Heatmap tương quan giữa các metrics"""
    metrics = [
        "support",
        "confidence",
        "lift",
        "conviction",
        "jaccard",
        "support_antecedent",
    ]
    corr_matrix = rules_df[metrics].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Ma trận tương quan giữa các metrics",
        height=400,
        margin=dict(b=40, l=20, r=20, t=60),
    )
    return fig


def plot_top_items_frequency(rules_df, top_n=15):
    """Biểu đồ tần suất xuất hiện của các items"""
    from collections import Counter

    all_items = []
    for _, row in rules_df.iterrows():
        try:
            ant_items = (
                ast.literal_eval(row["antecedent"])
                if isinstance(row["antecedent"], str)
                else [str(row["antecedent"])]
            )
            cons_items = (
                ast.literal_eval(row["consequent"])
                if isinstance(row["consequent"], str)
                else [str(row["consequent"])]
            )
            all_items.extend(ant_items + cons_items)
        except:
            all_items.extend([str(row["antecedent"]), str(row["consequent"])])

    item_counts = Counter(all_items)
    top_items = dict(item_counts.most_common(top_n))

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(top_items.keys()),
                y=list(top_items.values()),
                marker_color="lightblue",
                text=list(top_items.values()),
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=f"Top {top_n} sản phẩm xuất hiện nhiều nhất trong các luật",
        xaxis_title="Sản phẩm",
        yaxis_title="Số lần xuất hiện",
        height=400,
        margin=dict(b=40, l=20, r=20, t=60),
        xaxis_tickangle=-45,
    )
    return fig


def plot_support_confidence_scatter_3d(rules_df, top_n=200):
    """Scatter plot 3D: Support vs Confidence vs Lift"""
    rules = rules_df.sort_values("lift", ascending=False).head(top_n)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=rules["support"],
                y=rules["confidence"],
                z=rules["lift"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=rules["lift"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Lift"),
                ),
                text=[
                    f"{a} → {c}"
                    for a, c in zip(rules["antecedent"], rules["consequent"])
                ],
                hovertemplate="<b>%{text}</b><br>"
                + "Support: %{x:.3f}<br>"
                + "Confidence: %{y:.3f}<br>"
                + "Lift: %{z:.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Biểu đồ 3D: Support vs Confidence vs Lift",
        scene=dict(xaxis_title="Support", yaxis_title="Confidence", zaxis_title="Lift"),
        height=500,
        margin=dict(b=40, l=20, r=20, t=60),
    )
    return fig


def plot_lift_vs_confidence_bubble(rules_df, top_n=100):
    """Bubble chart: Lift vs Confidence với size = Support"""
    rules = rules_df.sort_values("lift", ascending=False).head(top_n)

    # Normalize support for bubble size
    min_support, max_support = rules["support"].min(), rules["support"].max()
    normalized_support = 10 + 30 * (rules["support"] - min_support) / (
        max_support - min_support + 1e-9
    )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=rules["confidence"],
                y=rules["lift"],
                mode="markers",
                marker=dict(
                    size=normalized_support,
                    color=rules["support"],
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(title="Support"),
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=[
                    f"{a} → {c}"
                    for a, c in zip(rules["antecedent"], rules["consequent"])
                ],
                hovertemplate="<b>%{text}</b><br>"
                + "Confidence: %{x:.3f}<br>"
                + "Lift: %{y:.2f}<br>"
                + "Support: %{marker.color:.3f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Bubble Chart: Lift vs Confidence (kích thước = Support)",
        xaxis_title="Confidence",
        yaxis_title="Lift",
        height=450,
        margin=dict(b=40, l=20, r=20, t=60),
    )
    return fig


def plot_antecedent_consequent_network(rules_df, top_n=50):
    """Network graph đơn giản hóa: Antecedent -> Consequent"""
    import plotly.graph_objects as go

    top_rules = rules_df.sort_values("lift", ascending=False).head(top_n)

    # Tạo nodes từ antecedent và consequent
    antecedents = set()
    consequents = set()
    edges = []

    for _, row in top_rules.iterrows():
        ant = str(row["antecedent"])
        cons = str(row["consequent"])
        antecedents.add(ant)
        consequents.add(cons)
        edges.append((ant, cons, row["lift"], row["confidence"]))

    # Tạo layout đơn giản
    all_nodes = list(antecedents.union(consequents))
    node_positions = {}

    # Antecedents ở bên trái, consequents ở bên phải
    ant_list = list(antecedents)
    cons_list = list(consequents)

    for i, ant in enumerate(ant_list):
        node_positions[ant] = (-1, i - len(ant_list) / 2)

    for i, cons in enumerate(cons_list):
        node_positions[cons] = (1, i - len(cons_list) / 2)

    # Vẽ edges
    edge_x, edge_y = [], []
    for ant, cons, lift, conf in edges:
        x0, y0 = node_positions[ant]
        x1, y1 = node_positions[cons]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Vẽ nodes
    node_x = [node_positions[node][0] for node in all_nodes]
    node_y = [node_positions[node][1] for node in all_nodes]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=all_nodes,
        textposition="middle center",
        hoverinfo="text",
        marker=dict(
            size=10,
            color=[
                "lightblue" if node in antecedents else "lightcoral"
                for node in all_nodes
            ],
            line=dict(width=2, color="white"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Network: Antecedent → Consequent",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Antecedents (trái) → Consequents (phải)",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
    )

    return fig


def main():
    st.title("📊 Dashboard Phân tích Luật Kết hợp (Association Rules)")

    # Load data
    try:
        df_raw = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        st.error(f"Không tìm thấy file CSV tại đường dẫn: {CSV_PATH}")
        st.stop()

    # === PHẦN 1: CONTROL PANEL ===
    st.header("🎛️ Bảng Điều Khiển")

    with st.expander("📋 Thông tin tổng quan", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng số luật", len(df_raw))
        with col2:
            st.metric("Support trung bình", f"{df_raw['support'].mean():.3f}")
        with col3:
            st.metric("Confidence trung bình", f"{df_raw['confidence'].mean():.3f}")
        with col4:
            st.metric("Lift trung bình", f"{df_raw['lift'].mean():.2f}")

    with st.expander("🔧 Bộ lọc chính", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📈 Metrics cơ bản")
            min_support = st.slider(
                "Min Support:",
                float(df_raw["support"].min()),
                float(df_raw["support"].max()),
                float(df_raw["support"].quantile(0.1)),
                0.001,
                format="%.3f",
            )
            min_conf = st.slider(
                "Min Confidence:",
                float(df_raw["confidence"].min()),
                float(df_raw["confidence"].max()),
                float(df_raw["confidence"].quantile(0.1)),
                0.05,
                format="%.2f",
            )
            min_lift = st.slider(
                "Min Lift:",
                float(df_raw["lift"].min()),
                float(df_raw["lift"].max()),
                float(df_raw["lift"].quantile(0.2)),
                0.1,
                format="%.1f",
            )

        with col2:
            st.subheader("📊 Metrics mở rộng")
            min_support_antecedent = st.slider(
                "Min Support Antecedent:",
                float(df_raw["support_antecedent"].min()),
                float(df_raw["support_antecedent"].max()),
                float(df_raw["support_antecedent"].min()),
                0.001,
                format="%.3f",
            )
            min_conviction = st.slider(
                "Min Conviction:",
                int(df_raw["conviction"].min()),
                int(df_raw["conviction"].max()),
                int(df_raw["conviction"].min()),
                1,
            )
            min_interest = st.slider(
                "Min Interest:",
                int(df_raw["interest"].min()),
                int(df_raw["interest"].max()),
                int(df_raw["interest"].min()),
                1,
            )

        with col3:
            st.subheader("🎯 Cài đặt khác")
            min_jaccard = st.slider(
                "Min Jaccard:",
                float(df_raw["jaccard"].min()),
                float(df_raw["jaccard"].max()),
                float(df_raw["jaccard"].min()),
                0.001,
                format="%.3f",
            )

    # Áp dụng filter
    df = df_raw[
        (df_raw["support"] >= min_support)
        & (df_raw["confidence"] >= min_conf)
        & (df_raw["lift"] >= min_lift)
        & (df_raw["conviction"] >= min_conviction)
        & (df_raw["support_antecedent"] >= min_support_antecedent)
        & (df_raw["interest"] >= min_interest)
        & (df_raw["jaccard"] >= min_jaccard)
    ]

    # Hiển thị kết quả filter
    st.info(f"✅ Sau khi lọc: {len(df)} luật (từ {len(df_raw)} luật ban đầu)")

    if len(df) == 0:
        st.warning(
            "⚠️ Không có luật nào thỏa mãn điều kiện lọc. Vui lòng điều chỉnh bộ lọc."
        )
        st.stop()

    max_n = min(300, len(df))

    # === PHẦN 2: NETWORK GRAPH ===
    st.header("🕸️ Network Graph Chính")

    with st.expander("⚙️ Cài đặt Network Graph", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            top_n = st.slider(
                "Số luật mạnh nhất (top N):",
                5,
                max_n if max_n >= 5 else 5,
                min(30, max_n),
                1,
            )
            min_degree = st.slider("Hiện label node có degree >=:", 1, 10, 3, 1)

        with col2:
            color_by = st.selectbox(
                "Highlight node theo:", ["out_degree", "in_degree"], index=0
            )
            label_degree_type = st.selectbox(
                "Hiện label cho node theo:",
                ["out_degree", "in_degree", "both"],
                index=2,
            )

        with col3:
            highlight_quantile = st.slider(
                "Highlight node top % lớn:", 0.7, 1.0, 0.9, 0.01
            )
            layout_type = st.selectbox(
                "Layout Network Graph:", ["spring", "circular"], index=0
            )

    # Vẽ network graph
    fig, G = plot_association_network_streamlit(
        df,
        top_n=top_n,
        min_degree=min_degree,
        color_by=color_by,
        highlight_quantile=highlight_quantile,
        label_degree_type=label_degree_type,
        layout_type=layout_type,
    )
    st.plotly_chart(fig, use_container_width=True)

    # === PHẦN 3: CHI TIẾT NODE ===
    st.header("🔍 Chi tiết Node")

    main_nodes = [
        n
        for n in G.nodes()
        if G.out_degree(n) >= min_degree or G.in_degree(n) >= min_degree
    ]

    if main_nodes:
        col1, col2 = st.columns([1, 2])

        with col1:
            selected_node = st.selectbox(
                "Chọn node để xem các luật liên quan:", main_nodes, key="node_select"
            )

            # Thông tin node
            st.metric("Luật đi từ node này", G.out_degree(selected_node))
            st.metric("Luật đến node này", G.in_degree(selected_node))

        with col2:
            detail_rules = df[
                df["antecedent"].apply(lambda x: selected_node in str(x))
                | df["consequent"].apply(lambda x: selected_node in str(x))
            ]

            st.subheader(f"📋 Các luật liên quan đến '{selected_node}'")
            st.dataframe(
                detail_rules[
                    ["antecedent", "consequent", "support", "confidence", "lift"]
                ].head(50),
                use_container_width=True,
            )

            # Xuất CSV
            if not detail_rules.empty:
                csv = detail_rules.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=f"📥 Tải về CSV các luật của '{selected_node}'",
                    data=csv,
                    file_name=f"rules_{selected_node}.csv",
                    mime="text/csv",
                )
    else:
        st.info("ℹ️ Không có node nào đủ điều kiện min_degree hiện tại!")

    # === PHẦN 4: BIỂU ĐỒ PHÂN TÍCH ===
    st.header("📈 Biểu đồ Phân tích Chi tiết")

    # Controls cho biểu đồ
    with st.expander("⚙️ Cài đặt biểu đồ", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            scatter_top_n = st.slider(
                "Số luật hiển thị trên biểu đồ:",
                min_value=10,
                max_value=min(1000, len(df)),
                value=min(200, len(df)),
                step=10,
                key="chart_slider",
            )
        with col2:
            top_items_n = st.slider(
                "Số sản phẩm top hiển thị:",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                key="top_items_slider",
            )

    # Tab cho các biểu đồ khác nhau
    tab1, tab3, tab4 = st.tabs(
        [
            "📊 Biểu đồ Cơ bản",
            "🕸️ Network Analysis",
            "📈 Thống kê Tổng quan",
        ]
    )

    with tab1:
        st.subheader("📊 Các biểu đồ phân tích cơ bản")

        # Hàng 1: Scatter Plot và Histogram
        col1, col2 = st.columns(2)
        with col1:
            scatter_fig = plot_scatter_rules(df, top_n=scatter_top_n)
            st.plotly_chart(scatter_fig, use_container_width=True)

        with col2:
            hist_fig = plot_lift_histogram(df, top_n=scatter_top_n)
            st.plotly_chart(hist_fig, use_container_width=True)

        # Hàng 2: Top Items và Bubble Chart
        col1, col2 = st.columns(2)
        with col1:
            top_items_fig = plot_top_items_frequency(df, top_n=top_items_n)
            st.plotly_chart(top_items_fig, use_container_width=True)

        with col2:
            bubble_fig = plot_lift_vs_confidence_bubble(df, top_n=scatter_top_n)
            st.plotly_chart(bubble_fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            scatter_3d_fig = plot_support_confidence_scatter_3d(df, top_n=scatter_top_n)
            st.plotly_chart(scatter_3d_fig, use_container_width=True)

        with col2:
            corr_fig = plot_metrics_correlation_heatmap(df)
            st.plotly_chart(corr_fig, use_container_width=True)

    with tab3:
        st.subheader("🕸️ Phân tích mạng lưới")

        # Network Graph chính (từ phần trước)
        fig, G = plot_association_network_streamlit(
            df,
            top_n=top_n,
            min_degree=min_degree,
            color_by=color_by,
            highlight_quantile=highlight_quantile,
            label_degree_type=label_degree_type,
            layout_type=layout_type,
        )
        st.plotly_chart(fig, use_container_width=True, key="fig3")

        # Network Graph đơn giản hóa
        st.subheader("🔗 Mạng lưới đơn giản hóa")
        simple_network_fig = plot_antecedent_consequent_network(
            df, top_n=min(10, len(df))
        )
        st.plotly_chart(simple_network_fig, use_container_width=True)

    with tab4:
        st.subheader("📈 Thống kê tổng quan")

        # Bảng thống kê tổng quan
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Thống kê Metrics")
            stats_df = df[
                ["support", "confidence", "lift", "conviction", "jaccard"]
            ].describe()
            st.dataframe(stats_df.round(3), use_container_width=True)

        with col2:
            st.subheader("🏆 Top 10 luật mạnh nhất")
            top_10_rules = df.nlargest(10, "lift")[
                ["antecedent", "consequent", "support", "confidence", "lift"]
            ]
            st.dataframe(top_10_rules, use_container_width=True)

    # === PHẦN 5: XUẤT DỮ LIỆU ===
    st.header("💾 Xuất Dữ liệu")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Xuất tất cả dữ liệu đã lọc
        csv_all = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải về tất cả luật đã lọc (CSV)",
            data=csv_all,
            file_name="filtered_association_rules.csv",
            mime="text/csv",
        )

    with col2:
        # Xuất top rules
        top_rules = df.sort_values("lift", ascending=False).head(scatter_top_n)
        csv_top = top_rules.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=f"📥 Tải về top {scatter_top_n} luật (CSV)",
            data=csv_top,
            file_name=f"top_{scatter_top_n}_rules.csv",
            mime="text/csv",
        )

    # === FOOTER ===
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Dashboard được tạo với ❤️ sử dụng Streamlit | Phân tích Association Rules
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
