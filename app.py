import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clustering Universe",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;400;600;700&display=swap');

html, body, .main, [data-testid="stAppViewContainer"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #050A14;
    color: #C8D8F0;
}

/* Fix for sliders appearing stuck or poorly styled */
.stSlider [data-baseweb="slider"] {
    margin-bottom: 25px;
}
.stSlider div[role="slider"] {
    background-color: #7DF9FF !important;
    border: 2px solid #A78BFA !important;
}

.stApp {
    background: radial-gradient(ellipse at top left, #0d1b3e 0%, #050A14 50%, #0a0514 100%);
}

h1 { font-family: 'Space Mono', monospace; color: #7DF9FF; letter-spacing: 2px; }
h2 { font-family: 'Space Mono', monospace; color: #A78BFA; }
h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; color: #67E8F9; font-size: 1.3rem; }

.stSidebar > div:first-child {
    background: linear-gradient(180deg, #0d1b3e 0%, #0a0a1a 100%);
    border-right: 1px solid #1e3a6e;
}

.concept-card {
    background: linear-gradient(135deg, rgba(13,27,62,0.8) 0%, rgba(10,5,20,0.9) 100%);
    border: 1px solid rgba(125,249,255,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    box-shadow: 0 0 20px rgba(125,249,255,0.05);
}

.algo-card {
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
}

.metric-box {
    background: rgba(103,232,249,0.06);
    border: 1px solid rgba(103,232,249,0.25);
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}

.prompt-block {
    background: rgba(0,0,0,0.4);
    border-left: 3px solid #7DF9FF;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #a0c4ff;
    margin: 10px 0;
}

.highlight {
    color: #7DF9FF;
    font-weight: 700;
}

.tag {
    display: inline-block;
    background: rgba(167,139,250,0.2);
    border: 1px solid rgba(167,139,250,0.5);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    color: #c4b5fd;
    margin: 2px;
}

.section-divider {
    border: none;
    border-top: 1px solid rgba(125,249,255,0.15);
    margin: 24px 0;
}

.stButton > button {
    background: linear-gradient(135deg, #1e3a6e, #2d1b69);
    color: #7DF9FF;
    border: 1px solid rgba(125,249,255,0.4);
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2d5a9e, #4c1d95);
    border-color: #7DF9FF;
    box-shadow: 0 0 15px rgba(125,249,255,0.3);
}

.stSelectbox label, .stSlider label, .stRadio label {
    color: #67E8F9 !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] { color: #7DF9FF; font-family: 'Space Mono', monospace; }
[data-testid="stMetricLabel"] { color: #a78bfa; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,27,62,0.5);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #67E8F9;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(125,249,255,0.15) !important;
    color: #7DF9FF !important;
    border-radius: 8px;
}

code {
    background: rgba(125,249,255,0.08) !important;
    color: #7DF9FF !important;
    border-radius: 4px;
    padding: 2px 6px;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ────────────────────────────────────────────────────────────────────
SPACE_COLORS = ['#7DF9FF', '#A78BFA', '#F472B6', '#34D399', '#FBBF24', '#FB7185', '#60A5FA', '#C084FC']
CMAP = ListedColormap(SPACE_COLORS[:6])

def space_fig(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#050A14')
    ax.set_facecolor('#080F20')
    ax.tick_params(colors='#67E8F9', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a6e')
    ax.xaxis.label.set_color('#67E8F9')
    ax.yaxis.label.set_color('#67E8F9')
    ax.title.set_color('#7DF9FF')
    return fig, ax

def starfield(ax, n=120):
    rng = np.random.default_rng(42)
    sx = rng.uniform(*ax.get_xlim(), n)
    sy = rng.uniform(*ax.get_ylim(), n)
    ax.scatter(sx, sy, s=rng.uniform(0.3, 2, n), c='white', alpha=0.25, zorder=0)

def render_prompt(text):
    st.markdown(f'<div class="prompt-block">💬 {text}</div>', unsafe_allow_html=True)

def render_card(content):
    st.markdown(f'<div class="concept-card">{content}</div>', unsafe_allow_html=True)

def render_algo_card(content):
    st.markdown(f'<div class="algo-card">{content}</div>', unsafe_allow_html=True)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🪐 Clustering Universe")
    st.markdown("*Navigate the cosmos of unsupervised learning*")
    st.markdown("---")

    topic = st.radio(
        "📡 SELECT TOPIC",
        [
            "🌌 Introduction to Clustering",
            "⚙️ K-Means Clustering",
            "🌿 Hierarchical Clustering",
            "🌊 DBSCAN",
            "📊 Cluster Evaluation Metrics",
            "🔭 Choosing the Right Algorithm",
            "🛰️ Real-World Applications",
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4a6a9e; font-family: Space Mono, monospace;'>
    🛸 Antigravity Prompt Engine<br>
    v2.0 · Clustering Edition<br>
    Each section powered by AI prompts
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TOPIC 1 — INTRODUCTION
# ════════════════════════════════════════════════════════════════════════════════
if topic == "🌌 Introduction to Clustering":
    st.title("🌌 Introduction to Clustering")
    st.markdown("#### *Discovering hidden constellations in your data*")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        render_prompt("Explain clustering as if you're an astronomer discovering new galaxies — what is it, why do we need it, and how does it differ from classification?")

        st.markdown("""
        <div class="concept-card">
        <h3>🔭 What is Clustering?</h3>
        Clustering is an <span class="highlight">unsupervised learning</span> technique that groups similar 
        data points together — without any pre-labeled examples. Like an astronomer grouping stars into 
        constellations purely by proximity, a clustering algorithm finds natural structure in raw data.

        <br><br><b>Key distinction:</b><br>
        • <span class="highlight">Classification</span> → learns from labeled data (supervised)<br>
        • <span class="highlight">Clustering</span> → discovers patterns with no labels (unsupervised)
        </div>
        """, unsafe_allow_html=True)

        render_prompt("Give me 5 real-world scenarios where clustering is the right tool, with one sentence explaining what the algorithm would discover in each case.")

        st.markdown("""
        <div class="concept-card">
        <h3>🌍 Real-World Use Cases</h3>
        <table style="width:100%; font-size:0.9rem; color:#C8D8F0;">
        <tr><td>🛍️ <b>E-commerce</b></td><td>Group shoppers by purchase behavior</td></tr>
        <tr><td>🧬 <b>Genomics</b></td><td>Identify gene expression profiles</td></tr>
        <tr><td>🌐 <b>Cybersecurity</b></td><td>Detect anomalous network traffic</td></tr>
        <tr><td>📰 <b>NLP</b></td><td>Cluster news articles by topic</td></tr>
        <tr><td>🏥 <b>Healthcare</b></td><td>Find patient sub-populations for treatment</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🎲 Live Demo: Cluster Emergence")
        n_clusters_demo = st.slider("Number of natural clusters", 2, 6, 3, key="intro_k")
        noise = st.slider("Data noise level", 0.1, 2.0, 0.6, key="intro_n")

        X_demo, y_demo = make_blobs(n_samples=300, centers=n_clusters_demo, cluster_std=noise) # Removed fixed random_state
        fig, ax = space_fig((6, 5))
        ax.scatter(X_demo[:, 0], X_demo[:, 1], s=18, c='#7DF9FF', alpha=0.6, edgecolors='none')
        ax.set_title(f"Raw Data: {n_clusters_demo} Clusters Emerged", fontsize=11)
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
        starfield(ax)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"""
        <div class="metric-box">
        <b style='color:#A78BFA'>The Challenge</b><br>
        <span style='font-size:0.85rem'>Can you spot the {n_clusters_demo} natural groups?<br>
        Clustering algorithms do this automatically — at any scale.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ The Clustering Landscape")

    cols = st.columns(4)
    algos = [
        ("⚙️", "Partitional", "K-Means, K-Medoids", "Divides data into K distinct groups"),
        ("🌿", "Hierarchical", "Ward, Complete, Average", "Builds a tree of nested clusters"),
        ("🌊", "Density-based", "DBSCAN, HDBSCAN", "Finds dense regions, handles noise"),
        ("🔀", "Distribution", "GMM, EM Algorithm", "Fits statistical distributions to data"),
    ]
    for col, (icon, name, examples, desc) in zip(cols, algos):
        with col:
            st.markdown(f"""
            <div class="metric-box" style="height:160px">
            <div style="font-size:2rem">{icon}</div>
            <b style='color:#7DF9FF'>{name}</b><br>
            <span style='color:#a78bfa; font-size:0.8rem'>{examples}</span><br>
            <span style='font-size:0.8rem; color:#8a9ab0'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — K-MEANS
# ════════════════════════════════════════════════════════════════════════════════
elif topic == "⚙️ K-Means Clustering":
    st.title("⚙️ K-Means Clustering")
    st.markdown("#### *Gravity wells pulling data points into orbits*")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📖 Concept & Algorithm", "🎮 Interactive Demo", "📐 Elbow Method"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            render_prompt("Explain the K-Means algorithm step-by-step using the metaphor of gravitational attraction — centroids as planets pulling nearby data points into orbit. Include the math for centroid update.")

            st.markdown("""
            <div class="concept-card">
            <h3>🪐 The Algorithm: 4 Orbital Steps</h3>

            <b>Step 1 — Initialize</b> 🌟<br>
            Place K centroids randomly (or via K-Means++)<br><br>

            <b>Step 2 — Assign</b> ⟶<br>
            Each point joins its nearest centroid:<br>
            <code>label(x) = argmin_k ||x − μk||²</code><br><br>

            <b>Step 3 — Update</b> 🔄<br>
            Recompute centroid as cluster mean:<br>
            <code>μk = (1/|Ck|) · Σ x∈Ck x</code><br><br>

            <b>Step 4 — Repeat</b> until convergence<br>
            Inertia stops decreasing significantly
            </div>
            """, unsafe_allow_html=True)

        with col2:
            render_prompt("List the key assumptions K-Means makes about data, and describe exactly when it will fail with a concrete example for each failure mode.")

            st.markdown("""
            <div class="concept-card">
            <h3>⚠️ Assumptions & Failure Modes</h3>
            <b style='color:#F472B6'>Assumes spherical clusters</b><br>
            Fails on elongated or crescent shapes (use DBSCAN)<br><hr style='border-color:#1e3a6e'>
            <b style='color:#F472B6'>Assumes equal-sized clusters</b><br>
            One large + one tiny cluster → centroids drift<br><hr style='border-color:#1e3a6e'>
            <b style='color:#F472B6'>Sensitive to outliers</b><br>
            Outliers distort centroid positions<br><hr style='border-color:#1e3a6e'>
            <b style='color:#F472B6'>Requires K upfront</b><br>
            Must try multiple K values with elbow/silhouette
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🎮 Watch K-Means Converge in Real-Time")
        col_ctrl, col_plot = st.columns([1, 2])

        with col_ctrl:
            k_val = st.slider("K (clusters)", 2, 8, 3)
            n_pts = st.slider("Data points", 100, 600, 250, step=50)
            data_shape = st.selectbox("Data shape", ["Blobs", "Moons (will fail)", "Circles (will fail)"])
            show_voronoi = st.checkbox("Show Voronoi regions", True)
            run_km = st.button("🚀 RUN K-MEANS")

        with col_plot:
            if data_shape == "Blobs":
                X_km, _ = make_blobs(n_samples=n_pts, centers=k_val, cluster_std=0.9, random_state=7)
            elif data_shape == "Moons (will fail)":
                X_km, _ = make_moons(n_samples=n_pts, noise=0.1, random_state=7)
            else:
                X_km, _ = make_circles(n_samples=n_pts, noise=0.05, factor=0.5, random_state=7)

            km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            labels_km = km.fit_predict(X_km)
            centers = km.cluster_centers_

            fig, ax = space_fig((7, 5))
            if show_voronoi:
                xx, yy = np.meshgrid(
                    np.linspace(X_km[:,0].min()-1, X_km[:,0].max()+1, 300),
                    np.linspace(X_km[:,1].min()-1, X_km[:,1].max()+1, 300)
                )
                Z = km.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                ax.contourf(xx, yy, Z, alpha=0.12, cmap=CMAP)

            for i in range(k_val):
                mask = labels_km == i
                ax.scatter(X_km[mask, 0], X_km[mask, 1], s=20, c=SPACE_COLORS[i % len(SPACE_COLORS)],
                           alpha=0.7, edgecolors='none', label=f'Cluster {i+1}')
            ax.scatter(centers[:, 0], centers[:, 1], s=220, c='white', marker='*',
                       edgecolors='#7DF9FF', linewidths=1.5, zorder=5, label='Centroids')
            ax.set_title(f"K-Means (K={k_val}) · Inertia={km.inertia_:.1f}", fontsize=11)
            ax.legend(fontsize=8, facecolor='#080F20', edgecolor='#1e3a6e', labelcolor='#C8D8F0')
            starfield(ax)
            st.pyplot(fig)
            plt.close()

        try:
            sil = silhouette_score(X_km, labels_km)
            c1, c2, c3 = st.columns(3)
            c1.metric("Inertia (WCSS)", f"{km.inertia_:.1f}")
            c2.metric("Silhouette Score", f"{sil:.3f}")
            c3.metric("Iterations", km.n_iter_)
        except:
            pass

    with tab3:
        st.markdown("### 📐 The Elbow Method — Finding Optimal K")
        render_prompt("Explain the elbow method for choosing K in K-Means. Why is finding the 'elbow' sometimes ambiguous, and what complementary metrics should be used alongside it?")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            X_elb, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
            k_range = range(1, 11)
            inertias, sil_scores = [], []
            for k in k_range:
                km_e = KMeans(n_clusters=k, random_state=42, n_init=10)
                km_e.fit(X_elb)
                inertias.append(km_e.inertia_)
                if k > 1:
                    sil_scores.append(silhouette_score(X_elb, km_e.labels_))

            fig, ax = space_fig((6, 4))
            ax.plot(list(k_range), inertias, 'o-', color='#7DF9FF', linewidth=2, markersize=7)
            ax.axvline(x=4, color='#F472B6', linestyle='--', alpha=0.7, label='Optimal K=4')
            ax.fill_between(list(k_range), inertias, alpha=0.08, color='#7DF9FF')
            ax.set_xlabel("K (number of clusters)"); ax.set_ylabel("Inertia (WCSS)")
            ax.set_title("🦾 Elbow Curve", fontsize=11)
            ax.legend(fontsize=8, facecolor='#080F20', edgecolor='#1e3a6e', labelcolor='#C8D8F0')
            starfield(ax)
            st.pyplot(fig)
            plt.close()

        with col_e2:
            fig2, ax2 = space_fig((6, 4))
            ax2.plot(list(k_range)[1:], sil_scores, 's-', color='#A78BFA', linewidth=2, markersize=7)
            best_k = list(k_range)[1:][np.argmax(sil_scores)]
            ax2.axvline(x=best_k, color='#34D399', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
            ax2.fill_between(list(k_range)[1:], sil_scores, alpha=0.08, color='#A78BFA')
            ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
            ax2.set_title("✨ Silhouette Analysis", fontsize=11)
            ax2.legend(fontsize=8, facecolor='#080F20', edgecolor='#1e3a6e', labelcolor='#C8D8F0')
            starfield(ax2)
            st.pyplot(fig2)
            plt.close()

        render_card("<b>💡 Rule of Thumb:</b> Use the elbow curve to narrow down 2–3 candidate values of K, then use the silhouette score to pick the best one. If both agree → high confidence.")


# ════════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — HIERARCHICAL
# ════════════════════════════════════════════════════════════════════════════════
elif topic == "🌿 Hierarchical Clustering":
    st.title("🌿 Hierarchical Clustering")
    st.markdown("#### *Building a cosmic family tree of data points*")
    st.markdown("---")

    col1, col2 = st.columns([2, 3])

    with col1:
        render_prompt("Explain agglomerative hierarchical clustering like you're building a family tree of stars — starting with individual stars and merging them into constellations, then into galaxies. Cover Ward, complete, and average linkage.")

        st.markdown("""
        <div class="concept-card">
        <h3>🌳 Agglomerative Strategy</h3>
        <b>Bottom-Up approach:</b><br>
        Start: each point is its own cluster<br>
        Loop: merge the two closest clusters<br>
        Stop: one cluster remains (full dendrogram)<br><br>
        <b>Linkage Methods:</b><br>
        🔹 <span class="highlight">Ward</span> — minimize within-cluster variance (best general)<br>
        🔹 <span class="highlight">Complete</span> — max distance between clusters (compact)<br>
        🔹 <span class="highlight">Average</span> — mean pairwise distance (balanced)<br>
        🔹 <span class="highlight">Single</span> — min distance, prone to chaining
        </div>
        """, unsafe_allow_html=True)

        linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
        n_clusters_h = st.slider("Cut tree at K clusters", 2, 7, 3)
        n_pts_h = st.slider("Points", 30, 120, 60, step=10)

    with col2:
        X_h, y_h = make_blobs(n_samples=n_pts_h, centers=n_clusters_h, cluster_std=0.8, random_state=21)
        Z = linkage(X_h, method=linkage_method)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#050A14')

        # Dendrogram
        axes[0].set_facecolor('#080F20')
        dendrogram(Z, ax=axes[0], color_threshold=Z[-(n_clusters_h-1), 2],
                   above_threshold_color='#4a6a9e',
                   leaf_rotation=90, leaf_font_size=6)
        axes[0].axhline(y=Z[-(n_clusters_h-1), 2], color='#F472B6', linestyle='--', alpha=0.8)
        axes[0].set_title(f"Dendrogram ({linkage_method} linkage)", color='#7DF9FF', fontsize=11)
        axes[0].tick_params(colors='#67E8F9')
        for spine in axes[0].spines.values():
            spine.set_edgecolor('#1e3a6e')

        # Cluster scatter
        hc = AgglomerativeClustering(n_clusters=n_clusters_h, linkage=linkage_method)
        labels_h = hc.fit_predict(X_h)
        axes[1].set_facecolor('#080F20')
        for i in range(n_clusters_h):
            mask = labels_h == i
            axes[1].scatter(X_h[mask, 0], X_h[mask, 1], s=40,
                            c=SPACE_COLORS[i % len(SPACE_COLORS)], alpha=0.8, label=f'Cluster {i+1}')
        axes[1].set_title("Resulting Clusters", color='#7DF9FF', fontsize=11)
        axes[1].legend(fontsize=8, facecolor='#080F20', edgecolor='#1e3a6e', labelcolor='#C8D8F0')
        axes[1].tick_params(colors='#67E8F9')
        for spine in axes[1].spines.values():
            spine.set_edgecolor('#1e3a6e')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    render_prompt("Compare agglomerative vs divisive hierarchical clustering. When would you prefer hierarchical over K-Means, and what are the computational trade-offs?")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_algo_card("<b>✅ Advantages</b><br>No need to specify K upfront<br>Produces interpretable dendrogram<br>Works with any distance metric<br>Captures nested structure")
    with c2:
        render_algo_card("<b>❌ Disadvantages</b><br>O(n³) time complexity<br>Cannot undo merges (greedy)<br>Sensitive to noise/outliers<br>Slow for large datasets")
    with c3:
        render_algo_card("<b>🎯 Best Used When</b><br>Dataset is small–medium<br>Hierarchical structure expected<br>Number of clusters unknown<br>Interpretability is crucial")


# ════════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — DBSCAN
# ════════════════════════════════════════════════════════════════════════════════
elif topic == "🌊 DBSCAN":
    st.title("🌊 DBSCAN")
    st.markdown("#### *Density-Based Spatial Clustering of Applications with Noise*")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📖 Core Concepts", "🎮 Interactive Explorer"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            render_prompt("Explain DBSCAN using the metaphor of a nebula — core points are dense star-forming regions, border points are on the outskirts, and noise points are isolated in the void. Include the mathematical definition of epsilon and min_samples.")

            st.markdown("""
            <div class="concept-card">
            <h3>🌀 Three Types of Points</h3>

            🔵 <span class="highlight">Core Point</span><br>
            Has ≥ min_samples within distance ε<br>
            Formula: |N(p, ε)| ≥ min_samples<br><br>

            🟡 <span class="highlight">Border Point</span><br>
            Within ε of a core point<br>
            but not dense enough itself<br><br>

            🔴 <span class="highlight">Noise Point</span><br>
            Neither core nor border<br>
            Labeled as -1 (outlier)<br><br>

            <b>Density Reachability:</b><br>
            q is density-reachable from p if<br>
            there's a chain of core points connecting them
            </div>
            """, unsafe_allow_html=True)

        with col2:
            render_prompt("List the scenarios where DBSCAN dramatically outperforms K-Means, and the scenarios where DBSCAN struggles. How does HDBSCAN improve on DBSCAN?")

            st.markdown("""
            <div class="concept-card">
            <h3>⚡ DBSCAN vs K-Means</h3>
            <table style="width:100%; font-size:0.85rem;">
            <tr><th style='color:#7DF9FF'>Property</th><th style='color:#A78BFA'>DBSCAN</th><th style='color:#34D399'>K-Means</th></tr>
            <tr><td>Cluster shape</td><td>Any shape</td><td>Spherical only</td></tr>
            <tr><td>Outlier handling</td><td>Detects noise</td><td>Forces assignment</td></tr>
            <tr><td>K required?</td><td>No</td><td>Yes</td></tr>
            <tr><td>Scalability</td><td>Medium</td><td>High</td></tr>
            <tr><td>Parameters</td><td>ε, min_samples</td><td>K</td></tr>
            <tr><td>Varying density</td><td>Struggles</td><td>Struggles</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🎮 Tune ε and min_samples Live")
        col_ctrl, col_vis = st.columns([1, 2])

        with col_ctrl:
            eps_val = st.slider("ε (epsilon)", 0.1, 2.0, 0.5, step=0.05)
            min_s = st.slider("min_samples", 2, 15, 5)
            db_shape = st.selectbox("Dataset", ["Moons", "Circles", "Blobs + Noise"])

            st.markdown("""
            <div class="metric-box">
            <b style='color:#67E8F9'>Tuning Guide</b><br>
            <span style='font-size:0.8rem'>
            ↑ ε → fewer, larger clusters<br>
            ↓ ε → more noise points<br>
            ↑ min_samples → stricter cores
            </span>
            </div>
            """, unsafe_allow_html=True)

        with col_vis:
            rng = np.random.default_rng(42)
            if db_shape == "Moons":
                X_db, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
            elif db_shape == "Circles":
                X_db, _ = make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=42)
            else:
                X_db, _ = make_blobs(n_samples=250, centers=3, cluster_std=0.5, random_state=42)
                noise_pts = rng.uniform(-4, 4, (30, 2))
                X_db = np.vstack([X_db, noise_pts])

            X_db = StandardScaler().fit_transform(X_db)
            db = DBSCAN(eps=eps_val, min_samples=min_s)
            labels_db = db.fit_predict(X_db)
            n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
            n_noise = np.sum(labels_db == -1)

            fig, ax = space_fig((7, 5))
            unique_labels = sorted(set(labels_db))
            for lbl in unique_labels:
                mask = labels_db == lbl
                if lbl == -1:
                    ax.scatter(X_db[mask, 0], X_db[mask, 1], s=20, c='#FF4444',
                               marker='x', alpha=0.6, label='Noise')
                else:
                    ax.scatter(X_db[mask, 0], X_db[mask, 1], s=20,
                               c=SPACE_COLORS[lbl % len(SPACE_COLORS)],
                               alpha=0.75, edgecolors='none', label=f'Cluster {lbl+1}')
            ax.set_title(f"DBSCAN · ε={eps_val} · min_samples={min_s}", fontsize=11)
            ax.legend(fontsize=8, facecolor='#080F20', edgecolor='#1e3a6e', labelcolor='#C8D8F0')
            starfield(ax)
            st.pyplot(fig)
            plt.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters Found", n_clusters_db)
        c2.metric("Noise Points", n_noise)
        c3.metric("Noise %", f"{100*n_noise/len(X_db):.1f}%")


# ════════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════════
elif topic == "📊 Cluster Evaluation Metrics":
    st.title("📊 Cluster Evaluation Metrics")
    st.markdown("#### *Measuring how well your constellations are drawn*")
    st.markdown("---")

    render_prompt("Explain the three main internal cluster evaluation metrics — silhouette score, Davies-Bouldin index, and Calinski-Harabasz index — with an intuitive analogy for each. State what a 'good' value looks like and the range of each metric.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="concept-card">
        <h3>🔮 Silhouette Score</h3>
        <b>Range:</b> -1 to +1 (higher = better)<br><br>
        For each point:<br>
        <code>s = (b - a) / max(a, b)</code><br><br>
        a = mean intra-cluster distance<br>
        b = mean nearest-cluster distance<br><br>
        <b>Analogy:</b> How much more at home is each star in its own galaxy vs the nearest other galaxy?
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-card">
        <h3>📏 Davies-Bouldin Index</h3>
        <b>Range:</b> 0 to ∞ (lower = better)<br><br>
        <code>DB = (1/K) Σ max(Rij)</code><br>
        where Rij = (σi + σj) / d(ci, cj)<br><br>
        Penalizes clusters that are<br>
        spread out AND close together<br><br>
        <b>Analogy:</b> Galaxies that are big AND packed together = bad score
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="concept-card">
        <h3>🌟 Calinski-Harabasz</h3>
        <b>Range:</b> 0 to ∞ (higher = better)<br><br>
        <code>CH = [tr(Bk)/(K-1)] / [tr(Wk)/(n-K)]</code><br><br>
        Ratio of between-cluster to within-cluster dispersion<br><br>
        <b>Analogy:</b> Reward galaxies that are far apart but internally compact
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Live Metric Comparison Across K Values")

    X_eval, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=55)
    k_vals = range(2, 10)
    sil_vals, db_vals, ch_vals = [], [], []
    for k in k_vals:
        km_e = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km_e.fit_predict(X_eval)
        sil_vals.append(silhouette_score(X_eval, lbl))
        db_vals.append(davies_bouldin_score(X_eval, lbl))
        ch_vals.append(calinski_harabasz_score(X_eval, lbl))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#050A14')
    metrics = [
        (sil_vals, '✦ Silhouette', '#7DF9FF', 'higher is better'),
        (db_vals, '✦ Davies-Bouldin', '#F472B6', 'lower is better'),
        (ch_vals, '✦ Calinski-Harabasz', '#34D399', 'higher is better'),
    ]
    for ax, (vals, name, color, hint) in zip(axes, metrics):
        ax.set_facecolor('#080F20')
        ax.plot(list(k_vals), vals, 'o-', color=color, linewidth=2, markersize=8)
        ax.fill_between(list(k_vals), vals, alpha=0.08, color=color)
        ax.axvline(x=4, color='white', linestyle=':', alpha=0.4, label='True K=4')
        ax.set_title(f"{name}\n({hint})", color=color, fontsize=10)
        ax.set_xlabel("K", color='#67E8F9')
        ax.tick_params(colors='#67E8F9')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a6e')
        ax.legend(fontsize=7, facecolor='#080F20', edgecolor='#1e3a6e', labelcolor='#C8D8F0')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    render_card("💡 <b>Key Insight:</b> All three metrics agree at K=4 for this dataset. In practice, they may disagree — use multiple metrics and domain knowledge together.")


# ════════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — CHOOSING THE RIGHT ALGORITHM
# ════════════════════════════════════════════════════════════════════════════════
elif topic == "🔭 Choosing the Right Algorithm":
    st.title("🔭 Choosing the Right Algorithm")
    st.markdown("#### *Navigating the galaxy of clustering choices*")
    st.markdown("---")

    render_prompt("Create a decision flowchart for choosing between K-Means, Hierarchical, DBSCAN, and GMM. The decisions should depend on: dataset size, expected cluster shape, whether outliers are present, and whether K is known.")

    # Visual comparison table
    st.markdown("### 🗂️ Algorithm Comparison Matrix")
    data = {
        "Algorithm": ["K-Means", "Agglomerative", "DBSCAN", "Mean Shift", "GMM"],
        "Cluster Shape": ["Spherical", "Any (with right linkage)", "Arbitrary", "Arbitrary", "Ellipsoidal"],
        "Handles Noise": ["❌", "❌", "✅", "✅", "❌"],
        "K Required": ["✅", "❌ (cut tree)", "❌", "❌", "✅"],
        "Scalability": ["⚡ High", "🐢 Low O(n³)", "⚡ Medium", "🐢 Low", "⚡ Medium"],
        "Best For": ["Large, compact data", "Hierarchical structure", "Irregular shapes + noise", "Unknown K, varied density", "Overlapping soft clusters"],
    }
    df = pd.DataFrame(data).set_index("Algorithm")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Shape Matters: Side-by-Side Algorithm Performance")

    datasets = {
        "Blobs (K-Means wins)": make_blobs(300, centers=3, cluster_std=0.7, random_state=42),
        "Moons (DBSCAN wins)": make_moons(300, noise=0.08, random_state=42),
        "Circles (DBSCAN wins)": make_circles(300, noise=0.05, factor=0.4, random_state=42),
    }

    chosen_ds = st.selectbox("Select dataset shape", list(datasets.keys()))
    X_cmp, _ = datasets[chosen_ds]
    X_cmp = StandardScaler().fit_transform(X_cmp)

    algos_cmp = {
        "K-Means (K=2)": KMeans(n_clusters=2, random_state=42, n_init=10),
        "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
        "Agglomerative (K=2)": AgglomerativeClustering(n_clusters=2),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#050A14')
    for ax, (aname, algo) in zip(axes, algos_cmp.items()):
        lbl = algo.fit_predict(X_cmp)
        ax.set_facecolor('#080F20')
        unique = sorted(set(lbl))
        for l in unique:
            mask = lbl == l
            col = '#FF4444' if l == -1 else SPACE_COLORS[l % len(SPACE_COLORS)]
            label = 'Noise' if l == -1 else f'Cluster {l+1}'
            ax.scatter(X_cmp[mask, 0], X_cmp[mask, 1], s=15, c=col, alpha=0.7, label=label)
        ax.set_title(aname, color='#7DF9FF', fontsize=10)
        ax.legend(fontsize=7, facecolor='#080F20', edgecolor='#1e3a6e', labelcolor='#C8D8F0')
        ax.tick_params(colors='#67E8F9')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a6e')
    plt.suptitle(f"Dataset: {chosen_ds}", color='#A78BFA', fontsize=12, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — REAL-WORLD APPLICATIONS
# ════════════════════════════════════════════════════════════════════════════════
elif topic == "🛰️ Real-World Applications":
    st.title("🛰️ Real-World Applications")
    st.markdown("#### *Where clustering lands in the real universe*")
    st.markdown("---")

    render_prompt("For each of these 5 industries — retail, healthcare, cybersecurity, NLP, and image processing — give a concrete clustering use case: dataset description, which algorithm to use and why, what the clusters represent, and how the business acts on the results.")

    apps = [
        ("🛍️", "Customer Segmentation", "Retail / E-commerce",
         "K-Means on RFM features (Recency, Frequency, Monetary)",
         ["VIP buyers", "Dormant users", "New customers", "Bargain hunters"],
         "Personalize campaigns, loyalty programs, targeted discounts"),
        ("🧬", "Patient Subgroup Discovery", "Healthcare",
         "Hierarchical / GMM on clinical + genomic features",
         ["High-risk group", "Treatment responders", "Chronic condition clusters"],
         "Tailor treatments, predict adverse events, reduce re-admissions"),
        ("🔐", "Anomaly Detection", "Cybersecurity",
         "DBSCAN on network traffic features (bytes, ports, duration)",
         ["Normal traffic", "Suspicious patterns", "Noise = potential attacks"],
         "Flag outliers for security review, reduce false positives"),
        ("📰", "Topic Discovery", "NLP / Media",
         "K-Means or LDA on TF-IDF embeddings of news articles",
         ["Politics", "Sports", "Technology", "Finance"],
         "Personalized news feeds, content recommendation"),
        ("🖼️", "Image Color Quantization", "Computer Vision",
         "K-Means on RGB pixel values",
         ["Color palettes representing dominant hues"],
         "Image compression, palette extraction, style transfer"),
    ]

    for icon, title, industry, method, clusters, action in apps:
        with st.expander(f"{icon} {title} — {industry}"):
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(f"""
                **Method:** `{method}`

                **Clusters discovered:**
                {chr(10).join(['• ' + c for c in clusters])}

                **Business action:** {action}
                """)
            with col_b:
                # Mini synthetic demo
                rng = np.random.default_rng(hash(title) % 999)
                X_app = rng.normal(size=(200, 2)) * rng.uniform(0.5, 1.5, (1, 2))
                for i in range(len(clusters)):
                    center = rng.uniform(-3, 3, 2)
                    X_app = np.vstack([X_app[:int(200/len(clusters))], rng.normal(center, 0.6, (int(200/len(clusters)), 2))])
                km_app = KMeans(n_clusters=len(clusters), random_state=42, n_init=10)
                lbl_app = km_app.fit_predict(X_app)
                fig, ax = space_fig((4, 3))
                for i, cl in enumerate(clusters):
                    mask = lbl_app == i
                    ax.scatter(X_app[mask, 0], X_app[mask, 1], s=12,
                               c=SPACE_COLORS[i % len(SPACE_COLORS)], alpha=0.7, label=cl[:12])
                ax.set_title(f"{title}", fontsize=9)
                ax.legend(fontsize=6, facecolor='#080F20', edgecolor='#1e3a6e',
                          labelcolor='#C8D8F0', loc='best')
                starfield(ax)
                st.pyplot(fig)
                plt.close()

    st.markdown("---")
    render_prompt("What are the top 3 mistakes practitioners make when applying clustering in production, and how do you fix each one?")

    pitfalls = [
        ("🔴", "Forgetting to scale features", "If features have different ranges (e.g. age 0-100 vs salary 0-100000), distance metrics are dominated by the large-scale feature. Always apply StandardScaler or MinMaxScaler before clustering."),
        ("🟡", "Using the wrong number of K", "Blindly picking K=3 or K=5 without validation. Use elbow + silhouette + domain knowledge together. Consider whether the business actually needs exactly K segments."),
        ("🟠", "Ignoring cluster stability", "A cluster solution that changes dramatically with different random seeds is unreliable. Run with multiple random_state values and check consistency of cluster shapes and sizes."),
    ]
    cols = st.columns(3)
    for col, (dot, title, desc) in zip(cols, pitfalls):
        with col:
            render_algo_card(f"<b>{dot} {title}</b><br><br><span style='font-size:0.85rem'>{desc}</span>")


# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family: 'Space Mono', monospace; font-size:0.72rem; color:#2a4a7e; padding: 12px 0">
🛸 CLUSTERING UNIVERSE · Powered by Antigravity Prompt Engine · Built with Streamlit + scikit-learn
</div>
""", unsafe_allow_html=True)