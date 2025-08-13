# app.py - フーリエ変換Webアプリケーション
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import time
from datetime import datetime

# Matplotlibの日本語フォント設定（オプション）
plt.rcParams['font.family'] = 'DejaVu Sans'
# 訪問者カウンター機能
def init_visitor_counter():
    """訪問者カウンターの初期化"""
    if 'visitor_count' not in st.session_state:
        st.session_state.visitor_count = 0
        st.session_state.visit_history = []
    
    # 新規訪問の判定（セッション開始から5秒以内は同一訪問とみなす）
    current_time = time.time()
    if 'last_visit_time' not in st.session_state or (current_time - st.session_state.last_visit_time) > 5:
        st.session_state.visitor_count += 1
        st.session_state.visit_history.append({
            'count': st.session_state.visitor_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date': datetime.now().strftime('%Y-%m-%d')
        })
        st.session_state.last_visit_time = current_time

def show_visitor_stats():
    """管理者用の訪問者統計表示"""
    with st.expander("📊 訪問者統計 (管理者用)", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("総訪問者数", st.session_state.visitor_count)
        
        with col2:
            # 今日の訪問者数
            today = datetime.now().strftime('%Y-%m-%d')
            today_visits = len([v for v in st.session_state.visit_history if v['date'] == today])
            st.metric("今日の訪問者", today_visits)
        
        with col3:
            # 最新訪問時刻
            if st.session_state.visit_history:
                latest_visit = st.session_state.visit_history[-1]['timestamp']
                st.metric("最新訪問", latest_visit[-8:])  # 時刻のみ表示
        
        # 訪問履歴の詳細表示
        if st.checkbox("詳細履歴を表示"):
            if st.session_state.visit_history:
                # 最新10件を表示
                recent_visits = st.session_state.visit_history[-10:]
                visit_df = pd.DataFrame(recent_visits)
                st.dataframe(visit_df, use_container_width=True)
                
                # CSVダウンロード機能
                if len(st.session_state.visit_history) > 0:
                    csv_data = pd.DataFrame(st.session_state.visit_history).to_csv(index=False)
                    st.download_button(
                        label="📥 訪問履歴をCSVでダウンロード",
                        data=csv_data,
                        file_name=f"visitor_log_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("まだ訪問履歴がありません")

# 訪問者カウンター初期化
init_visitor_counter()

# ページ設定
st.set_page_config(
    page_title="フーリエ変換 解析ツール",
    page_icon="📊",
    layout="wide"
)

# フーリエ変換クラス（軽量最適化版）
class FT_calc:
    def __init__(self, x, y):
        self.x = np.array(x, dtype=np.float32)  # メモリ効率向上
        self.y = np.array(y, dtype=np.float32)
        # 事前計算を簡素化
        self.dx = np.diff(self.x)
        self.y_truncated = self.y[:-1]

    def calculation(self, w):
        # シンプルなベクトル化（元のアルゴリズム保持）
        cos_terms = self.dx * self.y_truncated * np.cos(w * self.x[:-1])
        sin_terms = self.dx * self.y_truncated * np.sin(w * self.x[:-1])
        sum_cos = np.sum(cos_terms)
        sum_sin = np.sum(sin_terms)
        return np.sqrt(sum_cos**2 + sum_sin**2)

# アプリのタイトル
st.title("📊 フーリエ変換 解析ツール")
st.markdown("CSVファイルをアップロードして、フーリエ変換の結果を確認できます。")

# 訪問者統計の表示
show_visitor_stats()

# サイドバーの設定
st.sidebar.header("⚙️ 設定")

# CSVファイルアップロード
uploaded_file = st.sidebar.file_uploader(
    "CSVファイルを選択",
    type=['csv'],
    help="2列のデータ（x座標, y座標）が入ったCSVファイルをアップロードしてください"
)

# パラメータ設定（軽量デフォルト）
w_max = st.sidebar.number_input("最大周波数 (w_max)", min_value=10, max_value=1000, value=20, step=5)
w_step = st.sidebar.number_input("周波数ステップ", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# サンプルデータ生成機能
st.sidebar.markdown("---")
st.sidebar.subheader("📋 サンプルデータ")
if st.sidebar.button("サンプルデータを生成"):
    # サンプルデータ（正弦波 + ノイズ）
    sample_x = np.linspace(0, 4*np.pi, 100)
    sample_y = np.sin(2*sample_x) + 0.5*np.sin(5*sample_x) + 0.1*np.random.randn(100)
    
    sample_df = pd.DataFrame({
        'x': sample_x,
        'y': sample_y
    })
    
    # セッション状態に保存
    st.session_state['sample_data'] = sample_df

# メインコンテンツ
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 入力データ")
    
    # データの読み込み
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSVファイルが正常に読み込まれました")
        except Exception as e:
            st.error(f"❌ ファイル読み込みエラー: {e}")
    elif 'sample_data' in st.session_state:
        df = st.session_state['sample_data']
        st.info("📋 サンプルデータを使用中")
    
    if df is not None:
        # データの表示
        st.write("データプレビュー:")
        st.dataframe(df.head(10))
        
        # データの統計情報
        st.write("データ統計:")
        st.write(f"- データ点数: {len(df)}")
        st.write(f"- x範囲: {df.iloc[:, 0].min():.3f} ～ {df.iloc[:, 0].max():.3f}")
        st.write(f"- y範囲: {df.iloc[:, 1].min():.3f} ～ {df.iloc[:, 1].max():.3f}")
        
        # 入力データのプロット
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(df.iloc[:, 0], df.iloc[:, 1], 'b-', linewidth=1.5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('y vs x')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

with col2:
    st.subheader("🔄 フーリエ変換結果")
    
    if df is not None:
        # フーリエ変換の実行
        if st.button("🚀 フーリエ変換を実行", type="primary"):
            with st.spinner("計算中..."):
                try:
                    # データの準備
                    x = df.iloc[:, 0].tolist()
                    y = df.iloc[:, 1].tolist()
                    
                    # フーリエ変換の計算（シンプル最適化版）
                    ft = FT_calc(x, y)
                    w_values = np.arange(0, w_max, w_step)
                    
                    # プログレスバー付きで安全に計算
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    sum_results = []
                    
                    total_calculations = len(w_values)
                    for i, w in enumerate(w_values):
                        sum_results.append(ft.calculation(w))
                        
                        # 進捗更新（5%ごと）
                        if i % max(1, total_calculations // 20) == 0:
                            progress = (i + 1) / total_calculations
                            progress_bar.progress(progress)
                            status_text.text(f'計算中... {int(progress * 100)}% ({i+1}/{total_calculations})')
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # 結果のプロット
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.plot(w_values, sum_results, 'r-', linewidth=1.5)
                    ax2.set_xlabel('Frequency (w)')
                    ax2.set_ylabel('Magnitude')
                    ax2.set_title('result')
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    
                    # 統計情報
                    max_idx = np.argmax(sum_results)
                    st.write("📊 **結果の統計:**")
                    st.write(f"- 最大振幅: {max(sum_results):.3f}")
                    st.write(f"- 主要周波数: {w_values[max_idx]:.3f}")
                    st.write(f"- 平均振幅: {np.mean(sum_results):.3f}")
                    
                    # 結果データのダウンロード
                    result_df = pd.DataFrame({
                        'frequency': w_values,
                        'magnitude': sum_results
                    })
                    
                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 結果をCSVでダウンロード",
                        data=csv_data,
                        file_name="fourier_transform_result.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ 計算エラー: {e}")
    else:
        st.info(左のサイドバーからファイルを読み込むか、サンプルデータを生成してください)

# フッター
# フッター
st.markdown("---")

# 訪問者数を右下に小さく表示
col1, col2 = st.columns([3, 1])
with col2:
    st.caption(f"👥 訪問者数: {st.session_state.visitor_count}")

st.markdown("**使い方:**")
st.markdown("1. サイドバーからCSVファイルをアップロード（または サンプルデータを生成）")
st.markdown("2. 必要に応じて周波数範囲を調整")
st.markdown("3. 'フーリエ変換を実行'ボタンをクリック")
st.markdown("4. 結果を確認し、必要に応じてCSVファイルとしてダウンロード")

st.markdown("**📄 CSVファイル形式:** 1列目x値、2列目y値")
