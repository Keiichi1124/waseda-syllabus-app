import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import matplotlib.pyplot as plt
import json

# --- 設定 ---
st.set_page_config(page_title="WaseSearch AI", page_icon="🎓", layout="wide")

if "bookmarks" not in st.session_state:
    st.session_state["bookmarks"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- 関数群 ---
def get_type_label(exam, report):
    if exam == 0: return "Report", "📄 レポート重視"
    elif exam >= 80: return "Exam", "✍️ テスト重視"
    else: return "Balance", "⚖️ バランス型"

def cosine_similarity(a, b):
    if len(a) == 0 or len(b) == 0: return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def create_dynamic_pie_chart(score_details_json):
    try:
        scores = json.loads(score_details_json)
    except:
        return None
    if not scores: return None
    labels = list(scores.keys())
    sizes = list(scores.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f']
    fig, ax = plt.subplots(figsize=(2, 2))
    valid_sizes = []
    for s in sizes:
        if s > 0: valid_sizes.append(s)
    if not valid_sizes: return None
    ax.pie(valid_sizes, labels=None, autopct='%1.0f%%', colors=colors[:len(valid_sizes)], startangle=90, textprops={'fontsize': 8, 'color': 'white', 'weight': 'bold'})
    ax.axis('equal')
    fig.patch.set_alpha(0)
    return fig

# --- データ読み込み ---
try:
    df = pd.read_pickle("waseda_syllabus_ai.pkl")
    df = df.fillna({"day_period": "", "exam_score": 0, "report_score": 0, "normal_score": 0, "score_details": "{}"})
    trans_table = str.maketrans("１２３４５６７８９０", "1234567890")
    df["day_period"] = df["day_period"].astype(str).str.translate(trans_table)
    
    if len(df) > 0:
        type_results = df.apply(lambda x: get_type_label(x["exam_score"], x["report_score"]), axis=1).tolist()
        df["type_code"] = [r[0] for r in type_results]
        df["type_text"] = [r[1] for r in type_results]
        df["id"] = df.index.astype(str)
    else:
        df["type_code"] = []
        df["type_text"] = []
        df["id"] = []

except FileNotFoundError:
    st.error("❌ データなし。process_ai.py を実行してください。")
    st.stop()

# --- サイドバー ---
with st.sidebar:
    st.title("🎓 WaseSearch AI")
    
    # ★モード選択
    mode = st.radio("モード選択", ["🔍 授業検索 (一覧)", "🤖 AIコンシェルジュ (チャット)"])
    
    st.divider()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("OpenAI APIキー", type="password")

    if mode == "🔍 授業検索 (一覧)":
        st.header("絞り込み")
        keyword = st.text_input("検索ワード", placeholder="例: 心理学, 楽な授業")
        search_type = st.checkbox("AIふんわり検索を使う", value=True)
        target_day = st.selectbox("曜日", ["指定なし", "月", "火", "水", "木", "金", "土"])
        target_period = st.selectbox("時限", ["指定なし", "1", "2", "3", "4", "5", "6", "7"])
        type_filter = st.multiselect("タイプ", ["📄 レポート重視", "✍️ テスト重視", "⚖️ バランス型"], default=["📄 レポート重視", "✍️ テスト重視", "⚖️ バランス型"])

    st.divider()
    st.header("🔖 Myブックマーク")
    if len(st.session_state["bookmarks"]) == 0:
        st.info("登録なし")
    else:
        current_bookmarks = st.session_state["bookmarks"].copy()
        for i, item in enumerate(current_bookmarks):
            c1, c2 = st.columns([4, 1])
            c1.write(f"✅ {item}")
            if c2.button("🗑", key=f"del_bm_{i}"):
                st.session_state["bookmarks"].remove(item)
                st.rerun()

# ==========================================
#  モード 1: 通常検索 (これまでの機能)
# ==========================================
if mode == "🔍 授業検索 (一覧)":
    st.subheader("🔍 授業検索モード")

    filtered_df = df.copy()

    if target_day != "指定なし": filtered_df = filtered_df[filtered_df["day_period"].str.contains(target_day, na=False)]
    if target_period != "指定なし": filtered_df = filtered_df[filtered_df["day_period"].str.contains(target_period, na=False)]

    selected_types = []
    if "📄 レポート重視" in type_filter: selected_types.append("Report")
    if "✍️ テスト重視" in type_filter: selected_types.append("Exam")
    if "⚖️ バランス型" in type_filter: selected_types.append("Balance")
    filtered_df = filtered_df[filtered_df["type_code"].isin(selected_types)]

    if keyword:
        if search_type:
            if not api_key:
                st.error("⚠️ APIキーが必要です")
            else:
                with st.spinner("🧠 AI思考中..."):
                    try:
                        client = OpenAI(api_key=api_key)
                        res = client.embeddings.create(input=keyword, model="text-embedding-3-small")
                        query_vec = res.data[0].embedding
                        valid_indices = filtered_df[filtered_df["embedding"].apply(lambda x: len(x) > 0)].index
                        if len(valid_indices) > 0:
                            scores = filtered_df.loc[valid_indices, "embedding"].apply(lambda x: cosine_similarity(x, query_vec))
                            filtered_df.loc[valid_indices, "similarity"] = scores
                            filtered_df = filtered_df.sort_values("similarity", ascending=False)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            filtered_df = filtered_df[
                filtered_df["title"].str.contains(keyword, case=False) | 
                filtered_df["instructor"].str.contains(keyword, case=False) |
                filtered_df["ai_summary"].str.contains(keyword, case=False)
            ]

    st.markdown(f"**ヒット数: {len(filtered_df)} 件**")

    if len(filtered_df) == 0:
        st.warning("条件に合う授業が見つかりませんでした。")
    else:
        # 表示制限（重くなるので50件まで）
        display_df = filtered_df.head(50)
        
        for index, row in display_df.iterrows():
            anchor_id = f"course_{row['id']}"
            with st.container(border=True):
                c_head, c_btn = st.columns([4, 1])
                with c_head:
                    st.subheader(f"📖 {row['title']}", anchor=anchor_id)
                    st.caption(f"👨‍🏫 {row['instructor']} | ⏱ {row['day_period']}")
                with c_btn:
                    bm_key = f"{row['title']} ({row['day_period']})"
                    if bm_key in st.session_state["bookmarks"]:
                        if st.button("🗑 解除", key=f"btn_remove_{index}"):
                            st.session_state["bookmarks"].remove(bm_key)
                            st.rerun()
                    else:
                        if st.button("＋追加", key=f"btn_add_{index}"):
                            st.session_state["bookmarks"].append(bm_key)
                            st.rerun()

                t_code = row['type_code']
                if t_code == "Report": st.info(row['type_text'])
                elif t_code == "Exam": st.error(row['type_text'])
                else: st.warning(row['type_text'])
                
                st.markdown(f"**🤖 先輩AI:** {row['ai_summary']}")
                st.divider()
                
                col_chart, col_data, col_rec = st.columns([1.2, 1.8, 2])
                with col_chart:
                    if "score_details" in row and row["score_details"]:
                        fig = create_dynamic_pie_chart(row['score_details'])
                        if fig: st.pyplot(fig, use_container_width=True, transparent=True)
                with col_data:
                    if "score_details" in row and row["score_details"]:
                        try:
                            scores = json.loads(row["score_details"])
                            for k, v in scores.items():
                                st.write(f"🔹 {k}: **{v}%**")
                        except: st.write("-")
                with col_rec:
                    if len(row['embedding']) > 0:
                        st.markdown("**💡 似ている授業:**")
                        similarities = df[df["id"] != row["id"]]["embedding"].apply(lambda x: cosine_similarity(x, row['embedding']) if len(x)>0 else 0)
                        valid_similarities = similarities[similarities > 0]
                        top_similar = df.loc[valid_similarities.nlargest(5).index]
                        for _, sim_row in top_similar.iterrows():
                            # 同じモード内ではないのでリンクは機能しにくいが参考表示
                            st.caption(f"・{sim_row['title']}")
                st.link_button("🔗 公式シラバス", row['url'])

# ==========================================
#  モード 2: AIコンシェルジュ (チャット)
# ==========================================
elif mode == "🤖 AIコンシェルジュ (チャット)":
    st.subheader("🤖 履修相談 AIコンシェルジュ")
    st.caption("あなたの希望をチャットで伝えてください。AIがシラバス全体から最適な授業を探して提案します。")

    if not api_key:
        st.warning("⚠️ 左のサイドバーでOpenAI APIキーを入力してください。")
    else:
        # チャット履歴の表示
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # ユーザー入力
        if prompt := st.chat_input("例: 金曜日の午後で、レポートだけで単位が取れる面白い授業ある？"):
            # 1. ユーザーの入力を表示
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. RAG処理（検索 + 回答生成）
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🤔 シラバスを検索中...")

                try:
                    client = OpenAI(api_key=api_key)
                    
                    # A. 質問をベクトル化して検索
                    res = client.embeddings.create(input=prompt, model="text-embedding-3-small")
                    query_vec = res.data[0].embedding
                    
                    # 類似度計算 (全データ対象)
                    # 高速化のため、ベクトルがある行だけ対象
                    valid_df = df[df["embedding"].apply(lambda x: len(x) > 0)].copy()
                    if len(valid_df) > 0:
                        valid_df["similarity"] = valid_df["embedding"].apply(lambda x: cosine_similarity(x, query_vec))
                        # 上位 8 件を取得（コンテキストとしてAIに渡す）
                        top_results = valid_df.sort_values("similarity", ascending=False).head(8)
                    else:
                        top_results = pd.DataFrame()

                    # B. AIに渡す情報の作成
                    context_text = ""
                    if len(top_results) > 0:
                        context_text = "【検索された授業候補】\n"
                        for _, row in top_results.iterrows():
                            # 評価方法の詳細を取得
                            score_str = row['score_details'] if row['score_details'] else "詳細なし"
                            context_text += f"- 授業名: {row['title']}\n"
                            context_text += f"  教員: {row['instructor']} | 時間: {row['day_period']}\n"
                            context_text += f"  評価方法: {score_str}\n"
                            context_text += f"  概要: {row['ai_summary']}\n"
                            context_text += "---\n"
                    else:
                        context_text = "該当する授業が見つかりませんでした。"

                    # C. AIへの指示（プロンプト）
                    system_prompt = f"""
                    あなたは早稲田大学の「履修登録のプロ」です。
                    ユーザーの質問に対し、以下の【検索された授業候補】の情報を元に、具体的でおすすめの授業を提案してください。

                    ルール:
                    1. 必ず提供された授業データの中から答えること。架空の授業をでっち上げないこと。
                    2. 「〜という授業があります」だけでなく、「なぜそれがおすすめか（評価方法や内容）」を補足すること。
                    3. 質問と関係ない授業は紹介しないこと。
                    4. フランクで親しみやすい口調（先輩のような話し方）で。

                    {context_text}
                    """

                    # D. 回答生成
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )
                    
                    bot_response = response.choices[0].message.content
                    
                    # 表示と履歴保存
                    message_placeholder.markdown(bot_response)
                    st.session_state["chat_history"].append({"role": "assistant", "content": bot_response})
                    
                    # 参考データとしてヒットした授業を下にカード表示（任意）
                    with st.expander("📚 参考にした授業リスト"):
                        for _, row in top_results.iterrows():
                            st.write(f"**{row['title']}** ({row['day_period']}) - {row['instructor']}")

                except Exception as e:
                    message_placeholder.markdown(f"エラーが発生しました: {e}")