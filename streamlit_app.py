# Streamlit Web UI for Dual-Agent (Tsukuba × Revitalization)
# -----------------------------------------------------------
# 依存: pip install streamlit openai python-dotenv tiktoken rich
# 実行: streamlit run streamlit_app.py
# 環境: OPENAI_API_KEY を事前に設定しておく

import os
import io
from datetime import datetime
import streamlit as st

# 既存のロジックを再利用
from dual_agent_tsukuba_service import (
    build_indexes,
    AgentConfig,
    DebateOrchestrator,
    sys_prompt_tsukuba,
    sys_prompt_revitalization,
)

# --- 初期設定 ---------------------------------------------------------------
st.set_page_config(page_title="Tsukuba Dual-Agent", layout="wide")
st.title("🗻 筑波山麓 × 地方創生 デュアルAI")
st.caption("ローカル知見 × 制度・事例知見 で提案を共同作成するデモ UI")

# サイドバー: オプション
def sidebar_controls():
    with st.sidebar:
        st.header("⚙️ オプション")
        rounds = st.number_input("ラウンド数", min_value=1, max_value=5, value=2, step=1)
        k = st.number_input("各RAGの上位件数", min_value=1, max_value=10, value=4, step=1)
        model = st.text_input("生成モデル", value="gpt-4.1-mini")
        embed = st.text_input("埋め込みモデル", value="text-embedding-3-large")
        show_all_dialogue = st.checkbox("全ての会話を表示", value=False)
        return rounds, k, model, embed, show_all_dialogue

rounds, k, model, embed, show_all_dialogue = sidebar_controls()

# --- インデックスの構築（初回のみ） -----------------------------------------
@st.cache_resource(show_spinner=True)
def get_orchestrator(model_name: str):
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.stop()

    client = OpenAI()

    idx1, idx2 = build_indexes(client)
    a1 = AgentConfig(name="筑波山麓エキスパートAI", system_prompt=sys_prompt_tsukuba(), index=idx1)
    a2 = AgentConfig(name="地方創生エキスパートAI", system_prompt=sys_prompt_revitalization(), index=idx2)
    orch = DebateOrchestrator(client, a1, a2, gen_model=model_name)
    return orch

orch = get_orchestrator(model)

# --- 入力欄 -----------------------------------------------------------------
st.subheader("💬 質問（ユーザー入力）")
user_query = st.text_area(
    "例）空き家活用と観光回遊性を同時に高めるには？",
    height=100,
)
col1, col2 = st.columns([1,1])
run_clicked = col1.button("実行する 🚀", type="primary")
clear_clicked = col2.button("クリア")
if clear_clicked:
    st.session_state.pop("last_result", None)

# --- 実行 -------------------------------------------------------------------
if run_clicked and user_query.strip():
    with st.spinner("エージェントが議論中…"):
        result = orch.run(user_query, rounds=rounds, k=k)
        st.session_state["last_result"] = result

result = st.session_state.get("last_result")

# --- 表示 -------------------------------------------------------------------
if result:
    a1_topk = result["agent1_topk"]
    a2_topk = result["agent2_topk"]

    st.subheader("🔎 参考資料 上位ヒット")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 筑波山麓")
        for r in a1_topk:
            st.write(f"**{r['score']:.3f}** – {r['source']}")
    with c2:
        st.markdown("#### 地方創生")
        for r in a2_topk:
            st.write(f"**{r['score']:.3f}** – {r['source']}")

    st.subheader("🗣️ 議論ログ")
    dialogue = result["dialogue"]
    view_items = dialogue if show_all_dialogue else dialogue[-4:]
    for speaker, content in view_items:
        with st.expander(speaker, expanded=True):
            st.markdown(content)

    st.subheader("🧭 統合サマリ")
    st.markdown(result["summary"]) 

    # --- Markdown ダウンロード -------------------------------------------------
    def build_markdown(result_dict):
        lines = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"# 実行結果 ({ts})\n")
        lines.append("## 参考資料 上位ヒット: 筑波山麓\n")
        lines.extend([f"- {r['score']:.3f} | {r['source']}" for r in result_dict["agent1_topk"]])
        lines.append("\n## 参考資料 上位ヒット: 地方創生\n")
        lines.extend([f"- {r['score']:.3f} | {r['source']}" for r in result_dict["agent2_topk"]])
        lines.append("\n## 議論ログ\n")
        for speaker, content in result_dict["dialogue"]:
            lines.append(f"### {speaker}\n\n{content}\n")
        lines.append("\n## 統合サマリ\n")
        lines.append(result_dict["summary"])
        return "\n".join(lines)

    md_text = build_markdown(result)
    st.download_button(
        label="⬇️ Markdownをダウンロード",
        data=md_text,
        file_name="result.md",
        mime="text/markdown",
    )

    st.caption("※ PDF化はVS CodeのMarkdown PDF拡張やPandocをご利用ください")
else:
    st.info("左のサイドバーでオプションを設定し、上のテキストエリアに質問を書いて『実行する』を押してください。")
