"""
2エージェント対話サービス（筑波山麓 × 地方創生）
------------------------------------------------
概要:
- エージェント1: 筑波山麓の超ローカル専門家（現地協力会社のフィールドレポートやローカル資料を主データにRAG）
- エージェント2: 地方創生の政策・施策・事例の専門家（制度・補助金・成功事例コーパスを主データにRAG）
- 入力: ユーザーの課題・質問
- 出力: 2エージェントが数ラウンド討論し、合意/対立点を整理、実行可能な「よさげな案」を複数提示

依存:
- Python 3.10+
- pip install openai python-dotenv tiktoken
  （埋め込み/生成は OpenAI API を使用。環境変数 OPENAI_API_KEY を設定）

使い方（例）:
$ export OPENAI_API_KEY=sk-...  # or setx on Windows
$ python dual_agent_tsukuba_service.py --query "空き家活用と観光回遊性を同時に高めるには？" --rounds 2 --k 4

データの置き場所:
- ./data/tsukuba/               # 筑波山麓ローカル資料（txt/md/pdf→txt化）
- ./data/field_reports/         # 協力会社の現地レポート（txt/md）
- ./data/regeneration/          # 地方創生の制度/事例/論文サマリ

メモ:
- PDFは別途テキスト化してから data/ に配置してください
- ベクトルは毎回オンデマンドで作成します（小規模用途想定）。大規模運用は永続ストアに変更してください。
"""
from __future__ import annotations
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

# ---- OpenAI client ---------------------------------------------------------
# openai>=1.0 のモダンSDK
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("openai パッケージが見つかりません。`pip install openai` を実行してください")

# ---- 表示強化（Richは任意依存） -------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    console: Console | None = Console()
except Exception:
    console = None

from datetime import datetime
from pathlib import Path

# ---- 簡易ユーティリティ -----------------------------------------------------
import glob
import re
import math
import tiktoken


def read_corpus(dirpath: str) -> List[Tuple[str, str]]:
    """ディレクトリ配下の .txt/.md を読み込み (path, text) のリストを返す"""
    paths = sorted(
        list(glob.glob(os.path.join(dirpath, "**/*.txt"), recursive=True)) +
        list(glob.glob(os.path.join(dirpath, "**/*.md"), recursive=True))
    )
    docs: List[Tuple[str,str]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
                if txt.strip():
                    docs.append((p, txt))
        except Exception:
            pass
    return docs


def chunk_text(text: str, *, max_tokens: int = 600, overlap: int = 100, model: str = "gpt-4o-mini") -> List[str]:
    """トークン数ベースでテキストを分割（RAG向け）"""
    enc = tiktoken.encoding_for_model(model)
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        window = toks[i : i + max_tokens]
        chunks.append(enc.decode(window))
        i += max(1, max_tokens - overlap)
    return chunks


# ---- 埋め込み（OpenAI） ------------------------------------------------------

def embed_texts(client: OpenAI, texts: List[str], *, model: str = "text-embedding-3-large") -> List[List[float]]:
    # バッチ化（APIのトークン上限対策）
    BATCH = 64
    vecs: List[List[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return vecs


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class RAGIndex:
    name: str
    docs: List[Tuple[str, str]]
    chunks: List[str] = field(default_factory=list)
    meta: List[Dict] = field(default_factory=list)
    vecs: List[List[float]] = field(default_factory=list)

    def build(self, client: OpenAI, *, embed_model: str = "text-embedding-3-large"):
        self.chunks = []
        self.meta = []
        for path, text in self.docs:
            for ch in chunk_text(text):
                self.chunks.append(ch)
                self.meta.append({"source": path})
        if self.chunks:
            self.vecs = embed_texts(client, self.chunks, model=embed_model)
        else:
            self.vecs = []

    def search(self, client: OpenAI, query: str, *, top_k: int = 5, embed_model: str = "text-embedding-3-large") -> List[Dict]:
        if not self.vecs:
            return []
        qv = embed_texts(client, [query], model=embed_model)[0]
        scored = [
            (cosine(qv, v), i) for i, v in enumerate(self.vecs)
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, idx in scored[:top_k]:
            results.append({
                "score": float(score),
                "text": self.chunks[idx],
                "source": self.meta[idx]["source"],
            })
        return results


# ---- エージェント定義 --------------------------------------------------------

def sys_prompt_tsukuba() -> str:
    return (
        """
あなたは筑波山麓（つくば・石岡・桜川など）に極めて詳しいローカル専門家AIです。
目的: ユーザーの地域課題を、現地の実情・季節性・交通・土地利用・歴史文化・観光動線・地権者/自治会の文脈から、実行可能で小さく始められる提案へ落とし込む。
ルール:
- 主張には根拠（出典ファイル名・フィールドレポート）を必ず添える。
- ハード整備/ソフト事業/合意形成/資金計画/KPI を具体化し、最小実験(MVP)→評価→拡張の段取りを出す。
- 観光だけでなく、住民生活・防災・環境保全・農林業との両立を重視する。
- 机上推論だけにせず、現地検証が要る点は「現地確認TODO」に列挙する。
        """
    ).strip()


def sys_prompt_revitalization() -> str:
    return (
        """
あなたは地方創生・地域経済政策の専門家AIです。
目的: 既存制度（補助金・交付金・PPP/PFI・官民連携・DMO/観光地域づくり法人）や、国内外の成功失敗事例の示唆から、資金調達・事業スキーム・人材/ガバナンス設計を提案する。
ルール:
- 提案は実施主体・意思決定プロセス・関係者マップ・予算規模レンジ・評価指標を含める。
- 記載する制度名称や事例は、出典ファイル名を根拠として併記する。
- 現地事情の制約（季節・地理・観光動線・地権者調整）をローカル専門家の指摘と突き合わせ、過剰な机上設計を避ける。
        """
    ).strip()


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    index: RAGIndex


class DebateOrchestrator:
    def __init__(self, client: OpenAI, agent1: AgentConfig, agent2: AgentConfig, *, gen_model: str = "gpt-4.1-mini"):
        self.client = client
        self.agent1 = agent1
        self.agent2 = agent2
        self.gen_model = gen_model

    def _compose_context(self, retrieved: List[Dict]) -> str:
        lines = []
        for r in retrieved:
            lines.append(f"[score={r['score']:.3f}] {r['source']}\n{r['text']}\n")
        return "\n".join(lines[:8])

    def _chat(self, system: str, messages: List[Dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.gen_model,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()

    def run(self, user_query: str, *, rounds: int = 2, k: int = 5) -> Dict:
        # 1) RAG で各エージェントにコンテキストを供給
        r1 = self.agent1.index.search(self.client, user_query, top_k=k)
        r2 = self.agent2.index.search(self.client, user_query, top_k=k)
        ctx1 = self._compose_context(r1)
        ctx2 = self._compose_context(r2)

        # 2) 初期提案
        msg1 = self._chat(
            self.agent1.system_prompt,
            [
                {"role": "user", "content": f"ユーザー質問:\n{user_query}"},
                {"role": "user", "content": f"参考資料(ローカル):\n{ctx1}"},
                {"role": "user", "content": "形式: 箇条書きで3-5個のMVP提案。各提案に根拠(出典)・期待効果・リスク・現地確認TODO・概算費用レンジ(万円)を添える。"},
            ],
        )
        msg2 = self._chat(
            self.agent2.system_prompt,
            [
                {"role": "user", "content": f"ユーザー質問:\n{user_query}"},
                {"role": "user", "content": f"参考資料(制度/事例):\n{ctx2}"},
                {"role": "user", "content": "形式: 箇条書きで3-5個の事業スキーム提案。各提案に制度根拠(出典)・主体/関係者・資金/採算・KPI・想定スケジュールを添える。"},
            ],
        )

        dialogue = [(self.agent1.name, msg1), (self.agent2.name, msg2)]

        # 3) 相互批判と改良ラウンド
        for r in range(rounds):
            critique2 = self._chat(
                self.agent2.system_prompt,
                [
                    {"role": "user", "content": f"相手({self.agent1.name})の提案:\n{msg1}"},
                    {"role": "user", "content": f"参考資料(制度/事例):\n{ctx2}"},
                    {"role": "user", "content": "タスク: 実現性・制度適合性・資金調達の観点で批判と改良案を簡潔に提示。"},
                ],
            )
            critique1 = self._chat(
                self.agent1.system_prompt,
                [
                    {"role": "user", "content": f"相手({self.agent2.name})の提案:\n{msg2}"},
                    {"role": "user", "content": f"参考資料(ローカル):\n{ctx1}"},
                    {"role": "user", "content": "タスク: 現地実務・季節性・動線・地権者調整の観点で批判と改良案を簡潔に提示。"},
                ],
            )
            msg1 = self._chat(
                self.agent1.system_prompt,
                [
                    {"role": "user", "content": f"相手からの批評:\n{critique2}"},
                    {"role": "user", "content": "上記を踏まえ、あなたの提案をアップデートし、重複を整理して3-5個にまとめて。"},
                ],
            )
            msg2 = self._chat(
                self.agent2.system_prompt,
                [
                    {"role": "user", "content": f"相手からの批評:\n{critique1}"},
                    {"role": "user", "content": "上記を踏まえ、あなたの提案をアップデートし、重複を整理して3-5個にまとめて。"},
                ],
            )
            dialogue.append((self.agent2.name, critique2))
            dialogue.append((self.agent1.name, critique1))
            dialogue.append((self.agent1.name, msg1))
            dialogue.append((self.agent2.name, msg2))

        # 4) ファシリテータによる要約・統合プラン
        facilitator_sys = (
            """
あなたは中立のファシリテータです。議論ログを読み、
1) 合意点 / 相違点
2) 実行優先度順の「よさげな案」3-7個（各: 概要・主担当・必要リソース・概算費用・KPI・最初の30日間TODO）
3) リスク/前提条件/現地確認TODO
を日本語で簡潔にまとめてください。表はMarkdown可。
            """.strip()
        )
        transcript = "\n\n".join([f"## {speaker}\n{content}" for speaker, content in dialogue])
        summary = self._chat(
            facilitator_sys,
            [
                {"role": "user", "content": f"ユーザー質問: {user_query}"},
                {"role": "user", "content": f"議論ログ:\n{transcript}"},
            ],
        )

        return {
            "agent1_topk": r1,
            "agent2_topk": r2,
            "dialogue": dialogue,
            "summary": summary,
        }


# ---- メイン ---------------------------------------------------------------

def build_indexes(client: OpenAI) -> Tuple[RAGIndex, RAGIndex]:
    tsukuba_docs = read_corpus("data/tsukuba") + read_corpus("data/field_reports")
    regen_docs = read_corpus("data/regeneration")

    idx1 = RAGIndex(name="tsukuba", docs=tsukuba_docs)
    idx2 = RAGIndex(name="revitalization", docs=regen_docs)

    idx1.build(client)
    idx2.build(client)
    return idx1, idx2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="ユーザー質問/課題")
    parser.add_argument("--rounds", type=int, default=2, help="相互批判ラウンド数")
    parser.add_argument("--k", type=int, default=5, help="各エージェントのRAG上位件数")
    parser.add_argument("--model", default="gpt-4.1-mini", help="生成モデル")
    parser.add_argument("--embed", default="text-embedding-3-large", help="埋め込みモデル")
    parser.add_argument("--save-md", default=None, help="Markdownで結果を保存するパス（例: reports/result.md）")

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が未設定です")

    client = OpenAI()

    # インデックス構築
    idx1, idx2 = build_indexes(client)

    # エージェント構成
    a1 = AgentConfig(name="筑波山麓エキスパートAI", system_prompt=sys_prompt_tsukuba(), index=idx1)
    a2 = AgentConfig(name="地方創生エキスパートAI", system_prompt=sys_prompt_revitalization(), index=idx2)

    orchestrator = DebateOrchestrator(client, a1, a2, gen_model=args.model)
    result = orchestrator.run(args.query, rounds=args.rounds, k=args.k)

    # 出力
        # ------------------ 表示強化: Rich があればリッチ表示 ------------------
    if console is not None:
        console.rule("[bold]参考資料 上位ヒット")
        # 筑波山麓
        t1 = Table(title="筑波山麓", show_lines=True)
        t1.add_column("score", justify="right")
        t1.add_column("source", overflow="fold")
        for r in result["agent1_topk"]:
            t1.add_row(f"{r['score']:.3f}", r["source"])
        console.print(t1)
        # 地方創生
        t2 = Table(title="地方創生", show_lines=True)
        t2.add_column("score", justify="right")
        t2.add_column("source", overflow="fold")
        for r in result["agent2_topk"]:
            t2.add_row(f"{r['score']:.3f}", r["source"])
        console.print(t2)

        console.rule("[bold]議論ログ（直近の発言）")
        for speaker, content in result["dialogue"][-4:]:
            console.print(Panel.fit(
                content[:1500] + ("..." if len(content) > 1500 else ""),
                title=speaker
            ))

        console.rule("[bold]統合サマリ")
        console.print(Markdown(result["summary"]))
    else:
        # フォールバックの従来表示
        print("\n===== 参考資料 上位ヒット: 筑波山麓 =====")
        for r in result["agent1_topk"]:
            print(f"{r['score']:.3f} | {r['source']}")
        print("\n===== 参考資料 上位ヒット: 地方創生 =====")
        for r in result["agent2_topk"]:
            print(f"{r['score']:.3f} | {r['source']}")
        print("\n===== 議論ログ（抜粋） =====")
        for speaker, content in result["dialogue"][-4:]:
            print(f"\n## {speaker}\n{content[:1200]}\n...")
        print("\n===== 統合サマリ =====\n")
        print(result["summary"])

    # ------------------ Markdown保存（任意） ------------------
    if args.save_md:
        out_path = Path(args.save_md)
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md = [
            f"# 実行結果 ({ts})\n",
            "## 参考資料 上位ヒット: 筑波山麓\n",
            "\n".join([f"- {r['score']:.3f} | {r['source']}" for r in result["agent1_topk"]]),
            "\n\n## 参考資料 上位ヒット: 地方創生\n",
            "\n".join([f"- {r['score']:.3f} | {r['source']}" for r in result["agent2_topk"]]),
            "\n\n## 議論ログ（抜粋）\n",
            "\n\n".join([f"### {speaker}\n\n{content}" for speaker, content in result["dialogue"][-4:]]),
            "\n\n## 統合サマリ\n",
            result["summary"],
        ]
        out_path.write_text("\n".join(md), encoding="utf-8")
        if console is not None:
            console.print(Panel.fit(f"Markdownとして保存しました: {out_path}", title="保存完了"))
        else:
            print(f"\n[保存] {out_path}")



if __name__ == "__main__":
    main()
