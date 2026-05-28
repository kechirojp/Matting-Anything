# Matting Anything（マッティング・エニシング）

[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=XY2Q0HATGOk)
[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/shi-labs/Matting-Anything)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

[Jiachen Li](https://chrisjuniorli.github.io/)、
[Jitesh Jain](https://praeclarumjj3.github.io/)、
[Humphrey Shi](https://www.humphreyshi.com/home)

[[`プロジェクトページ`](https://chrisjuniorli.github.io/project/Matting-Anything/)]
[[`ArXiv`](https://arxiv.org/abs/2306.05399)]
[[`論文PDF`](https://arxiv.org/pdf/2306.05399.pdf)]
[[`動画`](https://www.youtube.com/watch?v=XY2Q0HATGOk)]
[[`デモ`](https://huggingface.co/spaces/shi-labs/Matting-Anything)]

![](./assets/teaser_arxiv_v2.png)

---

## 更新履歴

- **`2023/07/17`**: SAM ViT-L および SAM ViT-H ベースの MAM チェックポイントを追加。
- **`2023/06/28`**: [**Getting Started**](https://github.com/SHI-Labs/Matting-Anything/blob/main/GETTING_STARTED.md) を更新（学習・評価手順を追加）。
- **`2023/06/09`**: [**HuggingFace デモ**](https://huggingface.co/spaces/shi-labs/Matting-Anything) を公開。
- **`2023/06/08`**: [**ArXiv プレプリント**](https://arxiv.org/abs/2306.05399) を公開。
- **`2023/06/06`**: [**プロジェクトページ**](https://chrisjuniorli.github.io/project/Matting-Anything) および [**デモ動画**](https://www.youtube.com/watch?v=XY2Q0HATGOk) を公開。

---

## 目次

- [Matting Anything とは](#matting-anything-とは)
- [インストール](#インストール)
- [はじめに](#はじめに)
- [サードパーティプロジェクト](#サードパーティプロジェクト)

---

## Matting Anything とは

### 概要

本論文では、**Matting Anything Model（MAM）** を提案します。MAM は、視覚的または言語的なユーザープロンプトに基づき、画像内の任意のインスタンスのアルファマットを効率的かつ汎用的に推定するフレームワークです。MAM は従来の特化型画像マッティングネットワークに対して、以下の重要な利点を持ちます。

1. **汎用性**: 単一のモデルで、セマンティックマッティング・インスタンスマッティング・参照画像マッティングなど、さまざまな種類の画像マッティングに対応。
2. **軽量性**: Segment Anything Model（SAM）の特徴マップを活用し、軽量な **Mask-to-Matte（M2M）** モジュールを通じて反復的な精緻化によりアルファマットを予測。学習可能なパラメータはわずか **270 万個**。
3. **操作の簡易化**: SAM の統合により、ユーザーが指定するインタラクションをトライマップからボックス・ポイント・テキストプロンプトへと大幅に簡略化。

各種画像マッティングベンチマークでの評価の結果、MAM は各ベンチマークの複数の指標において最先端の特化型画像マッティングモデルと同等の性能を達成することが示されました。MAM は優れた汎化能力を持ち、少ないパラメータ数でさまざまな画像マッティングタスクを効果的に処理できる実用的なソリューションです。

### アーキテクチャ

<div align="center">
  <img src="assets/arxiv_fix.png" width="100%" height="100%"/>
</div><br/>

MAM のアーキテクチャは、事前学習済みの SAM と M2M モジュールで構成されています。入力画像 I が与えられると、SAM はボックスまたはポイントのユーザープロンプトに基づいてターゲットインスタンスのマスク予測を生成します。M2M モジュールは画像・マスク・特徴マップを結合した入力を受け取り、マルチスケール予測 α<sub>os8</sub>、α<sub>os4</sub>、α<sub>os1</sub> を出力します。反復的な精緻化プロセスにより、マルチスケール出力を取り込みながら最終的な精細なアルファマット α の精度を段階的に向上させます。

### 可視化

<div align="center">
  <img src="assets/teaser.gif" width="100%" height="100%"/>
</div>

<div align="center">
  <img src="assets/mam_vis_v2.png" width="100%" height="100%"/>
</div><br/>

SAM および MAM によるアルファマット予測の可視化を示します。赤いボックス内の差異に特に注目してください。可視化から、MAM がトライマップなしでも遷移領域においてより精緻な予測を達成していることが確認できます。また MAM は、SAM のマスク予測に含まれる穴（ホール）も効果的に補完しています。これらの比較により、アルファマット予測の精緻化・品質向上における MAM の優れた性能が示されています。

---

## インストール

MAM の完全なインストール手順については、[インストール手順](INSTALL.md) を参照してください。

---

## はじめに

データセットの準備・学習・推論の詳細については、[Getting Started](GETTING_STARTED.md) を参照してください。

---

## サードパーティプロジェクト

- [Matting-Anything-Colab](https://github.com/camenduru/Matting-Anything-colab)（[@camenduru](https://twitter.com/camenduru)）
- [Matting-Anything-Video](https://huggingface.co/spaces/fffiloni/Video-Matting-Anything)（[@fffiloni](https://twitter.com/fffiloni)）

---

## 引用

本研究を利用する場合は、以下の形式で引用してください。

```bibtex
@article{li2023matting,
      title={Matting Anything},
      author={Jiachen Li and Jitesh Jain and Humphrey Shi},
      journal={arXiv: 2306.05399},
      year={2023}
    }
```

---

## 謝辞

[SAM](https://github.com/facebookresearch/segment-anything)、[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)、[MGMatting](https://github.com/yucornetto/MGMatting)、[InstMatt](https://github.com/nowsyn/InstMatt/tree/main) の著者の方々にコードベースを公開していただいたことに感謝いたします。
