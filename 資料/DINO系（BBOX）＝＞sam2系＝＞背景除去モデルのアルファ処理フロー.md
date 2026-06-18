
# DINO 系（BBOX）＝＞sam2系＝＞背景除去モデルのアルファ処理フロー
作成日: 2026-06-16 
目的: DINO系（BBOX）からsam2系を経て背景除去モデルへ渡すアルファ処理フローの全体像を整理する。特に、SAM2のプロンプト理解と、SAM2マスクをM2Mがどう受け取っているかを中心に、正しいフローと注意点をまとめる。

## フロー図
資料\data\2026-06-16_20h19_42.png


1. モデル選定（使う／使わない）確定版

モデル
採否
役割
SAM3
❌ 使わない
ライセンス法務のたらい回しで時間が溶けるため除外
SAM2系
✅ 使う
インスタンスの選別・割り当て（どの対象か・どこか・重なりは誰のものか）
transparent-background（正式名称）
✅ 使う
エッジ/マット品質（髪・毛・半透明の縁）
Video Depth Anything-Small
✅ 追加
深度（前後関係。半透明・前後重なり時のみ必要）
役割分担の鉄則：SAM2に綺麗な縁を期待しない／transparent-backgroundに対象選別を期待しない／深度は既存2モデルからは出ないので専用モデルで取る。
---

2. SAM2のプロンプト理解（前回の誤解を訂正したもの）

- ポジ/ネガ点は「点の周りにマスクを作る」処理ではない。1対象=1マスクを作るための条件づけ（ヒント）。
- ポジ点＝「この座標は内側」、ネガ点＝「この座標は外側」。点単位のマスクは生まれない。
- よって「ポジ点ごとのマスクを統合」「ネガ点ごとのマスクを統合」という処理は存在しない。統合はインスタンス単位で起きる。
- multimask_output（候補3枚）は対象内の曖昧性を解く別レイヤー。対象同士の統合とは無関係。
- ネガ点は実装/版によって効きが弱い報告あり → 頼る設計なら最小再現で実機確認してから。
---

3. マスク統合の正しい順序（★ここが今回の核心）

logitのまま保持 → 統合 → 最後にアルファ化。情報を消す2値化／アルファ化は一番最後。
- softmaxはlogit（連続値）にかけるもの。2値化後はsoftmax不可（0/1にした時点で確信度が消える）。あなたが迷ってた「2値化してからsoftmax」は順番が逆。
- 「ピクセル単位softmax」=重なってる画素ごとに各対象＋背景のlogitを比べ、排他的にどれか1つへ振り分ける。これで重なりが自動解決。
- これ（重なり解決）は重なってる対象だけに必要。単独対象は素通り。

※「logitのまま保持」を初めて知った、とのことなので補足：logitは閾値で切る前の生スコア。これを持ち続けるからこそ、後段でsoftmax比較も連続アルファ化もできる。2値化は破壊的操作だと覚えておくと全体が腑に落ちます。

---

4. 痩せマスク対策（dilationは最後の手段）

- 第一手：threshold（閾値）を下げる。transparent-backgroundの2値化境界やSAM2 logitのsigmoid化を低めに使えば痩せが緩和。
- 第二手：logit→sigmoidで連続アルファ化して低めに使う（dilationより自然）。
- dilationは「縁の捏造」ではなく「取りこぼした面の欠損補填」に限定して小さく使う。髪・半透明をpaddingで"作る"のは不可、これはmattingの仕事。
---

5. コスト最適化（10秒→30分を実用速度に）効く順

1. propagationを使う（最効）。毎フレームGroundingDINO+SAM2を回さない。第1フレームでプロンプト→メモリで全フレーム伝播。検出は初回＋ドリフト時の再プロンプトだけ。
2. mattingはcrop領域だけに回す（全画面でかけない）。
3. 品質が要る対象だけmatting（髪・毛・半透明フラグ）。硬い物体はSAM2マスク＋小dilation。
4. バッチ処理：複数プロンプトを1フォワードでまとめる。
---

6. アルファ処理フロー図解

Video Depth Anything-Small を組み込んだ最新版フローです。
```
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, "Hiragino Kaku Gothic ProN", "Yu Gothic", Meiryo, sans-serif;
    background:#0f1419; color:#e6edf3; padding:20px; line-height:1.6;
  }
  .wrap { max-width:980px; margin:0 auto; }
  h1 { font-size:18px; margin-bottom:6px; color:#fff; }
  .sub { font-size:12px; color:#8b98a5; margin-bottom:24px; }
  .flow { display:flex; flex-direction:column; gap:14px; }
  .stage {
    border-radius:12px; padding:14px 16px; position:relative;
    border:1px solid rgba(255,255,255,0.08);
  }
  .stage .tag {
    display:inline-block; font-size:11px; font-weight:700;
    padding:2px 8px; border-radius:6px; margin-bottom:8px;
  }
  .stage h2 { font-size:15px; margin-bottom:4px; color:#fff; }
  .stage p { font-size:13px; color:#c2ccd6; }
  .stage .note { font-size:12px; color:#8b98a5; margin-top:6px; }
  .arrow { text-align:center; color:#5b6b7a; font-size:20px; line-height:1; }
  .row { display:flex; gap:14px; flex-wrap:wrap; }
  .row .stage { flex:1; min-width:240px; }
  .keep { background:rgba(46,160,67,0.10); border-color:rgba(46,160,67,0.35); }
  .keep .tag { background:rgba(46,160,67,0.25); color:#7ee787; }
  .sam { background:rgba(56,139,253,0.10); border-color:rgba(56,139,253,0.35); }
  .sam .tag { background:rgba(56,139,253,0.25); color:#79c0ff; }
  .depth { background:rgba(188,140,255,0.10); border-color:rgba(188,140,255,0.35); }
  .depth .tag { background:rgba(188,140,255,0.25); color:#d2a8ff; }
  .matte { background:rgba(255,166,87,0.10); border-color:rgba(255,166,87,0.35); }
  .matte .tag { background:rgba(255,166,87,0.25); color:#ffb77c; }
  .merge { background:rgba(247,129,102,0.10); border-color:rgba(247,129,102,0.35); }
  .merge .tag { background:rgba(247,129,102,0.25); color:#ff9b80; }
  .final { background:rgba(255,255,255,0.06); border-color:rgba(255,255,255,0.2); }
  .final .tag { background:rgba(255,255,255,0.18); color:#fff; }
  .badge {
    display:inline-block; font-size:10px; padding:1px 6px; border-radius:4px;
    background:rgba(255,255,255,0.1); color:#a9b4bf; margin-left:6px;
  }
  .danger { color:#ff7b72; font-weight:600; }
</style>
</head>
<body>
<div class="wrap">
  <h1>アルファ処理フロー（logit保持版 / 深度=Video Depth Anything-Small）</h1>
  <p class="sub">原則：2値化・アルファ化は最後。途中は全部 logit（連続値）で運ぶ。</p>

  <div class="flow">

    <div class="row">
      <div class="stage sam">
        <span class="tag">① SAM2 / 対象ごと</span>
        <h2>box + ポジ/ネガ点 → マスク logit</h2>
        <p>1対象=1回のforwardで logit を1枚。点は「内側/外側」の条件づけ。<span class="badge">2値化しない</span></p>
        <p class="note">propagationで全フレーム伝播（毎フレーム検出しない）</p>
      </div>
      <div class="stage depth">
        <span class="tag">①' 深度 / 並行</span>
        <h2>Video Depth Anything-Small → 深度マップ</h2>
        <p>各画素の前後関係。前後重なり・半透明の合成順に使う。</p>
        <p class="note">不要なシーンではスキップ可（多くは要らない）</p>
      </div>
    </div>

    <div class="arrow">▼</div>

    <div class="stage keep">
      <span class="tag">② logit 保持</span>
      <h2>全対象の logit ＋ 背景 logit を束ねる</h2>
      <p>ここでは絶対に切らない。連続値のまま次へ。<span class="danger">2値化＝破壊的操作</span></p>
    </div>

    <div class="arrow">▼</div>

    <div class="stage merge">
      <span class="tag">③ 重なり解決</span>
      <h2>ピクセル単位 softmax → 所有権マップ</h2>
      <p>重なる画素ごとに [対象A, 対象B, …, 背景] の logit を比較し、排他的に1つへ振り分け。</p>
      <p class="note">重なる領域だけ作用。単独対象は素通り。深度がある場合は半透明部の前後順をここで参照。</p>
    </div>

    <div class="arrow">▼</div>

    <div class="stage matte">
      <span class="tag">④ 縁の品質</span>
      <h2>crop領域だけ transparent-background → 連続アルファ</h2>
      <p>髪・毛・半透明フラグが立った対象のみ。硬い物体はSAM2 logit→sigmoid＋小dilationで代用。</p>
      <p class="note">痩せ対策：threshold下げ ＞ 連続アルファ ＞ dilation（最後の手段）</p>
    </div>

    <div class="arrow">▼</div>

    <div class="stage final">
      <span class="tag">⑤ 最終合成</span>
      <h2>所有権で領域割当 × アルファで縁 → 出力</h2>
      <p>③の所有権で「どの対象か」、④のアルファで「縁の透け」を与える。ここで初めてアルファ確定。</p>
    </div>

  </div>
</div>
</body>
</html>

```

---

7. 確度メモ

確実：logit保持と最後のアルファ化、softmaxはlogitにかける、ポジ/ネガ点は1対象1マスク条件づけ、既存2モデルは深度を持たない。
要実装検証：transparent-backgroundは「画像→マット」が本来I/Fで、SAM2マスクを入力に取る公式経路はない認識 → ④と⑤はあなたが後段で手合成する前提。Video Depth Anything-Smallの出力（相対深度）を③の前後順に使う際、スケールが相対値なので「前後の大小比較」には使えても「絶対距離」には使えない点に注意。
---
次は、④⑤の具体的な合成式（どの色空間で、premultiplied alphaか、所有権マップとアルファをどう掛け合わせるか）を詰めますか。それとも③のsoftmax温度をインスタンスごとにパラメータ化してバリエーション担保する設計（前に話してた方向）を先に組みますか。



---
@Claude-Opus-4.8
groudingDINOが取りこぼしたらものは
sam2系モデルでBBOXを手付けし　補強でネガティブポイント　ポジティブポイント
この流れいいか？
ハルシネーションがあるかもなので
ちゃんとネットを調べて回答してくれ
---
Thinking...

I need to verify what prompt types SAM2 actually supports—whether it can handle bounding box inputs, positive and negative point prompts, and how these can be combined together.

I'm checking whether the user's approach is viable—using GroundingDINO for initial detection, then manually adding bounding boxes in SAM2 when it misses something, combined with positive and negative points. SAM2's API does support box prompts through add_new_points_or_box, and while the playground documentation suggests box and point combinations aren't allowed, that appears to be a UI constraint rather than a fundamental limitation of the underlying model.

The key insight is that SAM2 internally converts boxes into corner points with special labels, so boxes and points are both encoded as point tokens—meaning you can add a box first and then refine it with additional positive or negative points. One important caveat: negative points in SAM2 are reportedly weak or sometimes ignored, which is worth keeping in mind. So the user's workflow is technically valid.
調べます。SAM2のboxプロンプト＋ポジ/ネガ点の併用が実際に公式仕様として成立するか、裏取りします。結論から言うと、そのフローは技術的に成立します。 ただし1点だけ「事実誤認しやすい注意点」があるので、そこを裏取り結果とあわせて整理します。
---

1. box手付け → ポジ/ネガ点で補強：成立する

SAM2の動画APIは add_new_points_or_box という関数で、box でもポイントでも同じ対象IDに対して与えられます。 SAM2VideoPredictorでは init_state の後に add_new_points_or_box でプロンプトを与え、同じフレーム上で即座に出力を得て、propagate_in_video で動画全体に伝播させる[5]構造です。
GroundingDINOが取りこぼした対象には、クリック・バウンディングボックス・マスクといったプロンプトで各フレームの対象境界を定義できる[4]ので、boxを手付けすればそのまま割り当てられます。
そのうえでのポジ/ネガ点による補強も正規の使い方です。SAM2はSAMと同様にネガティブ点（対象に属さない点）でのプロンプトに対応しており、これにより対象の境界を精密に定義できる[8]。実際のラベル付けも ラベル1が領域を追加するポジティブクリック、ラベル0が領域を除去するネガティブクリック[8]です。
---

2. ★最重要の注意点：box+点の「結合」は対象1つにつき逐次で行う

ここがハルシネーションしやすい箇所です。一部のUIガイドには 「SAM-2ではboxと点の結合プロンプトは使えない、一度に1つの組み合わせしか使えない」[3]という記述があります。ただしこれはGradioなどUI実装側の制約で、モデル本体の制約ではありません。
理由は内部実装にあります。SAM2（および現行のSAM3トラッカー）では、boxを受け取ると角の点や中心点に変換してラベルを割り当てており、そのラベルがboxの内側か外側かといった文脈をモデルに伝える[3]。つまりboxも内部的には「点トークン」として扱われるので、box → 同じ obj_id にポジ/ネガ点を追加、という逐次呼び出しなら正しく結合されます。
→ 実装方針： 同じ obj_id に対して、まず box を add_new_points_or_box で投入 → 同じ obj_id に points+labels を追加投入 → propagate。これがあなたのフローの正しい形です。
---

3. ★もう一つの注意点：ネガ点は「効きが弱い」報告がある

前回のセッションでも触れた点ですが、裏取りでも確認できました。公式リポジトリに 「2個程度のポジ点と4〜5個のネガ点で精緻化したいが、SAM2はネガ点プロンプトを考慮しないことがあり、この用途ではSAM(1)の方が優れている」[2]という報告（Issue #695）が立っています。
→ 設計上の含意： ネガ点を「主要な切り分け手段」として頼る設計にしないこと。box+ポジ点で対象を確定させ、ネガ点はあくまで微補正の位置づけにする。ネガ点に依存する切り分けが必要なら、最小再現で実機確認してから組み込む（前回の方針どおり）。
---

4. 整理：あなたのフローの確定形

ステップ
操作
確度
① GroundingDINO検出
自動でbox取得
―
② 取りこぼし対象
SAM2でbox手付け（add_new_points_or_boxのbox引数）
✅確実
③ 補強
同じobj_idにポジ点（label=1）で内側を確定
✅確実
④ 微補正
ネガ点（label=0）で誤包含を削る
⚠️効きが弱い場合あり
⑤ 伝播
propagate_in_videoで全フレーム展開
✅確実
結論：フローはOK。 ただし「box+点は逐次呼び出しで結合」「ネガ点は補助扱い、依存しない」の2点を守れば事実誤認になりません。
---
補足：③④はboxを手付けした第1フレーム上で行い、その後propagateするのが基本です。ドリフトしたら別フレームで再プロンプト（ポジ/ネガ点追加）すれば、メモリ機構が追従します。前回のセッション整理（propagation最優先のコスト方針）ともこのフローは整合しています。
次は、この手付けフローを既存のlogit保持パイプライン（前回図解の①SAM2段）にどう差し込むか、具体的な呼び出し順を組みますか？
---
Learn more:
1. sam2-playground/docs/PROMPT_GUIDE.md at master · jhj0517/sam2-playground
2. SAM 2: Segment Anything Model 2 | Ultralytics Docs
3. Negative point prompts are being ignored · Issue #695 · facebookresearch/sam2
4. Learn SAM 2 in Minutes: The Ultimate Starter Guide for 2025
5. docs/PROMPT_GUIDE.md · jhj0517/sam2-playground at 193878720cedea16d89232721bca364f0eef1a73
6. SAM3 Video Tracking: Negative Box Prompt Explained
7. Optimizing SAM: Elevating Prompt Engineering. | by Nandini Lokesh Reddy | Medium
8. SAM 2: Meta's Next-Gen Model for Video and Image Segmentation | DigitalOcean
9. Segment Anything 2: What Is the Secret Sauce? (A Deep Learner’s Guide) | by Avishek Biswas | TDS Archive | Medium
10. README.md · facebook/sam2-hiera-base-plus at main
11. Segment Anything 2 (SAM2) in Supervisely: The Fast and Accurate Object Segmentation Tool for Image and Video Labeling - Supervisely
12. Reference for ultralytics/models/sam/predict.py | Ultralytics Docs
13. SAM from Meta AI (Part 1): Segmentation with Prompts - PyImageSearch
14. Multiple bounding box as prompt · Issue #267 · facebookresearch/sam2
15. How to Use SAM 2 for Video Segmentation
16. samgeo2 module - segment-geospatial
17. facebook/sam3 · Hugging Face
18. SAM2 · Hugging Face
19. Segment Anything 2: What Is the Secret Sauce? (A Deep Learner's Guide) | Towards Data Science
20. Video segmentation with SAM 2 with boxes? · Issue #97 · facebookresearch/sam2