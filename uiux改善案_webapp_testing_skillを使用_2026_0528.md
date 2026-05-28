成果物

 - 静止画版: C:\Users\owner\.copilot\session-state\4fb2052a-a9d0-4273-8d32-4336f3ba180b\files\ui_static\
 - 動画版: C:\Users\owner\.copilot\session-state\4fb2052a-a9d0-4273-8d32-4336f3ba180b\files\ui_movie\
 - 各フォルダに *_initial.png, *_after_input.png, *_inspection.json を保存済み

確認結果

 - 静止画版: 137 要素。Input Image → prompt 指定 → Predict SAM2 Candidate Masks → Add to Union → Run transparent-background... が現状の想定フロー。
 - 動画版: 79 要素。Input Video → 第1フレームを取得 → prompt 指定 → 動画背景除去を実行 が現状の想定フロー。
 - 動画版は第1フレーム取得後、Prompt Status が 第 1 フレームを取得しました: 320x180, fps=8.00 に更新されることを確認。
 - モデル推論本体は checkpoint/GPU 依存のため、UI 導線確認範囲では実行していません。

必須入力削減・導線改善案

 1. 静止画版は Add to Union を必須に見せない。Predict SAM2 Candidate Masks 後に最高スコア候補を自動選択し、Run は既定で Best Candidate Mask を使う。
 2. Candidate Mask Indices の手入力をやめ、候補表にチェック選択を付ける。複合対象だけ Union を使う導線にする。
 3. Text Prompt は初期値 person . object . ではなく placeholder にし、「自動 bbox を使う場合だけ入力」と明示する。
 4. Point Label は Prompt Mode=box の時は非表示または disabled にする。
 5. 閾値・Top-K・Union Dilate・Min Area・JIT・crop などは “Advanced” に折りたたみ、初期表示は Input / Prompt / Run の3段に絞る。
 6. 動画版は動画アップロード時に第1フレームを自動取得し、第1フレームを取得 ボタンを任意の「再取得」に降格する。
 7. 動画版の output_mode=sequence 選択時は codec を自動非表示にし、「動画出力時のみ codec が必要」と表示する。
 8. 実行ボタンは前提未満では disabled にする。例: 画像/動画未入力、prompt 未指定、候補 mask 未生成時に押せないようにする。