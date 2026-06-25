# なにこれ
[![Check](https://github.com/ponkotuy/seq-photo-compression/actions/workflows/check.yml/badge.svg)](https://github.com/ponkotuy/seq-photo-compression/actions/workflows/check.yml)

画像の連続性を用いてNikonのRAW画像形式、NEFファイルを圧縮するコマンドラインツール。

写真は連続して撮影した場合、比較的類似した画像になっている可能性が高い。この関係を利用して高圧縮なRAW画像形式を新規に作成することを目標とする。

第一目標は、NEFの連続性を使った場合にどの程度ファイルサイズを削減できるかを調査することである。
そのため、元のNEFファイルとバイト完全一致する復元は当面の目標にしない。一方で、RAWの画素値は完全に復元できる可逆圧縮を目指す。ヘッダ、メタデータ、サムネイルなどはなるべく保持するが、復元後のファイルが一般的な現像ソフトで扱える程度に復元できればよい。

NEFファイルはざっくりヘッダと圧縮されたRAW画像のBodyによって構成されており、圧縮されたBodyを次の写真と比較しても意味ないため、まずRAW画像のBodyを展開する必要がある。展開したBodyに対して1枚目の写真と2枚目の写真を動画のコーデックが行うような差分のみ抽出して、2枚目をヘッダ類と差分圧縮したbodyという独自形式で保存する。

3枚目は2枚目との差分ではなく1枚目の差分とする。理由は3枚目のみ復元する場合に3枚以上復元しないといけない状況を避けるためである。これを繰り返していくと、差分が小さくなってどこかで独自形式がオリジナルより大きいファイルサイズになってしまう状況がうまれる筈である。このときは独自形式をやめてオリジナルを残し、次の写真を2枚目として扱う。

最終的な運用では、保存対象をkeyframeとして残すNEFファイルと、keyframeとの差分を持つ独自形式ファイルの2種類にすることで容量削減を狙う。独自形式ファイルは、このアプリケーションに通すことで対応するNEFファイルへ復元できるものとする。

# 復元方針
- RAW画素値は完全一致を必須とする
- 元NEFファイルとのバイト完全一致は必須としない
- ヘッダ、メタデータ、サムネイルなどは可能な範囲で保持する
- 復元後のファイルは、一般的な現像ソフトで開けることを目指す
- このソフトウェアからのみ読み書きできればよいため、独自形式の内部仕様は自由に決めてよい
- 独自形式ファイルは、同じ保存セット内にあるkeyframe NEFを参照して復元する
- 独自形式ファイルには、復元に必要なkeyframeのファイル名、ハッシュ、RAW形式情報などを記録する

# 開発要件
- 独自形式の拡張子は有名なやつと被らなければなんでもいい
- 言語の選定は以下を基準にする。上を優先する
  - NEFという特殊形式のBodyを展開できる（なるべくライブラリなど）
  - 動画的な差分圧縮ができる（なるべくライブラリなど）
  - 処理が高速
  - Linux環境でランタイムが入手しやすい
  - 開発しやすさ（特に型付きであること）
- 20260519というディレクトリにNEFファイル一式があるのでテストが可能
- このアプリケーションの機能として、独自形式に変換した元のNEFファイルを削除してはいけない
- オリジナルの削除はユーザーがアプリケーション外で判断して行う

# 開発段階

## 第一段階
- 連続する2枚のNEFを入力として扱う
- 1枚目をkeyframe NEFとしてそのまま残す
- 2枚目を、1枚目との差分を持つ独自形式として出力する
- 独自形式とkeyframe NEFから、2枚目に相当するNEFを復元できるようにする
- 復元したRAW画素値が、元の2枚目のRAW画素値と完全一致することを検証する
- 元の2枚目のNEFファイルは削除しない
- 圧縮前後のサイズ、削減率、復元したRAW画素値の一致可否をレポートする

## 第二段階
- `20260519` 内のNEFを対象に圧縮率を測定する
- ファイル名順で処理する。ただしカメラ機種、画像サイズ、RAW形式などが異なるファイル同士は差分対象にしない
- 基準ファイルを1枚選び、後続ファイルは基準ファイルとの差分として圧縮する
- 後続ファイルの独自形式には、復元に必要なkeyframe参照情報を含める
- 差分形式が元ファイルより大きくなる場合は、そのファイルを新しい基準ファイルとして扱う
- 複数keyframeと複数独自形式ファイルで構成される保存セット全体の圧縮率を測定する

# 現在の実装

第一段階の実験用CLIとして `spc` コマンドを実装している。互換用に `spc.py` からも同じCLIを呼び出せる。

必要なもの:
- `uv`
- LibRaw付属の `unprocessed_raw`
- `zstd`
- JPEG XL toolsの `cjxl` / `djxl`
- Pythonパッケージの NumPy / OpenCV（`uv sync` で導入）

Pythonパッケージは `uv` で管理する。

```sh
uv sync
```

開発用ツールも含める場合:

```sh
uv sync --dev
```

Pythonコードのフォーマット:

```sh
uv run ruff format .
```

フォーマット済みか確認する場合:

```sh
uv run ruff format --check .
```

コードメトリクスの確認:

```sh
uv run radon cc -s -a .
uv run radon mi -s .
```

例:

```sh
uv run spc encode 20260519/D8A_2000.NEF 20260519/D8A_2001.NEF -o d8a_2001.spcraw
uv run spc verify 20260519/D8A_2000.NEF 20260519/D8A_2001.NEF d8a_2001.spcraw
uv run spc restore 20260519/D8A_2000.NEF d8a_2001.spcraw -o restored_D8A_2001.NEF
```

差分を使わずに1枚のNEFを単体圧縮する場合:

```sh
uv run spc encode-single 20260519/D8A_1990.NEF -o d8a_1990.single.spcraw
uv run spc verify-single 20260519/D8A_1990.NEF d8a_1990.single.spcraw
uv run spc restore-single d8a_1990.single.spcraw -o restored_D8A_1990.NEF
```

複数のNEFを単体圧縮する場合は、各ファイルの拡張子を `.spcraw` に置き換えた名前で出力する:

```sh
uv run spc encode-single 20260519/*.NEF
```

`encode-single` は展開RAWをRGGB 4チャンネルPAMとしてJPEG XL Modular losslessで保存し、復元時は既存のNikon lossless 14bitエンコーダで元RAW stripを再生成する。既定では速度優先でMakerNoteの復元情報を信頼し、encode中の重いRAW strip再生成検証は行わない。保存後に `verify-single` を実行すると、復元RAW画素とRAW本体のバイナリ一致を確認できる。encode中にも確認したい場合は `--encode-verify pixels` または `--encode-verify strip` を指定する。RAW stripを再生成できない場合は既定で失敗するが、`--fallback raw-strip` を指定すると元RAW stripを直接zstd保存する。

ディレクトリ内のNEFをファイル名順に処理する場合:

```sh
uv run spc encode-dir 20260519 --diff-codec jxl --motion-mode ecc_affine
```

`encode-dir` は同じディレクトリに `TARGET.spcraw` を出力する。カメラ機種、RAW画像サイズ、RAW圧縮形式、bits per sampleが一致するファイル同士だけを差分対象にし、独自形式が元NEFの90%以上のサイズになる場合は `.spcraw` を残さず、そのNEFを新しいkeyframeとして扱う。閾値は `--max-archive-ratio` で変更できる。RAW画素値の一致検証も行う場合は `--verify` を指定する。

`encode` は既定でJPEG XL Modularを使い、RGGB 4チャンネル分離した動き補償残差を保存する。従来の単純差分zstd方式を使う場合は `--diff-codec zstd` を指定する。

JPEG XL Modularの指定例:

```sh
uv run spc encode 20260519/D8A_2041.NEF 20260519/D8A_2042.NEF \
  --diff-codec jxl --motion-mode ecc_affine --jxl-effort 6 \
  -o d8a_2042.spcraw
```

`--jxl-effort` は `1-10` を指定できる。`10` は圧縮率確認には有効だが遅いため、実用向けの既定値は `6` としている。

実装方針:
- `unprocessed_raw` でkeyframeと2枚目のRAW画素値を16bit PGMとして展開する
- 2枚目のRAW画素値からkeyframeのRAW画素値を引いた差分を `int16` として保存する
- 差分データと、2枚目NEFのRAW本体をゼロ化したshellを `zstd` で圧縮して独自形式に格納する
- `--diff-codec jxl` では、Python OpenCVでRGGBを4チャンネルに分離し、ECC affineでkeyframeをtargetへ合わせた予測画像との差分を16bit PAMへ格納し、JPEG XL Modular losslessで圧縮する
- 復元時はkeyframeからRAW画素値を再展開し、差分を加算して2枚目のRAW画素値を復元する
- 復元NEFは、2枚目NEFのshellに未圧縮RAWを末尾追加し、TIFFタグを未圧縮RAWへ向け直して出力する

現時点の制約:
- 単一stripのNEFのみ対応
- 復元NEFは元NEFとバイト完全一致しない
- 復元NEFは未圧縮RAWを含むため、元NEFより大きくなる可能性が高い
- 圧縮効果は独自形式ファイルのサイズで評価する
- 変換元NEFを削除する機能は持たない
