# 差分圧縮アルゴリズム

この文書は、このアプリケーションの中心であるNEF連番写真向け差分圧縮の考え方と、現在の `spc.py` に実装されている処理をまとめたものです。

## 目的

連続して撮影されたRAW写真は、構図や露出が近く、RAW画素値にも強い類似性があることが多い。この性質を利用して、1枚を基準画像、つまりkeyframeとしてそのまま保持し、後続画像はkeyframeとの差分だけを独自形式に保存する。

現時点の目標は、NEFファイルそのものをバイト完全一致で復元することではない。RAW画素値を可逆に復元できることを優先し、メタデータやサムネイルなどは可能な範囲で保持する。

## 基本方針

NEF内のRAW本体はカメラ独自形式で圧縮されているため、圧縮済みバイト列同士を直接比較しても、写真間の類似性をうまく利用できない。そのため、まず各NEFからRAW画素値を展開し、展開後の2次元配列に対して差分を取る。

現在の実装では、LibRaw付属の `unprocessed_raw` を使ってNEFを16bit PGMへ展開し、そのPGMを `uint16` のNumPy配列として読み込む。

差分は次の式で計算する。

```text
diff[y][x] = target_raw[y][x] - keyframe_raw[y][x]
```

ここで `keyframe_raw` は基準NEFのRAW画素値、`target_raw` は圧縮対象NEFのRAW画素値である。差分は `int16` として保存するため、各画素差分は `-32768` から `32767` の範囲に収まる必要がある。この範囲を超えた場合、現在の実装では圧縮不能としてエラーにする。

## エンコード処理

`encode` は、keyframe NEFとtarget NEFから `.spcraw` 独自形式を作る。

処理の流れは次の通り。

1. keyframe NEFを `unprocessed_raw` で展開し、`uint16` 配列として読む。
2. target NEFも同じ方法で展開し、`uint16` 配列として読む。
3. 2つのRAW配列の形状が一致することを確認する。
4. `target_raw - keyframe_raw` を `int32` で計算する。
5. 差分が `int16` の範囲に収まることを確認し、リトルエンディアン `int16` 配列へ変換する。
6. target NEFからTIFF IFDを走査し、RAW本体のstripを探す。
7. target NEFのRAW strip領域をゼロ埋めしたshellを作る。
8. shellとdiffをそれぞれ `zstd` で圧縮する。
9. 復元に必要なメタデータをJSONヘッダにまとめ、magic、ヘッダ長、ヘッダ、圧縮shell、圧縮diffの順に `.spcraw` へ書き込む。

target NEFのshellを保存する理由は、復元時にtarget側のヘッダ、メタデータ、サムネイル、その他の構造をなるべく再利用するためである。ただし、元の圧縮RAW本体はdiffから復元するため、shell内ではゼロ化してからzstd圧縮する。これにより、target NEF全体をそのまま保存するより小さくなることを狙う。

## 独自形式 `.spcraw`

現在の独自形式は、次の順序で構成される。

```text
MAGIC:      b"SPCNEF1\0"
HEADER_LEN: little-endian uint32
HEADER:    JSON UTF-8
SHELL:     zstd圧縮されたtarget NEF shell
DIFF:      圧縮された差分または残差データ
```

既定の `--diff-codec jxl` では、`DIFF` チャンクにJPEG XL Modularで圧縮した残差PAMが入る。この場合、JSONヘッダの `diff.compression` は `jxl_modular` になり、動き補償行列、残差範囲、JPEG XL effortなどを持つ。従来方式の `--diff-codec zstd` では、`DIFF` チャンクにzstd圧縮されたint16差分配列が入る。

JSONヘッダには、主に次の情報が入る。

- `version`: フォーマットバージョン
- `keyframe`: keyframeのパス、SHA-256、ファイルサイズ
- `target`: targetのパス、SHA-256、ファイルサイズ
- `raw`: RAW配列の幅、高さ、画素型、差分型
- `target_shell`: shellの圧縮方式、ゼロ化したRAW strip位置、復元時に書き換えるTIFFタグ位置
- `diff`: diffの圧縮方式、最小値、最大値
- `chunks`: 圧縮shellと圧縮diffのバイト長

`chunks` は書き込み時に実際の圧縮後サイズから設定される。読み込み時はこの長さを使って、shellとdiffを順に切り出す。

## 復元処理

`restore` は、keyframe NEFと `.spcraw` からtarget相当のNEF風ファイルを作る。

処理の流れは次の通り。

1. `.spcraw` を読み、magicとヘッダを確認する。
2. 指定されたkeyframe NEFのSHA-256がヘッダ内の値と一致することを確認する。
3. keyframe NEFを `unprocessed_raw` で再展開し、`uint16` 配列として読む。
4. `diff.compression` に応じて差分を展開する。`zstd` では `int16` 配列へ戻し、`jxl_modular` では `djxl` で残差PAMへ戻す。
5. `zstd` では `keyframe_raw + diff` を計算する。`jxl_modular` では保存された動き補償行列でkeyframeを予測画像へ変換し、残差を加算する。
6. 復元値が `uint16` の範囲に収まることを確認し、`uint16` 配列へ変換する。
7. 圧縮shellを `zstd` で展開する。
8. 復元RAW配列をリトルエンディアン `uint16` バイト列にする。
9. shell末尾へ未圧縮RAWバイト列を追加する。
10. TIFFタグを書き換え、RAW stripの位置とサイズを追加した未圧縮RAWへ向ける。
11. `Compression` タグを `1` に変更し、未圧縮RAWとして出力する。

復元後のファイルは、元NEFとバイト完全一致しない。特にRAW本体は元のカメラ圧縮形式ではなく、末尾に追加された未圧縮RAWになる。そのため、復元後ファイルは元NEFより大きくなる可能性が高い。

## 検証処理

`verify` は、`.spcraw` から復元されるRAW画素値がtarget NEFのRAW画素値と一致するかを確認する。

処理内容は、復元処理のうちRAW配列を復元する部分までを実行し、target NEFを `unprocessed_raw` で展開した配列と `np.array_equal` で比較する。一致しない場合は、異なる画素数を出力してエラーにする。

この検証は「RAW画素値の可逆性」を確認するものであり、復元NEFのバイト列一致を確認するものではない。

## JPEG XL Modular残差モード

`--diff-codec jxl` は、連続写真間の位置ズレを補正してから残差をJPEG XL Modular losslessで保存する実験モードである。

処理の流れは次の通り。

1. keyframeとtargetのRAWをRGGBの4チャンネルに分離する。
2. 4チャンネル平均のアライメント画像を作る。
3. `--motion-mode ecc_affine` ではOpenCVのECC affineでkeyframeからtargetへの変換行列を推定する。
4. その行列でkeyframe各チャンネルをwarpし、予測画像を作る。
5. `target - predictor` の符号付き残差へ `32768` を足し、16bit PAMの4チャンネルとして保存する。
6. PAMを `cjxl --distance=0 --modular=1 --effort=N` で可逆圧縮する。

復元時は `djxl` でPAMへ戻し、同じ行列で作った予測画像に残差を加算してRAWを復元する。動き補償行列は復元結果に直接影響するため、ヘッダには十分な精度で保存する。

`--jxl-effort` は `1-10` を指定できる。`10` は圧縮率測定向けには有効だが処理時間が長いため、CLIの既定値は実用寄りに `6` としている。

## TIFF/NEF内のRAW strip検出

`spc.py` は簡易TIFFパーサを持ち、NEF内のIFDとSubIFDを走査してRAW stripを探す。現在の判定は次の条件に基づく。

- `StripOffsets`
- `StripByteCounts`
- `Compression`

これらのタグを持つIFDを候補にし、その中で `StripByteCounts` が最も大きいものをRAW本体とみなす。

現在は単一stripのNEFのみ対応している。`StripOffsets` または `StripByteCounts` が複数値の場合は、第一段階の対象外としてエラーにする。

## keyframe戦略

現在の実装は2枚入力の第一段階であり、1つのkeyframeと1つのtargetだけを扱う。

最終的には、連番写真を保存セットとして扱い、次のような運用を想定している。

1. 先頭または基準に適したNEFをkeyframeとしてそのまま保存する。
2. 後続NEFは、そのkeyframeとの差分 `.spcraw` として保存する。
3. 差分形式が元NEFより大きくなる場合、そのNEFは新しいkeyframeとして保存する。
4. 以後の画像は新しいkeyframeとの差分として扱う。

差分を直前画像ではなくkeyframeから取るのは、任意の1枚を復元するために長い依存チェーンをたどる必要がないようにするためである。

## 現在の制約

- Nikon NEFを前提としている。
- RAW展開に外部コマンド `unprocessed_raw` が必要である。
- 圧縮と展開に外部コマンド `zstd` が必要である。
- JXL残差モードには外部コマンド `cjxl` / `djxl` と、Pythonパッケージの OpenCV が必要である。
- RAW配列の形状が一致する画像同士のみ差分化できる。
- 差分は `int16` に収まる必要がある。
- 単一stripのNEFのみ対応している。
- 復元NEFは元NEFとバイト完全一致しない。
- 復元NEFのRAW本体は未圧縮として末尾追加される。
- 圧縮率評価は、復元NEFではなく `.spcraw` のサイズで行う。
- 元NEFを削除する処理は持たない。

## 改善余地

今後の拡張候補は次の通り。

- 複数strip NEFへの対応
- 差分が `int16` に収まらない場合のfallback
- カメラ機種、画像サイズ、RAW形式ごとの自動グルーピング
- 保存セット全体のkeyframe自動選択
- `.spcraw` が元NEFより大きい場合の自動keyframe化
- 復元NEFをより元のNEF構造に近づける処理
- zstd以外の差分圧縮方式や予測符号化の比較
