"""
CLIP（Contrastive Language-Image Pre-training：対照的言語イメージ事前トレーニング）モデルを採用しています。
※"ViT-L/14@336px"モデルを使用しているため、写真は推奨サイズの336px×336pxにリサイズしました。
※検証を簡易化するため、写真には001からの番号を付けています。
"""
import os  # ファイルやディレクトリの操作用モジュール
import cv2  # 画像処理用モジュール(OpenCV)
import torch  # PyTorch深層学習フレームワーク
import clip  # CLIPモデル用ライブラリ
from PIL import Image, UnidentifiedImageError  # 画像操作用ライブラリ(Python Imaging Library)
import csv  # CSVファイル操作用モジュール

# CLIPモデルのロード
device = "cuda" if torch.cuda.is_available() else "cpu"  # CUDAが利用可能ならGPU、そうでない場合はCPUを使用
model, preprocess = clip.load("ViT-L/14@336px", device=device)  # CLIPモデルと前処理関数を読み込み

# 入力画像の準備
folder_path = "resize_01-45"  # 画像が保存されているフォルダのパス

if not os.path.exists(folder_path):  # フォルダが存在しない場合の確認
    print(f"指定されたフォルダ '{folder_path}' が存在しません")  # エラーメッセージを表示
    exit()  # プログラムを終了

# フォルダ内のアイテムにアクセスし情報を取得
files = os.listdir(folder_path)  # フォルダ内のファイル一覧を取得

# 結果を保存するためのリストを初期化
all_results = []  # 結果を保存するリスト

# フォルダ内にある画像ファイルを読み込む
for index, file in enumerate(files):  # フォルダ内の各ファイルに対してループを実行
    image_path = os.path.join(folder_path, file)  # ファイルのフルパスを取得
    try:
        image = cv2.imread(image_path)  # 画像を読み込む

        # 画像が読み込めない場合のエラーハンドリング
        if image is None:  # 画像が正常に読み込めなかった場合
            print(f"画像 '{file}' の読み込み中にエラーが発生しました")  # エラーメッセージを表示
            continue  # 次のファイルに進む

        # 画像の読み込みと前処理
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)  # 画像を前処理し、バッチ次元を追加してデバイスに移動
    except (UnidentifiedImageError, OSError) as e:
        print(f"画像 '{file}' の処理中にエラーが発生しました: {e}")  # エラーメッセージを表示
        continue  # 次のファイルに進む
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")  # 予期しないエラーメッセージを表示
        continue  # 次のファイルに進む

    # テキストのラベルを準備
    # 予測したいラベルをリストで指定
    labels = ["can", "bottle", "pet bottle", "cup", "tube", "shoes", "pen", "measure", "mug cup",
              "headphones", "paper bag", "socks", "paper", "scissors", "milk carton", "plush toy",
              "book", "fish", "dry cell battery", "keyholder", "button battery", "nail nippers",
              "spanner", "spring", "pan", "mouse", "notebook computer", "shaver", "calculator",
              "cardboard", "ring", "CD", "plastic bag", "hanger", "wood", "clothespin", "remote",
              "extension cord", "cushioning", "hand towel", "paper", "magazine", "drier", "Kleenex",
              "hexagon wrench", "diamond", "pick", "home electrical appliance", "mask", "toothbrush",
              "spray can", "cap", "plastic fork", "orange", "rubber band", "lip balm", "mirror", "shirt",
              "packing tape", "seal", "cutter knife", "ear pick", "umbrella", "hair comb", "hand fan",
              "shaver", "USB thumb drive", "camera", "nipper", "bis", "driver", "pincers", "lighter",
              "dumbbell", "vacuum cleaner"]  # ラベルのリスト

    # CLIPに渡すテキストプロンプト
    text_inputs = torch.cat([clip.tokenize(f"A photo of a {label}") for label in labels]).to(
        device)  # 各ラベルに対応するテキストプロンプトをトークン化して連結

    # モデルによる画像とテキストのエンコーディング
    try:
        with torch.no_grad():  # 勾配計算を無効化
            image_features = model.encode_image(image)  # 画像をエンコードして特徴ベクトルを取得
            text_features = model.encode_text(text_inputs)  # テキストをエンコードして特徴ベクトルを取得

            # 特徴ベクトルを正規化
            image_features /= image_features.norm(dim=-1, keepdim=True)  # 画像の特徴ベクトルを正規化
            text_features /= text_features.norm(dim=-1, keepdim=True)  # テキストの特徴ベクトルを正規化

            # 画像とテキストの類似度を計算
            similarities: torch.Tensor = (image_features @ text_features.T).squeeze(0)  # 画像とテキストのコサイン類似度を計算
    except Exception as e:
        print(f"モデルのエンコーディング中にエラーが発生しました: {e}")  # エラーメッセージを表示
        continue  # 次のファイルに進む

    # 最も類似度が高いラベルを取得
    best_match_index = similarities.argmax().item()  # 類似度が最大となるインデックスを取得
    predicted_label = labels[best_match_index]  # 予測ラベルを取得

    # ごみの分別品目
    if predicted_label in ["cup", "shoes", "pen", "headphones", "bag", "remote", "fish", "paper",
                           "plush toy", "keyholder", "mouse", "shaver", "calculator", "hanger",
                           "wood", "clothespin", "extension cord", "hand towel", "drier", "pick",
                           "mask", "rubber band", "lip balm", "packing tape", "seal", "ear pick",
                           "hair comb", "hand fan", "USB thumb drive", "camera", "lighter"]:
        category = "(燃やすごみ)"  # 燃やすごみのカテゴリー分け

    elif predicted_label in ["diamond", "orange", "mirror", "mug cup"]:
        category = "(燃えないごみ)"  # 燃えないごみのカテゴリー分け

    elif predicted_label in ["spray can"]:
        category = "(スプレー缶)"  # スプレー缶のカテゴリー分け

    elif predicted_label in ["dry cell battery", "button battery"]:
        category = "(乾電池)"  # 乾電池のカテゴリー分け

    elif predicted_label in ["toothbrush", "tube", "CD", "plastic bag", "cushioning", "plastic fork"]:
        category = "(プラスチック資源)"  # プラスチック資源のカテゴリー分け

    elif predicted_label in ["can", "bottle", "pet bottle"]:
        category = "(缶・びん・ペットボトル)"  # 缶・びん・ペットボトルのカテゴリー分け

    elif predicted_label in ["spanner", "bis", "measure", "scissors", "spring", "nail nippers",
                             "ring", "hexagon wrench", "cutter knife", "umbrella", "shaver",
                             "nipper", "driver", "pincers", "dumbbell"]:
        category = "(小さな金属類)"  # 小さな金属類のカテゴリー分け

    elif predicted_label in ["paper bag", "milk carton", "book", "cardboard", "paper", "magazine",
                             "Kleenex"]:
        category = "(古紙)"  # 古紙のカテゴリー分け

    elif predicted_label in ["socks", "cap", "shirt"]:
        category = "(古布)"  # 古布のカテゴリー分け

    elif predicted_label in ["pan", "vacuum cleaner"]:
        category = "(粗大ごみ)"  # 粗大ごみのカテゴリー分け

    elif predicted_label in ["home electrical appliance"]:
        category = "(市では取り扱えないもの)"  # 市で取り扱えないもののカテゴリー分け

    elif predicted_label in ["notebook computer"]:
        category = "(小型家電製品)"  # 小型家電製品のカテゴリー分け

    else:
        category = "(その他)"  # その他のカテゴリー分け

    # 結果をリストに追加
    all_results.append((index + 1, predicted_label, category))  # 結果をリストに追加

    print(f"画像ファイル名: {index + 1}, 予測ラベル名: {predicted_label}, 分別品目: {category}")  # 結果を表示

# 結果をCSVファイルに書き込み
try:
    with open("results.csv", mode="w", newline='',encoding="utf-8") as file:  # CSVファイルを新規作成または上書きモードで開く
        writer = csv.writer(file)  # CSVライターオブジェクトを作成
        # ヘッダーの書き込み
        writer.writerow(["画像ファイル名", "予測ラベル名", "分別品目"])  # ヘッダーを書き込む
        # 各行の書き込み
        writer.writerows(all_results)  # 結果リストの各行を書き込む
    print("結果が 'results.csv' に保存されました。")  # 成功メッセージを表示
except IOError as e:
    print(f"CSVファイルの書き込み中にエラーが発生しました: {e}")  # エラーメッセージを表示
