# 必要なライブラリをインポートします
import torch
import cv2
import os
import csv

try:
    # YOLOモデルをパスのモデル 'best.pt' を使用してロードします
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
    # モデルを評価モードに設定します
    model.eval()

# 'best.pt'ファイルが存在しない場合のエラーハンドリング
except Exception as e:
    print(f"'best.pt'ファイルの読み込み中にエラーが発生しました: {e}")
    # 読み込み中にエラーが発生した場合は終了します
    exit()

# 処理対象となる画像フォルダのパス
folder_path = 'images_resize_01-25'

# フォルダが存在するかどうかを確認します
if not os.path.exists(folder_path):
    print(f"指定されたフォルダ '{folder_path}' が存在しません")
    # フォルダが存在しない場合は終了します
    exit()

# 結果を保存するためのCSVファイルを準備します
output_csv = 'results.csv'
header = ['画像ファイル名', 'クラス名', '検出された本数']

with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    # フォルダ内の画像ファイルを処理
    # 指定されたフォルダ内のすべてのファイルを取得します
    files = os.listdir(folder_path)

    # フォルダ内にある画像ファイルを読み込む
    for file in files:
        image_path = os.path.join(folder_path, file)
        # OpenCVを使用して画像を読み込みます
        image = cv2.imread(image_path)

        # 画像が読み込めない場合のエラーハンドリング
        if image is None:
            print(f"画像 '{file}' の読み込み中にエラーが発生しました")
            continue # 次のファイルに進む

        try:
            # YOLOモデルによる物体検出
            # YOLOモデルを使用して画像内の物体を検出します
            results = model(image, size=640)

        # 物体検出中にエラーがでた場合のエラーハンドリング
        except Exception as e:
            print(f"画像 '{file}' で物体検出中にエラーが発生しました: {e}")
            continue # 次のファイルに進む

        # 検出結果の処理
        # 検出結果をPandas DataFrame形式で取得します
        detections = results.pandas().xyxy[0]  # pandas DataFrameで結果を取得

        # 各クラス名（検出された物体の種類）の出現回数をカウントします
        class_counts = detections['name'].value_counts()
        if class_counts.empty:
            writer.writerow([
                file,                   # 画像ファイル名
                'なし',                 # クラス名（‘なし’と記述）
                0,                      # 検出された本数
            ])
        else:
            for class_name, count in class_counts.items():
                writer.writerow([
                    file,               # 画像ファイル名
                    class_name,         # クラス名
                    count,              # 検出された本数
                ])

        # 検出結果の画像を表示します
        results.show()

        # 検出結果の画像をファイルとして保存します
        results.save()

print(f"検出結果が '{output_csv}' に保存されました。")
exit()
