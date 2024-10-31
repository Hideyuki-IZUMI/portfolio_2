"""
テニスゲームもどき：左矢印キー(←), 右矢印キー(→)でラケットを操作してプレーします。
"""

# tkinterモジュールのすべての機能をインポート
from tkinter import *
# winsoundモジュールをインポートし、音を鳴らす機能を使用可能に
import winsound

# ボールを表す辞書型データ
ball: dict[str, int] = {
    "dirx": 10,  # X方向のボールの速さ
    "diry": -10,  # Y方向のボールの速さ
    "x": 350,  # ボールのX座標（初期位置）
    "y": 300,  # ボールのY座標（初期位置）
    "w": 10,  # ボールの幅（半径）
}

# ラケットを表す辞書型データ
racket: dict[str, int] = {
    "x": 300,  # ラケットのX座標(初期位置)
    "y": 390,  # ラケットのY座標(固定位置)
    "w": 100,  # ラケットの幅
    "h": 5,  # ラケットの高さ
    "speed": 25,  # ラケットの移動速度
}

# ゲームオーバーフラグ
game_over: bool = False  # ゲームオーバーかどうかを判断するフラグ

# ウィンドウの作成
win: Tk = Tk()  # Tkinterのウィンドウを作成
cv: Canvas = Canvas(win, width=600, height=400)  # 描画エリア（キャンバス）を作成し、幅600、高さ400に設定
cv.pack()  # キャンバスをウィンドウに配置


# 画面を描画する
def draw_objects() -> None:
    cv.delete("all")  # 既存の描画内容をすべて消去

    # ボールを描画
    cv.create_oval(
        ball["x"] - ball["w"], ball["y"] - ball["w"],  # ボールの左上の座標
        ball["x"] + ball["w"], ball["y"] + ball["w"],  # ボールの右下の座標
        fill="red")  # ボールの色を赤に設定

    # ラケットを描画
    cv.create_rectangle(
        racket["x"] - racket["w"] // 2, racket["y"] - racket["h"] // 2,  # ラケットの左上の座標
        racket["x"] + racket["w"] // 2, racket["y"] + racket["h"] // 2,  # ラケットの右下の座標
        fill="green")  # ラケットの色を緑に設定

    # ゲームオーバーの表示
    if game_over:
        cv.create_text(300, 200, text="GAME OVER", font=("Helvetica", 40), fill="red")  # ゲームオーバーのテキストを表示


# ボールの移動を制御する関数
def move_ball() -> None:
    global game_over  # グローバル変数を参照
    if game_over:  # ゲームオーバーなら動作しない
        return

    # 仮の変数に移動後の値を記録
    bx: int = ball["x"] + ball["dirx"]  # X方向の移動後のボール位置
    by: int = ball["y"] + ball["diry"]  # Y方向の移動後のボール位置

    # 左右の壁に当たった場合
    if bx < 0 or bx > 600:
        ball["dirx"] *= -1  # X方向の進行方向を反転

    # 上の壁に当たった場合
    if by < 0:
        ball["diry"] *= -1  # Y方向の進行方向を反転

    # ラケットに当たったかをチェック
    if racket["y"] - racket["h"] // 2 <= by <= racket["y"] + racket["h"] // 2 and \
            racket["x"] - racket["w"] // 2 <= bx <= racket["x"] + racket["w"] // 2:
        ball["diry"] *= -1  # Y方向の進行方向を反転
        winsound.Beep(750, 100)  # ボールがラケットに当たった時に効果音を再生

    # 下の壁に当たった場合（ゲームオーバー）
    if by > 400:
        game_over = True  # ゲームオーバーフラグをTrueにしてゲームオーバーにする
    # 移動内容を反映
    if 0 <= bx <= 600:
        ball["x"] = bx  # ボールのX座標を更新
    if 0 <= by <= 400:
        ball["y"] = by  # ボールのY座標を更新


# ラケットの移動を制御する関数
def move_racket(event: Event) -> None:
    if game_over:  # ゲームオーバー時はラケットを動かさない
        return

    if event.keysym == "Left" and racket["x"] - racket["w"] // 2 > 0:
        racket["x"] -= racket["speed"]  # ラケットを左に移動

    elif event.keysym == "Right" and racket["x"] + racket["w"] // 2 < 600:
        racket["x"] += racket["speed"]  # ラケットを右に移動


# キーイベントを設定（左矢印と右矢印キーでラケットを操作）
win.bind("<Left>", move_racket)  # 左矢印キーが押されたらmove_racket関数を実行
win.bind("<Right>", move_racket)  # 右矢印キーが押されたらmove_racket関数を実行


# ゲームループを制御する関数
def game_loop() -> None:
    if not game_over:  # ゲームオーバーでなければ
        draw_objects()  # 画面のオブジェクトを描画
        move_ball()  # ボールを移動
        win.after(50, game_loop)  # 50ミリ秒後に再度game_loopを呼び出す
    else:
        draw_objects()  # ゲームオーバー時も最後に画面を描画


game_loop()  # ゲームループを開始
win.mainloop()  # ウィンドウを表示してイベント処理を開始
