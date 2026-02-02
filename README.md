# image_generator

生成AIを使用して画像生成するアプリケーション

基本的に、ローカルのみで実行できるようにします。
→モデル等についても実行前に自分で準備することとします。

- `StableDiffusion`を使用した画像生成を行います。
- `LoRA`を作成できる環境を目指しています。

## 開発環境について
- Windows OS

## 環境作成

1. Pythonの仮想環境を作成
    ```bash
    # 仮想環境を作成
    python -m venv venv

    # 作成した環境を有効化
    venv\Script\activate

    # 開発環境のpipを更新
    python -m pip install --upgrade pip

    # モジュールのインストール
    pip install -r requirements.txt
    ```
