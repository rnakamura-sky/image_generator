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
1. StableDiffusionのモデルファイルを取得して、フォルダに格納

    Hugging Faceのstable-diffusion-v1-5を使用します。
    初めてのため、[v1-5-pruned-emaonly.safetensors]を使用します
    - [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors)
    
    取得したら、下記フォルダに格納します。
    
    `(プロジェクトフォルダ)/rsc/models/stable-diffusion`

## 実行方法

```bash
python src\image_generate.py
```
