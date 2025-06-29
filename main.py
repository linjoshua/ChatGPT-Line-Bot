from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os
import uuid
from transformers import pipeline
from src.memory import Memory
from src.logger import logger
from src.storage import Storage, FileStorage
from src.utils import get_role_and_content

# 載入 .env 設定
load_dotenv('.env')

# 初始化 Hugging Face 模型
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY", "")
chat_model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct",
    tokenizer="mistralai/Mistral-7B-Instruct",
    use_auth_token=HF_TOKEN,
    max_length=512,
    temperature=0.7,
    do_sample=True
)

# 初始化 LINE
app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

# 教學語氣
DEFAULT_SYSTEM_MESSAGE = os.getenv('SYSTEM_MESSAGE', '你是一位非常有耐心又擅長教學的老師，懂得用比喻幫助學生理解，尤其擅長處理初學者的問題。')

# 狀態與記憶體
storage = Storage(FileStorage('db.json'))
memory = Memory(system_message=DEFAULT_SYSTEM_MESSAGE, memory_message_count=2)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature.")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    logger.info(f'{user_id}: {text}')
    try:
        memory.append(user_id, "user", text)

        # 將聊天歷史轉為 prompt
        prompt = "\n".join([m["content"] for m in memory.get(user_id)])
        full_prompt = f"{DEFAULT_SYSTEM_MESSAGE}\n\n{prompt}"

        # 呼叫 Hugging Face 模型
        result = chat_model(full_prompt)[0]['generated_text']
        # 去掉原本 prompt，保留模型回答部分
        response = result.replace(full_prompt, "").strip()
        role, cleaned_response = get_role_and_content(response)

        memory.append(user_id, role, cleaned_response)
        msg = TextSendMessage(text=cleaned_response)

    except Exception as e:
        msg = TextSendMessage(text=f'⚠️ 發生錯誤：{str(e)}')

    line_bot_api.reply_message(event.reply_token, msg)

@app.route("/", methods=['GET'])
def home():
    return "Hello SB收容所 👋"

if __name__ == "__main__":
    try:
        data = storage.load()
        for user_id in data.keys():
            pass  # 暫時不需要載入舊模型資料，因為 Hugging Face 不需要 token 註冊
    except FileNotFoundError:
        pass
    app.run(host='0.0.0.0', port=8080)


