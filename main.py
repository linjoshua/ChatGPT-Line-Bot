from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, AudioMessage
import os, uuid

from src.models import OpenAIModel
from src.memory import Memory
from src.logger import logger
from src.storage import Storage, FileStorage, MongoStorage
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader
from src.mongodb import mongodb

# 載入 .env 設定
load_dotenv('.env')

# 主要設定
app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

# GPT 模型名稱（預設 gpt-3.5-turbo）
DEFAULT_MODEL = os.getenv('OPENAI_MODEL_ENGINE', 'gpt-3.5-turbo')

# 預設語氣：你可以根據 SB收容所風格自訂
DEFAULT_SYSTEM_MESSAGE = os.getenv('SYSTEM_MESSAGE', '你是一位非常有耐心又擅長教學的老師，懂得用比喻幫助學生理解，尤其擅長處理初學者的問題。')

# 初始化模組
storage = None
youtube = Youtube(step=4)
website = Website()
memory = Memory(system_message=DEFAULT_SYSTEM_MESSAGE, memory_message_count=2)
model_management = {}
api_keys = {}

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    logger.info(f'{user_id}: {text}')

    try:
        if text.startswith('/註冊'):
            api_key = text[3:].strip()
            model = OpenAIModel(api_key=api_key)
            is_successful, _, _ = model.check_token_valid()
            if not is_successful:
                raise ValueError('Invalid API token')
            model_management[user_id] = model
            storage.save({user_id: api_key})
            msg = TextSendMessage(text='✅ Token 有效，註冊成功')

        elif text.startswith('/指令說明'):
            msg = TextSendMessage(text="指令：\n/註冊 + API Token\n👉 API Token 請先到 https://platform.openai.com/ 註冊登入後取得\n\n/系統訊息 + Prompt\n👉 Prompt 可以命令機器人扮演某個角色\n\n/清除\n👉 清除歷史訊息\n\n/圖像 + Prompt\n👉 以文字生成圖像\n\n語音輸入\n👉 語音轉文字後由 GPT 回答\n\n其他輸入\n👉 由 GPT 回答")

        elif text.startswith('/系統訊息'):
            memory.change_system_message(user_id, text[5:].strip())
            msg = TextSendMessage(text='✅ 系統訊息已更新')

        elif text.startswith('/清除'):
            memory.remove(user_id)
            msg = TextSendMessage(text='🧹 歷史訊息清除成功')

        elif text.startswith('/圖像'):
            prompt = text[3:].strip()
            memory.append(user_id, 'user', prompt)
            is_successful, response, error_message = model_management[user_id].image_generations(prompt)
            if not is_successful:
                raise Exception(error_message)
            url = response['data'][0]['url']
            msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
            memory.append(user_id, 'assistant', url)

        else:
            user_model = model_management[user_id]
            memory.append(user_id, 'user', text)
            url = website.get_url_from_text(text)
            if url:
                if youtube.retrieve_video_id(text):
                    is_successful, chunks, error_message = youtube.get_transcript_chunks(youtube.retrieve_video_id(text))
                    if not is_successful:
                        raise Exception(error_message)
                    reader = YoutubeTranscriptReader(user_model, DEFAULT_MODEL)
                    is_successful, response, error_message = reader.summarize(chunks)
                else:
                    chunks = website.get_content_from_url(url)
                    if not chunks:
                        raise Exception('無法撈取網站內容')
                    reader = WebsiteReader(user_model, DEFAULT_MODEL)
                    is_successful, response, error_message = reader.summarize(chunks)
            else:
                is_successful, response, error_message = user_model.chat_completions(memory.get(user_id), DEFAULT_MODEL)

            if not is_successful:
                raise Exception(error_message)

            role, response = get_role_and_content(response)
            msg = TextSendMessage(text=response)
            memory.append(user_id, role, response)

    except ValueError:
        msg = TextSendMessage(text='❌ Token 無效，請重新註冊，格式為 /註冊 sk-xxxxx')
    except KeyError:
        msg = TextSendMessage(text='⚠️ 請先註冊 Token，格式為 /註冊 sk-xxxxx')
    except Exception as e:
        memory.remove(user_id)
        if str(e).startswith('Incorrect API key provided'):
            msg = TextSendMessage(text='❌ API Token 錯誤，請重新註冊。')
        elif str(e).startswith('That model is currently overloaded'):
            msg = TextSendMessage(text='⚠️ 模型過載，請稍後再試')
        else:
            msg = TextSendMessage(text=f'⚠️ 發生錯誤：{str(e)}')

    line_bot_api.reply_message(event.reply_token, msg)

@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    user_id = event.source.user_id
    audio_content = line_bot_api.get_message_content(event.message.id)
    input_audio_path = f'{uuid.uuid4()}.m4a'
    with open(input_audio_path, 'wb') as fd:
        for chunk in audio_content.iter_content():
            fd.write(chunk)

    try:
        if not model_management.get(user_id):
            raise ValueError('Invalid API token')
        response = model_management[user_id].audio_transcriptions(input_audio_path, 'whisper-1')
        memory.append(user_id, 'user', response['text'])
        is_successful, response, error_message = model_management[user_id].chat_completions(memory.get(user_id), DEFAULT_MODEL)
        if not is_successful:
            raise Exception(error_message)
        role, response = get_role_and_content(response)
        memory.append(user_id, role, response)
        msg = TextSendMessage(text=response)
    except Exception as e:
        msg = TextSendMessage(text=f'⚠️ {str(e)}')
    finally:
        os.remove(input_audio_path)
        line_bot_api.reply_message(event.reply_token, msg)

@app.route("/", methods=['GET'])
def home():
    return 'Hello SB收容所 👋'

if __name__ == "__main__":
    if os.getenv('USE_MONGO'):
        mongodb.connect_to_database()
        storage = Storage(MongoStorage(mongodb.db))
    else:
        storage = Storage(FileStorage('db.json'))

    try:
        data = storage.load()
        for user_id in data.keys():
            model_management[user_id] = OpenAIModel(api_key=data[user_id])
    except FileNotFoundError:
        pass

    app.run(host='0.0.0.0', port=8080)

