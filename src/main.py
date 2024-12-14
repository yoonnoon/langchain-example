import json
import os

import aiohttp
from fastapi import FastAPI, Request
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_xai import ChatXAI

app = FastAPI()

xai_llm_model = ChatXAI(
    xai_api_key=os.getenv("XAI_API_KEY"),
    model="grok-beta",
    temperature=0  # 답변 랜덤성 제어
)

# context
api_choice_context = """
API 목록
주식 실시간 정보 API: 
GET https://polling.finance.naver.com/api/realtime/worldstock/stock

회사 ticker 목록
테슬라: TSLA.O
엔비디아: NVDA.O
퀀텀스케이프: QS
뉴스케일파워: SMR

주식 실시간 GET 요청시 url 뒤에 ticker를 붙인다. (payload는 없음)

가상화폐 실시간 정보 API:
POST https://m.stock.naver.com/front-api/realTime/crypto

가상화폐 ticker 목록
비트코인(업비트): BTC_KRW_UPBIT
비트코인(빗썸): BTC_KRW_BITHUMB
이더리움(업비트): ETH_KRW_UPBIT
이더리움(빗썸): ETH_KRW_BITHUMB 
에이다(업비트): ADA_KRW_UPBIT
에이다(빗썸): ADA_KRW_BITHUMB
리플(업비트): XRP_KRW_UPBIT
리플(빗썸): XRP_KRW_BITHUMB 

POST 요청시 payload 구성:
fqnfTickers 필드에 ticker를 리스트로 담는다.
"""

# Prompt Template 정의
api_choice_prompt_template = """
Context:
{context}

User input: 
{user_input}

User input 이 주가를 알려달라는 요청이면 주어진 API 목록에서 ticker를 선택해서, 아래와 같은 JSON format으로 응답해줘:
{{
  "selected_url": "api",
  "selected_http_method": "http method",
  "payload": "if http method post, return payload"
  "reason": "Reason for selecting the API"
}}
"""

api_choice_prompt = PromptTemplate(
    template=api_choice_prompt_template, input_variables=["context", "user_input"]
)

json_analysis_prompt_template = """
Context:
{context}

User input:
{user_input}
"""

json_analysis_prompt = PromptTemplate(
    template=json_analysis_prompt_template, input_variables=["user_input"]
)

api_choice_llm_chain = LLMChain(llm=xai_llm_model, prompt=api_choice_prompt)
json_analysis_llm_chain = LLMChain(llm=xai_llm_model, prompt=json_analysis_prompt)


@app.post("/")
async def health(request: Request):
    body = await request.body()
    user_input = body.decode("utf-8")
    ai_response = api_choice_llm_chain.run(context=api_choice_context, user_input=user_input)
    ai_response = ai_response.removeprefix("```json")
    ai_response = ai_response.removesuffix("```")
    ai_response = json.loads(ai_response)

    async with aiohttp.ClientSession() as session:
        async with session.request(
                method=ai_response['selected_http_method'],
                url=ai_response['selected_url'],
                json=ai_response['payload'] if ai_response['payload'] else None
        ) as response:
            stock_json_data = await response.read()

    return json_analysis_llm_chain.run(context=stock_json_data, user_input=user_input)
